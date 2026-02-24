"""
dataset.py — miniImageNet few-shot episode loader.

Supports two runtime modes:
1) online   : returns target image + support_paths (SigLIP is encoded in train loop)
2) tfrecord : returns target image + precomputed support embeddings
"""

import os
import numpy as np
import tensorflow as tf

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _scan_class_images(data_dir):
    class_dirs = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    class_to_images = []
    for cls_name in class_dirs:
        cls_path = os.path.join(data_dir, cls_name)
        imgs = sorted(
            os.path.join(cls_path, f)
            for f in os.listdir(cls_path)
            if f.endswith(_IMG_EXTS)
        )
        if len(imgs) < 6:
            raise ValueError(f"Class '{cls_name}' has {len(imgs)} images (need >= 6).")
        class_to_images.append(imgs)
    return class_dirs, class_to_images


def build_episode_table(data_dir, num_sets=100, seed=42):
    """
    Scan class folders, generate all (target, supports, class_id) episodes.

    Returns:
        episodes: list of (target_path, [5 support_paths], class_idx)
        class_names: sorted list of class folder names
    """
    rng = np.random.RandomState(seed)
    class_names, class_to_images = _scan_class_images(data_dir)

    episodes = []
    for cls_idx, imgs in enumerate(class_to_images):
        for _ in range(num_sets):
            chosen = [imgs[i] for i in rng.choice(len(imgs), 6, replace=False)]
            for rot in range(6):
                target = chosen[rot]
                supports = [chosen[j] for j in range(6) if j != rot]
                episodes.append((target, supports, cls_idx))
    return episodes, class_names


def _interleave_by_class(episodes, num_classes, seed):
    """Round-robin interleave episodes across classes for balanced batching."""
    rng = np.random.RandomState(seed)
    buckets = {c: [] for c in range(num_classes)}
    for ep in episodes:
        buckets[ep[2]].append(ep)
    for c in range(num_classes):
        rng.shuffle(buckets[c])

    result = []
    max_len = max(len(v) for v in buckets.values())
    for i in range(max_len):
        for c in range(num_classes):
            if i < len(buckets[c]):
                result.append(buckets[c][i])
    return result


def _decode_target(path, image_size, is_train):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - 0.5) / 0.5  # [-1, 1]
    if is_train:
        img = tf.image.random_flip_left_right(img)
    return img


def _episodes_to_arrays(episodes):
    target_paths = np.asarray([e[0] for e in episodes], dtype=np.bytes_)
    support_paths = np.asarray([e[1] for e in episodes], dtype=np.bytes_)
    class_ids = np.asarray([e[2] for e in episodes], dtype=np.int32)
    return target_paths, support_paths, class_ids


def _sample_online_episode(rng, class_to_images):
    cls_idx = int(rng.randint(len(class_to_images)))
    imgs = class_to_images[cls_idx]
    chosen = [imgs[i] for i in rng.choice(len(imgs), 6, replace=False)]
    target_idx = int(rng.randint(6))
    target = chosen[target_idx]
    supports = [chosen[j] for j in range(6) if j != target_idx]
    return np.bytes_(target), np.asarray(supports, dtype=np.bytes_), np.int32(cls_idx)


def _build_online_dataset(
    data_dir, batch_size, image_size=224, num_sets=100,
    is_train=True, seed=42, debug_n=0,
):
    class_names, class_to_images = _scan_class_images(data_dir)

    if is_train and not debug_n:
        print(
            f"[Dataset] online train: {len(class_names)} classes, dynamic episodes, seed={seed}"
        )

        def gen():
            rng = np.random.RandomState(seed)
            while True:
                yield _sample_online_episode(rng, class_to_images)

        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(5,), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )
    else:
        if is_train:
            rng = np.random.RandomState(seed)
            n_debug = max(int(debug_n), batch_size)
            episodes = [_sample_online_episode(rng, class_to_images) for _ in range(n_debug)]
            print(
                f"[Dataset] online train(debug): {len(class_names)} classes, "
                f"{len(episodes)} cached episodes"
            )
        else:
            episodes, _ = build_episode_table(data_dir, num_sets=num_sets, seed=seed)
            episodes = _interleave_by_class(episodes, len(class_names), seed + 1)
            if debug_n:
                episodes = episodes[:int(debug_n)]
            print(
                f"[Dataset] online val: {len(class_names)} classes, {len(episodes)} fixed episodes"
            )
        target_paths, support_paths, class_ids = _episodes_to_arrays(episodes)
        ds = tf.data.Dataset.from_tensor_slices((target_paths, support_paths, class_ids)).repeat()

    def parse_online(target_path, support_paths, class_id):
        return {
            "target": _decode_target(target_path, image_size=image_size, is_train=is_train),
            "support_paths": support_paths,
            "class_id": tf.cast(class_id, tf.int32),
        }

    ds = ds.map(parse_online, num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        options = tf.data.Options()
        options.experimental_deterministic = False
        ds = ds.with_options(options)
        if debug_n:
            ds = ds.shuffle(max(int(debug_n), 512), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, class_names


def _build_tfrecord_dataset(
    data_dir, batch_size, image_size=224, num_sets=100,
    is_train=True, seed=42, debug_n=0, load_support_seq=True,
    episode_tfrecord_pattern=None, tfrecord_compression_type="",
):
    del data_dir, num_sets
    if not episode_tfrecord_pattern:
        raise ValueError("`episode_tfrecord_pattern` is required when data_mode='tfrecord'.")

    files = tf.io.gfile.glob(episode_tfrecord_pattern)
    if not files:
        raise FileNotFoundError(f"No TFRecord files matched pattern: {episode_tfrecord_pattern}")
    print(f"[Dataset] tfrecord: {len(files)} shards from {episode_tfrecord_pattern}")

    ds = tf.data.TFRecordDataset(
        files,
        compression_type=tfrecord_compression_type or None,
        num_parallel_reads=tf.data.AUTOTUNE,
    )

    feature_spec = {
        "target_path": tf.io.FixedLenFeature([], tf.string),
        "class_id": tf.io.FixedLenFeature([], tf.int64),
        "supports_pooled": tf.io.FixedLenFeature([], tf.string),
        "supports_seq": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    }

    def parse_example(example_proto):
        ex = tf.io.parse_single_example(example_proto, feature_spec)
        target = _decode_target(ex["target_path"], image_size=image_size, is_train=is_train)
        supports_pooled = tf.io.decode_raw(ex["supports_pooled"], tf.float16)
        supports_pooled = tf.reshape(supports_pooled, [5, 768])

        if load_support_seq:
            has_seq = tf.greater(tf.strings.length(ex["supports_seq"]), 0)
            supports_seq = tf.cond(
                has_seq,
                lambda: tf.reshape(tf.io.decode_raw(ex["supports_seq"], tf.float16), [5, 196, 768]),
                lambda: tf.zeros([5, 196, 768], dtype=tf.float16),
            )
        else:
            supports_seq = tf.zeros([5, 196, 768], dtype=tf.float16)

        return {
            "target": target,
            "supports_seq": supports_seq,
            "supports_pooled": supports_pooled,
            "class_id": tf.cast(ex["class_id"], tf.int32),
        }

    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        options = tf.data.Options()
        options.experimental_deterministic = False
        ds = ds.with_options(options)
    ds = ds.repeat()
    if not debug_n:
        ds = ds.shuffle(8192, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, []


def build_dataset(
    data_dir, batch_size, image_size=224, num_sets=100,
    is_train=True, seed=42, debug_n=0, load_support_seq=True,
    data_mode="online", episode_tfrecord_pattern=None, tfrecord_compression_type="",
):
    """
    Build tf.data pipeline for FSDiT.

    Returns:
      online mode:
        {
          'target': (B,H,W,3) float32,
          'support_paths': (B,5) tf.string,
          'class_id': (B,) int32,
        }
      tfrecord mode:
        {
          'target': (B,H,W,3),
          'supports_seq': (B,5,196,768),
          'supports_pooled': (B,5,768),
          'class_id': (B,),
        }
    """
    if data_mode == "online":
        return _build_online_dataset(
            data_dir=data_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_sets=num_sets,
            is_train=is_train,
            seed=seed,
            debug_n=debug_n,
        )
    if data_mode == "tfrecord":
        return _build_tfrecord_dataset(
            data_dir=data_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_sets=num_sets,
            is_train=is_train,
            seed=seed,
            debug_n=debug_n,
            load_support_seq=load_support_seq,
            episode_tfrecord_pattern=episode_tfrecord_pattern,
            tfrecord_compression_type=tfrecord_compression_type,
        )
    raise ValueError(f"Unsupported data_mode: {data_mode}. Expected 'online' or 'tfrecord'.")
