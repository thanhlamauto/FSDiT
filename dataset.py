"""
dataset.py — miniImageNet few-shot episode loader.

Each class → 100 sets of 6 images → 6 rotations (1 target + 5 support).
Stratified sampling ensures balanced class representation per batch.
"""

import os
import numpy as np
import tensorflow as tf


def build_episode_table(data_dir, num_sets=100, seed=42):
    """
    Scan class folders, generate all (target, supports, class_id) episodes.

    Returns:
        episodes: list of (target_path, [5 support_paths], class_idx)
        class_names: sorted list of class folder names
    """
    rng = np.random.RandomState(seed)
    class_dirs = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )

    episodes = []
    for cls_idx, cls_name in enumerate(class_dirs):
        cls_path = os.path.join(data_dir, cls_name)
        imgs = sorted(
            os.path.join(cls_path, f)
            for f in os.listdir(cls_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG'))
        )
        assert len(imgs) >= 6, f"Class '{cls_name}' has {len(imgs)} images (need ≥ 6)"

        for _ in range(num_sets):
            chosen = [imgs[i] for i in rng.choice(len(imgs), 6, replace=False)]
            for rot in range(6):
                target = chosen[rot]
                supports = [chosen[j] for j in range(6) if j != rot]
                episodes.append((target, supports, cls_idx))

    return episodes, class_dirs


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


def build_dataset(
    data_dir, batch_size, image_size=224, num_sets=100,
    is_train=True, seed=42, debug_n=0,
):
    """
    Build tf.data pipeline for FSDiT training.

    Returns:
        dataset: yields {'target': (B,H,W,3), 'supports': (B,5,H,W,3), 'class_id': (B,)}
        class_names: list of class names
    """
    episodes, class_names = build_episode_table(data_dir, num_sets, seed)
    n_cls = len(class_names)
    n_ep = len(episodes)
    print(f"[Dataset] {data_dir}: {n_cls} classes, {n_ep} episodes")

    if debug_n > 0:
        episodes = episodes[:debug_n]

    episodes = _interleave_by_class(episodes, n_cls, seed + 1)

    # Build tensor slices
    targets = [e[0] for e in episodes]
    supports_flat = []
    for e in episodes:
        supports_flat.extend(e[1])
    class_ids = [e[2] for e in episodes]

    ds = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(targets),
        tf.data.Dataset.from_tensor_slices(tf.reshape(tf.constant(supports_flat), [-1, 5])),
        tf.data.Dataset.from_tensor_slices(tf.constant(class_ids, dtype=tf.int32)),
    ))

    def load_sample(target_path, support_paths, class_id):
        def read_img(path):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [image_size, image_size])
            img = tf.cast(img, tf.float32) / 255.0
            return (img - 0.5) / 0.5  # → [-1, 1]

        target = read_img(target_path)
        if is_train:
            target = tf.image.random_flip_left_right(target)
        supports = tf.map_fn(read_img, support_paths, fn_output_signature=tf.float32)
        return {'target': target, 'supports': supports, 'class_id': class_id}

    ds = ds.map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.repeat()
    if not debug_n:
        ds = ds.shuffle(min(len(episodes), n_cls * 50), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, class_names
