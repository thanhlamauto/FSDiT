"""
Build episode-level TFRecord shards for miniImageNet episodes.

Two input modes:
1) NPZ mode (default): read support embeddings from --embeddings_dir/{split}/*.npz.
2) Direct mode: set --direct_from_images=1 to encode SigLIP2 from support images
   and write TFRecord directly (no intermediate .npz cache).

Each record stores:
  - target_path (string)
  - class_id (int)
  - supports_pooled (bytes; float16 [5,768])
  - supports_seq (bytes; float16 [5,196,768]) if --store_seq=1 else empty bytes
"""

import argparse
import json
import math
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dataset import build_episode_table, _interleave_by_class


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def _npz_path_for(image_path, split_dir, embedding_split_dir):
    rel = os.path.relpath(image_path, split_dir)
    return os.path.join(embedding_split_dir, os.path.splitext(rel)[0] + ".npz")


def _read_image(path, image_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - 0.5) / 0.5
    return img.numpy()


def _load_support_arrays_from_npz(support_paths, split_dir, embedding_split_dir, store_seq):
    pooled_list = []
    seq_list = []
    for p in support_paths:
        npz_path = _npz_path_for(p, split_dir, embedding_split_dir)
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Missing embedding npz: {npz_path}")
        d = np.load(npz_path)
        pooled = d["pooled"].astype(np.float16, copy=False)
        if pooled.shape != (768,):
            raise ValueError(f"Invalid pooled shape in {npz_path}: {pooled.shape}")
        pooled_list.append(pooled)
        if store_seq:
            seq = d["seq"].astype(np.float16, copy=False)
            if seq.shape != (196, 768):
                raise ValueError(f"Invalid seq shape in {npz_path}: {seq.shape}")
            seq_list.append(seq)

    pooled_arr = np.stack(pooled_list, axis=0)  # (5, 768)
    if store_seq:
        seq_arr = np.stack(seq_list, axis=0)     # (5, 196, 768)
    else:
        seq_arr = None
    return pooled_arr, seq_arr


class DirectSiglipEncoder:
    """Encode support images directly via SigLIP2, batched for TFRecord export."""

    def __init__(
        self,
        variant="B/16",
        image_size=224,
        ckpt_path=None,
        batch_size=256,
        no_pmap=False,
        store_seq=True,
    ):
        import jax
        from encoder import SigLIP2Encoder

        self.jax = jax
        self.image_size = image_size
        self.store_seq = store_seq
        self.n_dev = jax.local_device_count()
        self.use_pmap = (not no_pmap) and self.n_dev > 1
        if self.use_pmap:
            self.global_bs = int(math.ceil(batch_size / self.n_dev) * self.n_dev)
            self.per_dev_bs = self.global_bs // self.n_dev
        else:
            self.global_bs = batch_size
            self.per_dev_bs = None

        print(
            f"[Direct SigLIP] variant={variant} res={image_size} "
            f"pmap={'on' if self.use_pmap else 'off'} global_batch={self.global_bs}"
        )
        self.siglip = SigLIP2Encoder.create(ckpt_path=ckpt_path, variant=variant, res=image_size)
        if store_seq:
            if self.use_pmap:
                @jax.pmap
                def _fn(x):
                    return self.siglip._encode_both(x)
            else:
                _fn = jax.jit(self.siglip._encode_both)
        else:
            if self.use_pmap:
                @jax.pmap
                def _fn(x):
                    return self.siglip._encode_batch(x)
            else:
                _fn = jax.jit(self.siglip._encode_batch)
        self._fn = _fn

        warmup = np.zeros((self.global_bs, image_size, image_size, 3), dtype=np.float32)
        _ = self._call_model(warmup)

    def _call_model(self, images):
        if self.use_pmap:
            inp = images.reshape(
                self.n_dev, self.per_dev_bs, self.image_size, self.image_size, 3
            )
            out = self._fn(inp)
            if self.store_seq:
                seq, pooled = out
                seq = np.asarray(self.jax.device_get(seq)).reshape(self.global_bs, 196, 768)
                pooled = np.asarray(self.jax.device_get(pooled)).reshape(self.global_bs, 768)
                return seq, pooled
            pooled = np.asarray(self.jax.device_get(out)).reshape(self.global_bs, 768)
            return None, pooled

        out = self._fn(images)
        if self.store_seq:
            seq, pooled = out
            return (
                np.asarray(self.jax.device_get(seq)),
                np.asarray(self.jax.device_get(pooled)),
            )
        return None, np.asarray(self.jax.device_get(out))

    def encode_paths(self, image_paths):
        n = len(image_paths)
        if n == 0:
            if self.store_seq:
                return (
                    np.zeros((0, 196, 768), dtype=np.float16),
                    np.zeros((0, 768), dtype=np.float16),
                )
            return None, np.zeros((0, 768), dtype=np.float16)

        seq_chunks = []
        pooled_chunks = []
        for st in range(0, n, self.global_bs):
            chunk = image_paths[st:st + self.global_bs]
            imgs = np.zeros((self.global_bs, self.image_size, self.image_size, 3), dtype=np.float32)
            for i, p in enumerate(chunk):
                try:
                    imgs[i] = _read_image(p, self.image_size)
                except Exception as exc:
                    raise RuntimeError(f"Failed to decode support image: {p}") from exc

            seq_all, pooled_all = self._call_model(imgs)
            valid = len(chunk)
            pooled_chunks.append(pooled_all[:valid].astype(np.float16, copy=False))
            if self.store_seq:
                seq_chunks.append(seq_all[:valid].astype(np.float16, copy=False))

        pooled = np.concatenate(pooled_chunks, axis=0)
        if self.store_seq:
            seq = np.concatenate(seq_chunks, axis=0)
            return seq, pooled
        return None, pooled


def build_split(
    split,
    data_dir,
    embeddings_dir,
    out_dir,
    num_sets,
    seed,
    num_shards,
    store_seq,
    compression,
    direct_encoder=None,
    encode_batch_size=256,
):
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        print(f"[Skip] split '{split}' not found at {split_dir}")
        return

    use_direct = direct_encoder is not None
    embedding_split_dir = None
    if not use_direct:
        embedding_split_dir = os.path.join(embeddings_dir, split)
        if not os.path.isdir(embedding_split_dir):
            raise FileNotFoundError(f"Missing embedding split dir: {embedding_split_dir}")

    episodes, class_names = build_episode_table(split_dir, num_sets=num_sets, seed=seed)
    episodes = _interleave_by_class(episodes, len(class_names), seed + 1)
    print(f"[{split}] classes={len(class_names)} episodes={len(episodes)}")

    split_out = os.path.join(out_dir, split)
    os.makedirs(split_out, exist_ok=True)

    writers = []
    shard_paths = []
    tf_opts = tf.io.TFRecordOptions(compression_type=compression) if compression else None
    for i in range(num_shards):
        p = os.path.join(split_out, f"{split}-{i:05d}-of-{num_shards:05d}.tfrecord")
        shard_paths.append(p)
        if tf_opts:
            writers.append(tf.io.TFRecordWriter(p, options=tf_opts))
        else:
            writers.append(tf.io.TFRecordWriter(p))

    episodes_per_chunk = 1
    if use_direct:
        episodes_per_chunk = max(1, encode_batch_size // 5)

    pbar = tqdm(total=len(episodes), desc=f"write-{split}", dynamic_ncols=True)
    try:
        for st in range(0, len(episodes), episodes_per_chunk):
            ep_chunk = episodes[st:st + episodes_per_chunk]
            seq_flat = None
            pooled_flat = None
            if use_direct:
                support_flat = [p for _, support_paths, _ in ep_chunk for p in support_paths]
                seq_flat, pooled_flat = direct_encoder.encode_paths(support_flat)

            for i, (target_path, support_paths, class_id) in enumerate(ep_chunk):
                if use_direct:
                    off = i * 5
                    pooled_arr = pooled_flat[off:off + 5]
                    seq_arr = seq_flat[off:off + 5] if store_seq else None
                else:
                    pooled_arr, seq_arr = _load_support_arrays_from_npz(
                        support_paths, split_dir, embedding_split_dir, store_seq=store_seq
                    )

                seq_bytes = seq_arr.tobytes(order="C") if store_seq else b""
                ex = tf.train.Example(features=tf.train.Features(feature={
                    "target_path": _bytes_feature(target_path.encode("utf-8")),
                    "class_id": _int64_feature(class_id),
                    "supports_pooled": _bytes_feature(pooled_arr.tobytes(order="C")),
                    "supports_seq": _bytes_feature(seq_bytes),
                }))
                idx = st + i
                writers[idx % num_shards].write(ex.SerializeToString())
            pbar.update(len(ep_chunk))
    finally:
        pbar.close()
        for w in writers:
            w.close()

    meta = {
        "split": split,
        "num_classes": len(class_names),
        "num_episodes": len(episodes),
        "num_sets": num_sets,
        "seed": seed,
        "num_shards": num_shards,
        "store_seq": bool(store_seq),
        "compression": compression,
        "source_mode": "direct_from_images" if use_direct else "npz",
        "shards": shard_paths,
    }
    with open(os.path.join(split_out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Build TFRecord episode shards for FSDiT.")
    parser.add_argument("--data_dir", required=True, help="miniImageNet split root with train/val/test.")
    parser.add_argument(
        "--embeddings_dir",
        default=None,
        help="Embedding root with train/val/test .npz (required in NPZ mode).",
    )
    parser.add_argument("--out_dir", required=True, help="Output root for TFRecord shards.")
    parser.add_argument("--splits", default="train,val", help="Comma-separated splits to export.")
    parser.add_argument("--num_sets", type=int, default=100, help="Sets per class (must match training setup).")
    parser.add_argument("--seed", type=int, default=42, help="Episode sampling seed.")
    parser.add_argument("--num_shards", type=int, default=64, help="Number of shards per split.")
    parser.add_argument("--store_seq", type=int, default=1, help="1=store support seq, 0=pooled-only.")
    parser.add_argument("--compression", default="GZIP", choices=["", "GZIP"], help="TFRecord compression.")
    parser.add_argument(
        "--direct_from_images",
        type=int,
        default=0,
        help="1=encode SigLIP from support images directly to TFRecord (no .npz).",
    )
    parser.add_argument("--variant", default="B/16", help="SigLIP2 variant for direct mode.")
    parser.add_argument("--image_size", type=int, default=224, help="SigLIP image size for direct mode.")
    parser.add_argument("--ckpt_path", default=None, help="Optional local SigLIP checkpoint .npz.")
    parser.add_argument("--encode_batch_size", type=int, default=256, help="Direct mode encode batch size.")
    parser.add_argument(
        "--no_pmap",
        action="store_true",
        help="Disable pmap in direct mode. By default pmap is used when possible.",
    )
    args = parser.parse_args()

    direct_mode = bool(args.direct_from_images)
    if not direct_mode and not args.embeddings_dir:
        parser.error("--embeddings_dir is required unless --direct_from_images=1.")

    os.makedirs(args.out_dir, exist_ok=True)
    split_list = [s.strip() for s in args.splits.split(",") if s.strip()]

    direct_encoder = None
    if direct_mode:
        direct_encoder = DirectSiglipEncoder(
            variant=args.variant,
            image_size=args.image_size,
            ckpt_path=args.ckpt_path,
            batch_size=args.encode_batch_size,
            no_pmap=args.no_pmap,
            store_seq=bool(args.store_seq),
        )

    for split in split_list:
        build_split(
            split=split,
            data_dir=args.data_dir,
            embeddings_dir=args.embeddings_dir,
            out_dir=args.out_dir,
            num_sets=args.num_sets,
            seed=args.seed,
            num_shards=args.num_shards,
            store_seq=bool(args.store_seq),
            compression=args.compression,
            direct_encoder=direct_encoder,
            encode_batch_size=args.encode_batch_size,
        )
    print("Done.")


if __name__ == "__main__":
    main()
