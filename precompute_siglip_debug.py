"""
Precompute SigLIP2 embeddings for miniImageNet support images.

Output format (saved next to each image):
  <image_name>.npz with keys:
    - seq:    (196, 768)
    - pooled: (768,)
"""

import argparse
import math
import os
import time

import jax
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from encoder import SigLIP2Encoder

tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")


def _is_image_file(name):
    return name.lower().endswith((".jpg", ".jpeg", ".png"))


def _collect_images(data_dir):
    paths = []
    for root, _, files in os.walk(data_dir, followlinks=True):
        for fname in files:
            if _is_image_file(fname):
                paths.append(os.path.join(root, fname))
    paths.sort()
    return paths


def _read_image(path, image_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - 0.5) / 0.5
    return img.numpy()


def _validate_outputs(seq_embs, pooled_emb):
    if seq_embs.shape != (196, 768):
        raise ValueError(f"Expected seq shape (196, 768), got {seq_embs.shape}")
    if pooled_emb.shape != (768,):
        raise ValueError(f"Expected pooled shape (768,), got {pooled_emb.shape}")
    if not np.isfinite(seq_embs).all() or not np.isfinite(pooled_emb).all():
        raise ValueError("NaN/Inf detected in embeddings")


def _resolve_npz_path(img_path, data_dir, out_dir):
    if out_dir:
        rel = os.path.relpath(img_path, data_dir)
        npz_path = os.path.join(out_dir, os.path.splitext(rel)[0] + ".npz")
    else:
        npz_path = os.path.splitext(img_path)[0] + ".npz"
    return npz_path


def main():
    parser = argparse.ArgumentParser(description="Precompute SigLIP2 sequence+pooled embeddings.")
    parser.add_argument("--data_dir", required=True, help="Root folder containing class folders.")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Optional writable root for .npz cache. If unset, saves next to images.",
    )
    parser.add_argument("--image_size", type=int, default=224, help="SigLIP resolution.")
    parser.add_argument("--variant", default="B/16", help="SigLIP2 vision variant.")
    parser.add_argument("--ckpt_path", default=None, help="Optional local SigLIP checkpoint .npz.")
    parser.add_argument("--batch_size", type=int, default=256, help="Global encode batch size.")
    parser.add_argument(
        "--no_pmap",
        action="store_true",
        help="Disable pmap. By default, pmap is used when multiple local devices exist.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=("float16", "float32"),
        help="Saved embedding dtype.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing .npz files.")
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue when a file fails. Default is fail-fast.",
    )
    args = parser.parse_args()

    t0 = time.time()
    np_dtype = np.float16 if args.dtype == "float16" else np.float32
    image_paths = _collect_images(args.data_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {args.data_dir}")

    print(f"Found {len(image_paths)} images under {args.data_dir}")
    print("Loading SigLIP2 encoder...")
    siglip = SigLIP2Encoder.create(
        ckpt_path=args.ckpt_path,
        variant=args.variant,
        res=args.image_size,
    )

    n_dev = jax.local_device_count()
    use_pmap = (not args.no_pmap) and n_dev > 1
    if use_pmap:
        global_bs = int(math.ceil(args.batch_size / n_dev) * n_dev)
        per_dev_bs = global_bs // n_dev
    else:
        global_bs = args.batch_size
        per_dev_bs = None

    print(
        f"JAX backend: {jax.default_backend()} | local_devices={n_dev} | "
        f"pmap={'on' if use_pmap else 'off'} | global_batch={global_bs}"
    )
    if use_pmap:
        print(f"Per-device batch: {per_dev_bs}")

    if use_pmap:
        @jax.pmap
        def _encode_fn(x):
            return siglip._encode_both(x)
    else:
        _encode_fn = jax.jit(siglip._encode_both)

    # Compile once with a fixed shape for stable throughput.
    warmup = np.zeros((global_bs, args.image_size, args.image_size, 3), dtype=np.float32)
    if use_pmap:
        warmup = warmup.reshape(n_dev, per_dev_bs, args.image_size, args.image_size, 3)
    _ = _encode_fn(warmup)

    work = []
    done = 0
    skipped = 0
    failed = 0
    for img_path in image_paths:
        npz_path = _resolve_npz_path(img_path, args.data_dir, args.out_dir)
        if os.path.exists(npz_path) and not args.overwrite:
            skipped += 1
            continue
        if args.out_dir:
            os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        work.append((img_path, npz_path))

    if not work:
        print("All embeddings already exist. Nothing to do.")
        return

    for st in tqdm(range(0, len(work), global_bs), dynamic_ncols=True):
        chunk = work[st:st + global_bs]
        imgs = np.zeros((global_bs, args.image_size, args.image_size, 3), dtype=np.float32)
        valid_slots = []
        slot_meta = []

        for slot, (img_path, npz_path) in enumerate(chunk):
            slot_meta.append((img_path, npz_path))
            try:
                imgs[slot] = _read_image(img_path, args.image_size)
                valid_slots.append(slot)
            except Exception as exc:
                failed += 1
                print(f"[ERROR] decode {img_path}: {exc}")
                if not args.continue_on_error:
                    raise

        if not valid_slots:
            continue

        try:
            if use_pmap:
                inp = imgs.reshape(n_dev, per_dev_bs, args.image_size, args.image_size, 3)
                seq_all, pooled_all = _encode_fn(inp)
                seq_all = np.asarray(jax.device_get(seq_all)).reshape(global_bs, 196, 768)
                pooled_all = np.asarray(jax.device_get(pooled_all)).reshape(global_bs, 768)
            else:
                seq_all, pooled_all = _encode_fn(imgs)
                seq_all = np.asarray(jax.device_get(seq_all))
                pooled_all = np.asarray(jax.device_get(pooled_all))
        except Exception as exc:
            failed += len(valid_slots)
            print(f"[ERROR] encode batch starting at idx={st}: {exc}")
            if not args.continue_on_error:
                raise
            continue

        for slot in valid_slots:
            img_path, npz_path = slot_meta[slot]
            try:
                seq_embs = seq_all[slot]
                pooled_emb = pooled_all[slot]
                _validate_outputs(seq_embs, pooled_emb)
                np.savez(
                    npz_path,
                    seq=seq_embs.astype(np_dtype),
                    pooled=pooled_emb.astype(np_dtype),
                )
                done += 1
            except Exception as exc:
                failed += 1
                print(f"[ERROR] save {img_path}: {exc}")
                if not args.continue_on_error:
                    raise

    dt = time.time() - t0
    print(
        f"Done in {dt:.1f}s | saved={done}, skipped={skipped}, failed={failed}, "
        f"dtype={args.dtype}"
    )


if __name__ == "__main__":
    main()
