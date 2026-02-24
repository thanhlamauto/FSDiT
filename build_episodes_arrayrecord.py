"""
Build episode-level ArrayRecord files for miniImageNet episodes.

Reads precomputed .npz embeddings (from precompute_siglip_pytorch.py or
precompute_siglip_debug.py) and packs them into ArrayRecord files.

Each record stores (serialized via msgpack):
  - target_path (string)
  - class_id (int)
  - supports_pooled (bytes; float16 [5,768])
  - supports_seq (bytes; float16 [5,196,768]) if --store_seq=1 else empty bytes

Usage:
  python build_episodes_arrayrecord.py \
      --data_dir /path/to/miniimagenet \
      --embeddings_dir /path/to/miniimagenet \
      --out_dir /path/to/episodes_arecord \
      --splits train,val \
      --store_seq 1
"""

import argparse
import json
import os

import msgpack
import numpy as np
from tqdm import tqdm

from dataset import build_episode_table, _interleave_by_class

try:
    from array_record.python.array_record_module import ArrayRecordWriter
except ImportError:
    raise ImportError(
        "array_record is required. Install with: pip install array_record"
    )


def _npz_path_for(image_path, split_dir, embedding_split_dir):
    rel = os.path.relpath(image_path, split_dir)
    return os.path.join(embedding_split_dir, os.path.splitext(rel)[0] + ".npz")


def _load_support_arrays(support_paths, split_dir, embedding_split_dir, store_seq):
    """Load precomputed embeddings for 5 support images."""
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
        seq_arr = np.stack(seq_list, axis=0)  # (5, 196, 768)
    else:
        seq_arr = None
    return pooled_arr, seq_arr


def _serialize_record(target_path, class_id, supports_pooled, supports_seq):
    """Serialize a single episode record using msgpack."""
    record = {
        "target_path": target_path,
        "class_id": int(class_id),
        "supports_pooled": supports_pooled.tobytes(),
        "supports_seq": supports_seq.tobytes() if supports_seq is not None else b"",
    }
    return msgpack.packb(record, use_bin_type=True)


def build_split(
    split,
    data_dir,
    embeddings_dir,
    out_dir,
    num_sets,
    seed,
    store_seq,
):
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        print(f"[Skip] split '{split}' not found at {split_dir}")
        return

    embedding_split_dir = os.path.join(embeddings_dir, split)
    if not os.path.isdir(embedding_split_dir):
        raise FileNotFoundError(
            f"Missing embedding split dir: {embedding_split_dir}"
        )

    episodes, class_names = build_episode_table(split_dir, num_sets=num_sets, seed=seed)
    episodes = _interleave_by_class(episodes, len(class_names), seed + 1)
    print(f"[{split}] classes={len(class_names)} episodes={len(episodes)}")

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    arecord_path = os.path.join(out_dir, f"{split}.arecord")

    # Write ArrayRecord
    writer = ArrayRecordWriter(arecord_path, "group_size:1")
    n_written = 0

    for target_path, support_paths, class_id in tqdm(
        episodes, desc=f"write-{split}", dynamic_ncols=True
    ):
        pooled_arr, seq_arr = _load_support_arrays(
            support_paths, split_dir, embedding_split_dir, store_seq=store_seq
        )
        record_bytes = _serialize_record(
            target_path, class_id, pooled_arr, seq_arr
        )
        writer.write(record_bytes)
        n_written += 1

    writer.close()

    # Write metadata
    meta = {
        "split": split,
        "num_classes": len(class_names),
        "num_episodes": len(episodes),
        "num_sets": num_sets,
        "seed": seed,
        "store_seq": bool(store_seq),
        "format": "arrayrecord",
        "serialization": "msgpack",
        "arecord_file": os.path.basename(arecord_path),
        "records_written": n_written,
    }
    meta_path = os.path.join(out_dir, f"{split}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    file_size_mb = os.path.getsize(arecord_path) / (1024 * 1024)
    print(f"[{split}] Written {n_written} records → {arecord_path} ({file_size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Build ArrayRecord episode files for FSDiT."
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="miniImageNet split root with train/val/test.",
    )
    parser.add_argument(
        "--embeddings_dir", required=True,
        help="Embedding root with train/val/test .npz files.",
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="Output directory for ArrayRecord files.",
    )
    parser.add_argument(
        "--splits", default="train,val",
        help="Comma-separated splits to export.",
    )
    parser.add_argument(
        "--num_sets", type=int, default=100,
        help="Sets per class (must match training setup).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Episode sampling seed.",
    )
    parser.add_argument(
        "--store_seq", type=int, default=1,
        help="1=store support seq tokens, 0=pooled-only.",
    )
    args = parser.parse_args()

    split_list = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split in split_list:
        build_split(
            split=split,
            data_dir=args.data_dir,
            embeddings_dir=args.embeddings_dir,
            out_dir=args.out_dir,
            num_sets=args.num_sets,
            seed=args.seed,
            store_seq=bool(args.store_seq),
        )

    print("Done.")


if __name__ == "__main__":
    main()
