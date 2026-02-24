"""
Shard a large ArrayRecord file into multiple smaller files.

Example:
  python shard_arrayrecord.py \
      --src /workspace/data/episodes_arecord/train.arecord \
      --out_dir /workspace/data/episodes_arecord \
      --num_shards 3

This will create:
  /workspace/data/episodes_arecord/train_shard_000.arecord
  /workspace/data/episodes_arecord/train_shard_001.arecord
  /workspace/data/episodes_arecord/train_shard_002.arecord
"""

import argparse
import json
import os

from array_record.python.array_record_module import ArrayRecordReader, ArrayRecordWriter


def shard_arrayrecord(src_path: str, out_dir: str, num_shards: int) -> None:
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"Source arecord not found: {src_path}")
    os.makedirs(out_dir, exist_ok=True)

    reader = ArrayRecordReader(src_path)
    num_records = reader.num_records()
    if num_records == 0:
        raise ValueError(f"Source arecord is empty: {src_path}")

    print(f"Source: {src_path}")
    print(f"Total records: {num_records}")
    print(f"Num shards: {num_shards}")

    # Evenly split by number of records
    records_per_shard = (num_records + num_shards - 1) // num_shards

    meta = {
        "source": os.path.basename(src_path),
        "num_records": int(num_records),
        "num_shards": int(num_shards),
        "shards": [],
    }

    for shard_idx in range(num_shards):
        start = shard_idx * records_per_shard
        end = min(start + records_per_shard, num_records)
        if start >= end:
            break

        shard_name = f"train_shard_{shard_idx:03d}.arecord"
        shard_path = os.path.join(out_dir, shard_name)
        print(f"[shard {shard_idx}] writing records [{start}, {end}) -> {shard_path}")

        writer = ArrayRecordWriter(shard_path, "group_size:1")
        written = 0
        for idx in range(start, end):
            raw_record = reader.read([idx])[0]
            writer.write(raw_record)
            written += 1
            if written % 10000 == 0:
                print(f"  shard {shard_idx}: {written}/{end-start} records")
        writer.close()

        meta["shards"].append(
            {
                "index": int(shard_idx),
                "path": shard_name,
                "start_idx": int(start),
                "end_idx": int(end),
                "num_records": int(end - start),
            }
        )

    # Save simple metadata next to shards
    meta_path = os.path.join(out_dir, "train_shards_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote shard metadata to {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Shard a large ArrayRecord into smaller files.")
    parser.add_argument("--src", required=True, help="Path to source .arecord (e.g., train.arecord).")
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for sharded .arecord files.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=3,
        help="Number of shards to create (3 keeps each well below 20GB for ~21GB source).",
    )
    args = parser.parse_args()

    shard_arrayrecord(args.src, args.out_dir, args.num_shards)


if __name__ == "__main__":
    main()

