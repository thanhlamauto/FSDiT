"""
dataset_grain.py — Grain-based dataset loader for FSDiT.

Reads episode-level ArrayRecord files (single or sharded) and provides
a Grain pipeline. Drop-in replacement for the tfrecord path in dataset.py.

Supports two layouts:
  1) Single file:   train.arecord
  2) Sharded files:  train_shard_000.arecord, train_shard_001.arecord, ...

Output contract (matches tfrecord mode):
  {
      'target': (B, H, W, 3) float32,
      'supports_pooled': (B, 5, 768) float16,
      'supports_seq': (B, 5, 196, 768) float16,
      'class_id': (B,) int32,
  }
"""

import glob
import os
from typing import List

import grain
import msgpack
import numpy as np
from PIL import Image

try:
    from array_record.python.array_record_module import ArrayRecordReader
except ImportError:
    raise ImportError(
        "array_record is required. Install with: pip install array_record"
    )


# ─── Image decode (PIL-based, no TF dependency) ──────────────────────────────

def _decode_image(path, image_size, is_train, rng=None):
    """Read, resize, normalize image. Optionally flip for training."""
    if isinstance(path, bytes):
        path = path.decode("utf-8")
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # → [-1, 1]
    if is_train and rng is not None and rng.random() < 0.5:
        arr = arr[:, ::-1, :]  # horizontal flip
    return arr


# ─── ArrayRecord data sources ────────────────────────────────────────────────

class EpisodeSource(grain.RandomAccessDataSource):
    """
    RandomAccessDataSource backed by a SINGLE ArrayRecord file.
    """

    def __init__(self, arecord_path):
        if not os.path.exists(arecord_path):
            raise FileNotFoundError(f"ArrayRecord file not found: {arecord_path}")
        self._reader = ArrayRecordReader(arecord_path)
        self._num_records = self._reader.num_records()
        if self._num_records == 0:
            raise ValueError(f"ArrayRecord file is empty: {arecord_path}")

    def __len__(self):
        return self._num_records

    def __getitem__(self, idx):
        raw = self._reader.read([idx])
        return msgpack.unpackb(raw[0], raw=False)


class ShardedEpisodeSource(grain.RandomAccessDataSource):
    """
    RandomAccessDataSource backed by MULTIPLE ArrayRecord shard files.

    Presents a unified view: indices [0, total_records) are mapped to the
    correct shard and local index automatically.

    Example:
      shard_000: 12000 records  → global idx [0, 12000)
      shard_001: 12000 records  → global idx [12000, 24000)
      shard_002: 12000 records  → global idx [24000, 36000)
    """

    def __init__(self, arecord_paths: List[str]):
        if not arecord_paths:
            raise ValueError("No ArrayRecord shard paths provided.")

        self._readers = []
        self._cumulative = [0]  # cumulative record counts
        total = 0
        for path in sorted(arecord_paths):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Shard not found: {path}")
            reader = ArrayRecordReader(path)
            n = reader.num_records()
            if n == 0:
                print(f"[Warning] Empty shard: {path}")
                continue
            self._readers.append(reader)
            total += n
            self._cumulative.append(total)
            print(f"  [Grain] shard {path}: {n} records")

        self._total = total
        if self._total == 0:
            raise ValueError("All shards are empty.")
        print(f"  [Grain] total: {self._total} records across {len(self._readers)} shards")

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        # Binary search for the correct shard
        lo, hi = 0, len(self._readers) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._cumulative[mid + 1]:
                hi = mid
            else:
                lo = mid + 1
        shard_idx = lo
        local_idx = idx - self._cumulative[shard_idx]
        raw = self._readers[shard_idx].read([local_idx])
        return msgpack.unpackb(raw[0], raw=False)


# ─── Grain transforms ────────────────────────────────────────────────────────

class DecodeEpisode(grain.MapTransform):
    """
    Decode a raw msgpack record into numpy arrays.

    Input: dict from EpisodeSource/ShardedEpisodeSource.__getitem__
    Output: dict with decoded numpy arrays matching the tfrecord contract.
    """

    def __init__(self, image_size=224, is_train=True, load_support_seq=True):
        self._image_size = image_size
        self._is_train = is_train
        self._load_support_seq = load_support_seq

    def map(self, record):
        # Decode target image
        target_path = record["target_path"]
        if isinstance(target_path, bytes):
            target_path = target_path.decode("utf-8")

        target = _decode_image(
            target_path,
            self._image_size,
            self._is_train,
            rng=np.random.RandomState(),
        )

        # Decode support embeddings
        pooled_bytes = record["supports_pooled"]
        supports_pooled = np.frombuffer(pooled_bytes, dtype=np.float16).reshape(5, 768).copy()

        seq_bytes = record["supports_seq"]
        if self._load_support_seq and len(seq_bytes) > 0:
            supports_seq = np.frombuffer(seq_bytes, dtype=np.float16).reshape(5, 196, 768).copy()
        else:
            supports_seq = np.zeros((5, 196, 768), dtype=np.float16)

        class_id = np.int32(record["class_id"])

        return {
            "target": target,
            "supports_pooled": supports_pooled,
            "supports_seq": supports_seq,
            "class_id": class_id,
        }


# ─── Auto-discover shards ────────────────────────────────────────────────────

def _discover_arecord_files(arecord_dir, split):
    """
    Auto-discover ArrayRecord files for a split.

    Search order:
      1) {split}_shard_*.arecord  (sharded)
      2) {split}.arecord          (single file)

    Returns list of paths.
    """
    # Try sharded pattern first
    pattern = os.path.join(arecord_dir, f"{split}_shard_*.arecord")
    shards = sorted(glob.glob(pattern))
    if shards:
        return shards

    # Try single file
    single = os.path.join(arecord_dir, f"{split}.arecord")
    if os.path.exists(single):
        return [single]

    raise FileNotFoundError(
        f"No ArrayRecord files found for split '{split}' in {arecord_dir}. "
        f"Expected {split}.arecord or {split}_shard_*.arecord"
    )


# ─── Pipeline builder ────────────────────────────────────────────────────────

def build_grain_dataset(
    arecord_dir,
    split,
    batch_size,
    image_size=224,
    is_train=True,
    seed=42,
    load_support_seq=True,
    num_threads=16,
    prefetch_buffer_size=500,
):
    """
    Build a Grain dataset pipeline from ArrayRecord file(s).

    Automatically detects single vs sharded layout.

    Args:
        arecord_dir: Directory containing .arecord files.
        split: Split name ('train' or 'val').
        batch_size: Batch size.
        image_size: Image resolution for target decode.
        is_train: Whether to apply training augmentations.
        seed: Random seed for shuffling.
        load_support_seq: Whether to load support sequence tokens.
        num_threads: Number of prefetch threads.
        prefetch_buffer_size: Prefetch buffer size.

    Returns:
        Grain DatasetIterator yielding batched numpy dicts.
    """
    paths = _discover_arecord_files(arecord_dir, split)
    print(f"[Grain] {split}: found {len(paths)} file(s)")

    if len(paths) == 1:
        source = EpisodeSource(paths[0])
    else:
        source = ShardedEpisodeSource(paths)

    print(f"[Grain] {split}: {len(source)} episodes, batch_size={batch_size}, is_train={is_train}")

    dataset = grain.MapDataset.source(source)

    if is_train:
        dataset = dataset.shuffle(seed=seed)

    dataset = dataset.map(
        DecodeEpisode(
            image_size=image_size,
            is_train=is_train,
            load_support_seq=load_support_seq,
        )
    )

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    iter_dataset = dataset.to_iter_dataset(
        grain.ReadOptions(
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size,
        )
    )

    return iter(iter_dataset)
