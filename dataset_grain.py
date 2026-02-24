"""
dataset_grain.py — Grain-based dataset loader for FSDiT.

Reads episode-level ArrayRecord files and provides a Grain pipeline.
Designed as a drop-in replacement for the tfrecord path in dataset.py.

Output contract (matches tfrecord mode):
  {
      'target': (B, H, W, 3) float32,
      'supports_pooled': (B, 5, 768) float16,
      'supports_seq': (B, 5, 196, 768) float16,
      'class_id': (B,) int32,
  }

Usage in train.py:
  from dataset_grain import build_grain_dataset
  train_ds = build_grain_dataset(
      arecord_path="episodes_arecord/train.arecord",
      batch_size=128, image_size=224, is_train=True, seed=42,
  )
  for batch in train_ds:
      train_step(batch)
"""

import os

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


# ─── ArrayRecord data source ─────────────────────────────────────────────────

class EpisodeSource(grain.RandomAccessDataSource):
    """
    RandomAccessDataSource backed by an ArrayRecord of msgpack episodes.

    Each record contains:
      - target_path: str
      - class_id: int
      - supports_pooled: bytes (float16 [5,768])
      - supports_seq: bytes (float16 [5,196,768]) or b""
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
        record = msgpack.unpackb(raw[0], raw=False)
        return record


# ─── Grain transforms ────────────────────────────────────────────────────────

class DecodeEpisode(grain.MapTransform):
    """
    Decode a raw msgpack record into numpy arrays.

    Input: dict from EpisodeSource.__getitem__
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
            rng=np.random.RandomState(),  # thread-local RNG for augmentation
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


# ─── Pipeline builder ────────────────────────────────────────────────────────

def build_grain_dataset(
    arecord_path,
    batch_size,
    image_size=224,
    is_train=True,
    seed=42,
    load_support_seq=True,
    num_threads=16,
    prefetch_buffer_size=500,
):
    """
    Build a Grain dataset pipeline from an ArrayRecord file.

    Args:
        arecord_path: Path to the .arecord file.
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
    source = EpisodeSource(arecord_path)
    print(
        f"[Grain] {arecord_path}: {len(source)} episodes, "
        f"batch_size={batch_size}, is_train={is_train}"
    )

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
