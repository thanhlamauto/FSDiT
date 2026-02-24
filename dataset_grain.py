"""
dataset_grain.py — Grain-based dataset loader for FSDiT.

Reads episode-level ArrayRecord files (single or sharded) and provides
a Grain pipeline. Drop-in replacement for the tfrecord path in dataset.py.

Supports two layouts:
  1) Single file:    train.arecord
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

import grain
import msgpack
import numpy as np
from PIL import Image


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
# grain.sources.ArrayRecordDataSource natively supports:
#   - single path:   ArrayRecordDataSource("/path/to/train.arecord")
#   - multiple paths: ArrayRecordDataSource(["/path/shard_000.arecord", ...])
# It returns raw bytes. We wrap it to deserialize msgpack.

class MsgpackEpisodeSource:
    """
    RandomAccessDataSource that reads msgpack-serialized episodes
    from ArrayRecord file(s).

    Implements __len__ and __getitem__ (the Grain RandomAccessDataSource protocol).
    Uses grain.sources.ArrayRecordDataSource under the hood for random access.
    """

    def __init__(self, arecord_paths):
        """
        Args:
            arecord_paths: Single path (str) or list of paths for shards.
        """
        if isinstance(arecord_paths, str):
            arecord_paths = [arecord_paths]

        for p in arecord_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"ArrayRecord file not found: {p}")

        # ArrayRecordDataSource handles multi-file concat + random access
        self._source = grain.sources.ArrayRecordDataSource(arecord_paths)
        print(f"  [Grain] loaded {len(arecord_paths)} file(s), {len(self._source)} records")

    def __len__(self):
        return len(self._source)

    def __getitem__(self, idx):
        raw_bytes = self._source[idx]
        return msgpack.unpackb(raw_bytes, raw=False)


# ─── Decode transform ────────────────────────────────────────────────────────
# Grain's .map() accepts any callable. No need to subclass a transform base.

class DecodeEpisode:
    """
    Decode a raw msgpack record dict into numpy arrays.

    Input: dict from MsgpackEpisodeSource.__getitem__
    Output: dict with decoded numpy arrays matching the tfrecord contract.
    """

    def __init__(self, image_size=224, is_train=True, load_support_seq=True):
        self._image_size = image_size
        self._is_train = is_train
        self._load_support_seq = load_support_seq

    def __call__(self, record):
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
    pattern = os.path.join(arecord_dir, f"{split}_shard_*.arecord")
    shards = sorted(glob.glob(pattern))
    if shards:
        return shards

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

    source = MsgpackEpisodeSource(paths)
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
