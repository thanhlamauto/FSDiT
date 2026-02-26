"""
precompute_fewshot_pytorch.py — Few-shot episode precompute (PyTorch).

Divides each class into k sets of (600/k) images. For each set, creates
(target, condition) pairs round-robin: each image is the target once, and
the remaining images in the set are condition sources (pooled mean of their
SigLIP2 embeddings).

Uses PyTorch + HuggingFace SigLIP2 checkpoint.

Usage (standalone GPU server — auto-downloads miniImageNet from Kaggle):
    python precompute_fewshot_pytorch.py \
        --download \
        --out_dir ./fewshot_arecord \
        --splits train,val \
        --k_sets 5 \
        --batch_size 64

Usage (data already downloaded):
    python precompute_fewshot_pytorch.py \
        --data_dir /path/to/miniimagenet_split \
        --out_dir ./fewshot_arecord \
        --splits train,val \
        --k_sets 5 \
        --batch_size 64

Kaggle API setup (required for --download):
    pip install kaggle
    export KAGGLE_USERNAME=your_username
    export KAGGLE_KEY=your_api_key
    # or place kaggle.json in ~/.kaggle/kaggle.json
"""

import argparse
import json
import os
import shutil
import subprocess
import time
import zipfile

import msgpack
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    from array_record.python.array_record_module import ArrayRecordWriter
except ImportError:
    raise ImportError("Install array_record: pip install array_record")


# ═══════════════════════════════════════════════════════════════════════════════
#  Download miniImageNet from Kaggle + split into train/val/test
# ═══════════════════════════════════════════════════════════════════════════════

def download_miniimagenet(dst_dir, train=60, val=16, test=20, seed=42):
    """
    Download miniImageNet from Kaggle and split into train/val/test.

    Requires:
        pip install kaggle
        KAGGLE_USERNAME + KAGGLE_KEY env vars or ~/.kaggle/kaggle.json

    Returns:
        split_dir: path to directory with train/ val/ test/ sub-dirs
    """
    raw_dir = os.path.join(dst_dir, "_raw")
    split_dir = os.path.join(dst_dir, "split")

    # Check if already split
    if os.path.isdir(os.path.join(split_dir, "train")):
        n_train = len(os.listdir(os.path.join(split_dir, "train")))
        print(f"[Download] Already split: {split_dir} ({n_train} train classes)")
        return split_dir

    # ── Step 1: Download from Kaggle ──
    os.makedirs(raw_dir, exist_ok=True)
    dataset_slug = "arjunashok33/miniimagenet"
    print(f"[Download] Downloading {dataset_slug} from Kaggle...")

    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", raw_dir],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "kaggle CLI not found. Install with: pip install kaggle\n"
            "Then set KAGGLE_USERNAME and KAGGLE_KEY env vars, or place "
            "kaggle.json in ~/.kaggle/kaggle.json"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Kaggle download failed: {e.stderr}")

    # ── Step 2: Unzip ──
    zip_path = os.path.join(raw_dir, "miniimagenet.zip")
    if os.path.exists(zip_path):
        print(f"[Download] Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(raw_dir)
        os.remove(zip_path)

    # Find the actual image directory (may be nested)
    img_root = raw_dir
    for candidate in [raw_dir, os.path.join(raw_dir, "miniimagenet")]:
        subdirs = [d for d in os.listdir(candidate)
                   if os.path.isdir(os.path.join(candidate, d)) and not d.startswith('_')]
        if len(subdirs) >= 20:
            img_root = candidate
            break

    print(f"[Download] Image root: {img_root}")

    # ── Step 3: Split into train/val/test ──
    all_classes = sorted([
        d for d in os.listdir(img_root)
        if os.path.isdir(os.path.join(img_root, d))
    ])
    n_total = len(all_classes)
    n_need = train + val + test
    print(f"[Download] Found {n_total} classes, splitting {train}/{val}/{test}")
    assert n_total >= n_need, f"Need {n_need} but found {n_total}"

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total)[:n_need]
    splits = {
        'train': [all_classes[i] for i in indices[:train]],
        'val':   [all_classes[i] for i in indices[train:train + val]],
        'test':  [all_classes[i] for i in indices[train + val:n_need]],
    }

    for split_name, class_list in splits.items():
        s_dir = os.path.join(split_dir, split_name)
        os.makedirs(s_dir, exist_ok=True)
        for cls_name in class_list:
            src = os.path.join(img_root, cls_name)
            dst = os.path.join(s_dir, cls_name)
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)
        print(f"  {split_name}: {len(class_list)} classes → {s_dir}")

    return split_dir


# ═══════════════════════════════════════════════════════════════════════════════
#  SigLIP2 PyTorch encoder
# ═══════════════════════════════════════════════════════════════════════════════

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPEG'}


class SiglipEncoderPT:
    """SigLIP2 B/16 encoder using HuggingFace transformers (PyTorch)."""

    def __init__(self, model_name="google/siglip2-base-patch16-224", device="cuda"):
        from transformers import AutoModel, AutoProcessor

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model = self.model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

        if self.device.type == "cuda":
            self.model = self.model.half()
            self._dtype = torch.float16
        else:
            self._dtype = torch.float32

        print(f"SigLIP2 loaded: device={self.device}, dtype={self._dtype}")

    @torch.no_grad()
    def encode_pil_batch(self, pil_images):
        """
        Encode a batch of PIL images.

        Returns:
            seq:    (N, 196, 768) float32 numpy — patch tokens
            pooled: (N, 768) float32 numpy — CLS / pooled token
        """
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self._dtype)

        vision_out = self.model.vision_model(
            pixel_values=pixel_values,
            return_dict=True,
        )
        seq = vision_out.last_hidden_state.float().cpu().numpy()
        pooled = vision_out.pooler_output.float().cpu().numpy()

        # Remove CLS token if present
        if seq.shape[1] == 197:
            seq = seq[:, 1:, :]
        elif seq.shape[1] != 196:
            raise ValueError(f"Unexpected seq length: {seq.shape[1]}")

        return seq, pooled


# ═══════════════════════════════════════════════════════════════════════════════
#  Image collection
# ═══════════════════════════════════════════════════════════════════════════════

def collect_class_images(split_dir):
    """Returns dict: class_name → sorted list of image paths."""
    classes = {}
    for cls_name in sorted(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        imgs = sorted([
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if os.path.splitext(f)[1] in IMAGE_EXTS
        ])
        if imgs:
            classes[cls_name] = imgs
    return classes


def load_pil(path, image_size=224):
    """Load image as PIL, resized to image_size × image_size."""
    img = Image.open(path).convert('RGB')
    img = img.resize((image_size, image_size), Image.BICUBIC)
    return img


# ═══════════════════════════════════════════════════════════════════════════════
#  Precompute all embeddings for a split
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_all_embeddings(image_paths, encoder, batch_size, image_size):
    """
    Encode all images and return:
        seq_all:    (N, 196, 768) float16
        pooled_all: (N, 768) float16
    """
    all_seq = []
    all_pooled = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="encode"):
        batch_paths = image_paths[i:i + batch_size]
        pil_imgs = [load_pil(p, image_size) for p in batch_paths]
        seq, pooled = encoder.encode_pil_batch(pil_imgs)
        all_seq.append(seq.astype(np.float16))
        all_pooled.append(pooled.astype(np.float16))

    return np.concatenate(all_seq), np.concatenate(all_pooled)


# ═══════════════════════════════════════════════════════════════════════════════
#  Episode generation (k-set round-robin)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_fewshot_episodes(class_images, class_to_id, k_sets, seed=42):
    """
    For each class, divide 600 images into k sets of (600/k) images.
    For each set, create round-robin (target, condition) pairs.

    Returns list of dicts:
      {
        'target_path': str,
        'target_idx': int,       # global index into flat image list
        'condition_idxs': [int], # global indices of condition images
        'class_id': int,
      }
    """
    rng = np.random.RandomState(seed)
    episodes = []

    # Build global index mapping
    all_paths = []
    path_to_global_idx = {}
    for cls_name in sorted(class_images.keys()):
        for p in class_images[cls_name]:
            path_to_global_idx[p] = len(all_paths)
            all_paths.append(p)

    for cls_name in sorted(class_images.keys()):
        imgs = class_images[cls_name]
        cid = class_to_id[cls_name]
        n = len(imgs)
        set_size = n // k_sets

        if set_size < 2:
            print(f"  Warning: class {cls_name} has {n} imgs, k={k_sets} → set_size={set_size} < 2, skipping")
            continue

        # Shuffle images within class
        idxs = rng.permutation(n)

        for s in range(k_sets):
            set_idxs = idxs[s * set_size: (s + 1) * set_size]
            set_paths = [imgs[j] for j in set_idxs]

            # Round-robin: each image in the set is target once
            for t_pos in range(len(set_paths)):
                target_path = set_paths[t_pos]
                cond_paths = [set_paths[j] for j in range(len(set_paths)) if j != t_pos]

                episodes.append({
                    'target_path': target_path,
                    'target_idx': path_to_global_idx[target_path],
                    'condition_idxs': [path_to_global_idx[p] for p in cond_paths],
                    'class_id': cid,
                })

    # Shuffle episodes for training
    rng.shuffle(episodes)
    return episodes, all_paths


def serialize_fewshot_record(episode, pooled_all, seq_all):
    """
    Serialize a few-shot episode.

    Condition = pooled mean of condition images' CLS tokens.
    Seq = all condition images' patch tokens.
    """
    cond_idxs = episode['condition_idxs']
    num_cond = len(cond_idxs)

    # Condition pooled: mean of all condition images' pooled embeddings
    cond_pooled = pooled_all[cond_idxs]  # (num_cond, 768) float16
    # Condition seq: all condition images' patch tokens
    cond_seq = seq_all[cond_idxs]  # (num_cond, 196, 768) float16

    record = {
        "target_path": episode['target_path'],
        "class_id": int(episode['class_id']),
        "supports_pooled": cond_pooled.tobytes(),
        "supports_seq": cond_seq.tobytes(),
    }
    return msgpack.packb(record, use_bin_type=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Build one split
# ═══════════════════════════════════════════════════════════════════════════════

def build_split_fewshot(split, data_dir, out_dir, encoder, batch_size,
                        image_size, k_sets, seed):
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        print(f"[Skip] split '{split}' not found at {split_dir}")
        return

    class_images = collect_class_images(split_dir)
    class_names = sorted(class_images.keys())
    class_to_id = {name: i for i, name in enumerate(class_names)}

    total_imgs = sum(len(v) for v in class_images.values())
    print(f"[{split}] {len(class_names)} classes, {total_imgs} images, k={k_sets}")

    # Generate episodes
    episodes, all_paths = generate_fewshot_episodes(
        class_images, class_to_id, k_sets, seed=seed
    )
    print(f"  Generated {len(episodes)} episodes")

    # Precompute all embeddings
    print(f"  Encoding {len(all_paths)} images...")
    seq_all, pooled_all = precompute_all_embeddings(
        all_paths, encoder, batch_size, image_size
    )
    print(f"  Embeddings: pooled={pooled_all.shape}, seq={seq_all.shape}")

    # Write ArrayRecord
    os.makedirs(out_dir, exist_ok=True)
    arecord_path = os.path.join(out_dir, f"{split}.arecord")
    writer = ArrayRecordWriter(arecord_path, "group_size:1")

    for ep in tqdm(episodes, desc=f"write-{split}"):
        record_bytes = serialize_fewshot_record(ep, pooled_all, seq_all)
        writer.write(record_bytes)

    writer.close()

    # Metadata
    meta = {
        "split": split,
        "mode": "few-shot-round-robin",
        "num_classes": len(class_names),
        "num_images_total": len(all_paths),
        "num_episodes": len(episodes),
        "k_sets": k_sets,
        "set_size": total_imgs // (len(class_names) * k_sets),
        "num_condition_images_per_episode": total_imgs // (len(class_names) * k_sets) - 1,
        "seed": seed,
        "format": "arrayrecord",
        "serialization": "msgpack",
    }
    meta_path = os.path.join(out_dir, f"{split}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    file_mb = os.path.getsize(arecord_path) / (1024 * 1024)
    print(f"[{split}] Written {len(episodes)} records → {arecord_path} ({file_mb:.1f} MB)")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Few-shot episode precompute with round-robin (target, condition) pairs."
    )
    parser.add_argument('--data_dir', default=None,
                        help='Split root with train/val/test sub-dirs. '
                             'If not set, use --download to auto-download.')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for ArrayRecord files.')
    parser.add_argument('--download', action='store_true',
                        help='Auto-download miniImageNet from Kaggle and split.')
    parser.add_argument('--download_dir', default='./data/miniimagenet',
                        help='Where to download/store miniImageNet.')
    parser.add_argument('--splits', default='train,val',
                        help='Comma-separated splits.')
    parser.add_argument('--k_sets', type=int, default=5,
                        help='Number of sets per class (600/k images per set).')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for SigLIP2 encoding.')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', default='google/siglip2-base-patch16-224',
                        help='HuggingFace model name.')
    parser.add_argument('--device', default='cuda',
                        help='PyTorch device (cuda or cpu).')
    parser.add_argument('--train_classes', type=int, default=60)
    parser.add_argument('--val_classes', type=int, default=16)
    parser.add_argument('--test_classes', type=int, default=20)
    args = parser.parse_args()

    # Auto-download if needed
    if args.download or args.data_dir is None:
        args.data_dir = download_miniimagenet(
            args.download_dir,
            train=args.train_classes,
            val=args.val_classes,
            test=args.test_classes,
            seed=args.seed,
        )
        print(f"Using data_dir: {args.data_dir}")

    print(f"Device: {args.device}")
    print(f"k_sets={args.k_sets} → set_size={600 // args.k_sets} → "
          f"{600 // args.k_sets - 1} condition images per episode")

    # Create encoder
    encoder = SiglipEncoderPT(args.model_name, args.device)

    for split in args.splits.split(','):
        split = split.strip()
        if split:
            t0 = time.time()
            build_split_fewshot(
                split, args.data_dir, args.out_dir, encoder,
                args.batch_size, args.image_size, args.k_sets, args.seed,
            )
            print(f"  {split} took {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
