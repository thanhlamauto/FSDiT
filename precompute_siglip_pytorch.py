"""
Precompute SigLIP2 embeddings using PyTorch (for local GPU, e.g. RTX 4090).

Output format (saved next to each image or in --out_dir):
  <image_name>.npz with keys:
    - seq:    (196, 768) float16
    - pooled: (768,)     float16

Compatible with build_episodes_arrayrecord.py and build_episode_tfrecord.py.

Usage:
  python precompute_siglip_pytorch.py \
      --data_dir /path/to/miniimagenet/train \
      --batch_size 128 \
      --dtype float16
"""

import argparse
import os
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ─── Image preprocessing (matches JAX encoder.py convention) ──────────────────

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


def _load_pil_image(path, image_size=224):
    """Load image as PIL (resized). Processor handles normalization."""
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BICUBIC)
    return img


def _validate_outputs(seq, pooled):
    if seq.shape != (196, 768):
        raise ValueError(f"Expected seq shape (196, 768), got {seq.shape}")
    if pooled.shape != (768,):
        raise ValueError(f"Expected pooled shape (768,), got {pooled.shape}")
    if not np.isfinite(seq).all() or not np.isfinite(pooled).all():
        raise ValueError("NaN/Inf detected in embeddings")


def _resolve_npz_path(img_path, data_dir, out_dir):
    if out_dir:
        rel = os.path.relpath(img_path, data_dir)
        npz_path = os.path.join(out_dir, os.path.splitext(rel)[0] + ".npz")
    else:
        npz_path = os.path.splitext(img_path)[0] + ".npz"
    return npz_path


# ─── PyTorch SigLIP2 encoder ─────────────────────────────────────────────────

class PytorchSiglipEncoder:
    """
    SigLIP2 B/16 encoder using HuggingFace transformers.
    Returns both sequence tokens (196, 768) and pooled embedding (768,).
    """

    def __init__(self, model_name="google/siglip2-base-patch16-224", device="cuda"):
        from transformers import AutoModel, AutoProcessor

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model = self.model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Use half precision on GPU for speed
        if self.device.type == "cuda":
            self.model = self.model.half()
            self._compute_dtype = torch.float16
        else:
            self._compute_dtype = torch.float32

        print(f"SigLIP2 loaded. Device={self.device}, dtype={self._compute_dtype}")

    @torch.no_grad()
    def encode_batch(self, pil_images):
        """
        Encode a batch of PIL images.

        Args:
            pil_images: list of PIL.Image objects (already resized to 224×224)

        Returns:
            seq:    (N, 196, 768) float32 numpy
            pooled: (N, 768) float32 numpy
        """
        # Let the processor handle normalization (correct SigLIP2 preprocessing)
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self._compute_dtype)

        # Use vision_model directly for reliable seq + pooled extraction
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            return_dict=True,
        )
        seq = vision_outputs.last_hidden_state.float().cpu().numpy()
        pooled = vision_outputs.pooler_output.float().cpu().numpy()

        # Remove CLS token if present (keep only 196 patch tokens)
        if seq.shape[1] == 197:  # 196 patches + 1 CLS
            seq = seq[:, 1:, :]
        elif seq.shape[1] != 196:
            raise ValueError(
                f"Unexpected seq length: {seq.shape[1]}. "
                f"Expected 196 or 197 for SigLIP2 B/16 at 224×224."
            )

        return seq, pooled


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Precompute SigLIP2 embeddings using PyTorch (GPU)."
    )
    parser.add_argument("--data_dir", required=True,
                        help="Root folder containing class folders with images.")
    parser.add_argument("--out_dir", default=None,
                        help="Optional writable root for .npz cache. If unset, saves next to images.")
    parser.add_argument("--image_size", type=int, default=224,
                        help="SigLIP input resolution.")
    parser.add_argument("--model_name", default="google/siglip2-base-patch16-224",
                        help="HuggingFace model name for SigLIP2.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Encode batch size.")
    parser.add_argument("--device", default="cuda",
                        help="PyTorch device (cuda, cpu).")
    parser.add_argument("--dtype", default="float16",
                        choices=("float16", "float32"),
                        help="Saved embedding dtype.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute existing .npz files.")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Continue when a file fails. Default is fail-fast.")
    args = parser.parse_args()

    t0 = time.time()
    np_dtype = np.float16 if args.dtype == "float16" else np.float32

    # Collect images
    image_paths = _collect_images(args.data_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {args.data_dir}")
    print(f"Found {len(image_paths)} images under {args.data_dir}")

    # Build work list (skip existing)
    work = []
    skipped = 0
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

    print(f"To process: {len(work)} images (skipped: {skipped})")

    # Initialize encoder
    encoder = PytorchSiglipEncoder(
        model_name=args.model_name,
        device=args.device,
    )

    done = 0
    failed = 0
    bs = args.batch_size

    for st in tqdm(range(0, len(work), bs), dynamic_ncols=True, desc="Encoding"):
        chunk = work[st:st + bs]
        pil_images = []
        valid_indices = []

        # Load images as PIL
        for i, (img_path, _) in enumerate(chunk):
            try:
                pil_img = _load_pil_image(img_path, args.image_size)
                pil_images.append(pil_img)
                valid_indices.append(i)
            except Exception as exc:
                failed += 1
                print(f"[ERROR] decode {img_path}: {exc}")
                if not args.continue_on_error:
                    raise

        if not valid_indices:
            continue

        # Encode batch
        try:
            seq_all, pooled_all = encoder.encode_batch(pil_images)
        except Exception as exc:
            failed += len(valid_indices)
            print(f"[ERROR] encode batch starting at idx={st}: {exc}")
            if not args.continue_on_error:
                raise
            continue

        # Save per-image
        for j, idx in enumerate(valid_indices):
            img_path, npz_path = chunk[idx]
            try:
                seq = seq_all[j]
                pooled = pooled_all[j]
                _validate_outputs(seq, pooled)
                np.savez(
                    npz_path,
                    seq=seq.astype(np_dtype),
                    pooled=pooled.astype(np_dtype),
                )
                done += 1
            except Exception as exc:
                failed += 1
                print(f"[ERROR] save {img_path}: {exc}")
                if not args.continue_on_error:
                    raise

    dt = time.time() - t0
    throughput = done / max(dt, 0.001)
    print(
        f"Done in {dt:.1f}s ({throughput:.0f} imgs/s) | "
        f"saved={done}, skipped={skipped}, failed={failed}, dtype={args.dtype}"
    )


if __name__ == "__main__":
    main()
