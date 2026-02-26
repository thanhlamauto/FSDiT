"""
precompute_selfrecon_jax.py — SigLIP2 self-reconstruction precompute (JAX/Flax).

For each image in the dataset, extract CLS token and patch tokens via SigLIP2
JAX checkpoint. Save as ArrayRecord where condition = target image.

Designed for Kaggle TPU v5e-8.

Usage:
    !python prepare_data.py --src /kaggle/input/datasets/arjunashok33/miniimagenet \
        --dst /kaggle/working/miniimagenet_split --train 60 --val 16 --test 20

    !python precompute_selfrecon_jax.py \
        --data_dir /kaggle/working/miniimagenet_split \
        --out_dir /kaggle/working/selfrecon_arecord \
        --splits train,val \
        --batch_size 64
"""

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import msgpack
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from array_record.python.array_record_module import ArrayRecordWriter
except ImportError:
    raise ImportError("Install array_record: pip install array_record")


# ═══════════════════════════════════════════════════════════════════════════════
#  SigLIP2 JAX encoder (via big_vision)
# ═══════════════════════════════════════════════════════════════════════════════

def setup_big_vision():
    repo = os.environ.get('BIG_VISION_DIR', '/kaggle/working/big_vision')
    if not os.path.exists(repo):
        print(f"Cloning big_vision → {repo}")
        os.system(f'git clone --quiet --branch=main --depth=1 '
                  f'https://github.com/google-research/big_vision {repo} > /dev/null 2>&1')
        os.system(f'pip install -q -r {repo}/big_vision/requirements.txt > /dev/null 2>&1')
    if repo not in sys.path:
        sys.path.insert(0, repo)


def create_siglip2_jax(variant='B/16', res=224):
    """Create SigLIP2 JAX model and load weights."""
    setup_big_vision()
    import big_vision.models.proj.image_text.two_towers as m
    import ml_collections

    txt_var, patch = variant.split('/')
    emb_dim = {'B': 768, 'L': 1024, 'So400m': 1152}[txt_var]

    cfg = ml_collections.ConfigDict(dict(
        image_model='vit',
        image=dict(pool_type='map', scan=True, variant=variant),
        text_model='proj.image_text.text_transformer',
        text=dict(scan=True, variant=txt_var, vocab_size=256_000),
        out_dim=[None, emb_dim],
        bias_init=-10,
    ))

    name = f'siglip2_{txt_var.lower()}{patch}_{res}.npz'
    ckpt_path = f'/tmp/{name}'
    if not os.path.exists(ckpt_path):
        url = f'https://storage.googleapis.com/big_vision/siglip2/{name}'
        print(f"Downloading {url}")
        os.system(f'wget -q {url} -O {ckpt_path}')

    print(f"Loading SigLIP2 {variant} {res}×{res}")
    model = m.Model(**cfg)
    params = m.load(None, ckpt_path, cfg)
    return model, params, emb_dim


def make_encode_fn(model, params, emb_dim):
    """Create JIT-compiled encode function returning (seq, pooled)."""

    @jax.jit
    def encode_batch(images):
        """(N, 224, 224, 3) → (seq: (N, 196, emb), pooled: (N, emb))"""
        outputs = model.apply({'params': params}, images, None)

        # Parse outputs to find sequence and pooled tensors
        leaves = jax.tree.leaves(outputs)
        bsz = images.shape[0]
        pooled = None
        seq = None
        for arr in leaves:
            shape = arr.shape
            if len(shape) == 2 and shape[0] == bsz and shape[1] == emb_dim:
                if pooled is None:
                    pooled = arr
            if len(shape) == 3 and shape[0] == bsz and shape[-1] == emb_dim and shape[1] == 196:
                if seq is None:
                    seq = arr
        return jax.lax.stop_gradient(seq), jax.lax.stop_gradient(pooled)

    return encode_batch


# ═══════════════════════════════════════════════════════════════════════════════
#  Image loading
# ═══════════════════════════════════════════════════════════════════════════════

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPEG'}


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


def load_and_preprocess(path, image_size=224):
    """Load image as normalized float32 array for SigLIP2."""
    img = Image.open(path).convert('RGB')
    img = img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr  # (224, 224, 3), [0, 1]


def load_and_preprocess_target(path, image_size=224):
    """Load image as [-1, 1] for DiT training target."""
    img = Image.open(path).convert('RGB')
    img = img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # → [-1, 1]
    return arr


# ═══════════════════════════════════════════════════════════════════════════════
#  Self-reconstruction record creation
# ═══════════════════════════════════════════════════════════════════════════════

def serialize_selfrecon_record(target_path, class_id, cls_token, patch_tokens):
    """
    Serialize a self-reconstruction record.

    The condition is the same image's SigLIP2 CLS + patch tokens.
    We store:
      - target_path: str
      - class_id: int
      - supports_pooled: bytes (1, 768) float16 — CLS token as "1-shot pooled"
      - supports_seq: bytes (1, 196, 768) float16 — patch tokens
    """
    # Shape: (1, 768) — single "support" that IS the target
    pooled = cls_token.reshape(1, -1).astype(np.float16)
    # Shape: (1, 196, 768)
    seq = patch_tokens.reshape(1, patch_tokens.shape[-2], -1).astype(np.float16)

    record = {
        "target_path": target_path,
        "class_id": int(class_id),
        "supports_pooled": pooled.tobytes(),
        "supports_seq": seq.tobytes(),
    }
    return msgpack.packb(record, use_bin_type=True)


def build_split_selfrecon(split, data_dir, out_dir, encode_fn, batch_size, image_size):
    """Build ArrayRecord for one split in self-reconstruction mode."""
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        print(f"[Skip] split '{split}' not found at {split_dir}")
        return

    class_images = collect_class_images(split_dir)
    class_names = sorted(class_images.keys())
    class_to_id = {name: i for i, name in enumerate(class_names)}

    # Flatten all images
    all_paths = []
    all_class_ids = []
    for cls_name in class_names:
        cid = class_to_id[cls_name]
        for path in class_images[cls_name]:
            all_paths.append(path)
            all_class_ids.append(cid)

    total = len(all_paths)
    print(f"[{split}] {len(class_names)} classes, {total} images")

    # Encode all images in batches
    all_cls_tokens = []
    all_patch_tokens = []

    for i in tqdm(range(0, total, batch_size), desc=f"encode-{split}"):
        batch_paths = all_paths[i:i + batch_size]
        batch_imgs = np.stack([
            load_and_preprocess(p, image_size) for p in batch_paths
        ])
        seq, pooled = encode_fn(batch_imgs)
        all_cls_tokens.append(np.array(pooled))
        all_patch_tokens.append(np.array(seq))

    all_cls_tokens = np.concatenate(all_cls_tokens, axis=0)
    all_patch_tokens = np.concatenate(all_patch_tokens, axis=0)
    print(f"  Encoded: cls={all_cls_tokens.shape}, patches={all_patch_tokens.shape}")

    # Write ArrayRecord
    os.makedirs(out_dir, exist_ok=True)
    arecord_path = os.path.join(out_dir, f"{split}.arecord")
    writer = ArrayRecordWriter(arecord_path, "group_size:1")

    for idx in tqdm(range(total), desc=f"write-{split}"):
        record_bytes = serialize_selfrecon_record(
            target_path=all_paths[idx],
            class_id=all_class_ids[idx],
            cls_token=all_cls_tokens[idx],
            patch_tokens=all_patch_tokens[idx],
        )
        writer.write(record_bytes)

    writer.close()

    # Metadata
    meta = {
        "split": split,
        "mode": "self-reconstruction",
        "num_classes": len(class_names),
        "num_images": total,
        "num_supports_per_record": 1,
        "format": "arrayrecord",
        "serialization": "msgpack",
        "class_names": class_names,
    }
    meta_path = os.path.join(out_dir, f"{split}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    file_mb = os.path.getsize(arecord_path) / (1024 * 1024)
    print(f"[{split}] Written {total} records → {arecord_path} ({file_mb:.1f} MB)")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Self-reconstruction precompute: each image's SigLIP2 CLS+patches as its own condition."
    )
    parser.add_argument('--data_dir', required=True,
                        help='Split root with train/val/test sub-dirs.')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for ArrayRecord files.')
    parser.add_argument('--splits', default='train,val',
                        help='Comma-separated splits.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for SigLIP2 encoding.')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--variant', default='B/16')
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"Num devices: {jax.device_count()}")

    # Create encoder
    model, params, emb_dim = create_siglip2_jax(args.variant, args.image_size)
    encode_fn = make_encode_fn(model, params, emb_dim)

    # Warmup
    print("Warmup JIT...")
    dummy = np.zeros((1, args.image_size, args.image_size, 3), dtype=np.float32)
    _ = encode_fn(dummy)
    print("Warmup done.")

    for split in args.splits.split(','):
        split = split.strip()
        if split:
            t0 = time.time()
            build_split_selfrecon(
                split, args.data_dir, args.out_dir,
                encode_fn, args.batch_size, args.image_size,
            )
            print(f"  {split} took {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
