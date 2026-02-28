import argparse
import os
import sys
import time
import random
import warnings

import jax
import jax.numpy as jnp
import flax
import numpy as np
import tensorflow as tf
from PIL import Image

# Suppress noisy diffusers/flax deprecation warnings emitted during module import.
warnings.filterwarnings(
    "ignore",
    message=".*Flax classes are deprecated and will be removed in Diffusers.*",
    category=FutureWarning,
)
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from model import DiT
from utils.train_state import TrainState
from utils.checkpoint import Checkpoint
from utils.stable_vae import StableVAE
from train import Trainer

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

def load_and_preprocess(path, image_size=224):
    """Load image as normalized float32 array for SigLIP2."""
    img = Image.open(path).convert('RGB')
    img = img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

# ═══════════════════════════════════════════════════════════════════════════════
#  Main Inference
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Few-shot Inference with FSDiT (Kaggle compatible).")
    parser.add_argument('--img_dir', default='/kaggle/input/datasets/arjunashok33/miniimagenet/test', help='Directory with subfolders of images or just images.')
    parser.add_argument('--ckpt_path', default='/kaggle/input/models/lucastnguyen/dit-few-shot/flax/default/1/ckpt_step_0050000.pkl', help='DiT Checkpoint path.')
    parser.add_argument('--out_path', default='/kaggle/working/output_shot.png', help='Generated image output array.')
    parser.add_argument('--num_shots', type=int, default=5, help='Number of condition images to randomly select.')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--variant', default='B/16')
    parser.add_argument('--cfg_scale', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--denoise_steps', type=int, default=50)
    args = parser.parse_args()

    np.random.seed(args.seed)
    print(f"JAX devices: {jax.devices()}")

    # 1. Initialize SigLIP2
    sig_model, sig_params, emb_dim = create_siglip2_jax(args.variant, args.image_size)
    encode_fn = make_encode_fn(sig_model, sig_params, emb_dim)

    # 2. Pick 5 random images
    all_imgs = []
    if os.path.exists(args.img_dir):
        for root, dirs, files in os.walk(args.img_dir):
            for f in files:
                if os.path.splitext(f)[1] in IMAGE_EXTS:
                    all_imgs.append(os.path.join(root, f))
    else:
        print(f"Warning: {args.img_dir} does not exist. Using dummy images for testing.")

    if len(all_imgs) >= args.num_shots:
        chosen = random.sample(all_imgs, args.num_shots)
    elif len(all_imgs) > 0:
        chosen = all_imgs * (args.num_shots // len(all_imgs) + 1)
        chosen = chosen[:args.num_shots]
    else:
        print(f"No images found! Creating {args.num_shots} dummy images.")
        chosen = [None] * args.num_shots

    batch_imgs = []
    for p in chosen:
        if p is not None:
            batch_imgs.append(load_and_preprocess(p, args.image_size))
        else:
            batch_imgs.append(np.zeros((args.image_size, args.image_size, 3), dtype=np.float32))
    
    batch_imgs = np.stack(batch_imgs)
    
    # 3. Encode via SigLIP and Pool
    # We pass 5 images at once
    seq_5, pooled_5 = encode_fn(batch_imgs)
    
    # Meaning them across the 5 shots
    pooled_model = np.mean(pooled_5, axis=0, keepdims=True).astype(np.float16)  # (1, 768)
    # The DiT model expects sequence embeddings to be concatenated: (1, 5 * 196, 768)
    # So we flatten the first dim and reshape
    seq_5 = np.array(seq_5)  # (5, 196, 768)
    seq_model = seq_5.reshape(1, -1, seq_5.shape[-1]).astype(np.float16)  # (1, 980, 768)

    print(f"Computed Condition. Pooled: {pooled_model.shape}, Seq: {seq_model.shape}")

    # 4. Initialize DiT and Stable VAE
    vae = StableVAE.create()
    vae_decode = jax.jit(vae.decode)

    cfg = dict(
        patch_size=2,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        siglip_dim=768,
        cond_dropout_prob=0.1,
        # Default sampling config
        num_t_bins=10,
        use_support_seq=1,
    )

    dit = DiT(
        patch_size=cfg['patch_size'], hidden_size=cfg['hidden_size'],
        depth=cfg['depth'], num_heads=cfg['num_heads'], mlp_ratio=cfg['mlp_ratio'],
        siglip_dim=cfg['siglip_dim'], cond_dropout_prob=cfg['cond_dropout_prob'],
    )

    rng = jax.random.PRNGKey(args.seed)
    rng, p_key, d_key = jax.random.split(rng, 3)

    img_s = args.image_size // 8 # VAE downsamples by 8
    img_c = 4 # VAE channels

    params = dit.init(
        {'params': p_key, 'cond_dropout': d_key},
        jnp.zeros((1, img_s, img_s, img_c)),
        jnp.zeros((1,)),
        jnp.zeros((1, cfg['siglip_dim'])),
        y_seq=jnp.zeros((1, seq_model.shape[1], cfg['siglip_dim'])),
    )['params']

    ts = TrainState.create(dit, params)
    ts_ema = TrainState.create(dit, params)
    trainer = Trainer(rng, ts, ts_ema, cfg)

    # 5. Load Checkpoint
    if os.path.exists(args.ckpt_path):
        print(f"Loading DiT checkpoint from {args.ckpt_path}")
        trainer = Checkpoint(args.ckpt_path, parallel=False).load_model(trainer)
    else:
        print(f"Warning: Checkpoint {args.ckpt_path} not found. Running with random weights.")

    # Replicate to local devices (even if 1)
    devices = jax.local_devices()
    n_dev = len(devices)
    trainer = flax.jax_utils.replicate(trainer, devices=devices)
    
    # Pad inputs to match n_dev
    if n_dev > 1:
        pooled_model = np.repeat(pooled_model, n_dev, axis=0)
        seq_model = np.repeat(seq_model, n_dev, axis=0)

    # Give trainer replicated rng
    trainer = trainer.replace(rng=jax.random.split(rng, n_dev))

    # 6. Sample Image (Euler flow matching)
    print("Sampling image...")
    # Add dummy device dim if ndev==1 => shape (ndev, B, H, W, C)
    eps_shape = (n_dev, 1, img_s, img_s, img_c)
    key = jax.random.PRNGKey(args.seed + 100)
    eps = jax.random.normal(key, eps_shape)

    pooled_pmap = pooled_model.reshape(n_dev, 1, -1)
    seq_pmap = seq_model.reshape(n_dev, 1, -1, seq_model.shape[-1])
    
    dt = 1.0 / args.denoise_steps

    x = eps
    for ti in range(args.denoise_steps):
        t_vec = jnp.full((n_dev, 1), ti / args.denoise_steps)
        # using sample_step from trainer 
        v = trainer.sample_step(
            x, t_vec, pooled_pmap, seq_pmap, True, args.cfg_scale
        )
        x = x + v * dt

    print("Decoding with VAE...")
    # x shape is (n_dev, 1, 28, 28, 4)
    # taking the first device's output
    latent = np.array(x)[0, 0] # (28, 28, 4)
    
    img_out = vae_decode(latent[None])[0] # (224, 224, 3) 
    img_out = np.array(jnp.clip(img_out * 0.5 + 0.5, 0, 1))

    # 7. Save Image
    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    img_pil = Image.fromarray((img_out * 255).astype(np.uint8))
    img_pil.save(args.out_path)
    print(f"Generated image saved at {args.out_path}")

if __name__ == '__main__':
    main()
