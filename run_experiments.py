import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
from PIL import Image
import matplotlib.pyplot as plt

# FSDiT specific imports
import configs.config as _config
from models import DiT
from train import Trainer, StableVAE
from utils.checkpoint import Checkpoint

import warnings
warnings.filterwarnings("ignore")

def cos_sim(a, b):
    """Compute cosine similarity between two 1D or 2D arrays."""
    a = a.flatten()
    b = b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def setup_siglip_jax():
    """Setup SigLIP2 JAX via big_vision and HuggingFace AutoTokenizer for txt."""
    import sys
    
    repo = os.environ.get('BIG_VISION_DIR', '/kaggle/working/big_vision')
    if not os.path.exists(repo):
        print(f"Cloning big_vision → {repo}")
        os.system(f'git clone --quiet --branch=main --depth=1 '
                   f'https://github.com/google-research/big_vision {repo} > /dev/null 2>&1')
        os.system(f'pip install -q -r {repo}/big_vision/requirements.txt > /dev/null 2>&1')
    if repo not in sys.path:
        sys.path.insert(0, repo)
        
    import big_vision.models.proj.image_text.two_towers as m
    from transformers import AutoTokenizer
    
    variant = "B/16"
    res = 224
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

    print(f"Loading SigLIP2 JAX Checkpoint: {variant} {res}×{res}")
    model = m.Model(**cfg)
    params = m.load(None, ckpt_path, cfg)
    
    print("Loading HuggingFace AutoTokenizer for text...")
    tokenizer = AutoTokenizer.from_pretrained("google/siglip2-base-patch16-224")
    
    return model, params, tokenizer

def encode_images_jax(image_paths_or_pils, model, params):
    """Encode PIL images or paths to get SigLIP2 patches and pooled CLS."""
    images = []
    for item in image_paths_or_pils:
        if isinstance(item, str):
            img = Image.open(item).convert("RGB")
        else:
            img = item.convert("RGB")
        img = img.resize((224, 224), Image.BICUBIC)
        # Normalize to [0, 1] for SigLIP
        arr = np.asarray(img, dtype=np.float32) / 255.0
        images.append(arr)
    images_np = np.stack(images)
    
    @jax.jit
    def encode(x):
        outputs = model.apply({'params': params}, x, None)
        leaves = jax.tree.leaves(outputs)
        seq = None
        pooled = None
        bsz = x.shape[0]
        # Extract features by parsing output tensor shapes
        for arr in leaves:
            sh = arr.shape
            if len(sh) == 2 and sh[0] == bsz and sh[1] == 768:
                pooled = arr if pooled is None else pooled
            if len(sh) == 3 and sh[0] == bsz and sh[-1] == 768 and sh[1] == 196:
                seq = arr if seq is None else seq
        return seq, pooled
        
    seq, pooled = encode(images_np)
    return np.asarray(seq), np.asarray(pooled)

def encode_text_jax(text, model, params, tokenizer):
    """Encode text string to get SigLIP2 sequence and pooled tokens."""
    # HF Tokenizer perfectly matches SigLIP2 subword encoding
    tokens = tokenizer([text], padding="max_length", max_length=64, return_tensors="np")["input_ids"]
    tokens = tokens.astype(np.int32)
    
    @jax.jit
    def encode(t):
        outputs = model.apply({'params': params}, None, t)
        leaves = jax.tree.leaves(outputs)
        seq = None
        pooled = None
        # Extract features
        for arr in leaves:
            sh = arr.shape
            if len(sh) == 2 and sh[0] == 1 and sh[1] == 768:
                pooled = arr if pooled is None else pooled
            if len(sh) == 3 and sh[0] == 1 and sh[-1] == 768 and sh[1] == 64:
                seq = arr if seq is None else seq
        return seq, pooled
        
    seq, pooled = encode(tokens)
    return np.asarray(seq), np.asarray(pooled)

def init_fsdit(ckpt_path):
    """Initialize DiT model and load weights."""
    print("Loading FSDiT model...")
    cfg = _config.get_config().model
    for k, v in _config.PRESETS[cfg.preset].items():
        cfg[k] = v
        
    cfg.use_support_seq = 1
    cfg.cfg_scale = 3.0 # Set sampling CFG
    cfg.denoise_steps = 50

    # DiT shapes
    img_s = cfg.image_size // 8 if cfg.use_vae else cfg.image_size
    img_c = 4 if cfg.use_vae else 3
    n_sup_tokens = 980 # 5 * 196

    rng = jax.random.PRNGKey(0)
    p_key, d_key = jax.random.split(rng)
    
    dit = DiT(
        patch_size=cfg.patch_size, hidden_size=cfg.hidden_size,
        depth=cfg.depth, num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
        siglip_dim=cfg.siglip_dim, cond_dropout_prob=0.0,
    )
    
    init_kwargs = {'y_seq': jnp.zeros((1, n_sup_tokens, cfg.siglip_dim))}
    params = dit.init(
        {'params': p_key, 'cond_dropout': d_key},
        jnp.zeros((1, img_s, img_s, img_c)),
        jnp.zeros((1,)),
        jnp.zeros((1, cfg.siglip_dim)),
        **init_kwargs,
    )['params']
    
    import optax
    from train import TrainState
    ts = TrainState.create(dit, params, tx=optax.sgd(0.0))
    ts_ema = TrainState.create(dit, params)
    trainer = Trainer(rng, ts, ts_ema, cfg)
    
    # Load ckpt
    cp = Checkpoint(ckpt_path)
    trainer = cp.load_model(trainer, ckpt_path)
    trainer = jax.device_put_replicated(trainer, jax.local_devices())
    
    # VAE
    vae = None
    if cfg.use_vae:
        vae = StableVAE.create()
        
    def decode_img(latent):
        if cfg.use_vae:
            latent = vae.decode(latent[None])[0]
        img = np.array(jnp.clip(latent * 0.5 + 0.5, 0, 1))
        return img
        
    return trainer, cfg, decode_img

def generate_sample(trainer, cfg, sup_pooled, sup_seq, seed=42):
    """Run Euler flow-matching generation given condition vectors."""
    n_dev = jax.local_device_count()
    
    # Pad seq to 980 if needed
    B, T, D = sup_seq.shape
    if T < 980:
        pad = np.zeros((B, 980 - T, D), dtype=sup_seq.dtype)
        sup_seq = np.concatenate([sup_seq, pad], axis=1)
    elif T > 980:
        sup_seq = sup_seq[:, :980, :]
        
    # Replicate for pmap
    p_pool = jnp.repeat(jnp.array(sup_pooled)[None], n_dev, axis=0) # (ndev, 1, 768)
    p_seq = jnp.repeat(jnp.array(sup_seq)[None], n_dev, axis=0)     # (ndev, 1, 980, 768)
    
    img_s = cfg.image_size // 8 if cfg.use_vae else cfg.image_size
    img_c = 4 if cfg.use_vae else 3
    
    key = jax.random.PRNGKey(seed)
    eps = jax.random.normal(key, (n_dev, 1, img_s, img_s, img_c))
    dt = 1.0 / cfg.denoise_steps

    x = eps
    for ti in range(cfg.denoise_steps):
        t_vec = jnp.full((n_dev, 1), ti / cfg.denoise_steps)
        pred = trainer.sample_step(
            x, t_vec, p_pool, p_seq, True, cfg.cfg_scale
        )
        x = x + pred * dt
        
    return x[0, 0] # return device 0 result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/kaggle/working/checkpoints/ckpt_step_0050000.pkl", help="Path to checkpoint")
    parser.add_argument("--img_dir", required=True, help="Directory containing at least 5 images of a class")
    parser.add_argument("--caption", required=True, help="Caption for Experiment 2")
    parser.add_argument("--out", default="experiments_output", help="Output directory for generated images")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    
    siglip_model, siglip_params, txt_tokenizer = setup_siglip_jax()
    trainer, cfg, decode_fn = init_fsdit(args.ckpt)
    
    print("\n" + "="*50)
    print("EXPERIMENT 1: 5-Shot Image Condition")
    print("="*50)
    
    # Get 5 images
    all_imgs = [os.path.join(args.img_dir, f) for f in os.listdir(args.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(all_imgs) < 5:
        raise ValueError(f"Need at least 5 images in {args.img_dir}")
    test_imgs = all_imgs[:5]
    print(f"Using 5 images from {args.img_dir}...")
    
    seq_5, pool_5 = encode_images_jax(test_imgs, siglip_model, siglip_params)
    
    # Mean pool the 5 images to form the condition
    mean_pool = np.mean(pool_5, axis=0, keepdims=True) # (1, 768)
    mean_seq = seq_5.reshape(1, -1, 768)               # (1, 1080/980, 768)
    
    # Generate
    print("Generating image for Exp 1...")
    latent_1 = generate_sample(trainer, cfg, mean_pool, mean_seq, seed=123)
    img_1_arr = decode_fn(latent_1)
    
    # Save Image
    exp1_path = os.path.join(args.out, "exp1_generated.png")
    Image.fromarray((img_1_arr * 255).astype(np.uint8)).save(exp1_path)
    print(f"Saved generated image to {exp1_path}")
    
    # Encode generated image and compute cosine similarity
    gen_seq_1, gen_pool_1 = encode_images_jax([exp1_path], siglip_model, siglip_params)
    sim_1 = cos_sim(gen_pool_1[0], mean_pool[0])
    print(f"==> Cosine Similarity (Gen vs Mean of 5): {sim_1:.4f}")
    
    
    print("\n" + "="*50)
    print("EXPERIMENT 2: Text Condition")
    print("="*50)
    print(f"Caption: '{args.caption}'")
    
    seq_txt, pool_txt = encode_text_jax(args.caption, siglip_model, siglip_params, txt_tokenizer)
    
    # Generate
    print("Generating image for Exp 2...")
    latent_2 = generate_sample(trainer, cfg, pool_txt, seq_txt, seed=456)
    img_2_arr = decode_fn(latent_2)
    
    # Save Image
    exp2_path = os.path.join(args.out, "exp2_generated.png")
    Image.fromarray((img_2_arr * 255).astype(np.uint8)).save(exp2_path)
    print(f"Saved generated image to {exp2_path}")
    
    # Encode generated image and compute cosine similarity
    gen_seq_2, gen_pool_2 = encode_images_jax([exp2_path], siglip_model, siglip_params)
    sim_2 = cos_sim(gen_pool_2[0], pool_txt[0])
    print(f"==> Cosine Similarity (Gen vs Text): {sim_2:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
