"""
train.py — FSDiT: Few-Shot Diffusion Transformer Training.

Flow-matching DiT conditioned on SigLIP2 support-set embeddings.
Data: miniImageNet (60 train / 16 val classes, 600 imgs each).
Episodes: 100 sets/class × 6 rotations → 1 target + 5 support.

Usage (Kaggle TPU v5e-8):
    python train.py --data_dir /kaggle/input/.../miniimagenet \
                    --save_dir /kaggle/working/ckpts \
                    --batch_size 128 --max_steps 200000
"""

from typing import Any
import os
import time
from absl import app, flags
from functools import partial
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
import flax
import optax
import wandb
from ml_collections import config_flags
import ml_collections
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import DiT
from dataset import build_dataset
from encoder import SigLIP2Encoder
from utils.train_state import TrainState, target_update
from utils.checkpoint import Checkpoint
from utils.stable_vae import StableVAE
from utils.wandb_utils import setup_wandb, default_wandb_config

# ═══════════════════════════════════════════════════════════════════════════════
#  Flags & Config
# ═══════════════════════════════════════════════════════════════════════════════

FLAGS = flags.FLAGS
# Paths
flags.DEFINE_string('data_dir', '/kaggle/input/datasets/arjunashok33/miniimagenet',
                    'miniImageNet root (contains train/, val/, test/).')
flags.DEFINE_string('load_dir', None,  'Resume from checkpoint.')
flags.DEFINE_string('save_dir', None,  'Save checkpoints here.')
flags.DEFINE_string('fid_stats', None, 'Precomputed FID stats .npz.')
flags.DEFINE_string('siglip_ckpt', None, 'SigLIP2 .npz path (auto-download if None).')
# Training
flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('batch_size', 128, 'Global batch size.')
flags.DEFINE_integer('max_steps', 200_000, 'Total training steps.')
flags.DEFINE_integer('num_sets', 100, 'Sets per class (each set = 6 images).')
flags.DEFINE_integer('debug_overfit', 0, 'Overfit on N samples (0 = off).')
# Logging
flags.DEFINE_integer('log_interval', 500, 'Train metric logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Validation + attention entropy interval.')
flags.DEFINE_integer('fid_interval', 25000, 'FID / sample grid interval.')
flags.DEFINE_integer('save_interval', 25000, 'Checkpoint interval.')

model_config = ml_collections.ConfigDict({
    # ── Optimizer ──
    'lr': 1e-4,              # peak learning rate
    'lr_min': 1e-6,          # cosine floor
    'warmup_steps': 5000,    # linear warmup (2.5% of 200k)
    'beta1': 0.9,
    'beta2': 0.99,
    'weight_decay': 0.01,    # AdamW regularization
    'grad_clip': 1.0,        # max gradient norm
    # ── DiT Architecture ──
    'hidden_size': 768,
    'patch_size': 2,
    'depth': 12,
    'num_heads': 12,
    'mlp_ratio': 4,
    'preset': 'big',
    # ── FSDiT Conditioning ──
    'siglip_dim': 768,       # SigLIP2 B/16 output dim
    'cond_dropout': 0.1,     # CFG: zero support 10% of time
    # ── Flow Matching ──
    'denoise_steps': 50,     # Euler steps for sampling
    'cfg_scale': 3.0,        # classifier-free guidance scale
    'ema_rate': 0.9999,      # EMA model update rate
    't_sampler': 'log-normal',
    # ── Misc ──
    'use_vae': 1,            # use SD VAE for latent diffusion
    'image_size': 224,
    'loss_ema_alpha': 0.99,  # EMA smoothing for logged train loss
    'num_t_bins': 10,        # t-bin resolution for loss breakdown
})

PRESETS = {
    'debug':  {'hidden_size': 64,   'patch_size': 8, 'depth': 2,  'num_heads': 2,  'mlp_ratio': 1},
    'big':    {'hidden_size': 768,  'patch_size': 2, 'depth': 12, 'num_heads': 12, 'mlp_ratio': 4},
    'large':  {'hidden_size': 1024, 'patch_size': 2, 'depth': 24, 'num_heads': 16, 'mlp_ratio': 4},
    'xlarge': {'hidden_size': 1152, 'patch_size': 2, 'depth': 28, 'num_heads': 16, 'mlp_ratio': 4},
}

wandb_config = default_wandb_config()
wandb_config.update({'project': 'fsdit', 'name': 'fsdit_miniimagenet'})
config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('model', model_config, lock_config=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  Flow Matching Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def flow_interpolate(x1, eps, t):
    """x_t = (1-t)*eps + t*x1  (linear interpolation noise→data)."""
    t = jnp.clip(t, 0, 0.99)
    return (1 - t) * eps + t * x1

def flow_velocity(x1, eps):
    """Ground-truth velocity: v = x1 - eps."""
    return x1 - eps

def attention_entropy(attn_weights):
    """Per-head entropy of attention. attn_weights: (B,H,Q,K) → (H,)"""
    return -jnp.sum(attn_weights * jnp.log(attn_weights + 1e-8), axis=-1).mean(axis=(0, 2))

def compute_t_bin_losses(loss_per_sample, t, num_bins):
    """Break down MSE loss by timestep bins. Returns (num_bins,) array."""
    idx = jnp.clip((t * num_bins).astype(jnp.int32), 0, num_bins - 1)
    bins = jnp.zeros(num_bins)
    for b in range(num_bins):
        mask = (idx == b).astype(jnp.float32)
        bins = bins.at[b].set(
            jnp.sum(loss_per_sample * mask) / jnp.maximum(jnp.sum(mask), 1.0))
    return bins


# ═══════════════════════════════════════════════════════════════════════════════
#  FSDiT Trainer (PyTreeNode — compatible with pmap/jit)
# ═══════════════════════════════════════════════════════════════════════════════

class Trainer(flax.struct.PyTreeNode):
    rng: Any
    model: TrainState        # live model (training)
    model_ema: TrainState    # EMA model (eval / sampling)
    config: dict = flax.struct.field(pytree_node=False)

    # ── Training step ──────────────────────────────────────────────────────
    @partial(jax.pmap, axis_name='data')
    def train_step(self, images, support_embed):
        """One training step. Returns (new_trainer, info_dict)."""
        new_rng, cond_key, time_key, noise_key = jax.random.split(self.rng, 4)
        num_bins = self.config['num_t_bins']

        def loss_fn(params):
            # Sample timesteps
            if self.config['t_sampler'] == 'log-normal':
                t = jax.nn.sigmoid(jax.random.normal(time_key, (images.shape[0],)))
            else:
                t = jax.random.uniform(time_key, (images.shape[0],))

            eps = jax.random.normal(noise_key, images.shape)
            x_t = flow_interpolate(images, eps, t[:, None, None, None])
            v_gt = flow_velocity(images, eps)
            v_pred = self.model(
                x_t, t, support_embed, train=True,
                rngs={'cond_dropout': cond_key}, params=params,
            )
            mse = (v_pred - v_gt) ** 2
            loss = jnp.mean(mse)

            # Per-sample, per-tbin breakdown
            loss_ps = jnp.mean(mse, axis=(1, 2, 3))
            tbin = compute_t_bin_losses(loss_ps, t, num_bins)

            return loss, {
                'loss': loss,
                'v_abs': jnp.abs(v_gt).mean(),
                'v_pred_abs': jnp.abs(v_pred).mean(),
                'tbin_loss': tbin,
            }

        grads, info = jax.grad(loss_fn, has_aux=True)(self.model.params)
        grads = jax.lax.pmean(grads, axis_name='data')
        info = jax.lax.pmean(info, axis_name='data')

        # Apply gradients
        updates, new_opt = self.model.tx.update(grads, self.model.opt_state, self.model.params)
        new_params = optax.apply_updates(self.model.params, updates)
        new_model = self.model.replace(
            step=self.model.step + 1, params=new_params, opt_state=new_opt)

        info['grad_norm'] = optax.global_norm(grads)
        info['param_norm'] = optax.global_norm(new_params)

        # EMA update
        tau = 1 - self.config['ema_rate']
        new_ema = target_update(self.model, self.model_ema, tau)

        return self.replace(rng=new_rng, model=new_model, model_ema=new_ema), info

    # ── Validation loss ────────────────────────────────────────────────────
    @partial(jax.pmap, axis_name='data')
    def val_loss(self, images, support_embed):
        """Compute val loss + t-bin breakdown (no dropout, uses EMA model)."""
        time_key, noise_key = jax.random.split(self.rng, 2)
        if self.config['t_sampler'] == 'log-normal':
            t = jax.nn.sigmoid(jax.random.normal(time_key, (images.shape[0],)))
        else:
            t = jax.random.uniform(time_key, (images.shape[0],))

        eps = jax.random.normal(noise_key, images.shape)
        x_t = flow_interpolate(images, eps, t[:, None, None, None])
        v_gt = flow_velocity(images, eps)
        v_pred = self.model_ema(x_t, t, support_embed, train=False, force_drop_ids=False)
        mse = jnp.mean((v_pred - v_gt) ** 2, axis=(1, 2, 3))
        loss = jnp.mean(mse)
        tbin = compute_t_bin_losses(mse, t, self.config['num_t_bins'])
        return loss, tbin

    # ── Attention entropy ──────────────────────────────────────────────────
    @partial(jax.pmap, axis_name='data')
    def get_attn_entropy(self, images, support_embed):
        """Returns (depth, num_heads) entropy matrix and (B,) timesteps."""
        time_key, noise_key = jax.random.split(self.rng, 2)
        if self.config['t_sampler'] == 'log-normal':
            t = jax.nn.sigmoid(jax.random.normal(time_key, (images.shape[0],)))
        else:
            t = jax.random.uniform(time_key, (images.shape[0],))
        eps = jax.random.normal(noise_key, images.shape)
        x_t = flow_interpolate(images, eps, t[:, None, None, None])

        _, attn_list = self.model_ema(
            x_t, t, support_embed, train=False, force_drop_ids=False, return_attn=True)
        entropies = jnp.stack([attention_entropy(aw) for aw in attn_list])  # (depth, H)
        return entropies, t

    # ── CFG sampling ───────────────────────────────────────────────────────
    @partial(jax.pmap, axis_name='data',
             in_axes=(0, 0, 0, 0), static_broadcasted_argnums=(4, 5))
    def sample_step(self, x, t_vec, support_embed, cfg=True, cfg_val=1.0):
        """One Euler step with optional CFG."""
        if not cfg or cfg_val == 0:
            return self.model_ema(x, t_vec, support_embed, train=False, force_drop_ids=False)
        B = x.shape[0]
        x2 = jnp.concatenate([x, x])
        t2 = jnp.concatenate([t_vec, t_vec])
        s2 = jnp.concatenate([support_embed, jnp.zeros_like(support_embed)])
        v = self.model_ema(x2, t2, s2, train=False, force_drop_ids=False)
        v_c, v_u = v[:B], v[B:]
        return v_u + cfg_val * (v_c - v_u)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(_):
    cfg = FLAGS.model
    for k, v in PRESETS[cfg.preset].items():
        cfg[k] = v

    np.random.seed(FLAGS.seed)
    devices = jax.local_devices()
    n_dev = len(devices)
    n_dev_global = jax.device_count()
    local_bs = FLAGS.batch_size // (n_dev_global // n_dev)
    print(f"Devices: {n_dev} local / {n_dev_global} global")
    print(f"Batch: {FLAGS.batch_size} global / {local_bs} local / {local_bs // n_dev} per-device")

    if jax.process_index() == 0:
        setup_wandb(cfg.to_dict(), **FLAGS.wandb)

    # ── Data ───────────────────────────────────────────────────────────────
    train_ds, train_cls = build_dataset(
        os.path.join(FLAGS.data_dir, 'train'), local_bs,
        image_size=cfg.image_size, num_sets=FLAGS.num_sets,
        is_train=True, seed=FLAGS.seed, debug_n=FLAGS.debug_overfit)
    val_ds, _ = build_dataset(
        os.path.join(FLAGS.data_dir, 'val'), local_bs,
        image_size=cfg.image_size, num_sets=FLAGS.num_sets,
        is_train=False, seed=FLAGS.seed + 1000)
    train_iter = iter(train_ds.as_numpy_iterator())
    val_iter = iter(val_ds.as_numpy_iterator())

    example = next(train_iter)
    example_img = example['target'][:1]  # (1, 224, 224, 3)

    # ── SigLIP2 (frozen) ──────────────────────────────────────────────────
    print("Loading SigLIP2 B/16 224×224...")
    siglip = SigLIP2Encoder.create(ckpt_path=FLAGS.siglip_ckpt, variant='B/16', res=224)

    # ── VAE (optional) ────────────────────────────────────────────────────
    if cfg.use_vae:
        vae = StableVAE.create()
        example_img = vae.encode(jax.random.PRNGKey(0), example_img)
        vae_rng = flax.jax_utils.replicate(jax.random.PRNGKey(42))
        vae_encode = jax.pmap(vae.encode)
        vae_decode = jax.jit(vae.decode)

    # ── DiT init ──────────────────────────────────────────────────────────
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, p_key, d_key = jax.random.split(rng, 3)
    print(f"Device memory: {jax.local_devices()[0].memory_stats()['bytes_limit'] / 2**30:.1f} GB")

    img_c = example_img.shape[-1]
    img_s = example_img.shape[1]

    dit = DiT(
        patch_size=cfg.patch_size, hidden_size=cfg.hidden_size,
        depth=cfg.depth, num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
        siglip_dim=cfg.siglip_dim, cond_dropout_prob=cfg.cond_dropout,
    )
    params = dit.init(
        {'params': p_key, 'cond_dropout': d_key},
        jnp.zeros((1, img_s, img_s, img_c)),   # x
        jnp.zeros((1,)),                         # t
        jnp.zeros((1, cfg.siglip_dim)),          # y (support embed)
    )['params']
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"DiT parameters: {n_params:,}")

    # ── Optimizer: warmup → cosine decay + grad clip + AdamW ──────────────
    warmup = optax.linear_schedule(0.0, cfg.lr, cfg.warmup_steps)
    cosine = optax.cosine_decay_schedule(
        cfg.lr, FLAGS.max_steps - cfg.warmup_steps, alpha=cfg.lr_min / cfg.lr)
    lr_schedule = optax.join_schedules([warmup, cosine], [cfg.warmup_steps])

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(lr_schedule, b1=cfg.beta1, b2=cfg.beta2, weight_decay=cfg.weight_decay),
    )

    ts = TrainState.create(dit, params, tx=tx)
    ts_ema = TrainState.create(dit, params)
    trainer = Trainer(rng, ts, ts_ema, cfg)

    if FLAGS.load_dir:
        trainer = Checkpoint(FLAGS.load_dir).load_model(trainer)
        print(f"Resumed from step {trainer.model.step}")

    trainer = flax.jax_utils.replicate(trainer, devices=devices)
    trainer = trainer.replace(rng=jax.random.split(rng, n_dev))

    # ── Helpers ────────────────────────────────────────────────────────────
    loss_ema = [None]  # mutable container for closure

    def decode_img(img):
        """Latent / pixel → displayable [0,1] numpy."""
        if cfg.use_vae:
            img = vae_decode(img[None])[0]
        return np.array(jnp.clip(img * 0.5 + 0.5, 0, 1))

    def encode_supports(batch_supports):
        """(B, 5, 224, 224, 3) → (B, siglip_dim)."""
        return siglip.encode_supports(batch_supports)

    # ── Eval function ─────────────────────────────────────────────────────
    def run_eval(step):
        val_batch = next(val_iter)
        val_img = val_batch['target']
        val_sup = encode_supports(val_batch['supports'])

        # Reshape for pmap
        val_img = val_img.reshape(n_dev, -1, *val_img.shape[1:])
        val_sup = val_sup.reshape(n_dev, -1, *val_sup.shape[1:])
        if cfg.use_vae:
            val_img = vae_encode(vae_rng, val_img)

        # Val loss
        v_loss, v_tbin = trainer.val_loss(val_img, val_sup)
        v_loss_f = float(np.array(v_loss).mean())
        v_tbin_f = np.array(v_tbin).mean(axis=0)

        if jax.process_index() == 0:
            gap = (loss_ema[0] - v_loss_f) if loss_ema[0] else 0.0
            log = {'val/loss': v_loss_f, 'val/train_val_gap': gap}
            for b in range(cfg.num_t_bins):
                log[f'val/loss_tbin_{b}'] = float(v_tbin_f[b])
            wandb.log(log, step=step)

        # Attention entropy
        try:
            ent_matrix, _ = trainer.get_attn_entropy(val_img, val_sup)
            ent = np.array(ent_matrix).mean(axis=0)  # avg devices → (depth, H)
            if jax.process_index() == 0:
                depth = cfg.depth
                alog = {}
                for li in [0, depth // 2, depth - 1]:
                    alog[f'attn/entropy_layer{li}'] = float(ent[li].mean())
                for h in range(cfg.num_heads):
                    alog[f'attn/entropy_head{h}_last'] = float(ent[-1, h])
                wandb.log(alog, step=step)
        except Exception as e:
            print(f"Attn entropy failed: {e}")

        # Sample grid
        if jax.process_index() == 0:
            _generate_samples(step, val_sup)

        del val_img, val_sup
        print(f"Eval done @ step {step}")

    def _generate_samples(step, sup_pmap):
        """Generate images with Euler sampling + CFG."""
        sup_viz = sup_pmap[:, :1]  # (ndev, 1, dim)
        key = jax.random.PRNGKey(42 + step)
        shape = (n_dev, 1, img_s, img_s, img_c)
        eps = jax.random.normal(key, shape)
        dt = 1.0 / cfg.denoise_steps

        for cfg_val in [0, cfg.cfg_scale]:
            x = eps
            for ti in range(cfg.denoise_steps):
                t_vec = jnp.full((n_dev, 1), ti / cfg.denoise_steps)
                x = x + trainer.sample_step(x, t_vec, sup_viz, True, cfg_val) * dt

            fig, axs = plt.subplots(1, min(n_dev, 8), figsize=(24, 4))
            if not hasattr(axs, '__len__'):
                axs = [axs]
            for j in range(min(n_dev, len(axs))):
                axs[j].imshow(decode_img(np.array(x)[j, 0]), vmin=0, vmax=1)
                axs[j].set_title(f"cfg={cfg_val}")
                axs[j].axis('off')
            plt.tight_layout()
            wandb.log({f'samples/cfg_{cfg_val}': wandb.Image(fig)}, step=step)
            plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════
    #  TRAIN LOOP
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print(f"  FSDiT training — {FLAGS.max_steps:,} steps, bs={FLAGS.batch_size}")
    print(f"{'═' * 60}\n")

    alpha = cfg.loss_ema_alpha
    for step in tqdm.tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, dynamic_ncols=True):
        t0 = time.time()

        # ── Get batch ──
        if not FLAGS.debug_overfit or step == 1:
            batch = next(train_iter)
            imgs = batch['target']
            sup = encode_supports(batch['supports'])
            imgs = imgs.reshape(n_dev, -1, *imgs.shape[1:])
            sup = sup.reshape(n_dev, -1, *sup.shape[1:])
            if cfg.use_vae:
                imgs = vae_encode(vae_rng, imgs)

        # ── Train step ──
        trainer, info = trainer.train_step(imgs, sup)
        dt_step = time.time() - t0

        # ── Log ──
        if step % FLAGS.log_interval == 0 and jax.process_index() == 0:
            train_loss = float(np.array(info['loss']).mean())
            if loss_ema[0] is None:
                loss_ema[0] = train_loss
            else:
                loss_ema[0] = alpha * loss_ema[0] + (1 - alpha) * train_loss

            log = {
                'train/loss': train_loss,
                'train/loss_ema': loss_ema[0],
                'train/grad_norm': float(np.array(info['grad_norm']).mean()),
                'train/lr': float(lr_schedule(step)),
                'train/step_time': dt_step,
                'train/param_norm': float(np.array(info['param_norm']).mean()),
            }
            tbin = np.array(info['tbin_loss']).mean(axis=0)
            for b in range(cfg.num_t_bins):
                log[f'train/loss_tbin_{b}'] = float(tbin[b])
            wandb.log(log, step=step)

        # ── Eval ──
        if step % FLAGS.eval_interval == 0 or step == 1000:
            run_eval(step)

        # ── Save ──
        if step % FLAGS.save_interval == 0 and FLAGS.save_dir and jax.process_index() == 0:
            single = flax.jax_utils.unreplicate(trainer)
            cp = Checkpoint(FLAGS.save_dir, parallel=False)
            cp.set_model(single)
            cp.save()
            del cp, single


if __name__ == '__main__':
    app.run(main)
