"""
train.py — FSDiT: Few-Shot Diffusion Transformer Training.

Flow-matching DiT conditioned on SigLIP2 support embeddings.
Supports:
  - online mode (default): dataset returns support_paths, SigLIP encoded at runtime
  - tfrecord mode        : dataset returns precomputed support embeddings
"""

from typing import Any
import os
import time
import warnings
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

# Suppress noisy diffusers/flax deprecation warnings emitted during module import.
warnings.filterwarnings(
    "ignore",
    message=".*Flax classes are deprecated and will be removed in Diffusers.*",
    category=FutureWarning,
)
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from model import GramDiT
from dataset import build_dataset
from utils.train_state import TrainState, target_update
from utils.checkpoint import Checkpoint
from utils.stable_vae import StableVAE
from utils.online_support_encoder import OnlineSupportEncoder
from utils.wandb_utils import setup_wandb, default_wandb_config
from utils.fid import get_fid_network, fid_from_stats
from utils.logging import (
    compute_condition_distribution_metrics,
    log_train_metrics,
    log_eval_metrics,
    log_attn_entropy,
    AsyncLogger,
)

# Lazy import for grain mode (only needed when --data_mode=grain)
_grain_dataset_mod = None
def _get_grain_dataset_mod():
    global _grain_dataset_mod
    if _grain_dataset_mod is None:
        import dataset_grain as m
        _grain_dataset_mod = m
    return _grain_dataset_mod

# ═══════════════════════════════════════════════════════════════════════════════
#  Flags & Config
# ═══════════════════════════════════════════════════════════════════════════════

FLAGS = flags.FLAGS
# Paths
flags.DEFINE_string('data_dir', '/kaggle/input/datasets/arjunashok33/miniimagenet',
                    'miniImageNet root (contains train/, val/, test/).')
flags.DEFINE_enum('data_mode', 'online', ['online', 'tfrecord', 'grain'],
                  'Data mode: online (support_paths + runtime SigLIP), tfrecord, or grain (ArrayRecord).')
flags.DEFINE_string('episode_tfrecord_dir', None,
                    'TFRecord episode root with train/*.tfrecord and val/*.tfrecord.')
flags.DEFINE_string('tfrecord_compression_type', 'GZIP',
                    'Compression type for TFRecord episode shards ("", "GZIP").')
flags.DEFINE_string('grain_arecord_dir', None,
                    'Directory containing train.arecord and val.arecord for grain mode.')
flags.DEFINE_string('grain_path_remap', None,
                    'Remap stored image paths. Single: "old:new". '
                    'Multiple: "old1:new1;old2:new2". '
                    'E.g. "/workspace/data:/kaggle/input/data;/home/user:/kaggle/user"')
flags.DEFINE_integer('online_cache_items', 1024,
                     'Max LRU cache items for online support embeddings.')
flags.DEFINE_integer('online_siglip_batch_size', 256,
                     'Mini-batch size for online SigLIP encoding.')
flags.DEFINE_bool('online_siglip_no_pmap', False,
                  'Disable pmap for online SigLIP encoding.')
flags.DEFINE_string('load_dir', None,  'Resume from checkpoint.')
flags.DEFINE_string('save_dir', None,  'Save checkpoints here.')
flags.DEFINE_string('fid_stats', None, 'Precomputed FID stats .npz.')
# Training
flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('batch_size', 128, 'Global batch size.')
flags.DEFINE_integer('max_steps', 200_000, 'Total training steps.')
flags.DEFINE_integer('num_sets', 100, 'Sets per class (each set = 6 images).')
flags.DEFINE_integer('debug_overfit', 0, 'Overfit on N samples (0 = off).')
flags.DEFINE_float('cond_dropout', None, 'Override cond_dropout (CFG dropout rate). Default: 0.1.')
flags.DEFINE_float('weight_decay', None, 'Override weight_decay. Default: 0.01.')
flags.DEFINE_float('lr', None, 'Override peak learning rate. Default: 1e-4.')
flags.DEFINE_bool('use_support_seq', False, 'Unused in gram-adaln (no cross-attn). Kept for data pipeline backward compat.')
flags.DEFINE_string('val_split', 'val', "Split name to use for validation: 'val' or 'test'.")
flags.DEFINE_integer('gram_rank_s', 16, 'Self-gram low-rank dimension (0 = disabled).')
flags.DEFINE_integer('gram_rank_c', 16, 'Cross-gram low-rank dimension (0 = disabled).')
flags.DEFINE_float('gram_dropout', 0.1, 'Dropout rate on gram branch outputs.')
flags.DEFINE_bool('suppress_diffusers_warnings', True, 'Suppress repeated diffusers Flax deprecation warnings.')
flags.DEFINE_bool('log_model_debug', True, 'Log model activation/condition debug metrics.')
# Logging
flags.DEFINE_integer('log_interval', 500, 'Train metric logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Validation + attention entropy interval.')
flags.DEFINE_integer('fid_interval', 25000, 'FID / sample grid interval.')
flags.DEFINE_integer('save_interval', 25000, 'Checkpoint interval.')
flags.DEFINE_integer('perf_log_interval', 100, 'Performance timing logging interval.')
flags.DEFINE_integer('cond_hist_interval', 5000, 'Condition distribution histogram logging interval.')

model_config = ml_collections.ConfigDict({
    # ── Optimizer ──
    'lr': 1e-4,              # peak learning rate
    'lr_min': 1e-6,          # cosine floor
    'warmup_steps': 5000,    # linear warmup (2.5% of 200k)
    'beta1': 0.9,
    'beta2': 0.99,
    'weight_decay': 0.01,    # AdamW regularization
    'grad_clip': 1.0,        # max gradient norm
    # ── GramDiT Architecture ──
    'hidden_size': 768,
    'patch_size': 2,
    'depth': 12,
    'num_heads': 12,
    'mlp_ratio': 4,
    'preset': 'big',
    # ── FSDiT Conditioning ──
    'siglip_dim': 768,       # CLS token dim (SigLIP2-B/16 or DINOv2-B)
    'cond_dropout': 0.1,     # CFG: zero support 10% of time
    # ── Gram Ranks ──
    'gram_rank_s': 16,       # self-gram rank (0 = disabled)
    'gram_rank_c': 16,       # cross-gram rank (0 = disabled)
    'gram_dropout': 0.1,     # dropout on gram branch outputs
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
    'log_model_debug': 1,    # return model debug tensors during training
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
    def train_step(self, images, support_pooled, support_seq):
        """One training step. Returns (new_trainer, info_dict).
        Note: support_seq is accepted for API compatibility but not passed to GramDiT.
        """
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
            sup_pooled = support_pooled.astype(images.dtype)
            # split one extra key for gram dropout (use distinct names to avoid Python closure issue)
            cond_key_inner, drop_key = jax.random.split(cond_key)
            # GramDiT uses only pooled CLS (no y_seq / context)
            if self.config.get('log_model_debug', 1):
                v_pred, dbg = self.model(
                    x_t, t, sup_pooled, train=True,
                    return_debug=True,
                    rngs={'cond_dropout': cond_key_inner, 'dropout': drop_key},
                    params=params,
                )
            else:
                v_pred = self.model(
                    x_t, t, sup_pooled, train=True,
                    rngs={'cond_dropout': cond_key_inner, 'dropout': drop_key},
                    params=params,
                )
                dbg = None
            mse = (v_pred - v_gt) ** 2
            loss = jnp.mean(mse)

            # Per-sample, per-tbin breakdown
            loss_ps = jnp.mean(mse, axis=(1, 2, 3))
            tbin = compute_t_bin_losses(loss_ps, t, num_bins)

            info = {
                'loss': loss,
                'v_abs': jnp.abs(v_gt).mean(),
                'v_pred_abs': jnp.abs(v_pred).mean(),
                'tbin_loss': tbin,
            }
            if dbg is not None:
                info.update({
                    'dbg/t_emb_abs_mean': dbg['t_emb_abs_mean'],
                    'dbg/y_emb_abs_mean': dbg['y_emb_abs_mean'],
                    'dbg/c_abs_mean': dbg['c_abs_mean'],
                    'dbg/c_l2_mean': dbg['c_l2_mean'],
                    'dbg/support_pooled_abs_mean': dbg['support_pooled_abs_mean'],
                    'dbg/support_pooled_l2_mean': dbg['support_pooled_l2_mean'],
                    'dbg/act_abs_per_layer': dbg['act_abs_per_layer'],
                    'dbg/act_rms_per_layer': dbg['act_rms_per_layer'],
                })
            return loss, info

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
    def val_loss(self, images, support_pooled, support_seq):
        """Compute val loss + t-bin breakdown (no dropout, uses EMA model).
        Note: support_seq accepted for API compatibility but not passed to GramDiT.
        """
        time_key, noise_key = jax.random.split(self.rng, 2)
        if self.config['t_sampler'] == 'log-normal':
            t = jax.nn.sigmoid(jax.random.normal(time_key, (images.shape[0],)))
        else:
            t = jax.random.uniform(time_key, (images.shape[0],))

        eps = jax.random.normal(noise_key, images.shape)
        x_t = flow_interpolate(images, eps, t[:, None, None, None])
        v_gt = flow_velocity(images, eps)
        sup_pooled = support_pooled.astype(images.dtype)
        v_pred = self.model_ema(
            x_t, t, sup_pooled,
            train=False, force_drop_ids=False,
        )
        mse = jnp.mean((v_pred - v_gt) ** 2, axis=(1, 2, 3))
        loss = jnp.mean(mse)
        tbin = compute_t_bin_losses(mse, t, self.config['num_t_bins'])
        return loss, tbin

    # ── Attention entropy ──────────────────────────────────────────────────
    @partial(jax.pmap, axis_name='data')
    def get_attn_entropy(self, images, support_pooled, support_seq):
        """Returns (depth, num_heads) entropy matrix and (B,) timesteps."""
        time_key, noise_key = jax.random.split(self.rng, 2)
        if self.config['t_sampler'] == 'log-normal':
            t = jax.nn.sigmoid(jax.random.normal(time_key, (images.shape[0],)))
        else:
            t = jax.random.uniform(time_key, (images.shape[0],))
        eps = jax.random.normal(noise_key, images.shape)
        x_t = flow_interpolate(images, eps, t[:, None, None, None])
        sup_pooled = support_pooled.astype(images.dtype)

        _, attn_list = self.model_ema(
            x_t, t, sup_pooled,
            train=False, force_drop_ids=False, return_attn=True,
        )
        entropies = jnp.stack([attention_entropy(aw) for aw in attn_list])  # (depth, H)
        return entropies, t

    # ── CFG sampling ───────────────────────────────────────────────────────
    @partial(jax.pmap, axis_name='data',
             in_axes=(0, 0, 0, 0, 0), static_broadcasted_argnums=(5, 6))
    def sample_step(self, x, t_vec, support_pooled, support_seq, cfg=True, cfg_val=1.0):
        """One Euler step with optional CFG.
        Note: support_seq accepted for API compatibility but not passed to GramDiT.
        """
        sup_pooled = support_pooled.astype(x.dtype)
        if not cfg or cfg_val == 0:
            return self.model_ema(
                x, t_vec, sup_pooled,
                train=False, force_drop_ids=False,
            )
        B = x.shape[0]
        x2 = jnp.concatenate([x, x])
        t2 = jnp.concatenate([t_vec, t_vec])
        pooled2 = jnp.concatenate([sup_pooled, jnp.zeros_like(sup_pooled)])
        v = self.model_ema(x2, t2, pooled2, train=False, force_drop_ids=False)
        v_c, v_u = v[:B], v[B:]
        return v_u + cfg_val * (v_c - v_u)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(_):
    cfg = FLAGS.model
    for k, v in PRESETS[cfg.preset].items():
        cfg[k] = v
    cfg.log_model_debug = int(FLAGS.log_model_debug)
    # Gram rank CLI overrides
    cfg.gram_rank_s = FLAGS.gram_rank_s
    cfg.gram_rank_c = FLAGS.gram_rank_c
    cfg.gram_dropout = FLAGS.gram_dropout
    # Other CLI overrides for tuning
    if FLAGS.cond_dropout is not None:
        cfg.cond_dropout = FLAGS.cond_dropout
    if FLAGS.weight_decay is not None:
        cfg.weight_decay = FLAGS.weight_decay
    if FLAGS.lr is not None:
        cfg.lr = FLAGS.lr

    if FLAGS.suppress_diffusers_warnings:
        warnings.filterwarnings(
            "ignore",
            message=".*Flax classes are deprecated and will be removed in Diffusers.*",
        )

    np.random.seed(FLAGS.seed)
    devices = jax.local_devices()
    n_dev = len(devices)
    n_dev_global = jax.device_count()
    local_bs = FLAGS.batch_size // (n_dev_global // n_dev)
    print(f"Devices: {n_dev} local / {n_dev_global} global")
    print(f"Batch: {FLAGS.batch_size} global / {local_bs} local / {local_bs // n_dev} per-device")
    if FLAGS.data_mode == 'tfrecord':
        if not FLAGS.episode_tfrecord_dir:
            raise ValueError(
                "When --data_mode=tfrecord, please set --episode_tfrecord_dir "
                "(expected train/*.tfrecord and val/*.tfrecord)."
            )
        print(f"Data mode: tfrecord ({FLAGS.episode_tfrecord_dir})")
    elif FLAGS.data_mode == 'grain':
        if not FLAGS.grain_arecord_dir:
            raise ValueError(
                "When --data_mode=grain, please set --grain_arecord_dir "
                "(expected directory with train.arecord and val.arecord)."
            )
        print(f"Data mode: grain ({FLAGS.grain_arecord_dir})")
    else:
        print("Data mode: online (support_paths + runtime SigLIP)")

    if jax.process_index() == 0:
        setup_wandb(cfg.to_dict(), **FLAGS.wandb)
    async_logger = AsyncLogger() if jax.process_index() == 0 else None

    # ── Data ───────────────────────────────────────────────────────────────
    if FLAGS.data_mode == 'grain':
        grain_mod = _get_grain_dataset_mod()
        # Parse path remap(s): "old:new" or "old1:new1;old2:new2"
        path_remaps = None
        if FLAGS.grain_path_remap:
            path_remaps = []
            for pair in FLAGS.grain_path_remap.split(';'):
                pair = pair.strip()
                if not pair:
                    continue
                parts = pair.split(':', 1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Each remap must be 'old:new', got: '{pair}'"
                    )
                path_remaps.append((parts[0], parts[1]))
                print(f"  Path remap: '{parts[0]}' → '{parts[1]}'")
        train_iter = grain_mod.build_grain_dataset(
            arecord_dir=FLAGS.grain_arecord_dir,
            split='train',
            batch_size=local_bs,
            image_size=cfg.image_size,
            is_train=True,
            seed=FLAGS.seed,
            load_support_seq=FLAGS.use_support_seq,
            path_prefix_remaps=path_remaps,
        )
        val_iter = grain_mod.build_grain_dataset(
            arecord_dir=FLAGS.grain_arecord_dir,
            split=FLAGS.val_split,
            batch_size=local_bs,
            image_size=cfg.image_size,
            is_train=False,
            seed=FLAGS.seed + 1000,
            load_support_seq=FLAGS.use_support_seq,
            path_prefix_remaps=path_remaps,
        )
    else:
        train_pattern = None
        val_pattern = None
        if FLAGS.data_mode == 'tfrecord':
            train_pattern = os.path.join(FLAGS.episode_tfrecord_dir, 'train', 'train-*.tfrecord')
            val_pattern = os.path.join(FLAGS.episode_tfrecord_dir, 'val', 'val-*.tfrecord')

        train_ds, _ = build_dataset(
            os.path.join(FLAGS.data_dir, 'train'), local_bs,
            image_size=cfg.image_size, num_sets=FLAGS.num_sets,
            is_train=True, seed=FLAGS.seed, debug_n=FLAGS.debug_overfit,
            load_support_seq=FLAGS.use_support_seq,
            data_mode=FLAGS.data_mode,
            episode_tfrecord_pattern=train_pattern,
            tfrecord_compression_type=FLAGS.tfrecord_compression_type,
        )
        val_ds, _ = build_dataset(
            os.path.join(FLAGS.data_dir, 'val'), local_bs,
            image_size=cfg.image_size, num_sets=FLAGS.num_sets,
            is_train=False, seed=FLAGS.seed + 1000,
            load_support_seq=FLAGS.use_support_seq,
            data_mode=FLAGS.data_mode,
            episode_tfrecord_pattern=val_pattern,
            tfrecord_compression_type=FLAGS.tfrecord_compression_type,
        )
        train_iter = iter(train_ds.as_numpy_iterator())
        val_iter = iter(val_ds.as_numpy_iterator())

    online_encoder = None
    if FLAGS.data_mode == 'online':
        print("Initializing OnlineSupportEncoder...")
        online_encoder = OnlineSupportEncoder(
            variant='B/16',
            image_size=cfg.image_size,
            cache_items=FLAGS.online_cache_items,
            batch_size=FLAGS.online_siglip_batch_size,
            no_pmap=FLAGS.online_siglip_no_pmap,
            warmup_need_seq=bool(FLAGS.use_support_seq),
        )

    example = next(train_iter)
    example_img = example['target'][:1]  # (1, 224, 224, 3)
    n_sup_tokens = 5 * 196
    if FLAGS.data_mode == 'tfrecord':
        example_sup_seq = example['supports_seq'][:1]  # (1, 5, 196, 768)
        n_sup_tokens = example_sup_seq.shape[1] * example_sup_seq.shape[2]

    # ── VAE (optional) ────────────────────────────────────────────────────
    if cfg.use_vae:
        vae = StableVAE.create()
        example_img = vae.encode(jax.random.PRNGKey(0), example_img)
        vae_rng = flax.jax_utils.replicate(jax.random.PRNGKey(42))
        vae_encode = jax.pmap(vae.encode)
        vae_decode = jax.jit(vae.decode)

    # ── GramDiT init ──────────────────────────────────────────────────────────────────
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, p_key, d_key = jax.random.split(rng, 3)
    print(f"Device memory: {jax.local_devices()[0].memory_stats()['bytes_limit'] / 2**30:.1f} GB")

    img_c = example_img.shape[-1]
    img_s = example_img.shape[1]

    dit = GramDiT(
        patch_size=cfg.patch_size, hidden_size=cfg.hidden_size,
        depth=cfg.depth, num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
        siglip_dim=cfg.siglip_dim, cond_dropout_prob=cfg.cond_dropout,
        gram_rank_s=cfg.gram_rank_s, gram_rank_c=cfg.gram_rank_c,
        gram_dropout=cfg.gram_dropout,
    )
    params = dit.init(
        {'params': p_key, 'cond_dropout': d_key, 'dropout': d_key},
        jnp.zeros((1, img_s, img_s, img_c)),   # x
        jnp.zeros((1,)),                        # t
        jnp.zeros((1, cfg.siglip_dim)),         # y_pooled (CLS token)
    )['params']
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"GramDiT parameters: {n_params:,} | gram_rank_s={cfg.gram_rank_s}, gram_rank_c={cfg.gram_rank_c}, gram_dropout={cfg.gram_dropout}")

    # ── Optimizer: warmup → cosine decay + grad clip + AdamW ──────────────
    max_steps = int(FLAGS.max_steps)
    warmup_steps = int(min(cfg.warmup_steps, max_steps))
    if max_steps <= cfg.warmup_steps:
        # Short benchmark runs: only warmup schedule (no cosine phase).
        lr_schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=cfg.lr,
            transition_steps=max(max_steps, 1),
        )
        print(
            f"[LR] warmup-only schedule: max_steps={max_steps} <= warmup_steps={cfg.warmup_steps}"
        )
    else:
        warmup = optax.linear_schedule(0.0, cfg.lr, warmup_steps)
        cosine = optax.cosine_decay_schedule(
            cfg.lr,
            max(max_steps - warmup_steps, 1),
            alpha=cfg.lr_min / cfg.lr,
        )
        lr_schedule = optax.join_schedules([warmup, cosine], [warmup_steps])

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

    def prepare_support_condition(batch):
        """
        Self-reconstruction mode: encode the TARGET image with SigLIP
        as the condition, so condition = exact target.

        Returns:
          pooled_global: (B,768) float32 (for stats)
          pooled_model : (B,768) float16 (for model input)
          seq_model    : (B,T,768) float16 (for model input)
          siglip_stats : dict or None
        """
        siglip_stats = None

        if online_encoder is not None:
            # Self-reconstruction: encode target image as its own condition
            # target_paths shape: (B,) strings — each is path to the target img
            # We wrap each target path as a (B, 1) "fake 1-shot support" array
            target_paths = batch.get('target_path', None)
            if target_paths is not None:
                # Make shape (B, 1) so online_encoder treats it as 1-shot
                if isinstance(target_paths, np.ndarray):
                    target_paths_1shot = target_paths.reshape(-1, 1)
                else:
                    target_paths_1shot = np.array(target_paths).reshape(-1, 1)
                pooled_5, seq_5, siglip_stats = online_encoder.encode_paths(
                    target_paths_1shot,
                    need_seq=bool(FLAGS.use_support_seq),
                )
            else:
                # Fallback: use stored support embeddings
                pooled_5 = batch['supports_pooled']
                seq_5 = batch['supports_seq'] if FLAGS.use_support_seq else None
        else:
            # No online encoder available, use stored embeddings
            pooled_5 = batch['supports_pooled']
            seq_5 = batch['supports_seq'] if FLAGS.use_support_seq else None

        pooled_global = np.mean(pooled_5, axis=1, dtype=np.float32)
        pooled_model = pooled_global.astype(np.float16)
        # GramDiT does not use seq tokens; return a dummy placeholder for API compat
        seq_model = np.zeros(
            (batch['target'].shape[0], 1, cfg.siglip_dim),
            dtype=np.float16,
        )
        return pooled_global, pooled_model, seq_model, siglip_stats

    # ── Eval function ─────────────────────────────────────────────────────
    def run_eval(step):
        if FLAGS.debug_overfit:
            # Overfit mode: eval on the SAME batch used for training
            val_batch = batch  # reuse last training batch (captured from closure)
        else:
            val_batch = next(val_iter)
        val_img = val_batch['target']
        val_class_ids = val_batch['class_id'].astype(np.int32)
        val_sup_pooled_global, val_sup_pooled, val_sup_seq, val_siglip_stats = (
            prepare_support_condition(val_batch)
        )

        # Reshape for pmap
        val_img = val_img.reshape(n_dev, -1, *val_img.shape[1:])
        val_sup_seq = val_sup_seq.reshape(n_dev, -1, *val_sup_seq.shape[1:])
        val_sup_pooled = val_sup_pooled.reshape(n_dev, -1, val_sup_pooled.shape[-1])
        if cfg.use_vae:
            val_img = vae_encode(vae_rng, val_img)

        # Val loss
        v_loss, v_tbin = trainer.val_loss(val_img, val_sup_pooled, val_sup_seq)
        v_loss_f = float(np.array(v_loss).mean())
        v_tbin_f = np.array(v_tbin).mean(axis=0)

        if jax.process_index() == 0:
            log_eval_metrics(
                step, v_loss_f, v_tbin_f, loss_ema, cfg,
                val_sup_pooled_global, val_class_ids,
                data_mode=FLAGS.data_mode, val_siglip_stats=val_siglip_stats,
                cond_hist_interval=FLAGS.cond_hist_interval,
                async_logger=async_logger,
            )

        # Attention entropy
        try:
            ent_matrix, _ = trainer.get_attn_entropy(val_img, val_sup_pooled, val_sup_seq)
            if jax.process_index() == 0:
                log_attn_entropy(step, ent_matrix, cfg, async_logger=async_logger)
        except Exception as e:
            print(f"Attn entropy failed: {e}")

        # Sample grid
        if jax.process_index() == 0:
            _generate_samples(step, val_sup_pooled, val_sup_seq)

        del val_img, val_sup_pooled, val_sup_seq
        print(f"Eval done @ step {step}")

    def _generate_samples(step, sup_pooled_pmap, sup_seq_pmap):
        """Generate images with Euler sampling + CFG."""
        sup_pooled_viz = sup_pooled_pmap[:, :1]  # (ndev, 1, dim)
        sup_seq_viz = sup_seq_pmap[:, :1]        # (ndev, 1, T, dim)
        key = jax.random.PRNGKey(42 + step)
        shape = (n_dev, 1, img_s, img_s, img_c)
        eps = jax.random.normal(key, shape)
        dt = 1.0 / cfg.denoise_steps

        for cfg_val in [0, cfg.cfg_scale]:
            x = eps
            for ti in range(cfg.denoise_steps):
                t_vec = jnp.full((n_dev, 1), ti / cfg.denoise_steps)
                x = x + trainer.sample_step(
                    x, t_vec, sup_pooled_viz, sup_seq_viz, True, cfg_val
                ) * dt

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

    def _compute_fid(step, n_samples=2048):
        """Compute FID: real stats from val_iter (cached), generated via Euler+CFG.

        Real image statistics are computed lazily from the validation iterator
        on the FIRST call and then cached as function attributes.
        Conditioning (sup_pooled/sup_seq) is fetched fresh from val_iter.
        """
        if jax.process_index() != 0:
            return

        # ── Lazy-init InceptionV3 ────────────────────────────────────────────
        if not hasattr(_compute_fid, 'inception'):
            print('[FID] loading InceptionV3 (pre-trained weights)...')
            _compute_fid.inception = get_fid_network()
            print('[FID] InceptionV3 ready.')
        inception_fn = _compute_fid.inception

        def _imgs_to_acts(imgs_list):
            """Decode list of [0,1] numpy images → InceptionV3 activations (N,2048)."""
            acts_all = []
            # Process in chunks of n_dev (one image per device)
            for start in range(0, len(imgs_list), n_dev):
                chunk = imgs_list[start:start + n_dev]
                # Pad to n_dev if needed (last chunk)
                while len(chunk) < n_dev:
                    chunk.append(chunk[-1])
                imgs_299 = []
                for img_px in chunk:
                    img_m1_1 = (img_px.astype(np.float32) * 2.0) - 1.0
                    img_299  = np.array(jax.image.resize(
                        img_m1_1, (299, 299, img_m1_1.shape[-1]), method='bilinear'
                    ))
                    imgs_299.append(img_299)
                imgs_np   = np.stack(imgs_299, axis=0)  # (n_dev, 299, 299, C)
                imgs_pmap = imgs_np[:, None]             # (n_dev, 1,   299, 299, C)
                acts = np.array(inception_fn(imgs_pmap)).reshape(n_dev, 2048)
                acts_all.append(acts)
            return np.concatenate(acts_all, axis=0)

        # ── Compute real image stats (once) ──────────────────────────────────
        if not hasattr(_compute_fid, 'mu_real'):
            print('[FID] computing real image statistics from val_iter...')
            real_imgs = []
            # Iterate val_iter until we have enough images
            val_it = val_iter  # closure over outer val_iter
            while len(real_imgs) < n_samples:
                try:
                    vbatch = next(val_it)
                except StopIteration:
                    print('[FID] val_iter exhausted before reaching n_samples.')
                    break
                # Decode target images from the batch
                targets = vbatch['target']                   # (local_bs, H, W, C_lat)
                if cfg.use_vae:
                    # vae_encode expects (n_dev, sub_bs, H, W, C)
                    # but here we just need decoded pixels → encode first? No:
                    # targets from val_iter are already raw images (not latents)
                    # Only train images go through vae_encode inside train loop.
                    # The val images are kept as pixels until vae_encode in run_eval.
                    # So here: targets may be pixel images (H=224, W=224, C=3)
                    # or latents depending on data mode. We just decode them.
                    pass
                for i in range(targets.shape[0]):
                    img = targets[i]
                    # If VAE used, targets from grain are already latent-encoded
                    # (done outside during data loading). Decode them.
                    if cfg.use_vae and img.shape[-1] == img_c:
                        # img is a latent: (H_lat, W_lat, img_c)
                        img_px = decode_img(img)             # → [0,1] RGB
                    else:
                        # Already a pixel image in [-1,1] or [0,1]
                        img_px = np.clip(img * 0.5 + 0.5, 0, 1).astype(np.float32)
                    real_imgs.append(img_px)
                    if len(real_imgs) >= n_samples:
                        break

            print(f'[FID] collected {len(real_imgs)} real images.')
            real_acts = _imgs_to_acts(real_imgs[:n_samples])
            _compute_fid.mu_real    = np.mean(real_acts, axis=0)
            _compute_fid.sigma_real = np.cov(real_acts, rowvar=False)
            print('[FID] real image stats ready.')

        mu_real    = _compute_fid.mu_real
        sigma_real = _compute_fid.sigma_real

        # ── Get conditioning from val_iter ───────────────────────────────────
        try:
            cond_batch = next(val_iter)
        except StopIteration:
            print('[FID] val_iter exhausted, skipping FID this step.')
            return
        _, sup_pooled_pmap, sup_seq_pmap, _ = prepare_support_condition(cond_batch)
        sup_pooled_pmap = sup_pooled_pmap.reshape(n_dev, -1, sup_pooled_pmap.shape[-1])
        sup_seq_pmap    = sup_seq_pmap.reshape(n_dev, -1, *sup_seq_pmap.shape[1:])

        # ── Generate samples with Euler + CFG ────────────────────────────────
        sup_pooled_viz = sup_pooled_pmap[:, :1]
        sup_seq_viz    = sup_seq_pmap[:, :1]
        dt = 1.0 / cfg.denoise_steps
        n_batches = int(np.ceil(n_samples / n_dev))
        print(f'[FID] generating {n_batches * n_dev} images @ step {step}...')

        gen_imgs = []
        for bi in range(n_batches):
            key = jax.random.PRNGKey(10000 + step + bi)
            x = jax.random.normal(key, (n_dev, 1, img_s, img_s, img_c))
            for ti in range(cfg.denoise_steps):
                t_vec = jnp.full((n_dev, 1), ti / cfg.denoise_steps)
                x = x + trainer.sample_step(
                    x, t_vec, sup_pooled_viz, sup_seq_viz, True, cfg.cfg_scale
                ) * dt
            for di in range(n_dev):
                gen_imgs.append(decode_img(np.array(x)[di, 0]))   # [0,1]

        gen_acts  = _imgs_to_acts(gen_imgs[:n_samples])
        mu_gen    = np.mean(gen_acts, axis=0)
        sigma_gen = np.cov(gen_acts, rowvar=False)
        fid = float(fid_from_stats(mu_real, sigma_real, mu_gen, sigma_gen))
        print(f'[FID] step {step}: FID = {fid:.2f}')
        wandb.log({'fid': fid}, step=step)

    # ═══════════════════════════════════════════════════════════════════════
    #  TRAIN LOOP
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print(f"  FSDiT training — {FLAGS.max_steps:,} steps, bs={FLAGS.batch_size}")
    print(f"{'═' * 60}\n")

    sup_pooled_global = None
    class_ids_global = None
    for step in tqdm.tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, dynamic_ncols=True):
        iter_t0 = time.time()
        data_time = 0.0
        siglip_time = 0.0
        siglip_stats = None
        vae_time = 0.0

        # ── Get batch ──
        if not FLAGS.debug_overfit or step == 1:
            t_data0 = time.time()
            batch = next(train_iter)
            data_time = time.time() - t_data0

            imgs = batch['target']
            class_ids_global = batch['class_id'].astype(np.int32)
            t_sig0 = time.time()
            sup_pooled_global, sup_pooled, sup_seq, siglip_stats = prepare_support_condition(batch)
            siglip_time = time.time() - t_sig0
            if siglip_stats is not None:
                siglip_time = float(siglip_stats['encode_time'])

            imgs = imgs.reshape(n_dev, -1, *imgs.shape[1:])
            sup_seq = sup_seq.reshape(n_dev, -1, *sup_seq.shape[1:])
            sup_pooled = sup_pooled.reshape(n_dev, -1, sup_pooled.shape[-1])
            if cfg.use_vae:
                t_vae0 = time.time()
                imgs = vae_encode(vae_rng, imgs)
                vae_time = time.time() - t_vae0

        # ── Train step ──
        t_step0 = time.time()
        trainer, info = trainer.train_step(imgs, sup_pooled, sup_seq)
        step_time = time.time() - t_step0
        dt_step = time.time() - iter_t0

        # ── Log ──
        if step % FLAGS.log_interval == 0 and jax.process_index() == 0:
            log_train_metrics(
                step, info, loss_ema, lr_schedule, cfg, dt_step=dt_step,
                sup_pooled_global=sup_pooled_global,
                class_ids_global=class_ids_global,
                log_model_debug=FLAGS.log_model_debug,
                cond_hist_interval=FLAGS.cond_hist_interval,
                async_logger=async_logger,
            )



        # ── Eval ──
        if step % FLAGS.eval_interval == 0 or step == 1000:
            run_eval(step)

        # ── Save + FID ──
        if step % FLAGS.save_interval == 0 and FLAGS.save_dir and jax.process_index() == 0:
            single = flax.jax_utils.unreplicate(trainer)
            ckpt_path = os.path.join(FLAGS.save_dir, f'checkpoint_step_{step}.pkl')
            cp = Checkpoint(ckpt_path, parallel=False)
            cp.set_model(single)
            cp.save()
            del cp, single
            # FID computed at same cadence as checkpoint saves (val stats lazily cached)
            _compute_fid(step)

    # Flush async logger before exit
    if async_logger is not None:
        async_logger.wait()


if __name__ == '__main__':
    app.run(main)
