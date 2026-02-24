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

from model import DiT
from dataset import build_dataset
from utils.train_state import TrainState, target_update
from utils.checkpoint import Checkpoint
from utils.stable_vae import StableVAE
from utils.online_support_encoder import OnlineSupportEncoder
from utils.wandb_utils import setup_wandb, default_wandb_config

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
                    'Remap stored image paths: "old_prefix:new_prefix". '
                    'E.g. "/workspace/data/miniimagenet:/kaggle/input/datasets/arjunashok33/miniimagenet"')
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
flags.DEFINE_bool('use_support_seq', True, 'Use support sequence context for cross-attention.')
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
    'use_support_seq': 1,    # whether to use support sequence context
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


def compute_condition_distribution_metrics(cond_vec, class_ids):
    """
    Numpy metrics for SigLIP pooled-condition distribution in a batch.
    Returns:
      metrics: scalar dict
      same_vals: cosine values for same-class pairs
      diff_vals: cosine values for diff-class pairs
    """
    x = np.asarray(cond_vec, dtype=np.float32)
    y = np.asarray(class_ids, dtype=np.int32).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        return {}, np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    bsz = x.shape[0]
    metrics = {
        'cond/support_pooled_abs_mean': float(np.mean(np.abs(x))),
        'cond/support_pooled_l2_mean': float(np.mean(np.linalg.norm(x, axis=-1))),
        'cond/support_pooled_dim_std_mean': float(np.mean(np.std(x, axis=0))),
    }

    if bsz < 2:
        metrics['cond/same_class_pair_ratio'] = 0.0
        metrics['cond/same_class_cos_mean'] = 0.0
        metrics['cond/same_class_cos_std'] = 0.0
        metrics['cond/diff_class_cos_mean'] = 0.0
        metrics['cond/diff_class_cos_std'] = 0.0
        return metrics, np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    x_norm = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)
    cos = x_norm @ x_norm.T
    same = (y[:, None] == y[None, :])
    iu = np.triu_indices(bsz, 1)
    cos_u = cos[iu]
    same_u = same[iu]
    same_vals = cos_u[same_u]
    diff_vals = cos_u[~same_u]
    num_pairs = float(cos_u.shape[0])
    metrics['cond/same_class_pair_ratio'] = float(same_vals.shape[0] / max(num_pairs, 1.0))
    metrics['cond/same_class_cos_mean'] = float(np.mean(same_vals)) if same_vals.size else 0.0
    metrics['cond/same_class_cos_std'] = float(np.std(same_vals)) if same_vals.size else 0.0
    metrics['cond/diff_class_cos_mean'] = float(np.mean(diff_vals)) if diff_vals.size else 0.0
    metrics['cond/diff_class_cos_std'] = float(np.std(diff_vals)) if diff_vals.size else 0.0
    return metrics, same_vals.astype(np.float32), diff_vals.astype(np.float32)


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
            sup_pooled = support_pooled.astype(images.dtype)
            sup_seq = support_seq.astype(images.dtype) if self.config.get('use_support_seq', 1) else None
            if self.config.get('log_model_debug', 1):
                v_pred, dbg = self.model(
                    x_t, t, sup_pooled, y_seq=sup_seq, train=True,
                    return_debug=True, rngs={'cond_dropout': cond_key}, params=params,
                )
            else:
                v_pred = self.model(
                    x_t, t, sup_pooled, y_seq=sup_seq, train=True,
                    rngs={'cond_dropout': cond_key}, params=params,
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
        """Compute val loss + t-bin breakdown (no dropout, uses EMA model)."""
        time_key, noise_key = jax.random.split(self.rng, 2)
        if self.config['t_sampler'] == 'log-normal':
            t = jax.nn.sigmoid(jax.random.normal(time_key, (images.shape[0],)))
        else:
            t = jax.random.uniform(time_key, (images.shape[0],))

        eps = jax.random.normal(noise_key, images.shape)
        x_t = flow_interpolate(images, eps, t[:, None, None, None])
        v_gt = flow_velocity(images, eps)
        sup_pooled = support_pooled.astype(images.dtype)
        sup_seq = support_seq.astype(images.dtype) if self.config.get('use_support_seq', 1) else None
        v_pred = self.model_ema(
            x_t, t, sup_pooled, y_seq=sup_seq,
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
        sup_seq = support_seq.astype(images.dtype) if self.config.get('use_support_seq', 1) else None

        _, attn_list = self.model_ema(
            x_t, t, sup_pooled, y_seq=sup_seq,
            train=False, force_drop_ids=False, return_attn=True,
        )
        entropies = jnp.stack([attention_entropy(aw) for aw in attn_list])  # (depth, H)
        return entropies, t

    # ── CFG sampling ───────────────────────────────────────────────────────
    @partial(jax.pmap, axis_name='data',
             in_axes=(0, 0, 0, 0, 0), static_broadcasted_argnums=(5, 6))
    def sample_step(self, x, t_vec, support_pooled, support_seq, cfg=True, cfg_val=1.0):
        """One Euler step with optional CFG."""
        sup_pooled = support_pooled.astype(x.dtype)
        use_seq = self.config.get('use_support_seq', 1)
        sup_seq = support_seq.astype(x.dtype) if use_seq else None
        if not cfg or cfg_val == 0:
            return self.model_ema(
                x, t_vec, sup_pooled, y_seq=sup_seq,
                train=False, force_drop_ids=False,
            )
        B = x.shape[0]
        x2 = jnp.concatenate([x, x])
        t2 = jnp.concatenate([t_vec, t_vec])
        pooled2 = jnp.concatenate([sup_pooled, jnp.zeros_like(sup_pooled)])
        if use_seq:
            seq2 = jnp.concatenate([sup_seq, jnp.zeros_like(sup_seq)])
        else:
            seq2 = None
        v = self.model_ema(x2, t2, pooled2, y_seq=seq2, train=False, force_drop_ids=False)
        v_c, v_u = v[:B], v[B:]
        return v_u + cfg_val * (v_c - v_u)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(_):
    cfg = FLAGS.model
    for k, v in PRESETS[cfg.preset].items():
        cfg[k] = v
    cfg.use_support_seq = int(FLAGS.use_support_seq)
    cfg.log_model_debug = int(FLAGS.log_model_debug)

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

    # ── Data ───────────────────────────────────────────────────────────────
    if FLAGS.data_mode == 'grain':
        grain_mod = _get_grain_dataset_mod()
        # Parse path remap: "old:new" → (old, new)
        path_remap = None
        if FLAGS.grain_path_remap:
            parts = FLAGS.grain_path_remap.split(':', 1)
            if len(parts) != 2:
                raise ValueError(
                    "--grain_path_remap must be 'old_prefix:new_prefix', "
                    f"got: {FLAGS.grain_path_remap}"
                )
            path_remap = (parts[0], parts[1])
            print(f"  Path remap: '{parts[0]}' → '{parts[1]}'")
        train_iter = grain_mod.build_grain_dataset(
            arecord_dir=FLAGS.grain_arecord_dir,
            split='train',
            batch_size=local_bs,
            image_size=cfg.image_size,
            is_train=True,
            seed=FLAGS.seed,
            load_support_seq=FLAGS.use_support_seq,
            path_prefix_remap=path_remap,
        )
        val_iter = grain_mod.build_grain_dataset(
            arecord_dir=FLAGS.grain_arecord_dir,
            split='val',
            batch_size=local_bs,
            image_size=cfg.image_size,
            is_train=False,
            seed=FLAGS.seed + 1000,
            load_support_seq=FLAGS.use_support_seq,
            path_prefix_remap=path_remap,
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
    init_kwargs = {}
    if FLAGS.use_support_seq:
        init_kwargs['y_seq'] = jnp.zeros((1, n_sup_tokens, cfg.siglip_dim))
    params = dit.init(
        {'params': p_key, 'cond_dropout': d_key},
        jnp.zeros((1, img_s, img_s, img_c)),   # x
        jnp.zeros((1,)),                        # t
        jnp.zeros((1, cfg.siglip_dim)),         # y_pooled
        **init_kwargs,
    )['params']
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"DiT parameters: {n_params:,}")

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
        Returns:
          pooled_global: (B,768) float32 (for stats)
          pooled_model : (B,768) float16 (for model input)
          seq_model    : (B,T,768) float16 (for model input)
          siglip_stats : dict or None
        """
        siglip_stats = None
        if FLAGS.data_mode == 'grain':
            pooled_5 = batch['supports_pooled']
            seq_5 = batch['supports_seq'] if FLAGS.use_support_seq else None
        elif FLAGS.data_mode == 'online':
            pooled_5, seq_5, siglip_stats = online_encoder.encode_paths(
                batch['support_paths'],
                need_seq=bool(FLAGS.use_support_seq),
            )
        else:
            pooled_5 = batch['supports_pooled']
            seq_5 = batch['supports_seq'] if FLAGS.use_support_seq else None

        pooled_global = np.mean(pooled_5, axis=1, dtype=np.float32)
        pooled_model = pooled_global.astype(np.float16)
        if FLAGS.use_support_seq:
            seq_model = seq_5.reshape(seq_5.shape[0], -1, seq_5.shape[-1]).astype(np.float16)
        else:
            seq_model = np.zeros(
                (batch['target'].shape[0], 1, cfg.siglip_dim),
                dtype=np.float16,
            )
        return pooled_global, pooled_model, seq_model, siglip_stats

    # ── Eval function ─────────────────────────────────────────────────────
    def run_eval(step):
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
            gap = (loss_ema[0] - v_loss_f) if loss_ema[0] else 0.0
            log = {'val/loss': v_loss_f, 'val/train_val_gap': gap}
            for b in range(cfg.num_t_bins):
                log[f'val/loss_tbin_{b}'] = float(v_tbin_f[b])
            cond_metrics, same_vals, diff_vals = compute_condition_distribution_metrics(
                val_sup_pooled_global, val_class_ids
            )
            for k, v in cond_metrics.items():
                log[f'val_{k}'] = v
            if FLAGS.data_mode == 'online' and val_siglip_stats is not None:
                log['val_perf/siglip_encode_time'] = float(val_siglip_stats['encode_time'])
                log['val_perf/siglip_cache_hit_rate'] = float(val_siglip_stats['cache_hit_rate'])
                log['val_perf/siglip_cache_items'] = float(val_siglip_stats['cache_items'])
                log['val_perf/siglip_unique_paths_per_batch'] = float(
                    val_siglip_stats['unique_paths_per_batch']
                )
            if step % FLAGS.cond_hist_interval == 0:
                log['val_cond/support_pooled_hist'] = wandb.Histogram(
                    val_sup_pooled_global.reshape(-1)
                )
                if same_vals.size:
                    log['val_cond/same_class_cos_hist'] = wandb.Histogram(same_vals)
                if diff_vals.size:
                    log['val_cond/diff_class_cos_hist'] = wandb.Histogram(diff_vals)
            wandb.log(log, step=step)

        # Attention entropy
        try:
            ent_matrix, _ = trainer.get_attn_entropy(val_img, val_sup_pooled, val_sup_seq)
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

    # ═══════════════════════════════════════════════════════════════════════
    #  TRAIN LOOP
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print(f"  FSDiT training — {FLAGS.max_steps:,} steps, bs={FLAGS.batch_size}")
    print(f"{'═' * 60}\n")

    alpha = cfg.loss_ema_alpha
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

            if FLAGS.log_model_debug:
                log['dbg/t_emb_abs_mean'] = float(np.array(info['dbg/t_emb_abs_mean']).mean())
                log['dbg/y_emb_abs_mean'] = float(np.array(info['dbg/y_emb_abs_mean']).mean())
                log['dbg/c_abs_mean'] = float(np.array(info['dbg/c_abs_mean']).mean())
                log['dbg/c_l2_mean'] = float(np.array(info['dbg/c_l2_mean']).mean())
                log['dbg/support_pooled_abs_mean_model'] = float(
                    np.array(info['dbg/support_pooled_abs_mean']).mean()
                )
                log['dbg/support_pooled_l2_mean_model'] = float(
                    np.array(info['dbg/support_pooled_l2_mean']).mean()
                )
                act_abs = np.array(info['dbg/act_abs_per_layer']).mean(axis=0)
                act_rms = np.array(info['dbg/act_rms_per_layer']).mean(axis=0)
                for li in range(cfg.depth):
                    log[f'act/layer{li}_abs_mean'] = float(act_abs[li])
                    log[f'act/layer{li}_rms'] = float(act_rms[li])

            if sup_pooled_global is not None and class_ids_global is not None:
                cond_metrics, same_vals, diff_vals = compute_condition_distribution_metrics(
                    sup_pooled_global, class_ids_global
                )
                log.update(cond_metrics)
                if step % FLAGS.cond_hist_interval == 0:
                    log['cond/support_pooled_hist'] = wandb.Histogram(
                        sup_pooled_global.reshape(-1)
                    )
                    if same_vals.size:
                        log['cond/same_class_cos_hist'] = wandb.Histogram(same_vals)
                    if diff_vals.size:
                        log['cond/diff_class_cos_hist'] = wandb.Histogram(diff_vals)
            wandb.log(log, step=step)

        if step % FLAGS.perf_log_interval == 0 and jax.process_index() == 0:
            perf_log = {
                'perf/data_time': data_time,
                'perf/siglip_encode_time': siglip_time,
                'perf/vae_time': vae_time,
                'perf/train_step_time': step_time,
                'perf/total_iter_time': dt_step,
            }
            if FLAGS.data_mode == 'online' and siglip_stats is not None:
                perf_log['perf/siglip_cache_hit_rate'] = float(siglip_stats['cache_hit_rate'])
                perf_log['perf/siglip_cache_items'] = float(siglip_stats['cache_items'])
                perf_log['perf/siglip_unique_paths_per_batch'] = float(
                    siglip_stats['unique_paths_per_batch']
                )
            wandb.log(perf_log, step=step)

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
