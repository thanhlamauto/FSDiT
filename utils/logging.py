"""
utils/logging.py — Logging helpers for FSDiT training.

Extracts metric computation and wandb logging from train.py for cleaner
separation. All functions are pure: they take data, compute metrics,
and log to wandb.
"""

import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    import wandb
except ImportError:
    wandb = None


# ═══════════════════════════════════════════════════════════════════════════════
#  Async wandb logger — prevents main thread from blocking on device sync
# ═══════════════════════════════════════════════════════════════════════════════

class AsyncLogger:
    """Log metrics to W&B in a background thread.

    Usage:
        logger = AsyncLogger()
        logger.log({'loss': jax_array, 'lr': 0.001}, step=step)
        # ... training continues immediately, logging happens in background

    The background thread materialises JAX arrays (blocking on device sync
    there, not on the main thread) and calls wandb.log.
    """

    def __init__(self, max_workers: int = 1):
        # Single worker so logs arrive in order.
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix='wandb_log')
        self._lock = threading.Lock()

    def log(self, log_dict: dict, step: int):
        """Submit a logging job to the background thread.

        JAX / numpy arrays are passed by reference; the background thread
        will materialise them (np.asarray / float conversion).  The main
        training loop returns immediately.
        """
        if wandb is None:
            return
        # Snapshot the dict so the caller can reuse keys safely.
        snapshot = dict(log_dict)
        self._pool.submit(self._do_log, snapshot, step)

    @staticmethod
    def _do_log(log_dict: dict, step: int):
        """Run on background thread: convert arrays → scalars, then log."""
        materialised = {}
        for k, v in log_dict.items():
            try:
                # Works for JAX arrays, numpy arrays, plain Python scalars,
                # and wandb.Histogram / wandb.Image objects.
                if hasattr(v, 'shape'):          # array-like
                    v = float(np.asarray(v).mean()) if v.ndim > 0 else float(np.asarray(v))
                materialised[k] = v
            except Exception:
                pass                             # skip unserializable values
        try:
            wandb.log(materialised, step=step)
        except Exception:
            pass

    def wait(self):
        """Block until all pending log jobs are flushed (call at end of training)."""
        self._pool.shutdown(wait=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Metric computation (pure numpy, no wandb dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_condition_distribution_metrics(cond_vec, class_ids):
    """
    Compute SigLIP pooled-condition distribution metrics for a batch.

    Args:
        cond_vec: (B, 768) float32 pooled support embeddings.
        class_ids: (B,) int32 class IDs.

    Returns:
        metrics: dict of scalar metrics
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
#  wandb log builders
# ═══════════════════════════════════════════════════════════════════════════════

def log_train_metrics(
    step, info, loss_ema, lr_schedule, cfg, dt_step=0.0,
    sup_pooled_global=None, class_ids_global=None,
    log_model_debug=True, cond_hist_interval=5000,
    async_logger=None,
):
    """
    Log training metrics to wandb (optionally async).

    If async_logger is provided (AsyncLogger instance), all wandb.log calls
    are dispatched to a background thread so the main training loop never
    blocks on JAX device synchronisation.
    """
    # loss_ema must be updated synchronously (it feeds back into future steps)
    alpha = cfg.loss_ema_alpha
    train_loss = float(np.array(info['loss']).mean())
    if loss_ema[0] is None:
        loss_ema[0] = train_loss
    else:
        loss_ema[0] = alpha * loss_ema[0] + (1 - alpha) * train_loss

    # Build log dict — raw JAX arrays OK; AsyncLogger materialises them on bg thread
    log = {
        'train/loss':      info['loss'],
        'train/loss_ema':  loss_ema[0],
        'train/grad_norm': info['grad_norm'],
        'train/lr':        float(lr_schedule(step)),
        'train/step_time': dt_step,
        'train/param_norm': info['param_norm'],
    }
    for b in range(cfg.num_t_bins):
        log[f'train/loss_tbin_{b}'] = info['tbin_loss']

    if log_model_debug:
        for key in ('t_emb_abs_mean', 'y_emb_abs_mean', 'c_abs_mean', 'c_l2_mean',
                     'support_pooled_abs_mean', 'support_pooled_l2_mean'):
            dbg_key = f'dbg/{key}'
            if dbg_key in info:
                dest = f'dbg/{key}_model' if 'support_pooled' in key else dbg_key
                log[dest] = info[dbg_key]

        if 'dbg/act_abs_per_layer' in info:
            log['_act_abs_per_layer'] = info['dbg/act_abs_per_layer']
            log['_act_rms_per_layer'] = info['dbg/act_rms_per_layer']
            log['_depth'] = cfg.depth

    if sup_pooled_global is not None and class_ids_global is not None:
        cond_metrics, same_vals, diff_vals = compute_condition_distribution_metrics(
            sup_pooled_global, class_ids_global
        )
        log.update(cond_metrics)
        if step % cond_hist_interval == 0 and wandb is not None:
            log['cond/support_pooled_hist'] = wandb.Histogram(
                np.asarray(sup_pooled_global).reshape(-1)
            )
            if same_vals.size:
                log['cond/same_class_cos_hist'] = wandb.Histogram(same_vals)
            if diff_vals.size:
                log['cond/diff_class_cos_hist'] = wandb.Histogram(diff_vals)

    _dispatch(log, step, async_logger)


def _dispatch(log_dict, step, async_logger):
    """Send log_dict to wandb via async_logger if available, else synchronously."""
    if async_logger is not None:
        async_logger.log(log_dict, step)
    elif wandb is not None:
        # Synchronous fallback: materialise arrays here
        materialised = {}
        for k, v in log_dict.items():
            if hasattr(v, 'shape'):
                v = float(np.asarray(v).mean())
            materialised[k] = v
        wandb.log(materialised, step=step)



def log_eval_metrics(step, v_loss_f, v_tbin_f, loss_ema, cfg,
                     val_sup_pooled_global, val_class_ids,
                     data_mode='grain', val_siglip_stats=None,
                     cond_hist_interval=5000, async_logger=None):
    """Log evaluation / validation metrics to wandb."""
    gap = (loss_ema[0] - v_loss_f) if loss_ema[0] else 0.0
    log = {'val/loss': v_loss_f, 'val/train_val_gap': gap}
    for b in range(cfg.num_t_bins):
        log[f'val/loss_tbin_{b}'] = float(v_tbin_f[b])

    cond_metrics, same_vals, diff_vals = compute_condition_distribution_metrics(
        val_sup_pooled_global, val_class_ids
    )
    for k, v in cond_metrics.items():
        log[f'val_{k}'] = v

    if step % cond_hist_interval == 0 and wandb is not None:
        log['val_cond/support_pooled_hist'] = wandb.Histogram(
            np.asarray(val_sup_pooled_global).reshape(-1)
        )
        if same_vals.size:
            log['val_cond/same_class_cos_hist'] = wandb.Histogram(same_vals)
        if diff_vals.size:
            log['val_cond/diff_class_cos_hist'] = wandb.Histogram(diff_vals)

    _dispatch(log, step, async_logger)


def log_attn_entropy(step, ent_matrix, cfg, async_logger=None):
    """Log attention entropy metrics to wandb."""
    ent = np.array(ent_matrix).mean(axis=0)  # avg devices → (depth, H)
    depth = cfg.depth
    alog = {}
    for li in [0, depth // 2, depth - 1]:
        alog[f'attn/entropy_layer{li}'] = float(ent[li].mean())
    for h in range(cfg.num_heads):
        alog[f'attn/entropy_head{h}_last'] = float(ent[-1, h])
    _dispatch(alog, step, async_logger)
