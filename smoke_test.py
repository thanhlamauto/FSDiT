"""
Smoke tests for FSDiT:
1) Dataset contract for selected data mode.
2) Support-condition contract (online SigLIP or TFRecord embeddings).
3) DiT forward pass with and without sequence context.
"""

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from dataset import build_dataset
from model import DiT
from utils.online_support_encoder import OnlineSupportEncoder

tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")


def _lazy_import_grain_dataset():
    import dataset_grain
    return dataset_grain


def _resolve_train_dir(data_dir):
    train_dir = os.path.join(data_dir, "train")
    return train_dir if os.path.isdir(train_dir) else data_dir


def check_dataset_contract_tfrecord(data_dir, episode_tfrecord_dir, batch_size, image_size, num_sets):
    train_dir = _resolve_train_dir(data_dir)
    if not episode_tfrecord_dir:
        raise ValueError("Please pass --episode_tfrecord_dir for tfrecord smoke test.")
    train_pattern = os.path.join(episode_tfrecord_dir, "train", "train-*.tfrecord")
    ds, _ = build_dataset(
        train_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_sets=num_sets,
        is_train=False,
        seed=0,
        debug_n=max(batch_size * 2, 2),
        data_mode="tfrecord",
        episode_tfrecord_pattern=train_pattern,
        tfrecord_compression_type="GZIP",
    )
    batch = next(iter(ds.as_numpy_iterator()))

    expected_keys = {"target", "supports_seq", "supports_pooled", "class_id"}
    got_keys = set(batch.keys())
    if got_keys != expected_keys:
        raise AssertionError(f"Dataset keys mismatch: got={got_keys}, expected={expected_keys}")

    target = batch["target"]
    supports_seq = batch["supports_seq"]
    supports_pooled = batch["supports_pooled"]

    if target.ndim != 4 or target.shape[-1] != 3:
        raise AssertionError(f"target must be (B,H,W,3), got {target.shape}")
    if supports_seq.shape[1:] != (5, 196, 768):
        raise AssertionError(f"supports_seq must be (B,5,196,768), got {supports_seq.shape}")
    if supports_pooled.shape[1:] != (5, 768):
        raise AssertionError(f"supports_pooled must be (B,5,768), got {supports_pooled.shape}")
    if not np.isfinite(target).all() or not np.isfinite(supports_seq).all() or not np.isfinite(supports_pooled).all():
        raise AssertionError("Non-finite values found in dataset batch.")

    print("[OK] Dataset contract (tfrecord):"
          f" target={target.shape}, supports_seq={supports_seq.shape}, supports_pooled={supports_pooled.shape}")
    return batch


def check_dataset_contract_online(data_dir, batch_size, image_size, num_sets):
    train_dir = _resolve_train_dir(data_dir)
    ds, _ = build_dataset(
        train_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_sets=num_sets,
        is_train=False,
        seed=0,
        debug_n=max(batch_size * 2, 2),
        data_mode="online",
    )
    batch = next(iter(ds.as_numpy_iterator()))

    expected_keys = {"target", "support_paths", "class_id"}
    got_keys = set(batch.keys())
    if got_keys != expected_keys:
        raise AssertionError(f"Dataset keys mismatch: got={got_keys}, expected={expected_keys}")

    target = batch["target"]
    support_paths = batch["support_paths"]
    if target.ndim != 4 or target.shape[-1] != 3:
        raise AssertionError(f"target must be (B,H,W,3), got {target.shape}")
    if support_paths.ndim != 2 or support_paths.shape[1] != 5:
        raise AssertionError(f"support_paths must be (B,5), got {support_paths.shape}")
    if not np.isfinite(target).all():
        raise AssertionError("Non-finite values found in online target batch.")
    print(f"[OK] Dataset contract (online): target={target.shape}, support_paths={support_paths.shape}")
    return batch


def check_dataset_contract_grain(grain_arecord_dir, batch_size, image_size, use_support_seq):
    grain_mod = _lazy_import_grain_dataset()
    ds_iter = grain_mod.build_grain_dataset(
        arecord_dir=grain_arecord_dir,
        split="train",
        batch_size=batch_size,
        image_size=image_size,
        is_train=False,
        seed=0,
        load_support_seq=use_support_seq,
    )
    batch = next(ds_iter)

    expected_keys = {"target", "supports_seq", "supports_pooled", "class_id"}
    got_keys = set(batch.keys())
    if got_keys != expected_keys:
        raise AssertionError(f"Dataset keys mismatch: got={got_keys}, expected={expected_keys}")

    target = batch["target"]
    supports_seq = batch["supports_seq"]
    supports_pooled = batch["supports_pooled"]

    if target.ndim != 4 or target.shape[-1] != 3:
        raise AssertionError(f"target must be (B,H,W,3), got {target.shape}")
    if supports_seq.shape[1:] != (5, 196, 768):
        raise AssertionError(f"supports_seq must be (B,5,196,768), got {supports_seq.shape}")
    if supports_pooled.shape[1:] != (5, 768):
        raise AssertionError(f"supports_pooled must be (B,5,768), got {supports_pooled.shape}")
    if not np.isfinite(target).all() or not np.isfinite(supports_seq).all() or not np.isfinite(supports_pooled).all():
        raise AssertionError("Non-finite values found in grain dataset batch.")

    print("[OK] Dataset contract (grain):"
          f" target={target.shape}, supports_seq={supports_seq.shape}, supports_pooled={supports_pooled.shape}")
    return batch


def build_condition(batch, data_mode, image_size, use_support_seq, online_cache_items, online_siglip_batch_size, online_siglip_no_pmap):
    bsz = min(2, batch["target"].shape[0])
    if data_mode in ("tfrecord", "grain"):
        pooled = batch["supports_pooled"][:bsz]
        y_pooled = pooled.mean(axis=1, dtype=np.float32)
        y_seq = None
        if use_support_seq:
            y_seq = batch["supports_seq"][:bsz].reshape(bsz, -1, 768).astype(np.float32)
        return y_pooled, y_seq

    enc = OnlineSupportEncoder(
        variant="B/16",
        image_size=image_size,
        cache_items=online_cache_items,
        batch_size=online_siglip_batch_size,
        no_pmap=online_siglip_no_pmap,
        warmup_need_seq=use_support_seq,
    )
    pooled, seq, stats = enc.encode_paths(batch["support_paths"][:bsz], need_seq=use_support_seq)
    _, _, stats2 = enc.encode_paths(batch["support_paths"][:bsz], need_seq=use_support_seq)
    y_pooled = pooled.mean(axis=1, dtype=np.float32)
    y_seq = seq.reshape(bsz, -1, 768).astype(np.float32) if use_support_seq else None
    print(
        "[OK] Online support encode:"
        f" pooled={pooled.shape}, seq={seq.shape},"
        f" cache_hit_rate={stats['cache_hit_rate']:.3f}, encode_time={stats['encode_time']:.3f}s"
    )
    print(
        "[OK] Online support cache re-encode:"
        f" cache_hit_rate={stats2['cache_hit_rate']:.3f}, encode_time={stats2['encode_time']:.3f}s"
    )
    if stats2["cache_hit_rate"] + 1e-6 < stats["cache_hit_rate"]:
        raise AssertionError("Online cache hit-rate did not improve on repeated batch encode.")
    if not np.isfinite(y_pooled).all():
        raise AssertionError("Non-finite pooled conditioning values.")
    if y_seq is not None and not np.isfinite(y_seq).all():
        raise AssertionError("Non-finite sequence conditioning values.")
    return y_pooled, y_seq


def check_model_forward(batch, y_pooled_np, y_seq_np):
    bsz = min(2, batch["target"].shape[0])
    x = jnp.asarray(batch["target"][:bsz], dtype=jnp.float32)
    y_pooled = jnp.asarray(y_pooled_np[:bsz], dtype=jnp.float32)
    y_seq = jnp.asarray(y_seq_np[:bsz], dtype=jnp.float32) if y_seq_np is not None else None
    t = jnp.linspace(0.1, 0.9, bsz, dtype=jnp.float32)

    dit = DiT(
        patch_size=8,
        hidden_size=64,
        depth=2,
        num_heads=2,
        mlp_ratio=1.0,
        siglip_dim=768,
        cond_dropout_prob=0.1,
    )

    rng = jax.random.PRNGKey(0)
    p_key, d_key = jax.random.split(rng)
    init_kwargs = {}
    if y_seq is not None:
        init_kwargs["y_seq"] = y_seq
    params = dit.init({"params": p_key, "cond_dropout": d_key}, x, t, y_pooled, train=True, **init_kwargs)["params"]

    if y_seq is not None:
        out_ctx = dit.apply(
            {"params": params},
            x,
            t,
            y_pooled,
            y_seq=y_seq,
            train=False,
        )
        if out_ctx.shape != x.shape:
            raise AssertionError(f"DiT output shape mismatch (with context): {out_ctx.shape} vs {x.shape}")
        if not np.isfinite(np.asarray(out_ctx)).all():
            raise AssertionError("Non-finite values in DiT forward output with context.")

    # Backward-compatible path: pooled only.
    out_pooled = dit.apply(
        {"params": params},
        x,
        t,
        y_pooled,
        train=False,
    )
    if out_pooled.shape != x.shape:
        raise AssertionError(f"DiT output shape mismatch (pooled-only): {out_pooled.shape} vs {x.shape}")

    if not np.isfinite(np.asarray(out_pooled)).all():
        raise AssertionError("Non-finite values in DiT pooled-only output.")
    if y_seq is not None:
        print(f"[OK] Model forward: with_context={out_ctx.shape}, pooled_only={out_pooled.shape}")
    else:
        print(f"[OK] Model forward: pooled_only={out_pooled.shape}")


def main():
    parser = argparse.ArgumentParser(description="Run FSDiT smoke tests.")
    parser.add_argument("--data_dir", required=True, help="miniImageNet split root or train folder.")
    parser.add_argument("--data_mode", choices=["online", "tfrecord", "grain"], default="online")
    parser.add_argument("--episode_tfrecord_dir", default=None, help="Episode TFRecord root (tfrecord mode only).")
    parser.add_argument("--grain_arecord_dir", default=None, help="ArrayRecord episode dir (grain mode only).")
    parser.add_argument("--use_support_seq", type=int, default=1)
    parser.add_argument("--online_cache_items", type=int, default=128)
    parser.add_argument("--online_siglip_batch_size", type=int, default=64)
    parser.add_argument("--online_siglip_no_pmap", action="store_true")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_sets", type=int, default=1)
    args = parser.parse_args()

    if args.data_mode == "tfrecord":
        batch = check_dataset_contract_tfrecord(
            args.data_dir,
            args.episode_tfrecord_dir,
            args.batch_size,
            args.image_size,
            args.num_sets,
        )
    elif args.data_mode == "grain":
        batch = check_dataset_contract_grain(
            args.grain_arecord_dir,
            args.batch_size,
            args.image_size,
            bool(args.use_support_seq),
        )
    else:
        batch = check_dataset_contract_online(
            args.data_dir,
            args.batch_size,
            args.image_size,
            args.num_sets,
        )

    y_pooled, y_seq = build_condition(
        batch,
        args.data_mode,
        args.image_size,
        bool(args.use_support_seq),
        args.online_cache_items,
        args.online_siglip_batch_size,
        args.online_siglip_no_pmap,
    )
    check_model_forward(batch, y_pooled, y_seq)
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
