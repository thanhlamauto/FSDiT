# FSDiT — Few-Shot Diffusion Transformer

Flow-matching DiT conditioned on SigLIP2 support embeddings for few-shot image generation
on miniImageNet.

Default runtime now uses **online SigLIP conditioning** (no embedding cache required on disk).

## Architecture

```
5 Support Image Paths → Online SigLIP2 encoder (+ LRU cache)
pooled → MLP → adaLN condition
seq    → OneLayerPerceiver → Cross-Attn context (optional with `--use_support_seq=1`)
Target Image → [SD VAE] → latent → DiT (adaLN-Zero + cross-attn) → v_pred
```

## Project Structure

```
FSDiT/
├── train.py          # Main training script
├── model.py          # DiT architecture + SupportProjector
├── dataset.py        # miniImageNet episode loader
├── encoder.py        # Frozen SigLIP2 B/16 encoder
├── utils/online_support_encoder.py # Runtime SigLIP encoder + LRU cache
├── precompute_siglip_debug.py  # Precompute seq/pooled embeddings to .npz
└── utils/
    ├── train_state.py
    ├── checkpoint.py
    ├── wandb_utils.py
    ├── stable_vae.py
    ├── fid.py
    └── pretrained_resnet.py
```

## Quick Start (Kaggle TPU v5e-8)

```python
# Cell 1: Clone & install
!git clone https://github.com/thanhlamauto/FSDiT.git /kaggle/working/FSDiT
!pip install -q wandb einops flax optax ml_collections jaxtyping typeguard diffusers

# Cell 2: Split data (miniImageNet is flat — need train/val/test split)
import os; os.chdir('/kaggle/working/FSDiT')
!python prepare_data.py \
    --src /kaggle/input/datasets/arjunashok33/miniimagenet \
    --dst /kaggle/working/miniimagenet \
    --train 60 --val 16 --test 20

# Cell 3: Train directly with online SigLIP conditioning (no TFRecord embedding shards)
!python train.py \
    --data_dir /kaggle/working/miniimagenet \
    --data_mode online \
    --save_dir /kaggle/working/ckpts \
    --batch_size 128 \
    --max_steps 200000 \
    --use_support_seq=1 \
    --online_cache_items=1024 \
    --online_siglip_batch_size=256 \
    --perf_log_interval=100 \
    --suppress_diffusers_warnings=1 \
    --wandb.name fsdit_online_siglip
```

### TFRecord fallback (optional)

Use this only when you explicitly want precomputed shard benchmarks:

1. `precompute_siglip_debug.py`
2. `build_episode_tfrecord.py`
3. `train.py --data_mode tfrecord --episode_tfrecord_dir ...`

## Hyperparameters

| Parameter | Value | Note |
|-----------|-------|------|
| LR | 1e-4 → 1e-6 | 5k warmup + cosine decay |
| Optimizer | AdamW | wd=0.01, β=(0.9, 0.99) |
| Grad clip | 1.0 | Global norm |
| Batch size | 128 | 8 TPU cores × 16/core |
| EMA | 0.9999 | For inference |
| CFG dropout | 10% | Support dropped to zero |
| CFG scale | 3.0 | At sampling time |
| Denoising | 50 steps | Euler method |

## Data: miniImageNet Episodes

- 60 train / 16 val / 20 test classes, 600 images/class
- 100 sets/class × 6 images/set × 6 rotations = **36,000 episodes**
- Each episode: 1 target + 5 support, stratified batching
- `data_mode=online` (default): dataset yields `target + support_paths`, SigLIP is encoded in `train.py`.
- `data_mode=tfrecord` (fallback): dataset reads precomputed episode shards (`--episode_tfrecord_dir`).
- `.npz` cache is optional and used only for TFRecord export flow.
- For TPU throughput in precompute:
  - keep `pmap` enabled (default)
  - use larger `--batch_size` (e.g. 256/512 depending on memory)
- Image decode/resize runs on CPU (TensorFlow input pipeline), while SigLIP2 encode runs on JAX backend (TPU/GPU/CPU).
- Each intermediate `.npz` file includes:
  - `seq`: `(196, 768)` (SigLIP2 patch tokens)
  - `pooled`: `(768,)` (SigLIP2 pooled embedding)

### Offline ArrayRecord layout (for GPU / Kaggle)

When using the PyTorch SigLIP2 precompute + ArrayRecord pipeline on GPU, we materialize
episode-level ArrayRecord files on disk:

- Precompute embeddings with `precompute_siglip_pytorch.py`:
  - Inputs: `/path/to/miniimagenet/{train,val}/*/*.JPEG`
  - Outputs: `.npz` next to each image with:
    - `seq`: `(196, 768)` float16
    - `pooled`: `(768,)` float16

- Build episodes as ArrayRecord with `build_episodes_arrayrecord.py`:
  - Example:
    - `--data_dir /workspace/data/miniimagenet`
    - `--embeddings_dir /workspace/data/miniimagenet`
    - `--out_dir /workspace/data/episodes_arecord`
    - `--splits train,val`
    - `--store_seq 1`
  - Output layout:
    - `/workspace/data/episodes_arecord/train.arecord`
    - `/workspace/data/episodes_arecord/train_meta.json`
    - `/workspace/data/episodes_arecord/val.arecord`
    - `/workspace/data/episodes_arecord/val_meta.json`

Each record in `*.arecord` is a msgpack blob with:

- `target_path`: string (absolute path to target image on disk)
- `class_id`: int
- `supports_pooled`: bytes (float16 `[5, 768]`)
- `supports_seq`: bytes (float16 `[5, 196, 768]`) or `b""` when `--store_seq=0`

These records are loaded in Python via:

```python
from dataset_grain import build_grain_dataset

train_ds = build_grain_dataset(
    arecord_path="/workspace/data/episodes_arecord/train.arecord",
    batch_size=128,
    image_size=224,
    is_train=True,
    seed=42,
    load_support_seq=True,
)
```

### Sharding train ArrayRecord for Kaggle (file size < 20GB)

Kaggle enforces a per-file size limit (~20GB). A full `train.arecord` with `store_seq=1`
can be larger than this, so we provide a small utility to shard it into multiple
independent ArrayRecord files:

- Script: `shard_arrayrecord.py`
- Example usage:

```bash
python shard_arrayrecord.py \
  --src /workspace/data/episodes_arecord/train.arecord \
  --out_dir /workspace/data/episodes_arecord \
  --num_shards 3
```

This reads `train.arecord` with `ArrayRecordReader` and writes sequential record ranges
into shard files:

- `/workspace/data/episodes_arecord/train_shard_000.arecord`  (~<20GB)
- `/workspace/data/episodes_arecord/train_shard_001.arecord`  (~<20GB)
- `/workspace/data/episodes_arecord/train_shard_002.arecord`  (<20GB)

Each shard is a valid ArrayRecord file and can be loaded independently with
`build_grain_dataset` by pointing `arecord_path` to the shard instead of the full
`train.arecord`. On Kaggle you typically upload:

- `train_shard_000.arecord`
- `train_shard_001.arecord`
- `train_shard_002.arecord`
- `val.arecord`

and then in your notebook:

```python
from dataset_grain import build_grain_dataset

train_ds = build_grain_dataset(
    arecord_path="/kaggle/input/your-dataset/train_shard_000.arecord",
    batch_size=128,
    image_size=224,
    is_train=True,
    seed=42,
)
```

## WandB Dashboard

- **Train**: loss (EMA), grad_norm, lr, step_time, loss per t-bin
- **Val**: loss, train-val gap, loss per t-bin
- **Attention**: entropy per layer (early/mid/late), per head (last layer)
- **Samples**: Generated image grids with CFG

## References

- [Flow Matching (Lipman et al., 2022)](https://arxiv.org/abs/2210.02747)
- [DiT (Peebles & Xie, 2023)](https://arxiv.org/abs/2212.09748)
- [SigLIP2 (Google, 2024)](https://arxiv.org/abs/2502.14786)
- Original JAX DiT: [kvfrans/jax-flow](https://github.com/kvfrans/jax-flow)
