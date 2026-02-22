# FSDiT — Few-Shot Diffusion Transformer

Flow-matching DiT conditioned on **SigLIP2 support-set embeddings** for few-shot image generation on miniImageNet.

## Architecture

```
5 Support Images → [SigLIP2 B/16 frozen] → mean pool → MLP → FiLM
Target Image     → [SD VAE]              → latent     → DiT (adaLN-Zero) → v_pred
```

## Project Structure

```
FSDiT/
├── train.py          # Main training script
├── model.py          # DiT architecture + SupportProjector
├── dataset.py        # miniImageNet episode loader
├── encoder.py        # Frozen SigLIP2 B/16 encoder
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

# Cell 3: Train
!python train.py \
    --data_dir /kaggle/working/miniimagenet \
    --save_dir /kaggle/working/ckpts \
    --batch_size 128 \
    --max_steps 200000 \
    --wandb.name fsdit_run1
```

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
