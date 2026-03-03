"""
kaggle_train_gram_adaln.py
==========================
Full training script for GramDiT-AdaLN on Kaggle TPU v5e-8.
Branch: feat/gram-dit-block-adaln

Usage (Kaggle notebook cell):
    !python kaggle_train_gram_adaln.py

Assumes:
  - Dataset (ArrayRecord) is precomputed at ARECORD_DIR
  - W&B API key set via os.environ or Kaggle secret
"""

import os, subprocess, sys

# ─── 0. Environment setup ────────────────────────────────────────────────────

# Prevent TF from grabbing TPU/GPU before JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["DIFFUSERS_VERBOSITY"]   = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Clone repo + install deps (idempotent)
REPO_DIR = "/kaggle/working/FSDiT"
BRANCH   = "feat/gram-dit-block-adaln"

def run(cmd, **kw):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kw)

if not os.path.isdir(REPO_DIR):
    run(f"git clone --branch {BRANCH} --single-branch "
        f"https://github.com/thanhlamauto/FSDiT.git {REPO_DIR}")
else:
    run(f"git -C {REPO_DIR} fetch origin {BRANCH} && "
        f"git -C {REPO_DIR} checkout {BRANCH} && "
        f"git -C {REPO_DIR} pull")

run("pip install -q grain einops ml_collections wandb arrayrecord "
    "flax optax tensorflow 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html")

sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

# ─── 1. Paths ────────────────────────────────────────────────────────────────

# ArrayRecord episodes precomputed by build_episodes_arrayrecord.py
# Expects: ARECORD_DIR/train.arecord  and  ARECORD_DIR/val.arecord
ARECORD_DIR = "/kaggle/input/fsdit-episodes/arecords"   # ← adjust to your dataset

# W&B (set WANDB_API_KEY in Kaggle Secrets or here)
WANDB_KEY = os.environ.get("WANDB_API_KEY", "")
if WANDB_KEY:
    run(f"wandb login {WANDB_KEY}")

SAVE_DIR  = "/kaggle/working/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── 2. Hyperparameters ──────────────────────────────────────────────────────

# TPU v5e-8 has 8 cores, each with ~16 GB HBM → total 128 GB
# Recommended batch size: 256 global (32 per core)
BATCH_SIZE  = 256
MAX_STEPS   = 200_000
PRESET      = "big"       # hidden=768, depth=12, heads=12 → 133.9M params
GRAM_RANK_S = 32          # self-gram rank
GRAM_RANK_C = 32          # cross-gram rank
LR          = 1e-4
WEIGHT_DECAY = 0.01
COND_DROPOUT = 0.1        # CFG conditioning dropout

# Path remap: stored paths in arecord → actual Kaggle paths
# Example: images stored as "/workspace/data/..." → "/kaggle/input/..."
GRAIN_PATH_REMAP = ""     # e.g. "/workspace/miniIN:/kaggle/input/miniimagenet"

# ─── 3. Build train command ──────────────────────────────────────────────────

cmd_parts = [
    "python", "train.py",

    # Data
    f"--data_mode=grain",
    f"--grain_arecord_dir={ARECORD_DIR}",
    *(["--grain_path_remap=" + GRAIN_PATH_REMAP] if GRAIN_PATH_REMAP else []),

    # Model architecture
    f"--model.preset={PRESET}",
    f"--gram_rank_s={GRAM_RANK_S}",
    f"--gram_rank_c={GRAM_RANK_C}",

    # Optimizer
    f"--batch_size={BATCH_SIZE}",
    f"--max_steps={MAX_STEPS}",
    f"--lr={LR}",
    f"--weight_decay={WEIGHT_DECAY}",
    f"--cond_dropout={COND_DROPOUT}",

    # Checkpointing
    f"--save_dir={SAVE_DIR}",
    f"--save_interval=25000",

    # Logging
    f"--log_interval=500",
    f"--eval_interval=5000",
    f"--perf_log_interval=100",
    f"--log_model_debug=True",

    # W&B run name
    "--wandb.name=gram-adaln-big",
    "--wandb.project=fsdit",
]

# ─── 4. Run ──────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  GramDiT-AdaLN Training — TPU v5e-8")
print(f"  Branch : {BRANCH}")
print(f"  Preset : {PRESET}  |  gram_rank_s={GRAM_RANK_S}  gram_rank_c={GRAM_RANK_C}")
print(f"  Batch  : {BATCH_SIZE} global  ({BATCH_SIZE//8} per TPU core)")
print(f"  Steps  : {MAX_STEPS:,}")
print("="*60 + "\n")

cmd = " ".join(cmd_parts)
print(f"$ {cmd}\n")
os.execvp("python", cmd_parts)   # replace process (so Ctrl-C works cleanly)
