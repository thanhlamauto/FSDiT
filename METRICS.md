# FSDiT — Logged Metrics Reference

All metrics are logged to **Weights & Biases** (wandb). Categories match the wandb panel groups.

---

## `train/` — Training Loss & Optimization

| Metric | Ý nghĩa | Giá trị lý tưởng |
|--------|---------|-------------------|
| `train/loss` | MSE loss giữa velocity dự đoán vs ground-truth | ↓ giảm dần |
| `train/loss_ema` | EMA-smoothed loss (α=0.99) — ít nhiễu hơn | ↓ đường cong mượt |
| `train/grad_norm` | L2 norm của gradient (sau clip) | Ổn định, < `grad_clip` (1.0) |
| `train/param_norm` | L2 norm toàn bộ params | Tăng nhẹ rồi ổn định |
| `train/lr` | Learning rate hiện tại (warmup → cosine decay) | Theo schedule |
| `train/step_time` | Wall-clock time cho 1 iteration (giây) | Càng nhỏ càng tốt |
| `train/loss_tbin_{i}` | Loss chia theo timestep bin (i=0..9) | Xem distribution |

**Cách đọc `loss_tbin`:** Bin 0 = t gần 0 (noise), bin 9 = t gần 1 (data). Nếu bin 0 loss cao → model khó denoise noise thuần. Nếu bin 9 cao → model khó refine gần data.

---

## `val/` — Validation

| Metric | Ý nghĩa |
|--------|---------|
| `val/loss` | Validation MSE (dùng EMA model, no dropout) |
| `val/train_val_gap` | `train_loss_ema - val_loss`. Gap > 0 = overfitting |
| `val/loss_tbin_{i}` | Validation loss theo timestep bin |

---

## `cond/` & `val_cond/` — Conditioning Quality

Đánh giá chất lượng SigLIP support embeddings — quan trọng vì model điều kiện hoá theo đây.

| Metric | Ý nghĩa | Giá trị lý tưởng |
|--------|---------|-------------------|
| `cond/support_pooled_abs_mean` | |embedding| trung bình | > 0, ổn định |
| `cond/support_pooled_l2_mean` | L2 norm trung bình | > 0 |
| `cond/support_pooled_dim_std_mean` | Trung bình std mỗi dimension | Cao = diverse |
| `cond/same_class_pair_ratio` | Tỷ lệ cặp cùng class trong batch | ~`1/num_classes` |
| `cond/same_class_cos_mean` | Cosine similarity trung bình cặp cùng class | Cao (>0.5) = SigLIP phân biệt tốt |
| `cond/diff_class_cos_mean` | Cosine similarity trung bình cặp khác class | Thấp (<0.3) |
| `cond/same_class_cos_std` | Std cosine cùng class | Thấp = nhất quán |
| `cond/diff_class_cos_std` | Std cosine khác class | Thấp |

**Histograms** (mỗi `cond_hist_interval` steps):
- `cond/support_pooled_hist` — phân bố giá trị embedding
- `cond/same_class_cos_hist` — phân bố cosine cùng class
- `cond/diff_class_cos_hist` — phân bố cosine khác class

> **Mục đích:** Nếu `same_class_cos_mean ≈ diff_class_cos_mean` → SigLIP embeddings không phân biệt classes → model khó học conditioned generation.

---

## `dbg/` — Model Debug (Internal Activations)

Chỉ log khi `--log_model_debug=True`. Giúp debug training instability.

| Metric | Ý nghĩa | Cảnh báo |
|--------|---------|----------|
| `dbg/t_emb_abs_mean` | |timestep embedding| trung bình | → 0 = dead |
| `dbg/y_emb_abs_mean` | |condition embedding| trung bình | → 0 = condition bị ignore |
| `dbg/c_abs_mean` | |combined condition| (t+y) trung bình | Nên ổn định |
| `dbg/c_l2_mean` | L2 norm combined condition | Nên ổn định |
| `dbg/support_pooled_abs_mean_model` | |SigLIP pooled| trung bình (as seen by model) | — |
| `dbg/support_pooled_l2_mean_model` | L2 norm SigLIP pooled (model side) | — |

---

## `act/` — Per-Layer Activation Statistics

| Metric | Ý nghĩa | Cảnh báo |
|--------|---------|----------|
| `act/layer{i}_abs_mean` | |activation| trung bình layer i | Nên ổn định theo depth |
| `act/layer{i}_rms` | RMS activation layer i | Tăng vọt = instability |

> **Mục đích:** Phát hiện vanishing/exploding activations. Nếu `act/layer11_rms` >> `act/layer0_rms` → exploding. Nếu → 0 → vanishing.

---

## `attn/` — Attention Entropy

| Metric | Ý nghĩa | Giá trị |
|--------|---------|---------|
| `attn/entropy_layer{i}` | Entropy trung bình các head ở layer i | Cao = đa dạng |
| `attn/entropy_head{h}_last` | Entropy head h ở layer cuối | — |

> **Entropy cao** = attention phân bổ đều (tốt ở layers đầu). **Entropy thấp** = attention tập trung (tốt ở layers cuối → model biết focus vào đâu).

---

## `perf/` — Performance Timing

| Metric | Ý nghĩa | Target |
|--------|---------|--------|
| `perf/data_time` | Thời gian load + preprocess 1 batch (s) | < 0.1s |
| `perf/siglip_encode_time` | Thời gian SigLIP encode (online mode) (s) | — |
| `perf/vae_time` | Thời gian VAE encode (nếu `use_vae`) (s) | — |
| `perf/train_step_time` | Thời gian JAX train step (forward+backward+update) (s) | Bottleneck |
| `perf/total_iter_time` | Tổng wall-clock 1 iteration (s) | — |
| `perf/siglip_cache_hit_rate` | Tỷ lệ cache hit SigLIP (online mode) | > 0.5 tốt |
| `perf/siglip_cache_items` | Số items trong SigLIP LRU cache | — |

---

## `samples/` — Generated Images

| Metric | Ý nghĩa |
|--------|---------|
| `samples/cfg_0` | Generated images không CFG (unconditional) |
| `samples/cfg_{scale}` | Generated images với Classifier-Free Guidance |

> So sánh `cfg_0` vs `cfg_{scale}`: nếu trông giống nhau → model chưa học được conditioning.
