"""
model.py — DiT variants for FSDiT.

BRANCH: feat/gram-dit-block-adaln
  - GramDiTBlock: Gram matrix branches (self-gram + cross-gram) + AdaLN conditioning
  - Cross-attention REMOVED; conditioning via CLS token (SigLIP2/DINOv2) through AdaLN
  - Self-gram:  X' = RMSNorm(X·(X^T·A_s)·B_s) + Y
  - Cross-gram: Z  = RMSNorm(X'·(C^T·C_r)·D_r)   [no residual add; C = CLS token]
  - adaLN: c = LN(t_emb) + cond_scale·LN(MLP(cls_token))

Legacy DiT/DiTBlock kept for backward compatibility.
"""

import math
from typing import Any, Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


# ═══════════════════════════════════════════════════════════════════════════════
#  Embeddings
# ═══════════════════════════════════════════════════════════════════════════════

class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep → MLP → hidden_size vector."""
    hidden_size: int
    freq_dim: int = 256

    @nn.compact
    def __call__(self, t):
        # Sinusoidal encoding
        t = t.astype(jnp.float32) * 10000
        half = self.freq_dim // 2
        freqs = jnp.exp(-math.log(10000) *
                        jnp.arange(half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        # MLP
        x = nn.Dense(self.hidden_size,
                     kernel_init=nn.initializers.normal(0.02))(emb)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size,
                     kernel_init=nn.initializers.normal(0.02))(x)
        return x  # (B, hidden_size)


class SupportProjector(nn.Module):
    """
    Projects mean-pooled SigLIP2 embedding → DiT hidden_size.
    Handles CFG dropout: zeros out conditioning with prob `dropout_prob`.
    """
    hidden_size: int
    siglip_dim: int = 768
    dropout_prob: float = 0.1

    @nn.compact
    def __call__(self, support_embed, train=False, force_drop_ids=None):
        """
        Args:
            support_embed: (B, siglip_dim)
            train: enable dropout
            force_drop_ids: bool or (B,) int — True=uncond, False=cond
        Returns: (B, hidden_size)
        """
        # CFG dropout
        if force_drop_ids is not None:
            if isinstance(force_drop_ids, bool):
                if force_drop_ids:
                    support_embed = jnp.zeros_like(support_embed)
            else:
                mask = (force_drop_ids == 1)[:, None]
                support_embed = jnp.where(mask, 0.0, support_embed)
        elif train and self.dropout_prob > 0:
            drop = jax.random.bernoulli(
                self.make_rng('cond_dropout'), self.dropout_prob,
                (support_embed.shape[0],)
            )
            support_embed = jnp.where(drop[:, None], 0.0, support_embed)
        # MLP
        x = nn.Dense(self.hidden_size * 2,
                     kernel_init=nn.initializers.normal(0.02))(support_embed)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_size,
                     kernel_init=nn.initializers.normal(0.02))(x)
        return x


class LabelEmbedder(nn.Module):
    """Class-label embedding + CFG dropout. (Kept for backward compatibility.)"""
    dropout_prob: float
    num_classes: int
    hidden_size: int

    @nn.compact
    def __call__(self, labels, train, force_drop_ids=None):
        if force_drop_ids is not None:
            drop = force_drop_ids == 1
            labels = jnp.where(drop, self.num_classes, labels)
        elif train and self.dropout_prob > 0:
            drop = jax.random.bernoulli(
                self.make_rng('label_dropout'), self.dropout_prob,
                (labels.shape[0],)
            )
            labels = jnp.where(drop, self.num_classes, labels)
        return nn.Embed(
            self.num_classes + 1, self.hidden_size,
            embedding_init=nn.initializers.normal(0.02)
        )(labels)


# ═══════════════════════════════════════════════════════════════════════════════
#  Positional Encoding
# ═══════════════════════════════════════════════════════════════════════════════

def _sincos_1d(dim, pos):
    omega = 1.0 / \
        (10000 ** (jnp.arange(dim // 2, dtype=jnp.float32) / (dim / 2)))
    out = pos.reshape(-1)[:, None] * omega[None]
    return jnp.concatenate([jnp.sin(out), jnp.cos(out)], axis=1)


def get_2d_sincos_pos_embed(rng, dim, length):
    """(1, H*W, dim) sincos positional embedding for 2D grid."""
    gs = int(length ** 0.5)
    assert gs * gs == length
    grid_h, grid_w = jnp.arange(
        gs, dtype=jnp.float32), jnp.arange(gs, dtype=jnp.float32)
    gw, gh = jnp.meshgrid(grid_w, grid_h)
    emb = jnp.concatenate([
        _sincos_1d(dim // 2, gh.reshape(-1)),
        _sincos_1d(dim // 2, gw.reshape(-1)),
    ], axis=1)
    return emb[None]  # (1, H*W, dim)


# ═══════════════════════════════════════════════════════════════════════════════
#  Core DiT Blocks
# ═══════════════════════════════════════════════════════════════════════════════

def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class RMSNorm(nn.Module):
    """Root Mean Square Normalisation with learnable per-channel scale."""
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        scale = self.param('scale', nn.initializers.ones, (x.shape[-1],))
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        return scale * x / rms


class PatchEmbed(nn.Module):
    """Conv-based 2D → patch tokens."""
    patch_size: int
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        ps = self.patch_size
        x = nn.Conv(self.embed_dim, (ps, ps), (ps, ps), padding="VALID",
                    kernel_init=nn.initializers.xavier_uniform())(x)
        return rearrange(x, 'b h w c -> b (h w) c')


class OneLayerPerceiver(nn.Module):
    """One-layer latent Perceiver that compresses support token context."""
    num_latents: int
    hidden_size: int
    num_heads: int = 12
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, ctx):
        B = ctx.shape[0]
        hd = self.hidden_size // self.num_heads
        latents = self.param(
            'latents',
            nn.initializers.truncated_normal(stddev=0.02),
            (1, self.num_latents, self.hidden_size),
        )
        x = jnp.broadcast_to(latents, (B, self.num_latents, self.hidden_size))

        q = nn.Dense(self.hidden_size, use_bias=False)(
            nn.LayerNorm(use_bias=False, use_scale=False)(x)
        )
        kv = nn.Dense(2 * self.hidden_size, use_bias=False)(
            nn.LayerNorm(use_bias=False, use_scale=False)(ctx)
        )
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k, v = [rearrange(t, 'b s (h d) -> b h s d', h=self.num_heads)
                for t in jnp.split(kv, 2, axis=-1)]

        attn_w = jax.nn.softmax(
            jnp.einsum('bhqd,bhsd->bhqs', q, k) * (hd ** -0.5),
            axis=-1,
        )
        attn_out = rearrange(
            jnp.einsum('bhqs,bhsd->bhqd', attn_w, v),
            'b h n d -> b n (h d)',
        )
        attn_out = nn.Dense(
            self.hidden_size, kernel_init=nn.initializers.xavier_uniform())(attn_out)
        x = x + attn_out

        mlp_dim = int(self.hidden_size * self.mlp_ratio)
        h = nn.Dense(
            mlp_dim,
            kernel_init=nn.initializers.he_normal(),
        )(nn.LayerNorm(use_bias=False, use_scale=False)(x))
        h = nn.gelu(h)
        h = nn.Dense(
            self.hidden_size,
            kernel_init=nn.initializers.xavier_uniform(),
        )(h)
        return x + h


# ═══════════════════════════════════════════════════════════════════════════════
#  Gram DiT Block (feat/gram-dit-block-adaln)
# ═══════════════════════════════════════════════════════════════════════════════

class GramDiTBlock(nn.Module):
    """Gram-matrix DiT block with AdaLN conditioning.

    Pipeline (no cross-attention):
      Y  = X + α₁ · MSA(adaLN(X, γ₁, β₁))
      X' = RMSNorm(X · (X^T·A_s) · B_s) + Y        ← self-gram branch
      Z  = RMSNorm(X' · (C^T·C_r) · D_r)            ← cross-gram branch (C = CLS token, no residual)
      out = Z + α₂ · MLP(adaLN(Z, γ₂, β₂))

    The gram branches are zero-init at B_s / D_r so they are identity-skip
    at the start of training.
    """
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    gram_rank_s: int = 32   # rank for self-gram
    gram_rank_c: int = 32   # rank for cross-gram

    @nn.compact
    def __call__(self, x, c, cls_token, return_attn=False):
        """
        Args:
            x:         (B, N, D)  patch tokens
            c:         (B, D)     global adaLN conditioning signal
            cls_token: (B, D)     CLS embedding from SigLIP2/DINOv2 for cross-gram
        Returns:
            x: (B, N, D)
        """
        D = self.hidden_size
        r_s = self.gram_rank_s
        r_c = self.gram_rank_c

        # ── adaLN modulation (6-way projection, zero-init → stable at init) ──
        c_proj = nn.Dense(
            6 * D, kernel_init=nn.initializers.constant(0.)
        )(nn.silu(c))  # (B, 6D)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = jnp.split(c_proj, 6, axis=-1)

        # ── Self-Attention ──
        x_norm = modulate(nn.LayerNorm(use_bias=False, use_scale=False)(x), shift_a, scale_a)
        hd = D // self.num_heads
        qkv = nn.Dense(3 * D, kernel_init=nn.initializers.xavier_uniform())(x_norm)
        q, k, v = [rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads)
                   for t in jnp.split(qkv, 3, axis=-1)]
        attn_w = jax.nn.softmax(
            jnp.einsum('bhqd,bhkd->bhqk', q, k) * (hd ** -0.5), axis=-1)
        attn_out = rearrange(
            jnp.einsum('bhqk,bhkd->bhqd', attn_w, v), 'b h n d -> b n (h d)')
        attn_out = nn.Dense(D, kernel_init=nn.initializers.xavier_uniform())(attn_out)
        Y = x + gate_a[:, None] * attn_out  # (B, N, D)

        # ── Self-Gram Branch: X' = RMSNorm( X · (X^T·A_s) · B_s ) + Y ──
        if r_s > 0:
            A_s = self.param('A_s', nn.initializers.truncated_normal(0.02), (D, r_s))
            B_s = self.param('B_s', nn.initializers.zeros, (r_s, D))
            # Efficient: avoid N×N gram matrix
            u_s = jnp.einsum('bnd,dr->bnr', x, A_s)       # (B, N, r_s)
            gram_s = jnp.einsum('bnr,rd->bnd', u_s, B_s)  # (B, N, D)
            X_prime = RMSNorm()(gram_s) + Y
        else:
            X_prime = Y

        # ── Cross-Gram Branch: Z = RMSNorm( X' · (C^T·C_r) · D_r ) ──
        # C = cls_token: (B, D) — per-sample, so u_c is per-sample
        if r_c > 0:
            C_r = self.param('C_r', nn.initializers.truncated_normal(0.02), (D, r_c))
            D_r = self.param('D_r', nn.initializers.zeros, (r_c, D))
            # u_c: per-sample projection matrix  (B, D, r_c)
            u_c = jnp.einsum('bd,dr->bdr', cls_token, C_r)
            # v_c: project patch tokens into gram space  (B, N, r_c)
            v_c = jnp.einsum('bnd,bdr->bnr', X_prime, u_c)
            gram_c = jnp.einsum('bnr,rd->bnd', v_c, D_r)  # (B, N, D)
            Z = RMSNorm()(gram_c)                          # no residual add per diagram
        else:
            Z = X_prime

        # ── MLP with adaLN ──
        z_norm = modulate(nn.LayerNorm(use_bias=False, use_scale=False)(Z), shift_m, scale_m)
        mlp_dim = int(D * self.mlp_ratio)
        h = nn.Dense(mlp_dim, kernel_init=nn.initializers.xavier_uniform())(z_norm)
        h = nn.gelu(h)
        h = nn.Dense(D, kernel_init=nn.initializers.xavier_uniform())(h)
        out = Z + gate_m[:, None] * h

        return (out, attn_w) if return_attn else out


class DiTBlock(nn.Module):
    """adaLN-Zero DiT block with CLS-only cross-attention.

    Self-attention operates on [CLS, patch_tokens]. Cross-attention only uses
    the CLS token (index 0) as query against support context, then writes
    the result back to the CLS position. Patches receive condition info
    through self-attention with CLS.
    """
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, c, context=None, return_attn=False):
        # adaLN modulation params
        c_proj = nn.Dense(
            6 * self.hidden_size, kernel_init=nn.initializers.constant(0.)
        )(nn.silu(c))
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = jnp.split(
            c_proj, 6, axis=-1)

        # ── Self-Attention on [CLS, patches] ──
        x_norm = modulate(
            nn.LayerNorm(use_bias=False, use_scale=False)(x), shift_a, scale_a
        )
        hd = self.hidden_size // self.num_heads
        qkv = nn.Dense(3 * self.hidden_size,
                       kernel_init=nn.initializers.xavier_uniform())(x_norm)
        q, k, v = [rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads)
                   for t in jnp.split(qkv, 3, axis=-1)]
        attn_w = jax.nn.softmax(jnp.einsum(
            'bhqd,bhkd->bhqk', q, k) * (hd ** -0.5), axis=-1)
        attn_out = rearrange(jnp.einsum(
            'bhqk,bhkd->bhqd', attn_w, v), 'b h n d -> b n (h d)')
        attn_out = nn.Dense(
            self.hidden_size, kernel_init=nn.initializers.xavier_uniform())(attn_out)
        x = x + gate_a[:, None] * attn_out

        # ── CLS-only Cross-Attention on support context ──
        if context is not None:
            cls_token = x[:, :1]  # (B, 1, D)
            q_c = nn.Dense(
                self.hidden_size, kernel_init=nn.initializers.xavier_uniform())(
                nn.LayerNorm(use_bias=False, use_scale=False)(cls_token)
            )
            kv_c = nn.Dense(
                2 * self.hidden_size, kernel_init=nn.initializers.xavier_uniform())(
                nn.LayerNorm(use_bias=False, use_scale=False)(context)
            )
            q_c = rearrange(q_c, 'b n (h d) -> b h n d', h=self.num_heads)
            k_c, v_c = [rearrange(t, 'b s (h d) -> b h s d', h=self.num_heads)
                        for t in jnp.split(kv_c, 2, axis=-1)]
            c_attn_w = jax.nn.softmax(
                jnp.einsum('bhqd,bhsd->bhqs', q_c, k_c) * (hd ** -0.5),
                axis=-1,
            )
            c_attn_out = rearrange(
                jnp.einsum('bhqs,bhsd->bhqd', c_attn_w, v_c),
                'b h n d -> b n (h d)',
            )
            c_attn_out = nn.Dense(
                self.hidden_size,
                kernel_init=nn.initializers.zeros,
            )(c_attn_out)
            # Write cross-attn result back to CLS position only
            x = x.at[:, :1].add(c_attn_out)

        # ── MLP ──
        x_norm2 = modulate(
            nn.LayerNorm(use_bias=False, use_scale=False)(x), shift_m, scale_m
        )
        mlp_dim = int(self.hidden_size * self.mlp_ratio)
        h = nn.Dense(
            mlp_dim, kernel_init=nn.initializers.xavier_uniform())(x_norm2)
        h = nn.gelu(h)
        h = nn.Dense(self.hidden_size,
                     kernel_init=nn.initializers.xavier_uniform())(h)
        x = x + gate_m[:, None] * h

        return (x, attn_w) if return_attn else x


class FinalLayer(nn.Module):
    """adaLN + linear projection to patch pixels."""
    patch_size: int
    out_channels: int
    hidden_size: int

    @nn.compact
    def __call__(self, x, c):
        c_proj = nn.Dense(
            2 * self.hidden_size, kernel_init=nn.initializers.constant(0)
        )(nn.silu(c))
        shift, scale = jnp.split(c_proj, 2, axis=-1)
        x = modulate(nn.LayerNorm(use_bias=False,
                     use_scale=False)(x), shift, scale)
        x = nn.Dense(
            self.patch_size ** 2 * self.out_channels,
            kernel_init=nn.initializers.constant(0)
        )(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  GramDiT Model (feat/gram-dit-block-adaln)
# ═══════════════════════════════════════════════════════════════════════════════

class GramDiT(nn.Module):
    """
    Gram-matrix Diffusion Transformer with AdaLN conditioning.

    Architecture:
      1. PatchEmbed → patch tokens + sincos pos embed
      2. Conditioning: c = LN(t_emb) + cond_scale·LN(SupportProjector(y_pooled))
         - y_pooled: mean-pooled CLS token from SigLIP2 or DINOv2 (B, siglip_dim)
         - cls_token passed raw to each GramDiTBlock for cross-gram branch
      3. N × GramDiTBlock: self-attn + self-gram + cross-gram (no cross-attn)
      4. FinalLayer → unpatchify

    No learnable CLS prepend, no y_seq/context pipeline.
    """
    patch_size: int
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    siglip_dim: int = 768      # CLS token dim (SigLIP2-B/16 or DINOv2-B)
    cond_dropout_prob: float = 0.1
    gram_rank_s: int = 32      # self-gram rank
    gram_rank_c: int = 32      # cross-gram rank
    learn_sigma: bool = False

    @nn.compact
    def __call__(
        self, x, t, y_pooled,
        train=False, force_drop_ids=None, return_attn=False, return_debug=False
    ):
        """
        Args:
            x:         (B, H, W, C)      noisy image / latent
            t:         (B,)              timestep ∈ [0, 1]
            y_pooled:  (B, siglip_dim)   mean-pooled CLS token from encoder
            train:     bool
            force_drop_ids: bool or (B,) int — CFG unconditional
        """
        B, S, _, C_in = x.shape
        C_out = C_in * (2 if self.learn_sigma else 1)
        ps = self.patch_size
        n_patches = (S // ps) ** 2
        n_side = S // ps

        # Patch embed + sincos pos embed
        pos = self.param("pos_embed", get_2d_sincos_pos_embed,
                         self.hidden_size, n_patches)
        pos = jax.lax.stop_gradient(pos)
        x = PatchEmbed(ps, self.hidden_size)(x) + pos  # (B, N, D)

        # Timestep embed
        t_emb = TimestepEmbedder(self.hidden_size)(t)  # (B, D)

        # ── Unified CFG dropout ──────────────────────────────────────────────
        # Compute ONE drop mask, apply consistently to both:
        #   cls_emb (→ AdaLN conditioning c)  and  cls_token (→ cross-gram C)
        # This prevents the inconsistency where one is dropped but not the other.
        if force_drop_ids is not None:
            # Normalise to a (B,) int array so isinstance() is never called at
            # trace time, avoiding an extra XLA retrace when type changes.
            if isinstance(force_drop_ids, bool):
                # bool → broadcast scalar: True=drop-all, False=keep-all
                drop_ids = jnp.ones((B,), dtype=jnp.int32) if force_drop_ids \
                           else jnp.zeros((B,), dtype=jnp.int32)
            else:
                drop_ids = force_drop_ids  # already (B,) int
            unified_mask = (drop_ids == 1)[:, None]   # (B, 1) bool
            cls_token = jnp.where(unified_mask, 0.0, y_pooled)
            # SupportProjector: pass the same normalised array, disable internal dropout
            cls_emb = SupportProjector(
                self.hidden_size, self.siglip_dim, self.cond_dropout_prob
            )(y_pooled, train=False, force_drop_ids=drop_ids)
        elif train and self.cond_dropout_prob > 0:
            # Training dropout: draw ONE mask, share for both branches
            drop = jax.random.bernoulli(
                self.make_rng('cond_dropout'), self.cond_dropout_prob, (B,)
            )
            unified_mask = drop[:, None]              # (B, 1)
            cls_token = jnp.where(unified_mask, 0.0, y_pooled)
            # Convert bool mask to int ids for SupportProjector
            drop_ids = drop.astype(jnp.int32)
            cls_emb = SupportProjector(
                self.hidden_size, self.siglip_dim, self.cond_dropout_prob
            )(y_pooled, train=False, force_drop_ids=drop_ids)
        else:
            cls_token = y_pooled
            cls_emb = SupportProjector(
                self.hidden_size, self.siglip_dim, self.cond_dropout_prob
            )(y_pooled, train=False, force_drop_ids=False)

        # Project raw cls_token to hidden_size for cross-gram
        cls_token_proj = nn.Dense(
            self.hidden_size,
            kernel_init=nn.initializers.normal(0.02),
            name='cls_token_proj'
        )(cls_token)  # (B, D)

        # Global conditioning signal: c = LN(t_emb) + cond_scale * LN(cls_emb)
        t_emb_n = nn.LayerNorm(name='t_emb_ln')(t_emb)
        y_emb_n = nn.LayerNorm(name='y_emb_ln')(cls_emb)
        cond_scale = self.param('cond_scale', nn.initializers.constant(2.0), ())
        c = t_emb_n + cond_scale * y_emb_n  # (B, D)

        debug = None
        act_abs = []
        act_rms = []
        if return_debug:
            c_f32 = c.astype(jnp.float32)
            yp = y_pooled.astype(jnp.float32)
            debug = {
                "t_emb_abs_mean": jnp.mean(jnp.abs(t_emb_n.astype(jnp.float32))),
                "y_emb_abs_mean": jnp.mean(jnp.abs(y_emb_n.astype(jnp.float32))),
                "c_abs_mean": jnp.mean(jnp.abs(c_f32)),
                "c_l2_mean": jnp.mean(jnp.linalg.norm(c_f32, axis=-1)),
                "support_pooled_abs_mean": jnp.mean(jnp.abs(yp)),
                "support_pooled_l2_mean": jnp.mean(jnp.linalg.norm(yp, axis=-1)),
            }

        # Transformer blocks
        attn_list = []
        for _ in range(self.depth):
            if return_attn:
                x, aw = GramDiTBlock(
                    self.hidden_size, self.num_heads, self.mlp_ratio,
                    self.gram_rank_s, self.gram_rank_c
                )(x, c, cls_token_proj, return_attn=True)
                attn_list.append(aw)
            else:
                x = GramDiTBlock(
                    self.hidden_size, self.num_heads, self.mlp_ratio,
                    self.gram_rank_s, self.gram_rank_c
                )(x, c, cls_token_proj)
            if return_debug:
                x_f32 = x.astype(jnp.float32)
                act_abs.append(jnp.mean(jnp.abs(x_f32)))
                act_rms.append(jnp.sqrt(jnp.mean(jnp.square(x_f32))))

        # Final layer → unpatchify
        x = FinalLayer(ps, C_out, self.hidden_size)(x, c)
        x = jnp.reshape(x, (B, n_side, n_side, ps, ps, C_out))
        x = jnp.einsum('bhwpqc->bhpwqc', x)
        x = rearrange(x, 'B H P W Q C -> B (H P) (W Q) C')

        if return_debug:
            debug["act_abs_per_layer"] = jnp.stack(act_abs)
            debug["act_rms_per_layer"] = jnp.stack(act_rms)

        if return_attn and return_debug:
            return x, attn_list, debug
        if return_attn:
            return x, attn_list
        if return_debug:
            return x, debug
        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  Legacy DiT Model (kept for backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

class DiT(nn.Module):
    """
    Diffusion Transformer with CLS-token cross-attention.

    Architecture:
      1. PatchEmbed → patch tokens
      2. Prepend learnable CLS token
      3. N × DiTBlock: self-attn on [CLS, patches]; CLS-only cross-attn with context
      4. Strip CLS → FinalLayer → unpatchify
    """
    patch_size: int
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    # Class-label mode
    class_dropout_prob: float = 0.0
    num_classes: int = 0
    # FSDiT mode
    siglip_dim: int = 0
    cond_dropout_prob: float = 0.1
    learn_sigma: bool = False

    @nn.compact
    def __call__(
        self, x, t, y_pooled, y_seq=None,
        train=False, force_drop_ids=None, return_attn=False, return_debug=False
    ):
        """
        x: (B, H, W, C)   noisy image / latent
        t: (B,)            timestep ∈ [0, 1]
        y_pooled: (B,) int  OR  (B, siglip_dim) float
        y_seq: (B, T, siglip_dim) float or None
        """
        B, S, _, C_in = x.shape
        C_out = C_in * (2 if self.learn_sigma else 1)
        ps = self.patch_size
        n_patches = (S // ps) ** 2
        n_side = S // ps

        # Patch embed + pos embed
        pos = self.param("pos_embed", get_2d_sincos_pos_embed,
                         self.hidden_size, n_patches)
        pos = jax.lax.stop_gradient(pos)
        x = PatchEmbed(ps, self.hidden_size)(x) + pos  # (B, n_patches, D)

        # ── Prepend CLS token ──
        cls_token = self.param(
            "cls_token",
            nn.initializers.truncated_normal(stddev=0.02),
            (1, 1, self.hidden_size),
        )
        cls_tokens = jnp.broadcast_to(cls_token, (B, 1, self.hidden_size))
        x = jnp.concatenate([cls_tokens, x], axis=1)  # (B, 1+n_patches, D)

        # Timestep embed
        t_emb = TimestepEmbedder(self.hidden_size)(t)

        # Conditioning embed
        if self.siglip_dim > 0:
            y_emb = SupportProjector(
                self.hidden_size, self.siglip_dim, self.cond_dropout_prob
            )(y_pooled, train=train, force_drop_ids=force_drop_ids)
        else:
            y_emb = LabelEmbedder(
                self.class_dropout_prob, self.num_classes, self.hidden_size
            )(y_pooled, train=train, force_drop_ids=force_drop_ids)

        # Normalize both to same scale before combining
        t_emb = nn.LayerNorm(name='t_emb_ln')(t_emb)
        y_emb = nn.LayerNorm(name='y_emb_ln')(y_emb)
        cond_scale = self.param('cond_scale', nn.initializers.constant(2.0), ())
        c = t_emb + cond_scale * y_emb

        debug = None
        act_abs = []
        act_rms = []
        if return_debug:
            c_f32 = c.astype(jnp.float32)
            t_f32 = t_emb.astype(jnp.float32)
            y_f32 = y_emb.astype(jnp.float32)
            debug = {
                "t_emb_abs_mean": jnp.mean(jnp.abs(t_f32)),
                "y_emb_abs_mean": jnp.mean(jnp.abs(y_f32)),
                "c_abs_mean": jnp.mean(jnp.abs(c_f32)),
                "c_l2_mean": jnp.mean(jnp.linalg.norm(c_f32, axis=-1)),
            }
            if self.siglip_dim > 0 and hasattr(y_pooled, "ndim") and y_pooled.ndim == 2:
                yp = y_pooled.astype(jnp.float32)
                debug["support_pooled_abs_mean"] = jnp.mean(jnp.abs(yp))
                debug["support_pooled_l2_mean"] = jnp.mean(jnp.linalg.norm(yp, axis=-1))
            else:
                debug["support_pooled_abs_mean"] = jnp.array(0.0, dtype=jnp.float32)
                debug["support_pooled_l2_mean"] = jnp.array(0.0, dtype=jnp.float32)

        # ── Support context for CLS cross-attention ──
        context = None
        if y_seq is not None:
            context = OneLayerPerceiver(
                num_latents=n_patches,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
            )(y_seq)
            if force_drop_ids is not None:
                if isinstance(force_drop_ids, bool):
                    if force_drop_ids:
                        context = jnp.zeros_like(context)
                else:
                    mask = (force_drop_ids == 1)[:, None, None]
                    context = jnp.where(mask, 0.0, context)
            elif train and self.cond_dropout_prob > 0:
                drop = jax.random.bernoulli(
                    self.make_rng('cond_dropout'),
                    self.cond_dropout_prob,
                    (context.shape[0],),
                )
                context = jnp.where(drop[:, None, None], 0.0, context)

        # Transformer blocks
        attn_list = []
        for _ in range(self.depth):
            if return_attn:
                x, aw = DiTBlock(self.hidden_size, self.num_heads,
                                 self.mlp_ratio)(x, c, context=context, return_attn=True)
                attn_list.append(aw)
            else:
                x = DiTBlock(self.hidden_size, self.num_heads,
                             self.mlp_ratio)(x, c, context=context)
            if return_debug:
                x_f32 = x.astype(jnp.float32)
                act_abs.append(jnp.mean(jnp.abs(x_f32)))
                act_rms.append(jnp.sqrt(jnp.mean(jnp.square(x_f32))))

        # ── Strip CLS token, keep only patch tokens ──
        x = x[:, 1:]  # (B, n_patches, D)

        # Unpatchify
        x = FinalLayer(ps, C_out, self.hidden_size)(x, c)
        x = jnp.reshape(x, (B, n_side, n_side, ps, ps, C_out))
        x = jnp.einsum('bhwpqc->bhpwqc', x)
        x = rearrange(x, 'B H P W Q C -> B (H P) (W Q) C')

        if return_debug:
            debug["act_abs_per_layer"] = jnp.stack(act_abs)
            debug["act_rms_per_layer"] = jnp.stack(act_rms)

        if return_attn and return_debug:
            return x, attn_list, debug
        if return_attn:
            return x, attn_list
        if return_debug:
            return x, debug
        return x
