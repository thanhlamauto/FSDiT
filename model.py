"""
model.py — DiT (Diffusion Transformer) with FiLM conditioning.

Supports two modes:
  1. Class-label conditioning (original DiT)
  2. Support-set conditioning via SigLIP2 embeddings (FSDiT)

Architecture: PatchEmbed → N × DiTBlock(adaLN-Zero) → FinalLayer → unpatchify
Conditioning: c = TimestepEmbed(t) + SupportProjector(siglip_embed)
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
        freqs = jnp.exp(-math.log(10000) * jnp.arange(half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        # MLP
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(emb)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
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
        x = nn.Dense(self.hidden_size * 2, kernel_init=nn.initializers.normal(0.02))(support_embed)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
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
    omega = 1.0 / (10000 ** (jnp.arange(dim // 2, dtype=jnp.float32) / (dim / 2)))
    out = pos.reshape(-1)[:, None] * omega[None]
    return jnp.concatenate([jnp.sin(out), jnp.cos(out)], axis=1)


def get_2d_sincos_pos_embed(rng, dim, length):
    """(1, H*W, dim) sincos positional embedding for 2D grid."""
    gs = int(length ** 0.5)
    assert gs * gs == length
    grid_h, grid_w = jnp.arange(gs, dtype=jnp.float32), jnp.arange(gs, dtype=jnp.float32)
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


class DiTBlock(nn.Module):
    """adaLN-Zero DiT block. Optionally returns attention weights."""
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, c, return_attn=False):
        # adaLN modulation params
        c_proj = nn.Dense(
            6 * self.hidden_size, kernel_init=nn.initializers.constant(0.)
        )(nn.silu(c))
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = jnp.split(c_proj, 6, axis=-1)

        # ── Self-Attention ──
        x_norm = modulate(
            nn.LayerNorm(use_bias=False, use_scale=False)(x), shift_a, scale_a
        )
        if return_attn:
            attn_out, attn_w = self._attention_with_weights(x_norm)
        else:
            attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                kernel_init=nn.initializers.xavier_uniform(),
            )(x_norm, x_norm)
            attn_w = None
        x = x + gate_a[:, None] * attn_out

        # ── MLP ──
        x_norm2 = modulate(
            nn.LayerNorm(use_bias=False, use_scale=False)(x), shift_m, scale_m
        )
        mlp_dim = int(self.hidden_size * self.mlp_ratio)
        h = nn.Dense(mlp_dim, kernel_init=nn.initializers.xavier_uniform())(x_norm2)
        h = nn.gelu(h)
        h = nn.Dense(self.hidden_size, kernel_init=nn.initializers.xavier_uniform())(h)
        x = x + gate_m[:, None] * h

        return (x, attn_w) if return_attn else x

    def _attention_with_weights(self, x):
        """Manual MHA that returns attention weights for entropy tracking."""
        hd = self.hidden_size // self.num_heads
        qkv = nn.Dense(3 * self.hidden_size, kernel_init=nn.initializers.xavier_uniform())(x)
        q, k, v = [rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads)
                    for t in jnp.split(qkv, 3, axis=-1)]
        attn_w = jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk', q, k) * (hd ** -0.5), axis=-1)
        out = rearrange(jnp.einsum('bhqk,bhkd->bhqd', attn_w, v), 'b h n d -> b n (h d)')
        out = nn.Dense(self.hidden_size, kernel_init=nn.initializers.xavier_uniform())(out)
        return out, attn_w  # attn_w: (B, H, Q, K)


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
        x = modulate(nn.LayerNorm(use_bias=False, use_scale=False)(x), shift, scale)
        x = nn.Dense(
            self.patch_size ** 2 * self.out_channels,
            kernel_init=nn.initializers.constant(0)
        )(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  Full DiT Model
# ═══════════════════════════════════════════════════════════════════════════════

class DiT(nn.Module):
    """
    Diffusion Transformer.

    When `siglip_dim > 0`: FSDiT mode — conditioned on support-set embeddings.
    When `siglip_dim == 0`: Original mode — conditioned on class labels.
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
    def __call__(self, x, t, y, train=False, force_drop_ids=None, return_attn=False):
        """
        x: (B, H, W, C)   noisy image / latent
        t: (B,)            timestep ∈ [0, 1]
        y: (B,) int  OR  (B, siglip_dim) float
        """
        B, S, _, C_in = x.shape
        C_out = C_in * (2 if self.learn_sigma else 1)
        ps = self.patch_size
        n_patches = (S // ps) ** 2
        n_side = S // ps

        # Patch embed + pos embed
        pos = self.param("pos_embed", get_2d_sincos_pos_embed, self.hidden_size, n_patches)
        pos = jax.lax.stop_gradient(pos)
        x = PatchEmbed(ps, self.hidden_size)(x) + pos

        # Timestep embed
        t_emb = TimestepEmbedder(self.hidden_size)(t)

        # Conditioning embed
        if self.siglip_dim > 0:
            y_emb = SupportProjector(
                self.hidden_size, self.siglip_dim, self.cond_dropout_prob
            )(y, train=train, force_drop_ids=force_drop_ids)
        else:
            y_emb = LabelEmbedder(
                self.class_dropout_prob, self.num_classes, self.hidden_size
            )(y, train=train, force_drop_ids=force_drop_ids)
        c = t_emb + y_emb

        # Transformer blocks
        attn_list = []
        for _ in range(self.depth):
            if return_attn:
                x, aw = DiTBlock(self.hidden_size, self.num_heads, self.mlp_ratio)(x, c, True)
                attn_list.append(aw)
            else:
                x = DiTBlock(self.hidden_size, self.num_heads, self.mlp_ratio)(x, c)

        # Unpatchify
        x = FinalLayer(ps, C_out, self.hidden_size)(x, c)
        x = jnp.reshape(x, (B, n_side, n_side, ps, ps, C_out))
        x = jnp.einsum('bhwpqc->bhpwqc', x)
        x = rearrange(x, 'B H P W Q C -> B (H P) (W Q) C')

        return (x, attn_list) if return_attn else x
