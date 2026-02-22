"""
encoder.py — Frozen SigLIP2 B/16 vision encoder.

Encodes 5 support images → mean-pooled (B, 768) embedding.
All weights frozen (stop_gradient), only used for inference.
"""

import os
import sys
import jax
import jax.numpy as jnp
import ml_collections
from functools import partial

# Lazy big_vision imports
_model_mod = None


def _setup_big_vision():
    """Clone big_vision repo if needed and import."""
    global _model_mod
    if _model_mod is not None:
        return
    repo = os.environ.get('BIG_VISION_DIR', '/kaggle/working/big_vision')
    if not os.path.exists(repo):
        print(f"Cloning big_vision → {repo}")
        os.system(f'git clone --quiet --branch=main --depth=1 '
                   f'https://github.com/google-research/big_vision {repo} > /dev/null 2>&1')
        os.system(f'pip install -q -r {repo}/big_vision/requirements.txt > /dev/null 2>&1')
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import big_vision.models.proj.image_text.two_towers as m  # noqa
    import big_vision.pp.ops_general  # noqa: register
    import big_vision.pp.ops_image    # noqa: register
    _model_mod = m


class SigLIP2Encoder:
    """
    Frozen SigLIP2 B/16 224×224 vision encoder.

    Usage:
        enc = SigLIP2Encoder.create()
        embed = enc.encode_supports(support_images)  # (B, 768)
    """
    EMBED_DIM = 768  # B/16

    def __init__(self, model, params):
        self.model = model
        self.params = params

    @classmethod
    def create(cls, ckpt_path=None, variant='B/16', res=224):
        _setup_big_vision()
        txt_var, patch = variant.split('/')
        emb_dim = {'B': 768, 'L': 1024, 'So400m': 1152}[txt_var]

        cfg = ml_collections.ConfigDict(dict(
            image_model='vit',
            image=dict(pool_type='map', scan=True, variant=variant),
            text_model='proj.image_text.text_transformer',
            text=dict(scan=True, variant=txt_var, vocab_size=256_000),
            out_dim=[None, emb_dim],
            bias_init=-10,
        ))

        if ckpt_path is None:
            name = f'siglip2_{txt_var.lower()}{patch}_{res}.npz'
            ckpt_path = f'/tmp/{name}'
            if not os.path.exists(ckpt_path):
                url = f'https://storage.googleapis.com/big_vision/siglip2/{name}'
                print(f"Downloading {url}")
                os.system(f'wget -q {url} -O {ckpt_path}')

        print(f"Loading SigLIP2 {variant} {res}×{res}")
        model = _model_mod.Model(**cfg)
        params = _model_mod.load(None, ckpt_path, cfg)
        return cls(model, params)

    def _encode_batch(self, images):
        """(N, 224, 224, 3) → (N, 768), frozen."""
        zimg, _, _ = self.model.apply({'params': self.params}, images, None)
        return jax.lax.stop_gradient(zimg)

    @partial(jax.jit, static_argnums=(0,))
    def encode_supports(self, support_images):
        """
        (B, 5, 224, 224, 3) → mean-pool → (B, 768)
        """
        B = support_images.shape[0]
        flat = support_images.reshape(B * 5, *support_images.shape[2:])
        embs = self._encode_batch(flat).reshape(B, 5, -1)
        return jnp.mean(embs, axis=1)

    def encode_supports_pmap(self, support_images):
        """pmap version: (ndev, local_B, 5, 224, 224, 3) → (ndev, local_B, 768)"""
        @partial(jax.pmap, axis_name='data')
        def _fn(imgs):
            B = imgs.shape[0]
            flat = imgs.reshape(B * 5, *imgs.shape[2:])
            z, _, _ = self.model.apply({'params': self.params}, flat, None)
            z = jax.lax.stop_gradient(z).reshape(B, 5, -1)
            return jnp.mean(z, axis=1)
        return _fn(support_images)
