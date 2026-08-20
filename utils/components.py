from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import EinMix

from utils.config import NetworkConfig


class FFN(nn.Module):
    """Feed-forward network with SwiGLU activation.

    EinMix replaces the standard Linear layers, keeping the einops convention.
    Hidden dimension is dim * expansion_factor * 2 (split into two halves for SwiGLU).
    """
    def __init__(self, dim: int, expansion_factor: int) -> None:
        super().__init__()
        hidden = dim * expansion_factor
        # Project up: (b, t, d) -> (b, t, 2 * hidden) for the two SwiGLU halves
        self.w1 = EinMix("b t d -> b t h", weight_shape="d h", d=dim, h=hidden * 2)
        # Project down: (b, t, hidden) -> (b, t, d)
        self.w2 = EinMix("b t h -> b t d", weight_shape="h d", h=hidden, d=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, t, d) — batch, tokens, embedding dimension
        h = self.w1(x)                                  # (b, t, 2 * hidden)
        a, b = h.chunk(2, dim=-1)                       # each (b, t, hidden) — the two SwiGLU branches
        return self.w2(torch.nn.functional.silu(a) * b) # (b, t, d)


class MHSA(nn.Module):
    """Multi-head self-attention.

    EinMix computes the Q/K/V projection and the output projection.
    The attention pattern is the standard scaled dot-product.
    """
    def __init__(self, dim: int, num_heads: int, dim_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim_heads = dim_heads
        inner = num_heads * dim_heads
        # QKV projection: (b, t, d) -> (b, t, 3 * inner)
        self.to_qkv = EinMix("b t d -> b t qkv", weight_shape="d qkv", d=dim, qkv=inner * 3)
        # Output projection: (b, h, t, d_head) -> (b, t, dim) — merges heads back
        self.to_out = EinMix("b h t d -> b t hd", weight_shape="h d hd", h=num_heads, d=dim_heads, hd=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, t, d) — batch, tokens, embedding dimension
        b, t, _ = x.shape
        qkv = self.to_qkv(x)                           # (b, t, 3 * inner)
        q, k, v = qkv.chunk(3, dim=-1)                 # each (b, t, inner)
        # Reshape to multi-head: (b, t, inner) -> (b, num_heads, t, dim_heads)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)  # (b, h, t, d_head)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.num_heads)  # (b, h, t, d_head)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.num_heads)  # (b, h, t, d_head)
        # Scaled dot-product attention
        scale = self.dim_heads ** -0.5
        attn = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale  # (b, h, t, t) attention logits
        attn = torch.softmax(attn, dim=-1)                                  # (b, h, t, t) attention weights
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)        # (b, h, t, d_head)
        return self.to_out(out)                              # (b, t, d) — merged heads


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: attention + FFN, each with residual connections."""
    def __init__(self, dim: int, num_heads: int, dim_heads: int, expansion_factor: int) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = MHSA(dim, num_heads, dim_heads)
        self.norm2 = nn.RMSNorm(dim)
        self.ffn = FFN(dim, expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, t, d)
        x = x + self.attn(self.norm1(x))   # residual around attention
        x = x + self.ffn(self.norm2(x))    # residual around FFN
        return x                            # (b, t, d)


def sincos_embedding(coordinate: torch.Tensor, period: float, dim: int) -> torch.Tensor:
    """Sine and cosine features of a periodic coordinate at the harmonics of its period.

    Args:
        coordinate: (...) any shape — the coordinate value (e.g. hours since Jan 1)
        period: the period in the same units (e.g. 24.0 for diurnal, 365.25*24 for seasonal)
        dim: output dimension (must be even)
    Returns:
        (..., dim) — sin and cos at dim//2 frequencies, concatenated
    """
    freqs = 2 * math.pi / period * torch.arange(1, dim // 2 + 1, device=coordinate.device, dtype=coordinate.dtype)
    # (...,) @ (dim//2,) -> (..., dim//2) — angles for each frequency
    angles = torch.einsum("..., f -> ... f", coordinate, freqs)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (..., dim)


class ViT(nn.Module):
    """Vision Transformer for weather forecasting.

    Standard (separable_embed=False):
        to_tokens: (b, v, H, W) -> (b, h*w, d)       — patch embedding mixes all variables
        pos_emb:  (h*w, d)                            — learned positional embedding
        blocks:   (b, h*w, d) -> (b, h*w, d)          — transformer blocks at width d
        to_fields: (b, h*w, d) -> (b, v, H, W)        — unembedding back to fields

    Separable (separable_embed=True):
        to_tokens: (b, v, H, W) -> (b, h*w, v*d)     — per-variable embedding, no mixing
        pos_emb:  (h*w, v*d)                          — positional embedding at width v*d
        blocks:   (b, h*w, v*d) -> (b, h*w, v*d)     — transformer blocks at width v*d
        to_fields: (b, h*w, v*d) -> (b, v, H, W)     — per-variable unembedding

    Time embedding (metadata_embed=True):
        time -> sin/cos features (2 * dim_metadata) -> linear -> added to every token
    """
    def __init__(self, config: NetworkConfig, num_variables: int, field_size: tuple[int, int]) -> None:
        super().__init__()
        self.num_variables = num_variables
        self.config = config
        hh, ww = config.patch_size                     # patch size in lat, lon
        H, W = field_size                              # full field size
        self.h, self.w = H // hh, W // ww              # number of patches
        self.hh, self.ww = hh, ww

        if config.separable_embed:
            # Per-variable patch embedding: each variable gets its own (hh, ww, d) weight,
            # concatenated to (v*d) total width. No variable mixing at this layer.
            total_dim = num_variables * config.dim     # e.g. 9 * 16 = 144
            self.to_tokens = EinMix(
                "b v (h hh) (w ww) -> b (h w) (v d)", # v is kept separate from d
                weight_shape="v hh ww d",              # (v, hh, ww, d) — one weight per variable
                v=num_variables, hh=hh, ww=ww, d=config.dim,
            )
            self.pos_emb = nn.Parameter(torch.randn(self.h * self.w, total_dim) * 0.02)
            self.blocks = nn.ModuleList([
                TransformerBlock(total_dim, config.num_heads, config.dim_heads, config.expansion_factor)
                for _ in range(config.num_layers)
            ])
            self.norm = nn.RMSNorm(total_dim)
            # Per-variable unembedding: (v*d) -> separate (v, hh, ww) per variable
            self.to_fields = EinMix(
                "b (h w) (v d) -> b v (h hh) (w ww)",
                weight_shape="v hh ww d",              # (v, hh, ww, d) — one weight per variable
                v=num_variables, h=self.h, w=self.w, hh=hh, ww=ww, d=config.dim,
            )
        else:
            # Standard patch embedding: all variables mixed into a single d-dimensional token
            self.to_tokens = EinMix(
                "b v (h hh) (w ww) -> b (h w) d",     # v is contracted with the weight
                weight_shape="v hh ww d",              # (v, hh, ww, d) — shared across positions
                v=num_variables, hh=hh, ww=ww, d=config.dim,
            )
            self.pos_emb = nn.Parameter(torch.randn(self.h * self.w, config.dim) * 0.02)
            self.blocks = nn.ModuleList([
                TransformerBlock(config.dim, config.num_heads, config.dim_heads, config.expansion_factor)
                for _ in range(config.num_layers)
            ])
            self.norm = nn.RMSNorm(config.dim)
            self.to_fields = EinMix(
                "b (h w) d -> b v (h hh) (w ww)",     # v is broadcast from the weight
                weight_shape="v hh ww d",
                v=num_variables, h=self.h, w=self.w, hh=hh, ww=ww, d=config.dim,
            )

        self.metadata_embed = config.metadata_embed
        if self.metadata_embed:
            dim_meta = config.dim_metadata
            total_dim = num_variables * config.dim if config.separable_embed else config.dim
            # Project 2*dim_metadata time features to the token width
            self.time_proj = nn.Linear(2 * dim_meta, total_dim)

    def forward(self, x: torch.Tensor, time: torch.Tensor | None = None) -> torch.Tensor:
        # x: (b, v, H, W) — batch, variable, latitude, longitude
        # time: (b,) or None — hours since Jan 1 of the year, e.g. [0.0, 24.0, ...]
        b = x.shape[0]
        tokens = self.to_tokens(x)                     # (b, h*w, d) or (b, h*w, v*d)
        tokens = tokens + self.pos_emb.unsqueeze(0)    # broadcast-add positional embedding: (h*w, d) -> (1, h*w, d)

        if self.metadata_embed and time is not None:
            # Diurnal features: period = 24 hours, dim_metadata dims
            diurnal = sincos_embedding(time, 24.0, self.config.dim_metadata)            # (b, dim_metadata)
            # Seasonal features: period = 365.25 days in hours, dim_metadata dims
            seasonal = sincos_embedding(time, 365.25 * 24.0, self.config.dim_metadata)  # (b, dim_metadata)
            meta = torch.cat([diurnal, seasonal], dim=-1)                                # (b, 2 * dim_metadata)
            # Project to token width and broadcast-add to every token
            tokens = tokens + self.time_proj(meta).unsqueeze(1)  # (b, 1, d) added to (b, h*w, d)

        for block in self.blocks:
            tokens = block(tokens)                     # (b, h*w, d) or (b, h*w, v*d) — through each transformer block
        tokens = self.norm(tokens)                     # final RMSNorm
        return self.to_fields(tokens)                  # (b, v, H, W) — back to field space


class Persistence(nn.Module):
    """Persistence baseline: returns the input unchanged (forecast = last observed state)."""
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x
