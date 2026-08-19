from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import EinMix

from utils.config import NetworkConfig


class FFN(nn.Module):
    def __init__(self, dim: int, expansion_factor: int) -> None:
        super().__init__()
        hidden = dim * expansion_factor
        self.w1 = EinMix("b t d -> b t h", weight_shape="d h", d=dim, h=hidden * 2)
        self.w2 = EinMix("b t h -> b t d", weight_shape="h d", h=hidden, d=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w1(x)
        a, b = h.chunk(2, dim=-1)
        return self.w2(torch.nn.functional.silu(a) * b)


class MHSA(nn.Module):
    def __init__(self, dim: int, num_heads: int, dim_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim_heads = dim_heads
        inner = num_heads * dim_heads
        self.to_qkv = EinMix("b t d -> b t qkv", weight_shape="d qkv", d=dim, qkv=inner * 3)
        self.to_out = EinMix("b h t d -> b t hd", weight_shape="h d hd", h=num_heads, d=dim_heads, hd=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.num_heads)
        scale = self.dim_heads ** -0.5
        attn = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dim_heads: int, expansion_factor: int) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = MHSA(dim, num_heads, dim_heads)
        self.norm2 = nn.RMSNorm(dim)
        self.ffn = FFN(dim, expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, config: NetworkConfig, num_variables: int, field_size: tuple[int, int]) -> None:
        super().__init__()
        self.num_variables = num_variables
        hh, ww = config.patch_size
        H, W = field_size
        self.h, self.w = H // hh, W // ww
        self.hh, self.ww = hh, ww

        self.to_tokens = EinMix(
            "b v (h hh) (w ww) -> b (h w) d",
            weight_shape="v hh ww d",
            v=num_variables, hh=hh, ww=ww, d=config.dim,
        )
        self.pos_emb = nn.Parameter(torch.randn(self.h * self.w, config.dim) * 0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.dim, config.num_heads, config.dim_heads, config.expansion_factor)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.RMSNorm(config.dim)
        self.to_fields = EinMix(
            "b (h w) d -> b v (h hh) (w ww)",
            weight_shape="v hh ww d",
            v=num_variables, h=self.h, w=self.w, hh=hh, ww=ww, d=config.dim,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        tokens = self.to_tokens(x)
        tokens = tokens + self.pos_emb.unsqueeze(0)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return self.to_fields(tokens)


class Persistence(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
