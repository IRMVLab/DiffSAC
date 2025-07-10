import math

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


"""
Adapted from: https://github.com/openai/openai/blob/55363aa496049423c37124b440e9e30366db3ed6/orc/orc/diffusion/vit.py
"""

from checkpoint import checkpoint


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.attention = QKVMultiheadAttention(device=device, dtype=dtype, heads=heads, n_ctx=n_ctx)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x,), (), True)
        x = self.c_proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, width: int, init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, device=device, dtype=dtype)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, heads: int, n_ctx: int):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.heads = heads
        self.n_ctx = n_ctx

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


class PointDiffusionTransformer(nn.Module):
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        input_channels: int = 1,
        output_channels: int = 1,
        input_channels_points: int = 2,
        output_channels_points: int = 2,
        n_ctx: int = 200,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        time_token_cond: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx
        self.time_token_cond = time_token_cond
        self.time_embed = MLP(
            device=device, dtype=dtype, width=width, init_scale=init_scale * math.sqrt(1.0 / width)
        )
        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.backbone = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx + int(time_token_cond),
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.input_proj_points = nn.Linear(input_channels_points, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels, device=device, dtype=dtype)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def forward(self, points: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
        """
        :param x: an [N x C x T] tensor.    # 噪声
        :param t: an [N] tensor.            # 时间
        :return: an [N x C' x T] tensor.    # 生成的点云
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        return self._forward_with_cond(points, x, [(t_embed, self.time_token_cond)])

    def _forward_with_cond(
        self, points: torch.Tensor, x: torch.Tensor, cond_as_token: List[Tuple[torch.Tensor, bool]]
    ) -> torch.Tensor:
        
        h = self.input_proj(x.permute(0, 2, 1))
        h_points = self.input_proj_points(points.permute(0, 2, 1))

        h = h + h_points
        
        for emb, as_token in cond_as_token:
            if not as_token:
                h = h + emb[:, None]
        
        extra_tokens = [
            (emb[:, None] if len(emb.shape) == 2 else emb)
            for emb, as_token in cond_as_token
            if as_token
        ]
        if len(extra_tokens):
            h = torch.cat(extra_tokens + [h], dim=1)

        h = self.ln_pre(h)
        h = self.backbone(h)
        h = self.ln_post(h)

        if len(extra_tokens):
            h = h[:, sum(h.shape[1] for h in extra_tokens) :]
        h = self.output_proj(h)
        h = h.permute(0, 2, 1)
        return h
