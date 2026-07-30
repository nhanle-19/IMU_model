# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Reusable building blocks for the LSTM/Foundation model family.

Placeholder transformer/attention blocks, the einops rearrange wrapper, the
trunk-wiring helper ``IMU_instantiate_trunk``, and the ``ResBlock`` / ``FcBlock``
primitives shared by the ResNet+LSTM trunks and output heads.
"""
from functools import partial

import einops
import torch
import torch.nn as nn


# Placeholder classes for missing transformer_base module
class MultiheadAttention(nn.Module):
    """Placeholder for MultiheadAttention."""

    def __init__(self, embed_dim, num_heads, bias=True, add_bias_kv=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Placeholder implementation
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        return self.linear(x)


class SimpleTransformer(nn.Module):
    """Placeholder for SimpleTransformer."""

    def __init__(
        self,
        embed_dim,
        num_blocks,
        ffn_dropout_rate=0.0,
        drop_path_rate=0.0,
        attn_target=None,
        pre_transformer_layer=None,
        post_transformer_layer=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # Placeholder implementation
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        return self.linear(x)


class EinOpsRearrange(nn.Module):
    """nn.Module wrapper around ``einops.rearrange`` for use in Sequential."""

    def __init__(self, rearrange_expr: str, **kwargs) -> None:
        super().__init__()
        self.rearrange_expr = rearrange_expr
        self.kwargs = kwargs

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        return einops.rearrange(x, self.rearrange_expr, **self.kwargs)


def IMU_instantiate_trunk(
    embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path
):
    """Build a SimpleTransformer trunk wired with multi-head attention blocks."""
    return SimpleTransformer(
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        ffn_dropout_rate=0.0,
        drop_path_rate=drop_path,
        attn_target=partial(
            MultiheadAttention,
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=True,
            add_bias_kv=add_bias_kv,
        ),
        pre_transformer_layer=nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-6) if pre_transformer_ln else nn.Identity(),
            EinOpsRearrange("b l d -> l b d"),
        ),
        post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
    )


class ResBlock(nn.Module):
    """Basic 1D residual block (two 3-kernel convs) with optional downsample."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                planes,
                planes * self.expansion,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(planes * self.expansion),
        )
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.convs(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.clone() + identity
        out = self.relu(out)

        return out


class FcBlock(nn.Module):
    """Three-layer fully-connected head (with ReLU + dropout) mapping features to output_dim."""

    def __init__(
        self, in_dim, out_dim, mid_dim=256, dropout=0.5, cfg=None
    ):
        super(FcBlock, self).__init__()

        self.mid_dim = mid_dim  # 256
        self.in_dim = in_dim
        self.out_dim = out_dim

        # fc layers
        self.fcs = nn.Sequential(
            nn.Linear(self.in_dim, self.mid_dim),
            # nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(self.mid_dim, self.mid_dim),
            # nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(self.mid_dim, self.out_dim),
        )

    def forward(self, x):
        # Handle case where x might be a tuple (from LSTM output)
        if isinstance(x, tuple):
            x = x[0]  # Take the first element (output tensor)
        x = x.view(x.size(0), -1)
        x = self.fcs(x)
        return x
