# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Feature-gated model backbone registry.

Builders are registered by `model_name` key and imported lazily (each builder
imports its heavy deps inside the function body), so a missing optional
dependency only fails when that specific key is selected — not at import time.

`build_backbone(key, cfg)` returns the constructed nn.Module WITHOUT any device
placement or multi-GPU wrapping; config/configer.build_model keeps that tail.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from torch import nn

_BUILDERS: dict[str, Callable] = {}


def register(key: str) -> Callable:
    def _decorator(fn: Callable) -> Callable:
        _BUILDERS[key] = fn
        return fn
    return _decorator


def available() -> list[str]:
    return sorted(_BUILDERS)


def build_backbone(key: str, cfg: dict) -> nn.Module:
    if key not in _BUILDERS:
        raise ValueError(
            f"Unknown model_name '{key}'. Available: {available()}"
        )
    return _BUILDERS[key](cfg)


# Import builder modules for their registration side effects. Each module's
# builder imports its own heavy deps lazily, so importing the module here is
# cheap and safe.
#
# LSTM is the core backbone and is always shipped. The Transformer backbone is
# optional: an LSTM-only release can drop tartan_imu/model/backbones/transformer.py
# (and its transformer_odom / model_transformer deps) and this registry still
# imports fine — selecting model_name="Transformer" then fails with a clean
# "Unknown model_name" ValueError from build_backbone rather than an ImportError.
from tartan_imu.model.backbones import lstm as _lstm  # noqa: E402,F401

try:
    from tartan_imu.model.backbones import transformer as _transformer  # noqa: E402,F401
except ImportError:
    # Transformer backbone not shipped in this build; leave it unregistered.
    pass
