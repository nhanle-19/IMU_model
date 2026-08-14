"""Shared training utilities."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from imu_velocity_diffusion.data import build_dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_batch_size(cfg: dict, stage: str, device: torch.device | str) -> int:
    """Resolve train.batch_size from stage-specific per-device settings.

    The diffusion stack is single-process/single-device. If multiple GPUs are
    visible, select one with --device; this resolver intentionally does not
    multiply by GPU count.
    """
    train_cfg = cfg["train"]
    stage_key = f"{stage}_batch_size_per_gpu"
    per_device = train_cfg.get(stage_key, train_cfg.get("batch_size_per_gpu"))
    if per_device is None:
        return int(train_cfg["batch_size"])

    resolved = int(per_device)
    train_cfg["batch_size"] = resolved
    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible_gpus > 1 and torch.device(device).type == "cuda":
        print(
            f"{stage}_batch_size={resolved} per selected GPU "
            f"(visible_gpus={visible_gpus}; this trainer uses one device)"
        )
    else:
        print(f"{stage}_batch_size={resolved}")
    return resolved


def make_loader(
    cfg: dict, split: str, shuffle: bool, *, distributed: bool = False
) -> DataLoader:
    dataset = build_dataset(cfg, split)
    train_cfg = cfg["train"]
    sampler = (
        DistributedSampler(dataset, shuffle=shuffle)
        if distributed and split == "train"
        else None
    )
    return DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=shuffle if sampler is None else False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(train_cfg.get("drop_last", False)) and split == "train",
        sampler=sampler,
    )


def output_dir(cfg: dict) -> Path:
    path = Path(cfg["train"]["out_dir"]).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def batch_to_device(
    batch: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }
