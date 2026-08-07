"""Shared training utilities."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

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


def make_loader(cfg: dict, split: str, shuffle: bool) -> DataLoader:
    dataset = build_dataset(cfg, split)
    train_cfg = cfg["train"]
    return DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(train_cfg.get("drop_last", False)) and split == "train",
    )


def output_dir(cfg: dict) -> Path:
    path = Path(cfg["train"]["out_dir"]).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
