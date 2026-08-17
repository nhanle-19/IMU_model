"""Shared training utilities."""

from __future__ import annotations

import datetime
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
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


def setup_distributed(requested: str = "auto") -> tuple[torch.device, int, int, int]:
    """Initialize torchrun distributed state when WORLD_SIZE requests it."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return get_device(requested), rank, world_size, local_rank

    if requested == "cpu":
        backend = "gloo"
        device = torch.device("cpu")
    elif requested != "auto":
        raise ValueError(
            "Distributed training assigns devices from LOCAL_RANK; use --device auto "
            "for GPU torchrun launches, or --device cpu for CPU smoke tests."
        )
    else:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            visible_gpus = torch.cuda.device_count()
            if local_rank >= visible_gpus:
                raise ValueError(
                    f"LOCAL_RANK={local_rank} but only {visible_gpus} CUDA "
                    "device(s) are visible. Lower --nproc_per_node or fix "
                    "CUDA_VISIBLE_DEVICES."
                )
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    timeout_seconds = int(os.environ.get("TORCH_DISTRIBUTED_TIMEOUT_SECONDS", "21600"))
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )
    return device, rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def distributed_barrier() -> None:
    if is_distributed():
        dist.barrier()


def resolve_batch_size(cfg: dict, stage: str, device: torch.device | str) -> int:
    """Resolve train.batch_size from stage-specific per-device settings.

    Stage-specific values are per process/device. In torchrun DDP each rank
    receives this batch size; otherwise it is the selected device's batch size.
    """
    train_cfg = cfg["train"]
    stage_key = f"{stage}_batch_size_per_gpu"
    per_device = train_cfg.get(stage_key, train_cfg.get("batch_size_per_gpu"))
    if per_device is None:
        return int(train_cfg["batch_size"])

    resolved = int(per_device)
    train_cfg["batch_size"] = resolved
    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if is_main_process():
        if is_distributed():
            print(f"{stage}_batch_size={resolved} per rank")
        elif visible_gpus > 1 and torch.device(device).type == "cuda":
            print(
                f"{stage}_batch_size={resolved} per selected GPU "
                f"(visible_gpus={visible_gpus}; this trainer uses one device)"
            )
        else:
            print(f"{stage}_batch_size={resolved}")
    return resolved


def make_loader(
    cfg: dict,
    split: str,
    shuffle: bool,
    *,
    distributed: bool = False,
) -> DataLoader:
    dataset = build_dataset(cfg, split)
    train_cfg = cfg["train"]
    sampler = None
    if distributed and split == "train":
        sampler = DistributedSampler(dataset, shuffle=shuffle)
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
