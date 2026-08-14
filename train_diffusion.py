"""Train the IMU-conditioned velocity diffusion model."""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from imu_velocity_diffusion.checkpoint import load_checkpoint, save_checkpoint
from imu_velocity_diffusion.config import load_config
from imu_velocity_diffusion.data import require_average_velocity_targets
from imu_velocity_diffusion.diffusion import DiffusionSchedule
from imu_velocity_diffusion.factory import build_diffusion_model
from imu_velocity_diffusion.models import VelocityDiffusionModel
from imu_velocity_diffusion.training import (
    batch_to_device,
    get_device,
    make_loader,
    output_dir,
    resolve_batch_size,
    set_seed,
)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed(requested_device: str, local_rank_arg: int) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=get_device(requested_device),
        )

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed diffusion training requires CUDA/NCCL GPUs.")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", str(local_rank_arg)))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device(f"cuda:{local_rank}"),
    )


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def reduce_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def evaluate(
    model: VelocityDiffusionModel,
    loader,
    schedule: DiffusionSchedule,
    num_candidates: int,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    total_noise_loss = 0.0
    total_min_mse = 0.0
    total_count = 0
    platform_min_mse: dict[str, float] = defaultdict(float)
    platform_count: dict[str, int] = defaultdict(int)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = batch_to_device(batch, device)
            imu = batch["imu"]
            velocity = batch["velocity"]
            timesteps = schedule.sample_timesteps(velocity.shape[0])
            noisy_velocity, noise = schedule.add_noise(velocity, timesteps)
            pred_noise = model(imu, noisy_velocity, timesteps)
            noise_loss = F.mse_loss(pred_noise, noise, reduction="none").mean(dim=1)

            candidates = schedule.sample(model, imu, num_candidates=num_candidates)
            candidate_mse = ((candidates - velocity[:, None, :]) ** 2).mean(dim=-1)
            min_mse = candidate_mse.min(dim=1).values

            total_noise_loss += float(noise_loss.sum().item())
            total_min_mse += float(min_mse.sum().item())
            total_count += int(velocity.shape[0])
            platforms = batch.get("platform")
            if platforms is None:
                platforms = ["unknown"] * int(velocity.shape[0])
            for platform, sample_mse in zip(platforms, min_mse.cpu().tolist()):
                platform_min_mse[str(platform)] += float(sample_mse)
                platform_count[str(platform)] += 1
    platform_rmse = {
        platform: math.sqrt(platform_min_mse[platform] / count)
        for platform, count in platform_count.items()
    }
    return {
        "val_noise_loss": total_noise_loss / max(total_count, 1),
        "val_candidate_min_rmse": math.sqrt(total_min_mse / max(total_count, 1)),
        "num_eval_samples": float(total_count),
        "platform_candidate_min_rmse": platform_rmse,
        "platform_eval_samples": dict(platform_count),
    }


def print_platform_candidate_metrics(metrics: dict[str, Any]) -> None:
    rmses = metrics.get("platform_candidate_min_rmse", {})
    counts = metrics.get("platform_eval_samples", {})
    if not rmses:
        return
    print("val_candidate_min_rmse_by_platform:")
    for platform in sorted(rmses):
        print(
            f"  {platform}: rmse={rmses[platform]:.6f} "
            f"eval_samples={int(counts.get(platform, 0))}"
        )


def append_metrics(path, metrics: dict[str, float]) -> None:
    fieldnames = [
        "epoch",
        "train_noise_loss",
        "val_noise_loss",
        "val_candidate_min_rmse",
        "num_eval_samples",
        "best_val_candidate_min_rmse",
        "train_seconds",
        "eval_seconds",
        "epoch_seconds",
        "validated",
    ]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: metrics.get(name, "") for name in fieldnames})


def split_root(cfg: dict, split: str) -> Path:
    data_cfg = cfg["data"]
    split_dirs = data_cfg.get("split_dirs", {})
    return Path(data_cfg["root"]).expanduser() / split_dirs.get(split, split)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-rank", dest="local_rank", default=0, type=int)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if args.eval_only and not args.resume_from:
        parser.error("--eval-only requires --resume-from")

    cfg = load_config(args.config)
    require_average_velocity_targets(cfg)
    set_seed(int(cfg.get("seed", 42)))
    dist_ctx = setup_distributed(args.device, args.local_rank)
    device = dist_ctx.device
    resolve_batch_size(cfg, "diffusion", device)

    train_loader = make_loader(
        cfg, "train", shuffle=True, distributed=dist_ctx.is_distributed
    )
    val_loader = make_loader(cfg, "val", shuffle=False) if dist_ctx.is_main else None
    input_channels = int(train_loader.dataset.input_channels)

    model = build_diffusion_model(cfg, input_channels, device)
    schedule = DiffusionSchedule(**cfg["diffusion"], device=device)

    out_dir = output_dir(cfg)
    metrics_path = out_dir / "diffusion_metrics.csv"
    best_metric = float("inf")
    start_epoch = 0
    if args.resume_from:
        checkpoint = load_checkpoint(args.resume_from, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        checkpoint_metrics = checkpoint.get("metrics", {})
        for metric_name in ("best_val_candidate_min_rmse", "val_candidate_min_rmse"):
            if metric_name in checkpoint_metrics:
                best_metric = float(checkpoint_metrics[metric_name])
                break
        if dist_ctx.is_main:
            print(f"resumed_from={args.resume_from} start_epoch={start_epoch}")

    if dist_ctx.is_distributed:
        model = DistributedDataParallel(model, device_ids=[dist_ctx.local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    if args.resume_from and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epochs = int(cfg["train"]["epochs"])
    val_candidates = int(cfg["policy"].get("num_candidates", 16))
    val_every_n_epochs = int(cfg["train"].get("val_every_n_epochs", 1))
    val_max_batches_cfg = cfg["train"].get("val_max_batches")
    val_max_batches = (
        None if val_max_batches_cfg in (None, 0) else int(val_max_batches_cfg)
    )

    if dist_ctx.is_main:
        print(
            f"device={device} world_size={dist_ctx.world_size} "
            f"train_samples={len(train_loader.dataset)} "
            f"val_samples={len(val_loader.dataset)} "
            f"batch_size_per_gpu={cfg['train']['batch_size']} "
            f"global_batch_size={cfg['train']['batch_size'] * dist_ctx.world_size} "
            f"epochs={epochs} diffusion_steps={cfg['diffusion']['steps']}"
        )
        print(
            f"train_data_dir={split_root(cfg, 'train')} "
            f"val_data_dir={split_root(cfg, 'val')}"
        )
        print(
            f"validation_every={val_every_n_epochs} epoch(s) "
            f"val_candidates={val_candidates} "
            f"val_max_batches={val_max_batches if val_max_batches is not None else 'all'}"
        )
        print(f"outputs={out_dir} metrics_csv={metrics_path}")

    if args.eval_only:
        if dist_ctx.is_main:
            eval_started_at = time.perf_counter()
            metrics = evaluate(
                unwrap_model(model),
                val_loader,
                schedule,
                val_candidates,
                device,
                max_batches=val_max_batches,
            )
            eval_seconds = time.perf_counter() - eval_started_at
            print(
                f"eval_only checkpoint_epoch={start_epoch} "
                f"val_noise_loss={metrics['val_noise_loss']:.6f} "
                f"val_candidate_min_rmse={metrics['val_candidate_min_rmse']:.6f} "
                f"eval_samples={int(metrics['num_eval_samples'])} "
                f"seconds={eval_seconds:.1f}"
            )
            print_platform_candidate_metrics(metrics)
        cleanup_distributed(dist_ctx)
        return

    for epoch in range(start_epoch + 1, epochs + 1):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        epoch_started_at = time.perf_counter()
        model.train()
        running_loss = 0.0
        running_count = 0
        progress = tqdm(
            train_loader,
            desc=f"diffusion epoch {epoch}/{epochs}",
            disable=not dist_ctx.is_main,
        )
        for batch in progress:
            batch = batch_to_device(batch, device)
            velocity = batch["velocity"]
            timesteps = schedule.sample_timesteps(velocity.shape[0])
            noisy_velocity, noise = schedule.add_noise(velocity, timesteps)
            pred_noise = model(batch["imu"], noisy_velocity, timesteps)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg["train"].get("grad_clip", 1.0))
            )
            optimizer.step()

            running_loss += float(loss.item()) * int(velocity.shape[0])
            running_count += int(velocity.shape[0])
            progress.set_postfix(loss=running_loss / max(running_count, 1))

        train_seconds = time.perf_counter() - epoch_started_at
        should_validate = (
            epoch == 1 or epoch == epochs or epoch % val_every_n_epochs == 0
        )
        global_running_loss = reduce_sum(running_loss, device)
        global_running_count = reduce_sum(float(running_count), device)
        metrics: dict[str, float] = {}
        eval_seconds = 0.0
        if should_validate and dist_ctx.is_main:
            eval_started_at = time.perf_counter()
            metrics = evaluate(
                unwrap_model(model),
                val_loader,
                schedule,
                val_candidates,
                device,
                max_batches=val_max_batches,
            )
            eval_seconds = time.perf_counter() - eval_started_at
        metrics["train_noise_loss"] = global_running_loss / max(
            global_running_count, 1.0
        )
        metrics["epoch"] = float(epoch)
        metrics["train_seconds"] = train_seconds
        metrics["eval_seconds"] = eval_seconds
        metrics["epoch_seconds"] = time.perf_counter() - epoch_started_at
        metrics["validated"] = float(should_validate)

        if dist_ctx.is_main:
            if should_validate:
                print(
                    f"epoch={epoch} train_noise_loss={metrics['train_noise_loss']:.6f} "
                    f"val_noise_loss={metrics['val_noise_loss']:.6f} "
                    f"val_candidate_min_rmse={metrics['val_candidate_min_rmse']:.6f} "
                    f"eval_samples={int(metrics['num_eval_samples'])} "
                    f"train_seconds={metrics['train_seconds']:.1f} "
                    f"eval_seconds={metrics['eval_seconds']:.1f} "
                    f"seconds={metrics['epoch_seconds']:.1f}"
                )
                print_platform_candidate_metrics(metrics)
            else:
                print(
                    f"epoch={epoch} train_noise_loss={metrics['train_noise_loss']:.6f} "
                    f"validation=skipped "
                    f"train_seconds={metrics['train_seconds']:.1f} "
                    f"seconds={metrics['epoch_seconds']:.1f}"
                )
            is_best = (
                should_validate and metrics["val_candidate_min_rmse"] < best_metric
            )
            if is_best:
                best_metric = metrics["val_candidate_min_rmse"]
            metrics["best_val_candidate_min_rmse"] = best_metric
            append_metrics(metrics_path, metrics)

            save_checkpoint(
                out_dir / "diffusion_last.pt",
                model=unwrap_model(model),
                optimizer=optimizer,
                epoch=epoch,
                cfg=cfg,
                metrics=metrics,
            )
            if is_best:
                save_checkpoint(
                    out_dir / "diffusion_best.pt",
                    model=unwrap_model(model),
                    optimizer=optimizer,
                    epoch=epoch,
                    cfg=cfg,
                    metrics=metrics,
                )
        if dist_ctx.is_distributed:
            dist.barrier()

    cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()
