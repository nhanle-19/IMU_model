"""Train the IMU + diffusion-candidate velocity distribution refiner."""

from __future__ import annotations

import argparse
import math
import os
import time
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from imu_velocity_diffusion.checkpoint import load_checkpoint, save_checkpoint
from imu_velocity_diffusion.config import load_config
from imu_velocity_diffusion.data import (
    assert_compatible_velocity_contract,
    model_window_size,
    require_average_velocity_targets,
)
from imu_velocity_diffusion.diffusion import DiffusionSchedule
from imu_velocity_diffusion.factory import build_diffusion_model, build_refiner_model
from imu_velocity_diffusion.models import VelocityDiffusionModel
from imu_velocity_diffusion.training import (
    batch_to_device,
    cleanup_distributed,
    distributed_barrier,
    is_distributed,
    is_main_process,
    make_loader,
    output_dir,
    reduce_sum,
    resolve_batch_size,
    set_seed,
    setup_distributed,
    unwrap_model,
)


def load_diffusion(
    path: str,
    cfg: dict,
    input_channels: int,
    device: torch.device,
) -> tuple[VelocityDiffusionModel, DiffusionSchedule]:
    ckpt = load_checkpoint(path, device)
    diffusion_cfg = ckpt.get("cfg", cfg)
    require_average_velocity_targets(diffusion_cfg)
    assert_compatible_velocity_contract(cfg, diffusion_cfg)
    if model_window_size(cfg["data"]) != model_window_size(diffusion_cfg["data"]):
        raise ValueError(
            "train_policy.py feeds IMU windows directly into the diffusion model, "
            "so data.sample_freq must match the diffusion checkpoint. Exported "
            "diffusion velocity candidates can still be consumed by a downstream "
            "40 Hz actual model."
        )
    model = build_diffusion_model(diffusion_cfg, input_channels, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    schedule = DiffusionSchedule(**diffusion_cfg["diffusion"], device=device)
    return model, schedule


@torch.no_grad()
def make_candidates(
    diffusion: VelocityDiffusionModel,
    schedule: DiffusionSchedule,
    imu: torch.Tensor,
    target_velocity: torch.Tensor,
    cfg: dict,
) -> torch.Tensor:
    num_candidates = int(cfg["policy"].get("num_candidates", 16))
    candidates = schedule.sample(
        diffusion,
        imu,
        num_candidates=num_candidates,
        max_sample_batch_size=int(
            cfg["policy"].get("candidate_sample_batch_size", 8192)
        ),
    )

    if cfg["policy"].get("bootstrap_with_target_candidate", False):
        std = float(cfg["policy"].get("target_candidate_std", 0.05))
        replacement = target_velocity + std * torch.randn_like(target_velocity)
        candidates[:, 0, :] = replacement
    return candidates


def refiner_loss(
    outputs: dict[str, torch.Tensor],
    candidates: torch.Tensor,
    target: torch.Tensor,
    cfg: dict,
):
    velocity_loss = F.mse_loss(outputs["velocity"], target)
    residual = outputs["corrected_candidates"] - candidates
    residual_penalty = residual.pow(2).mean()
    global_residual = outputs["velocity"] - outputs["weighted_velocity"]
    global_residual_penalty = global_residual.pow(2).mean()
    weights = outputs["weights"].clamp_min(1e-8)
    attention_entropy = -(weights * weights.log()).sum(dim=1).mean()
    total = (
        velocity_loss
        + float(cfg["policy"].get("residual_penalty_weight", 0.01)) * residual_penalty
        + float(cfg["policy"].get("global_residual_penalty_weight", 0.0))
        * global_residual_penalty
        - float(cfg["policy"].get("attention_entropy_weight", 0.0)) * attention_entropy
    )
    return total, {
        "velocity_loss": float(velocity_loss.detach().item()),
        "residual_penalty": float(residual_penalty.detach().item()),
        "global_residual_penalty": float(global_residual_penalty.detach().item()),
        "attention_entropy": float(attention_entropy.detach().item()),
    }


def evaluate(
    refiner,
    diffusion,
    schedule,
    loader,
    cfg,
    device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    refiner.eval()
    total_mse = 0.0
    total_oracle_mse = 0.0
    total_distribution_mean_mse = 0.0
    total_weighted_velocity_mse = 0.0
    total_count = 0
    platform_mse: dict[str, float] = defaultdict(float)
    platform_oracle_mse: dict[str, float] = defaultdict(float)
    platform_distribution_mean_mse: dict[str, float] = defaultdict(float)
    platform_weighted_velocity_mse: dict[str, float] = defaultdict(float)
    platform_count: dict[str, int] = defaultdict(int)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = batch_to_device(batch, device)
            candidates = make_candidates(
                diffusion, schedule, batch["imu"], batch["velocity"], cfg
            )
            outputs = refiner(batch["imu"], candidates)
            mse = ((outputs["velocity"] - batch["velocity"]) ** 2).mean(dim=-1)
            distribution_mean_mse = (
                (outputs["distribution_mean"] - batch["velocity"]) ** 2
            ).mean(dim=-1)
            weighted_velocity_mse = (
                (outputs["weighted_velocity"] - batch["velocity"]) ** 2
            ).mean(dim=-1)
            oracle_mse = (
                ((candidates - batch["velocity"][:, None, :]) ** 2)
                .mean(dim=-1)
                .min(dim=1)
                .values
            )
            total_mse += float(mse.sum().item())
            total_oracle_mse += float(oracle_mse.sum().item())
            total_distribution_mean_mse += float(distribution_mean_mse.sum().item())
            total_weighted_velocity_mse += float(weighted_velocity_mse.sum().item())
            total_count += int(mse.shape[0])
            platforms = batch.get("platform")
            if platforms is None:
                platforms = ["unknown"] * int(mse.shape[0])
            for (
                platform,
                sample_mse,
                sample_oracle_mse,
                sample_distribution_mean_mse,
                sample_weighted_velocity_mse,
            ) in zip(
                platforms,
                mse.cpu().tolist(),
                oracle_mse.cpu().tolist(),
                distribution_mean_mse.cpu().tolist(),
                weighted_velocity_mse.cpu().tolist(),
            ):
                platform = str(platform)
                platform_mse[platform] += float(sample_mse)
                platform_oracle_mse[platform] += float(sample_oracle_mse)
                platform_distribution_mean_mse[platform] += float(
                    sample_distribution_mean_mse
                )
                platform_weighted_velocity_mse[platform] += float(
                    sample_weighted_velocity_mse
                )
                platform_count[platform] += 1
    return {
        "rmse": math.sqrt(total_mse / max(total_count, 1)),
        "candidate_oracle_rmse": math.sqrt(total_oracle_mse / max(total_count, 1)),
        "distribution_mean_rmse": math.sqrt(
            total_distribution_mean_mse / max(total_count, 1)
        ),
        "weighted_velocity_rmse": math.sqrt(
            total_weighted_velocity_mse / max(total_count, 1)
        ),
        "platform_rmse": {
            platform: math.sqrt(total / platform_count[platform])
            for platform, total in platform_mse.items()
        },
        "platform_candidate_oracle_rmse": {
            platform: math.sqrt(total / platform_count[platform])
            for platform, total in platform_oracle_mse.items()
        },
        "platform_distribution_mean_rmse": {
            platform: math.sqrt(total / platform_count[platform])
            for platform, total in platform_distribution_mean_mse.items()
        },
        "platform_weighted_velocity_rmse": {
            platform: math.sqrt(total / platform_count[platform])
            for platform, total in platform_weighted_velocity_mse.items()
        },
        "platform_eval_samples": dict(platform_count),
    }


def print_platform_refiner_metrics(metrics: dict[str, Any]) -> None:
    rmses = metrics.get("platform_rmse", {})
    oracle_rmses = metrics.get("platform_candidate_oracle_rmse", {})
    mean_rmses = metrics.get("platform_distribution_mean_rmse", {})
    weighted_rmses = metrics.get("platform_weighted_velocity_rmse", {})
    counts = metrics.get("platform_eval_samples", {})
    if not rmses:
        return
    print(
        f"val_rmse_by_platform average={metrics['rmse']:.6f} "
        f"distribution_mean_average={metrics['distribution_mean_rmse']:.6f} "
        f"weighted_velocity_average={metrics['weighted_velocity_rmse']:.6f} "
        f"candidate_oracle_average={metrics['candidate_oracle_rmse']:.6f}:"
    )
    for platform in sorted(rmses):
        print(
            f"  {platform}: rmse={rmses[platform]:.6f} "
            f"distribution_mean_rmse={mean_rmses.get(platform, float('nan')):.6f} "
            f"weighted_velocity_rmse={weighted_rmses.get(platform, float('nan')):.6f} "
            f"candidate_oracle_rmse={oracle_rmses.get(platform, float('nan')):.6f} "
            f"eval_samples={int(counts.get(platform, 0))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--diffusion-checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume-from", default="")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override train.epochs; useful when resuming past the config value.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    require_average_velocity_targets(cfg)
    device, rank, world_size, local_rank = setup_distributed(args.device)
    distributed = is_distributed()
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(
        f"rank={rank} local_rank={local_rank} device={device} "
        f"cuda_current_device="
        f"{torch.cuda.current_device() if device.type == 'cuda' else 'cpu'} "
        f"CUDA_VISIBLE_DEVICES={visible_devices}",
        flush=True,
    )
    set_seed(int(cfg.get("seed", 42)) + rank)
    resolve_batch_size(cfg, "refiner", device)

    train_loader = make_loader(
        cfg, "train", shuffle=True, distributed=distributed
    )
    val_loader = make_loader(cfg, "val", shuffle=False)
    input_channels = int(train_loader.dataset.input_channels)

    diffusion, schedule = load_diffusion(
        args.diffusion_checkpoint, cfg, input_channels, device
    )
    refiner = build_refiner_model(cfg, input_channels, device)
    if distributed:
        refiner = torch.nn.parallel.DistributedDataParallel(
            refiner,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )
    optimizer = torch.optim.AdamW(
        refiner.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )

    out_dir = output_dir(cfg)
    best_rmse = float("inf")
    start_epoch = 0
    if args.resume_from:
        checkpoint = load_checkpoint(args.resume_from, device)
        unwrap_model(refiner).load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        checkpoint_metrics = checkpoint.get("metrics", {})
        if "rmse" in checkpoint_metrics:
            best_rmse = float(checkpoint_metrics["rmse"])
        if is_main_process():
            print(f"resumed_from={args.resume_from} start_epoch={start_epoch}")
    epochs = int(cfg["train"]["epochs"])
    val_every_n_epochs = max(1, int(cfg["train"].get("val_every_n_epochs", 1)))
    val_max_batches_cfg = cfg["train"].get("val_max_batches")
    val_max_batches = (
        None if val_max_batches_cfg in (None, 0) else int(val_max_batches_cfg)
    )

    if is_main_process():
        print(
            f"device={device} world_size={world_size} "
            f"train_samples={len(train_loader.dataset)} "
            f"val_samples={len(val_loader.dataset)} "
            f"batch_size_per_rank={cfg['train']['batch_size']} "
            f"epochs={epochs} diffusion_steps={schedule.steps} "
            f"num_candidates={int(cfg['policy'].get('num_candidates', 16))}"
        )
        print(
            f"validation_every={val_every_n_epochs} epoch(s) "
            f"val_max_batches={val_max_batches if val_max_batches is not None else 'all'}"
        )

    if start_epoch >= epochs:
        if is_main_process():
            print(
                f"resume checkpoint epoch {start_epoch} is already >= "
                f"target epochs {epochs}; increase --epochs to continue."
            )
        cleanup_distributed()
        return

    for epoch in range(start_epoch + 1, epochs + 1):
        if distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        epoch_started_at = time.perf_counter()
        refiner.train()
        total_loss = 0.0
        total_count = 0
        progress = tqdm(
            train_loader,
            desc=f"refiner epoch {epoch}/{epochs}",
            disable=not is_main_process(),
        )
        for batch in progress:
            batch = batch_to_device(batch, device)
            candidates = make_candidates(
                diffusion, schedule, batch["imu"], batch["velocity"], cfg
            )
            outputs = refiner(batch["imu"], candidates)
            loss, loss_parts = refiner_loss(outputs, candidates, batch["velocity"], cfg)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                refiner.parameters(), float(cfg["train"].get("grad_clip", 1.0))
            )
            optimizer.step()

            batch_size = int(batch["velocity"].shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            progress.set_postfix(loss=total_loss / max(total_count, 1), **loss_parts)

        train_totals = torch.tensor(
            [total_loss, float(total_count)], device=device, dtype=torch.float64
        )
        reduce_sum(train_totals)
        should_validate = (
            epoch == 1 or epoch == epochs or epoch % val_every_n_epochs == 0
        )
        metrics: dict[str, float] = {}
        if should_validate and is_main_process():
            metrics = evaluate(
                unwrap_model(refiner),
                diffusion,
                schedule,
                val_loader,
                cfg,
                device,
                max_batches=val_max_batches,
            )
        if is_main_process():
            metrics["train_loss"] = float(
                train_totals[0].item() / max(train_totals[1].item(), 1.0)
            )
            metrics["epoch_seconds"] = time.perf_counter() - epoch_started_at
            metrics["validated"] = float(should_validate)
            if should_validate:
                print(
                    f"epoch={epoch} train_loss={metrics['train_loss']:.6f} "
                    f"val_rmse={metrics['rmse']:.6f} "
                    f"distribution_mean_rmse={metrics['distribution_mean_rmse']:.6f} "
                    f"weighted_velocity_rmse={metrics['weighted_velocity_rmse']:.6f} "
                    f"candidate_oracle_rmse={metrics['candidate_oracle_rmse']:.6f} "
                    f"seconds={metrics['epoch_seconds']:.1f}"
                )
                print_platform_refiner_metrics(metrics)
            else:
                print(
                    f"epoch={epoch} train_loss={metrics['train_loss']:.6f} "
                    f"validation=skipped seconds={metrics['epoch_seconds']:.1f}"
                )

            save_checkpoint(
                out_dir / "refiner_last.pt",
                model=refiner,
                optimizer=optimizer,
                epoch=epoch,
                cfg=cfg,
                metrics=metrics,
            )
            if should_validate and metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                save_checkpoint(
                    out_dir / "refiner_best.pt",
                    model=refiner,
                    optimizer=optimizer,
                    epoch=epoch,
                    cfg=cfg,
                    metrics=metrics,
                )
        distributed_barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
