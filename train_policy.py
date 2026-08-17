"""Train the IMU + diffusion-candidate velocity distribution refiner."""

from __future__ import annotations

import argparse
import math
import os
import time

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
    weights = outputs["weights"].clamp_min(1e-8)
    attention_entropy = -(weights * weights.log()).sum(dim=1).mean()
    total = (
        velocity_loss
        + float(cfg["policy"].get("residual_penalty_weight", 0.01)) * residual_penalty
        - float(cfg["policy"].get("attention_entropy_weight", 0.0)) * attention_entropy
    )
    return total, {
        "velocity_loss": float(velocity_loss.detach().item()),
        "residual_penalty": float(residual_penalty.detach().item()),
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
) -> dict[str, float]:
    refiner.eval()
    total_mse = 0.0
    total_oracle_mse = 0.0
    total_count = 0
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
            oracle_mse = (
                ((candidates - batch["velocity"][:, None, :]) ** 2)
                .mean(dim=-1)
                .min(dim=1)
                .values
            )
            total_mse += float(mse.sum().item())
            total_oracle_mse += float(oracle_mse.sum().item())
            total_count += int(mse.shape[0])
    totals = torch.tensor(
        [total_mse, total_oracle_mse, float(total_count)],
        device=device,
        dtype=torch.float64,
    )
    reduce_sum(totals)
    total_mse = float(totals[0].item())
    total_oracle_mse = float(totals[1].item())
    total_count = int(totals[2].item())
    return {
        "rmse": math.sqrt(total_mse / max(total_count, 1)),
        "candidate_oracle_rmse": math.sqrt(total_oracle_mse / max(total_count, 1)),
        "num_eval_samples": float(total_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--diffusion-checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
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
    val_loader = make_loader(
        cfg, "val", shuffle=False, distributed_eval=distributed
    )
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
            f"val_max_batches_per_rank="
            f"{val_max_batches if val_max_batches is not None else 'all'}"
        )

    for epoch in range(1, epochs + 1):
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
        if should_validate:
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
                    f"candidate_oracle_rmse={metrics['candidate_oracle_rmse']:.6f} "
                    f"eval_samples={int(metrics['num_eval_samples'])} "
                    f"seconds={metrics['epoch_seconds']:.1f}"
                )
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
