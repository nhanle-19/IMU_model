"""Train the IMU-conditioned velocity diffusion model."""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

from imu_velocity_diffusion.checkpoint import save_checkpoint
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
    set_seed,
)


def evaluate(
    model: VelocityDiffusionModel,
    loader,
    schedule: DiffusionSchedule,
    num_candidates: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_noise_loss = 0.0
    total_min_mse = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
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
    return {
        "noise_loss": total_noise_loss / max(total_count, 1),
        "candidate_min_rmse": math.sqrt(total_min_mse / max(total_count, 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    require_average_velocity_targets(cfg)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(args.device)

    train_loader = make_loader(cfg, "train", shuffle=True)
    val_loader = make_loader(cfg, "val", shuffle=False)
    input_channels = int(train_loader.dataset.input_channels)

    model = build_diffusion_model(cfg, input_channels, device)
    schedule = DiffusionSchedule(**cfg["diffusion"], device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )

    out_dir = output_dir(cfg)
    best_metric = float("inf")
    epochs = int(cfg["train"]["epochs"])
    val_candidates = int(cfg["policy"].get("num_candidates", 16))

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        progress = tqdm(train_loader, desc=f"diffusion epoch {epoch}/{epochs}")
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

        metrics = evaluate(model, val_loader, schedule, val_candidates, device)
        metrics["train_noise_loss"] = running_loss / max(running_count, 1)
        print(
            f"epoch={epoch} train_noise_loss={metrics['train_noise_loss']:.6f} "
            f"val_noise_loss={metrics['noise_loss']:.6f} "
            f"val_candidate_min_rmse={metrics['candidate_min_rmse']:.6f}"
        )

        save_checkpoint(
            out_dir / "diffusion_last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            cfg=cfg,
            metrics=metrics,
        )
        if metrics["candidate_min_rmse"] < best_metric:
            best_metric = metrics["candidate_min_rmse"]
            save_checkpoint(
                out_dir / "diffusion_best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                cfg=cfg,
                metrics=metrics,
            )


if __name__ == "__main__":
    main()
