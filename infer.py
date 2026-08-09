"""Run diffusion-first velocity inference on one NPZ trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from imu_velocity_diffusion.checkpoint import load_checkpoint
from imu_velocity_diffusion.config import load_config
from imu_velocity_diffusion.data import (
    assert_compatible_velocity_contract,
    model_window_size,
    require_average_velocity_targets,
    resolve_downsample_step,
)
from imu_velocity_diffusion.diffusion import DiffusionSchedule
from imu_velocity_diffusion.factory import build_diffusion_model, build_refiner_model
from imu_velocity_diffusion.training import get_device


def load_imu_windows(cfg: dict, input_npz: str) -> tuple[torch.Tensor, np.ndarray]:
    data_cfg = cfg["data"]
    with np.load(input_npz, allow_pickle=True) as npz:
        imu = np.asarray(npz[data_cfg.get("imu_key", "imu")])
    if imu.ndim >= 3 and imu.shape[0] == 1:
        imu = imu[0]
    imu = imu.reshape(imu.shape[0], -1).astype(np.float32)
    columns = data_cfg.get("imu_columns")
    if columns is not None:
        imu = imu[:, list(columns)]

    window_size = int(data_cfg["window_size"])
    stride = int(data_cfg.get("stride", 1))
    downsample_step = resolve_downsample_step(data_cfg)
    starts = np.arange(0, len(imu) - window_size + 1, stride, dtype=np.int64)
    if len(starts) == 0:
        raise ValueError("Input trajectory is shorter than data.window_size")
    windows = np.stack(
        [imu[s : s + window_size : downsample_step].T for s in starts], axis=0
    )
    return torch.from_numpy(windows), starts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--diffusion-checkpoint", required=True)
    parser.add_argument("--refiner-checkpoint", required=True)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    require_average_velocity_targets(cfg)
    device = get_device(args.device)

    diffusion_ckpt = load_checkpoint(args.diffusion_checkpoint, device)
    diffusion_cfg = diffusion_ckpt.get("cfg", cfg)
    require_average_velocity_targets(diffusion_cfg)
    assert_compatible_velocity_contract(cfg, diffusion_cfg)
    windows, starts = load_imu_windows(diffusion_cfg, args.input_npz)
    input_channels = int(windows.shape[1])
    diffusion = build_diffusion_model(diffusion_cfg, input_channels, device)
    diffusion.load_state_dict(diffusion_ckpt["model_state_dict"])
    diffusion.eval()
    schedule = DiffusionSchedule(**diffusion_cfg["diffusion"], device=device)

    policy_ckpt = load_checkpoint(args.refiner_checkpoint, device)
    policy_cfg = policy_ckpt.get("cfg", cfg)
    require_average_velocity_targets(policy_cfg)
    assert_compatible_velocity_contract(cfg, policy_cfg)
    refiner = build_refiner_model(policy_cfg, input_channels, device)
    refiner.load_state_dict(policy_ckpt["model_state_dict"])
    refiner.eval()

    batch_size = int(cfg.get("inference", {}).get("batch_size", 256))
    num_candidates = args.num_candidates or int(cfg["policy"].get("num_candidates", 16))
    all_candidates, all_weights, all_velocity = [], [], []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            imu = windows[start : start + batch_size].to(device)
            candidates = schedule.sample(diffusion, imu, num_candidates=num_candidates)
            outputs = refiner(imu, candidates)
            all_candidates.append(candidates.cpu().numpy())
            all_weights.append(outputs["weights"].cpu().numpy())
            all_velocity.append(outputs["velocity"].cpu().numpy())

    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        window_starts=starts,
        candidate_velocities=np.concatenate(all_candidates, axis=0),
        candidate_weights=np.concatenate(all_weights, axis=0),
        refined_velocity=np.concatenate(all_velocity, axis=0),
        candidate_semantics="average_body_velocity_over_window",
        source_window_size=int(diffusion_cfg["data"]["window_size"]),
        source_stride=int(diffusion_cfg["data"].get("stride", 1)),
        diffusion_imu_downsample_step=resolve_downsample_step(diffusion_cfg["data"]),
        diffusion_model_window_size=model_window_size(diffusion_cfg["data"]),
    )
    print(f"wrote {args.output_npz}")


if __name__ == "__main__":
    main()
