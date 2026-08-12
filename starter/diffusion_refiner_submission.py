#!/usr/bin/env python3
"""Diffusion + refiner velocity stack -> Kaggle-style submission CSV.

Runs the trained diffusion candidate generator and VelocityDistributionRefiner
over an indexed set of one-second IMU windows and writes:

    window_id, vx, vy, vz
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imu_velocity_diffusion.checkpoint import load_checkpoint
from imu_velocity_diffusion.config import load_config
from imu_velocity_diffusion.data import (
    assert_compatible_velocity_contract,
    require_average_velocity_targets,
    resolve_downsample_step,
)
from imu_velocity_diffusion.diffusion import DiffusionSchedule
from imu_velocity_diffusion.factory import build_diffusion_model, build_refiner_model
from imu_velocity_diffusion.training import get_device


def _load_windows(path: str) -> dict[str, list[tuple[int, int]]]:
    by_traj: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"traj_id", "win_idx", "window_id"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            by_traj[row["traj_id"]].append((int(row["win_idx"]), int(row["window_id"])))
    return {traj_id: sorted(pairs) for traj_id, pairs in by_traj.items()}


def _build_windows_from_split(
    split_root: Path,
    data_cfg: dict,
) -> dict[str, list[tuple[int, int]]]:
    imu_key = data_cfg.get("imu_key", "imu")
    window_size = int(data_cfg["window_size"])
    stride = int(data_cfg.get("stride", window_size))
    by_traj: dict[str, list[tuple[int, int]]] = {}
    next_window_id = 0
    for npz_path in sorted(split_root.rglob(str(data_cfg.get("file_glob", "*.npz")))):
        with np.load(npz_path, allow_pickle=True) as npz:
            if imu_key not in npz:
                continue
            imu = np.asarray(npz[imu_key])
        if imu.ndim >= 3 and imu.shape[0] == 1:
            length = int(imu.shape[1])
        else:
            length = int(imu.shape[0])
        if length < window_size:
            continue
        num_windows = ((length - window_size) // stride) + 1
        traj_id = str(npz_path.relative_to(split_root).with_suffix(""))
        pairs = []
        for win_idx in range(num_windows):
            pairs.append((win_idx, next_window_id))
            next_window_id += 1
        by_traj[traj_id] = pairs
    if not by_traj:
        raise ValueError(f"No valid NPZ windows found under {split_root}")
    return by_traj


def _split_root(cfg: dict, split: str) -> Path:
    data_cfg = cfg["data"]
    split_dirs = data_cfg.get("split_dirs", {})
    return Path(data_cfg["root"]).expanduser() / split_dirs.get(split, split)


def _windows_path(cfg: dict, split: str) -> Path:
    return Path(cfg["data"]["root"]).expanduser() / "index" / f"{split}_windows.csv"


def _resolve_npz(split_root: Path, traj_id: str) -> Path:
    traj_path = Path(traj_id)
    candidates = []
    if traj_path.suffix == ".npz":
        candidates.append(split_root / traj_path)
    else:
        candidates.append(split_root / f"{traj_id}.npz")
        candidates.append(split_root / traj_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    name = traj_path.name if traj_path.suffix == ".npz" else f"{traj_path.name}.npz"
    matches = sorted(split_root.rglob(name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SystemExit(
            f"{traj_id}: multiple matching NPZ files under {split_root}; "
            "use a windows traj_id with a relative subdirectory."
        )
    raise FileNotFoundError(f"{traj_id}: no NPZ found under {split_root}")


def _infer_input_channels(state_dict: dict[str, torch.Tensor]) -> int:
    key = "imu_encoder.net.0.weight"
    if key not in state_dict:
        raise KeyError(f"Cannot infer input channels: checkpoint missing {key!r}")
    return int(state_dict[key].shape[1])


def _load_imu_windows(
    npz_path: str,
    pairs: list[tuple[int, int]],
    data_cfg: dict,
) -> torch.Tensor:
    imu_key = data_cfg.get("imu_key", "imu")
    with np.load(npz_path, allow_pickle=True) as npz:
        imu = np.asarray(npz[imu_key])
    if imu.ndim >= 3 and imu.shape[0] == 1:
        imu = imu[0]
    imu = imu.reshape(imu.shape[0], -1).astype(np.float32)

    columns = data_cfg.get("imu_columns")
    if columns is not None:
        imu = imu[:, list(columns)]

    window_size = int(data_cfg["window_size"])
    stride = int(data_cfg.get("stride", window_size))
    downsample_step = resolve_downsample_step(data_cfg)
    windows = []
    for win_idx, _window_id in pairs:
        start = int(win_idx) * stride
        end = start + window_size
        segment = imu[start:end]
        if segment.shape[0] < window_size:
            pad = window_size - segment.shape[0]
            edge = segment[-1:] if segment.shape[0] else np.zeros((1, imu.shape[1]))
            segment = np.concatenate([segment, np.repeat(edge, pad, axis=0)], axis=0)
        windows.append(segment[::downsample_step].T)
    return torch.from_numpy(np.stack(windows, axis=0))


@torch.no_grad()
def _predict_windows(
    diffusion,
    refiner,
    schedule: DiffusionSchedule,
    windows: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    num_candidates: int,
) -> np.ndarray:
    velocities = []
    for start in range(0, len(windows), batch_size):
        imu = windows[start : start + batch_size].to(device)
        candidates = schedule.sample(diffusion, imu, num_candidates=num_candidates)
        outputs = refiner(imu, candidates)
        velocities.append(outputs["velocity"].cpu().numpy())
    return np.concatenate(velocities, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run diffusion+refiner over indexed windows and write submission CSV"
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="config split to read when --split-root/--windows are omitted",
    )
    parser.add_argument(
        "--split-root",
        "--test_root",
        dest="split_root",
        default=None,
        help="dir of <traj_id>.npz files; defaults to config data.root/split_dirs[split]",
    )
    parser.add_argument(
        "--windows",
        default=None,
        help="index CSV; defaults to config data.root/index/{split}_windows.csv",
    )
    parser.add_argument("--out", required=True, help="output submission CSV")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--diffusion-checkpoint", required=True)
    parser.add_argument("--refiner-checkpoint", required=True)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args.config)
    require_average_velocity_targets(cfg)
    device = get_device(args.device)
    split_root = (
        Path(args.split_root).expanduser() if args.split_root else _split_root(cfg, args.split)
    )
    windows_path = Path(args.windows).expanduser() if args.windows else _windows_path(cfg, args.split)

    diffusion_ckpt = load_checkpoint(args.diffusion_checkpoint, device)
    diffusion_cfg = diffusion_ckpt.get("cfg", cfg)
    require_average_velocity_targets(diffusion_cfg)
    assert_compatible_velocity_contract(cfg, diffusion_cfg)
    input_channels = _infer_input_channels(diffusion_ckpt["model_state_dict"])
    diffusion = build_diffusion_model(diffusion_cfg, input_channels, device)
    diffusion.load_state_dict(diffusion_ckpt["model_state_dict"])
    diffusion.eval()
    schedule = DiffusionSchedule(**diffusion_cfg["diffusion"], device=device)

    refiner_ckpt = load_checkpoint(args.refiner_checkpoint, device)
    refiner_cfg = refiner_ckpt.get("cfg", cfg)
    require_average_velocity_targets(refiner_cfg)
    assert_compatible_velocity_contract(cfg, refiner_cfg)
    refiner = build_refiner_model(refiner_cfg, input_channels, device)
    refiner.load_state_dict(refiner_ckpt["model_state_dict"])
    refiner.eval()

    batch_size = args.batch_size or int(cfg.get("inference", {}).get("batch_size", 256))
    num_candidates = args.num_candidates or int(cfg["policy"].get("num_candidates", 16))

    rows: dict[int, tuple[float, float, float]] = {}
    if windows_path.exists():
        by_traj = _load_windows(str(windows_path))
        print(f"split={args.split} root={split_root} windows={windows_path}")
    elif args.windows is not None:
        raise FileNotFoundError(f"windows CSV not found: {windows_path}")
    else:
        by_traj = _build_windows_from_split(split_root, diffusion_cfg["data"])
        print(
            f"split={args.split} root={split_root} windows=generated_from_npz "
            f"trajectories={len(by_traj)}"
        )
    for traj_id, pairs in sorted(by_traj.items()):
        npz_path = _resolve_npz(split_root, traj_id)
        windows = _load_imu_windows(str(npz_path), pairs, diffusion_cfg["data"])
        pred = _predict_windows(
            diffusion,
            refiner,
            schedule,
            windows,
            device=device,
            batch_size=batch_size,
            num_candidates=num_candidates,
        )
        for (_win_idx, window_id), velocity in zip(pairs, pred):
            rows[window_id] = (float(velocity[0]), float(velocity[1]), float(velocity[2]))
        print(f"{traj_id}: windows={len(pairs)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["window_id", "vx", "vy", "vz"])
        for window_id in sorted(rows):
            vx, vy, vz = rows[window_id]
            writer.writerow([window_id, f"{vx:.6f}", f"{vy:.6f}", f"{vz:.6f}"])
    print(f"wrote {args.out} rows={len(rows)}")


if __name__ == "__main__":
    main()
