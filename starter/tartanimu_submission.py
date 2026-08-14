#!/usr/bin/env python3
# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.
"""Released pretrained TartanIMU baseline -> Kaggle submission.

Loads the released unified 4-head model (auto-downloaded from the Hugging Face
Hub unless a local ``--checkpoint``/``--config`` is given), runs it over a set
of trajectories, and writes a ``window_id, vx, vy, vz`` submission.

Head routing
------------
The released model is multi-head: one output head per embodiment, so a head has
to be chosen per trajectory. How that choice is made depends on the split:

* **Labelled splits (train / val).** ``index/train_windows.csv`` and
  ``index/val_windows.csv`` carry a ``platform`` column; this script routes with
  it automatically. Use this to reproduce numbers and to self-score with
  ``kaggle_metric_tartanimu_score.py``.
* **The competition test split.** ``index/test_windows.csv`` is anonymized — it
  has no ``platform`` column and the ``.npz`` files carry no ``platform_id`` —
  so no routing signal ships with the data. Pass ``--head`` to force one head
  for every trajectory, or supply your own per-trajectory routing via
  ``--routing``.

  Note the competition rules require predictions to come from a **single model
  with one shared set of weights**. Inferring the embodiment inside your network
  is allowed and is precisely the point of the benchmark; recovering the
  platform in order to dispatch to four separately-trained experts is not.

Run from the ``superxslam/TartanIMU`` repo root after ``pip install -e .``.
Weights: https://huggingface.co/Tartan-IMU/TartanIMU

Windows CSV : window_id, traj_id, win_idx [, platform]
Trajectory  : <test_root>/<traj_id>.npz with key ``imu`` (N,6) = [ax,ay,az,gx,gy,gz]
Output      : window_id, vx, vy, vz
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tartan_imu.config import configer

HF_REPO = "Tartan-IMU/TartanIMU"
HF_CHECKPOINT = "checkpoints/unified.pt"
HF_CONFIG = "config/unified.yaml"

PLATFORMS = ("car", "dog", "drone", "human")
PLATFORM_BY_INDEX = {idx: name for idx, name in enumerate(PLATFORMS)}
_AUX_KEYS = {"_platform_logits", "_platform_probs"}


def _resolve_weights(checkpoint: str | None, config: str | None) -> tuple[str, str]:
    """Return (config_path, checkpoint_path), downloading from HF if not local."""
    if checkpoint and config:
        return config, checkpoint
    from huggingface_hub import hf_hub_download

    config = config or hf_hub_download(HF_REPO, HF_CONFIG)
    checkpoint = checkpoint or hf_hub_download(HF_REPO, HF_CHECKPOINT)
    return config, checkpoint


def _load_runtime_config(config_path: str) -> dict:
    """Load a config and merge its model YAML when it is an experiment YAML."""
    from tartan_imu.config import configer

    cfg = configer.load_config(config_path)
    model_yaml = cfg.get("model", {}).get("model_yaml")
    if model_yaml and "model_param" not in cfg:
        model_yaml_path = Path(model_yaml).expanduser()
        if not model_yaml_path.is_absolute() and not model_yaml_path.exists():
            model_yaml_path = Path(config_path).expanduser().parent / model_yaml_path
        with open(model_yaml_path, "r", encoding="utf-8") as handle:
            model_cfg = yaml.load(handle, Loader=yaml.Loader)
        configer.update_recursive(cfg, model_cfg)
    return cfg


def _load_windows(path: str) -> tuple[dict, bool]:
    """Read a windows index CSV.

    Args:
        path: Path to a ``*_windows.csv`` index file.

    Returns:
        A ``({traj_id: (platform_or_None, [(win_idx, window_id), ...])}, has_platform)``
        pair, with each window list sorted by ``win_idx``. ``platform_or_None`` is
        populated only when the CSV carries a ``platform`` column.
    """
    windows: dict = defaultdict(list)
    platform: dict = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        has_platform = "platform" in (reader.fieldnames or [])
        for row in reader:
            windows[row["traj_id"]].append((int(row["win_idx"]), int(row["window_id"])))
            if has_platform:
                platform[row["traj_id"]] = row["platform"]
    return {t: (platform.get(t), sorted(p)) for t, p in windows.items()}, has_platform


def _load_routing(path: str) -> dict:
    """Read a user-supplied ``traj_id,platform`` routing CSV."""
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"traj_id", "platform"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"--routing CSV must have columns traj_id,platform (missing {sorted(missing)})"
            )
        return {row["traj_id"]: row["platform"] for row in reader}


def _split_defaults(data_root: str, split: str) -> tuple[str, str]:
    root = Path(data_root).expanduser()
    return str(root / split), str(root / "index" / f"{split}_windows.csv")


def _resolve_npz(split_root: str, traj_id: str) -> str:
    root = Path(split_root).expanduser()
    traj_path = Path(traj_id)
    candidates = []
    if traj_path.suffix == ".npz":
        candidates.append(root / traj_path)
    else:
        candidates.append(root / f"{traj_id}.npz")
        candidates.append(root / traj_path)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    name = traj_path.name if traj_path.suffix == ".npz" else f"{traj_path.name}.npz"
    matches = sorted(root.rglob(name))
    if len(matches) == 1:
        return str(matches[0])
    if len(matches) > 1:
        raise SystemExit(
            f"{traj_id}: multiple matching NPZ files under {root}; use a windows "
            "traj_id with a relative subdirectory or pass a narrower --split-root."
        )
    raise FileNotFoundError(f"{traj_id}: no NPZ found under {root}")


def _build_windows_from_split(
    split_root: str,
    win_size: int,
) -> tuple[dict, bool]:
    root = Path(split_root).expanduser()
    windows: dict = {}
    next_window_id = 1
    has_platform = False
    for npz_path in sorted(root.rglob("*.npz")):
        with np.load(npz_path, allow_pickle=True) as npz:
            if "imu" not in npz:
                continue
            imu = np.asarray(npz["imu"])
        length = int(imu.shape[1] if imu.ndim >= 3 and imu.shape[0] == 1 else imu.shape[0])
        num_windows = length // win_size
        if num_windows <= 0:
            continue

        rel = npz_path.relative_to(root).with_suffix("")
        platform = rel.parts[0] if rel.parts and rel.parts[0] in PLATFORMS else None
        has_platform = has_platform or platform is not None
        pairs = []
        for win_idx in range(num_windows):
            pairs.append((win_idx, next_window_id))
            next_window_id += 1
        windows[str(rel)] = (platform, pairs)

    if not windows:
        raise ValueError(f"No valid IMU windows found under {root}")
    return windows, has_platform


def _moving_average_lowpass(feat: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return feat
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    padded = np.pad(feat, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    filtered = np.empty_like(feat, dtype=np.float32)
    for channel in range(feat.shape[1]):
        filtered[:, channel] = np.convolve(padded[:, channel], kernel, mode="valid")
    return filtered


def _windows_for_traj(feature, win_idxs, win_size, step, *, cfg):
    """Slice a trajectory feature array into (num_windows, channels, frames)."""
    pcfg = cfg.get("model_param", {}).get("platform_conditioning", {})
    lp_cfg = cfg.get("data", {}).get("low_pass_filter", {})
    use_platform_windows = bool(pcfg.get("enabled", False))
    low_pass_before_downsample = bool(lp_cfg.get("enabled", use_platform_windows))
    low_pass_kernel_size = int(lp_cfg.get("kernel_size", step))
    velocity_feature = (
        _moving_average_lowpass(feature, low_pass_kernel_size)
        if low_pass_before_downsample
        else feature
    )
    out = np.zeros((len(win_idxs), feature.shape[1], win_size // step), dtype=np.float32)
    platform_out = (
        np.zeros((len(win_idxs), feature.shape[1], win_size), dtype=np.float32)
        if use_platform_windows
        else None
    )
    for row, k in enumerate(win_idxs):
        seg = feature[k * win_size: k * win_size + win_size]
        velocity_seg = velocity_feature[k * win_size: k * win_size + win_size]
        if seg.shape[0] < win_size:
            pad = win_size - seg.shape[0]
            edge = seg[-1:] if seg.shape[0] else np.zeros((1, feature.shape[1]))
            seg = np.concatenate([seg, np.repeat(edge, pad, axis=0)], axis=0)
            velocity_edge = (
                velocity_seg[-1:]
                if velocity_seg.shape[0]
                else np.zeros((1, feature.shape[1]))
            )
            velocity_seg = np.concatenate(
                [velocity_seg, np.repeat(velocity_edge, pad, axis=0)], axis=0
            )
        out[row] = velocity_seg[::step, :].T
        if platform_out is not None:
            platform_out[row] = seg.T
    return out, platform_out


@torch.no_grad()
def _predict_traj(model, windows, platform_windows, platform, seq_len, device, batch_seqs, cfg):
    """Predict body-frame velocity for one trajectory's windows."""
    m = windows.shape[0]
    pad = (-m) % seq_len
    if pad:
        windows = np.concatenate([windows, np.repeat(windows[-1:], pad, axis=0)], 0)
        if platform_windows is not None:
            platform_windows = np.concatenate(
                [platform_windows, np.repeat(platform_windows[-1:], pad, axis=0)], 0
            )
    groups = windows.reshape(-1, seq_len, windows.shape[1], windows.shape[2])
    platform_groups = None
    if platform_windows is not None:
        platform_groups = platform_windows.reshape(
            -1, seq_len, platform_windows.shape[1], platform_windows.shape[2]
        )
    preds = []
    for start in range(0, groups.shape[0], batch_seqs):
        chunk = torch.from_numpy(groups[start: start + batch_seqs]).to(device)
        platform_chunk = None
        if platform_groups is not None:
            platform_chunk = torch.from_numpy(
                platform_groups[start: start + batch_seqs]
            ).to(device)
        heads = model(chunk, compute_all_heads=True, platform_x=platform_chunk)
        logits = heads.pop("_platform_logits", None)
        for key in _AUX_KEYS:
            heads.pop(key, None)
        if platform is not None:
            preds.append(heads[platform].reshape(-1, 3).cpu().numpy())
            continue

        pcfg = cfg.get("model_param", {}).get("platform_conditioning", {})
        if logits is None or not pcfg.get("route_by_prediction", False):
            raise SystemExit(
                "No platform label/head was provided and this model did not emit "
                "platform logits for route_by_prediction."
            )

        predicted = torch.argmax(logits.detach(), dim=-1).reshape(-1).cpu().numpy()
        flat_heads = {
            name: value.reshape(-1, 3).cpu().numpy()
            for name, value in heads.items()
            if name in PLATFORMS
        }
        pred = np.zeros((len(predicted), 3), dtype=np.float32)
        for platform_idx, platform_name in PLATFORM_BY_INDEX.items():
            mask = predicted == platform_idx
            if np.any(mask):
                pred[mask] = flat_heads[platform_name][mask]
        preds.append(pred)
    return np.concatenate(preds, axis=0)[:m]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrained TartanIMU baseline submission")
    parser.add_argument("--split", choices=("train", "val", "test"), default=None,
                        help="read data/<split> and data/index/<split>_windows.csv")
    parser.add_argument("--data-root", default="data",
                        help="dataset root used with --split")
    parser.add_argument("--split-root", "--test_root", dest="test_root", default=None,
                        help="dir of <traj_id>.npz files")
    parser.add_argument("--windows", default=None,
                        help="index/test_windows.csv (or a labelled *_windows.csv)")
    parser.add_argument("--out", required=True, help="output submission CSV")
    parser.add_argument("--head", choices=PLATFORMS, default=None,
                        help="force one head for every trajectory; required for "
                             "unlabelled splits unless the config routes by prediction")
    parser.add_argument("--routing", default=None,
                        help="CSV with columns traj_id,platform giving your own "
                             "per-trajectory head choice; overrides --head")
    parser.add_argument("--checkpoint", default=None, help="local .pt (else download from HF)")
    parser.add_argument("--config", default=None, help="local yaml (else download from HF)")
    parser.add_argument("--device", default=None, help="cuda:0 / cpu (auto if unset)")
    parser.add_argument("--batch_seqs", type=int, default=256)
    args = parser.parse_args()

    if args.split is not None:
        default_root, default_windows = _split_defaults(args.data_root, args.split)
        args.test_root = args.test_root or default_root
        args.windows = args.windows or default_windows
    if args.test_root is None or args.windows is None:
        raise SystemExit(
            "Pass --split {train,val,test}, or pass both --split-root and --windows."
        )

    config_path, checkpoint_path = _resolve_weights(args.checkpoint, args.config)
    cfg = _load_runtime_config(config_path)
    cfg["train"]["use_multi_gpu"] = False
    step = int(cfg["data"]["imu_freq"] / cfg["data"]["sample_freq"])
    seq_len = int(cfg["train"]["seq_len"])
    win_size = int(cfg["model_param"]["window_time"] * cfg["data"]["imu_freq"])

    if os.path.exists(args.windows):
        by_traj, has_platform = _load_windows(args.windows)
        print(f"windows={args.windows}")
    elif args.split is not None and args.windows == _split_defaults(args.data_root, args.split)[1]:
        by_traj, has_platform = _build_windows_from_split(args.test_root, win_size)
        print(
            f"windows=generated_from_npz split={args.split} "
            f"root={args.test_root} trajectories={len(by_traj)}"
        )
    else:
        raise FileNotFoundError(f"windows CSV not found: {args.windows}")

    routing = _load_routing(args.routing) if args.routing else {}
    route_by_prediction = bool(
        cfg.get("model_param", {})
        .get("platform_conditioning", {})
        .get("route_by_prediction", False)
    )
    if not has_platform and not routing and args.head is None and not route_by_prediction:
        raise SystemExit(
            f"{args.windows} has no 'platform' column — the competition test index is "
            "anonymized, so this multi-head model has no head to route to.\n"
            "Pass --head {car,dog,drone,human} to force one head, --routing "
            "traj_to_platform.csv to supply your own per-trajectory choice, or use "
            "a config/checkpoint with model_param.platform_conditioning.route_by_prediction.\n"
            "Per the competition rules, predictions must come from a single model with "
            "one shared set of weights."
        )

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = configer.build_model(types.SimpleNamespace(local_rank=0), cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    model.to(device).eval()
    print(f"loaded {checkpoint_path} | win_size={win_size} step={step} "
          f"seq_len={seq_len} device={device} route_by_prediction={route_by_prediction}")

    rows: dict = {}
    for traj_id, (csv_platform, pairs) in sorted(by_traj.items()):
        platform = routing.get(traj_id) or args.head or csv_platform
        if platform not in PLATFORMS:
            if platform is not None or not route_by_prediction:
                raise SystemExit(
                    f"{traj_id}: no valid head to route to (got {platform!r}); "
                    f"expected one of {list(PLATFORMS)}"
                )
            platform = None
        imu = np.asarray(np.load(_resolve_npz(args.test_root, traj_id))["imu"],
                         dtype=np.float32)
        # model expects [gyro | accel]; npz stores [accel | gyro]
        feature = np.concatenate([imu[:, 3:6], imu[:, 0:3]], axis=1)
        win_idxs = [wi for wi, _ in pairs]
        windows, platform_windows = _windows_for_traj(
            feature, win_idxs, win_size, step, cfg=cfg
        )
        pred = _predict_traj(
            model,
            windows,
            platform_windows,
            platform,
            seq_len,
            device,
            args.batch_seqs,
            cfg,
        )
        for (_wi, wid), v in zip(pairs, pred):
            rows[wid] = (float(v[0]), float(v[1]), float(v[2]))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["window_id", "vx", "vy", "vz"])
        for wid in sorted(rows):
            vx, vy, vz = rows[wid]
            writer.writerow([wid, f"{vx:.6f}", f"{vy:.6f}", f"{vz:.6f}"])
    print(f"wrote {args.out} rows={len(rows)}")


if __name__ == "__main__":
    main()
