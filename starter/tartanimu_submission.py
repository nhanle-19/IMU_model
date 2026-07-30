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
  ``kaggle_metric_ate20.py``.
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
import types
from collections import defaultdict

import numpy as np
import torch

from tartan_imu.config import configer

HF_REPO = "Tartan-IMU/TartanIMU"
HF_CHECKPOINT = "checkpoints/unified.pt"
HF_CONFIG = "config/unified.yaml"

PLATFORMS = ("car", "dog", "drone", "human")


def _resolve_weights(checkpoint: str | None, config: str | None) -> tuple[str, str]:
    """Return (config_path, checkpoint_path), downloading from HF if not local."""
    if checkpoint and config:
        return config, checkpoint
    from huggingface_hub import hf_hub_download

    config = config or hf_hub_download(HF_REPO, HF_CONFIG)
    checkpoint = checkpoint or hf_hub_download(HF_REPO, HF_CHECKPOINT)
    return config, checkpoint


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


def _windows_for_traj(feature, win_idxs, win_size, step):
    """Slice a trajectory feature array into (num_windows, channels, frames)."""
    out = np.zeros((len(win_idxs), feature.shape[1], win_size // step), dtype=np.float32)
    for row, k in enumerate(win_idxs):
        seg = feature[k * win_size: k * win_size + win_size]
        if seg.shape[0] < win_size:
            pad = win_size - seg.shape[0]
            edge = seg[-1:] if seg.shape[0] else np.zeros((1, feature.shape[1]))
            seg = np.concatenate([seg, np.repeat(edge, pad, axis=0)], axis=0)
        out[row] = seg[::step, :].T
    return out


@torch.no_grad()
def _predict_traj(model, windows, platform, seq_len, device, batch_seqs):
    """Predict body-frame velocity for one trajectory's windows."""
    m = windows.shape[0]
    pad = (-m) % seq_len
    if pad:
        windows = np.concatenate([windows, np.repeat(windows[-1:], pad, axis=0)], 0)
    groups = windows.reshape(-1, seq_len, windows.shape[1], windows.shape[2])
    preds = []
    for start in range(0, groups.shape[0], batch_seqs):
        chunk = torch.from_numpy(groups[start: start + batch_seqs]).to(device)
        heads = model(chunk, compute_all_heads=True)
        preds.append(heads[platform].reshape(-1, 3).cpu().numpy())
    return np.concatenate(preds, axis=0)[:m]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrained TartanIMU baseline submission")
    parser.add_argument("--test_root", required=True, help="dir of <traj_id>.npz files")
    parser.add_argument("--windows", required=True,
                        help="index/test_windows.csv (or a labelled *_windows.csv)")
    parser.add_argument("--out", required=True, help="output submission CSV")
    parser.add_argument("--head", choices=PLATFORMS, default=None,
                        help="force one head for every trajectory; required when the "
                             "windows CSV has no platform column (the test split)")
    parser.add_argument("--routing", default=None,
                        help="CSV with columns traj_id,platform giving your own "
                             "per-trajectory head choice; overrides --head")
    parser.add_argument("--checkpoint", default=None, help="local .pt (else download from HF)")
    parser.add_argument("--config", default=None, help="local yaml (else download from HF)")
    parser.add_argument("--device", default=None, help="cuda:0 / cpu (auto if unset)")
    parser.add_argument("--batch_seqs", type=int, default=256)
    args = parser.parse_args()

    by_traj, has_platform = _load_windows(args.windows)
    routing = _load_routing(args.routing) if args.routing else {}
    if not has_platform and not routing and args.head is None:
        raise SystemExit(
            f"{args.windows} has no 'platform' column — the competition test index is "
            "anonymized, so this multi-head model has no head to route to.\n"
            "Pass --head {car,dog,drone,human} to force one head, or --routing "
            "traj_to_platform.csv to supply your own per-trajectory choice.\n"
            "Per the competition rules, predictions must come from a single model with "
            "one shared set of weights."
        )

    config_path, checkpoint_path = _resolve_weights(args.checkpoint, args.config)
    cfg = configer.load_config(config_path)
    cfg["train"]["use_multi_gpu"] = False
    step = int(cfg["data"]["imu_freq"] / cfg["data"]["sample_freq"])
    seq_len = int(cfg["train"]["seq_len"])
    win_size = int(cfg["model_param"]["window_time"] * cfg["data"]["imu_freq"])

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = configer.build_model(types.SimpleNamespace(local_rank=0), cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    model.to(device).eval()
    print(f"loaded {checkpoint_path} | win_size={win_size} step={step} "
          f"seq_len={seq_len} device={device}")

    rows: dict = {}
    for traj_id, (csv_platform, pairs) in sorted(by_traj.items()):
        platform = routing.get(traj_id) or args.head or csv_platform
        if platform not in PLATFORMS:
            raise SystemExit(
                f"{traj_id}: no valid head to route to (got {platform!r}); "
                f"expected one of {list(PLATFORMS)}"
            )
        imu = np.asarray(np.load(os.path.join(args.test_root, f"{traj_id}.npz"))["imu"],
                         dtype=np.float32)
        # model expects [gyro | accel]; npz stores [accel | gyro]
        feature = np.concatenate([imu[:, 3:6], imu[:, 0:3]], axis=1)
        win_idxs = [wi for wi, _ in pairs]
        pred = _predict_traj(model, _windows_for_traj(feature, win_idxs, win_size, step),
                             platform, seq_len, device, args.batch_seqs)
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
