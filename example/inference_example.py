#!/usr/bin/env python3
# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.
"""Minimal inference example for the TartanIMU foundation model.

Loads a released checkpoint (unified or single-platform expert), runs it on one
``.npz`` trajectory, and prints the predicted body-frame velocity. This reuses
the repository's own config/model builders so the released weights load exactly
as they were trained.

Run from the cloned ``superxslam/TartanIMU`` repo root (``pip install -e .``):

    python example/inference_example.py \
        --config ./tartanimu_weights/config/unified.yaml \
        --model  ./tartanimu_weights/checkpoints/unified.pt \
        --npz    <path/to/trajectory.npz> \
        --motion_type human
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tartan_imu.config.configer import build_model, load_config  # noqa: E402
from tartan_imu.dataloader import dataset_AirLab as dataset_utils  # noqa: E402

MOTION_TYPE_ID = {"car": 1, "dog": 2, "drone": 3, "human": 4}


class _Args:
    """Minimal args object expected by ``build_model``."""

    local_rank = 0


def load_tracker(config_path: str, model_path: str, device: torch.device):
    """Build the model from config and load a released checkpoint.

    Args:
        config_path: Path to the experiment YAML (e.g. ``config/unified.yaml``).
        model_path: Path to a released ``.pt`` checkpoint.
        device: Torch device to place the model on.

    Returns:
        A tuple ``(model, config)`` with the model in eval mode on ``device``.
    """
    config = load_config(config_path)
    model = build_model(_Args(), config)
    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, config


def predict_npz(model, config, npz_path: str, motion_type: str,
                device: torch.device) -> np.ndarray:
    """Run the model on one ``.npz`` trajectory.

    Args:
        model: The loaded TartanIMU model.
        config: The loaded config dict.
        npz_path: Path to a trajectory ``.npz`` (AirLab reader keys).
        motion_type: One of ``car`` / ``dog`` / ``drone`` / ``human``.
        device: Torch device.

    Returns:
        Predicted body-frame velocity, shape ``(num_windows, 3)``.
    """
    basic = dataset_utils.BasicSequenceData(config, [npz_path], mode="test")
    dataset = dataset_utils.ResNetLSTMSeqToSeqDataset(
        config, basic, basic.get_merged_index_map(), mode="test"
    )
    loader = DataLoader(dataset, batch_size=config.get("test", {}).get("batch_size", 256))

    preds = []
    with torch.no_grad():
        for batch in loader:
            imu = batch[0].to(device)
            label = batch[3].to(device)
            out = model(imu, motion_type=label, predict_cov=False, compute_all_heads=False)
            preds.append(out[motion_type].cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.empty((0, 3))


def main() -> None:
    parser = argparse.ArgumentParser(description="TartanIMU inference example")
    parser.add_argument("--config", required=True, help="experiment YAML path")
    parser.add_argument("--model", required=True, help="checkpoint .pt path")
    parser.add_argument("--npz", required=True, help="trajectory .npz path")
    parser.add_argument("--motion_type", default="human", choices=list(MOTION_TYPE_ID))
    parser.add_argument("--device", default=None, help="cuda / cpu (auto if unset)")
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, config = load_tracker(args.config, args.model, device)
    velocity = predict_npz(model, config, args.npz, args.motion_type, device)

    print(f"device            : {device}")
    print(f"windows predicted : {velocity.shape[0]}")
    print(f"velocity shape    : {velocity.shape}  (body-frame vx, vy, vz)")
    if velocity.size:
        print(f"mean |velocity|   : {np.linalg.norm(velocity, axis=1).mean():.4f} m/s")
        print(f"first 3 windows   :\n{np.round(velocity[:3], 4)}")


if __name__ == "__main__":
    main()
