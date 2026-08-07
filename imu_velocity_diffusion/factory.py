"""Model construction helpers shared by training and inference scripts."""

from __future__ import annotations

import torch

from imu_velocity_diffusion.models import VelocityDiffusionModel, VelocitySelector


def build_diffusion_model(
    cfg: dict, input_channels: int, device: torch.device
) -> VelocityDiffusionModel:
    model_cfg = cfg["model"]
    return VelocityDiffusionModel(
        input_channels=input_channels,
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        time_dim=int(model_cfg.get("time_dim", 64)),
        velocity_dim=int(model_cfg.get("velocity_dim", 3)),
    ).to(device)


def build_selector_model(
    cfg: dict, input_channels: int, device: torch.device
) -> VelocitySelector:
    model_cfg = cfg["model"]
    policy_cfg = cfg["policy"]
    return VelocitySelector(
        input_channels=input_channels,
        hidden_dim=int(policy_cfg.get("hidden_dim", model_cfg.get("hidden_dim", 128))),
        candidate_dim=int(policy_cfg.get("candidate_dim", 64)),
        velocity_dim=int(model_cfg.get("velocity_dim", 3)),
    ).to(device)
