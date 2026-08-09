# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Lightweight runtime validation for training/test YAML configs."""

from __future__ import annotations

import os

import yaml


REQUIRED_TOP_LEVEL_KEYS = ("data", "model", "train", "schemes")


def validate_config(cfg: dict) -> bool:
    """Validate required config shape for training entrypoints."""
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in cfg:
            raise ValueError(f"Missing top-level config key: {key}")

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    required_data_keys = ("dataset", "data_path")
    for key in required_data_keys:
        if key not in data_cfg:
            raise ValueError(f"Missing data config key: {key}")

    required_train_keys = ("out_dir", "batch_size", "epochs", "optimizer")
    for key in required_train_keys:
        if key not in train_cfg:
            raise ValueError(f"Missing train config key: {key}")

    if "model_yaml" not in model_cfg:
        raise ValueError("Missing model.model_yaml config key")

    optimizer_cfg = train_cfg["optimizer"]
    if "learning_rate" not in optimizer_cfg:
        raise ValueError("Missing optimizer learning_rate")

    return True


def validate_data_paths(cfg: dict) -> bool:
    """Validate data path references while allowing optional categories."""
    data_path = cfg["data"].get("data_path")
    if isinstance(data_path, str):
        if not os.path.exists(data_path):
            raise ValueError(f"Configured data path does not exist: {data_path}")
        return True

    if isinstance(data_path, dict):
        existing_paths = 0
        for _, path_value in data_path.items():
            if path_value:
                if not os.path.exists(path_value):
                    raise ValueError(f"Configured category data path does not exist: {path_value}")
                existing_paths += 1
        if existing_paths == 0:
            raise ValueError("No category data paths were configured")
        return True

    raise ValueError("Unsupported data.data_path type, expected str or dict")


def validate_split_dirs(cfg: dict) -> bool:
    """Ensure each category data_path contains the configured train/val/test subdirs."""
    data_cfg = cfg["data"]
    data_path = data_cfg.get("data_path")
    subdirs = [
        data_cfg.get("train_dir", "train"),
        data_cfg.get("validation_dir", "val"),
        data_cfg.get("test_dir", "test"),
    ]
    roots: list[str] = []
    if isinstance(data_path, str):
        roots = [data_path]
    elif isinstance(data_path, dict):
        roots = [p for p in data_path.values() if p]
    for root in roots:
        for sub in subdirs:
            full = os.path.join(root, sub)
            if not os.path.isdir(full):
                raise ValueError(f"Missing split dir: {full}")
    return True


# Stage -> required input_dim. The model builds bn_input for input_dim, so a
# mismatch crashes deep in the forward pass with an opaque BatchNorm error
# ("running_mean should contain N elements not 6"). Catch it at startup instead.
#   stage 1: IMU only                       -> 6
#   stage 2: + 29 joint angles q            -> 6 + 29 = 35
#   stage 3: + 29 joints x (q, dq) masked   -> 6 + 58 = 64
STAGE_INPUT_DIMS = {1: 6, 2: 35, 3: 64}


def validate_model_dims(cfg: dict) -> bool:
    """Stage-aware sanity check on model input/output dims.

    Each stage has a fixed input_dim (see STAGE_INPUT_DIMS); a mismatch fails
    fast here rather than inside the model forward pass. Output is always 3
    (xyz velocity).
    """
    mp = cfg.get("model_param") or cfg["model"].get("model_param")
    if mp is None:
        model_yaml = cfg["model"].get("model_yaml")
        if not model_yaml or not os.path.isfile(model_yaml):
            raise ValueError(f"Configured model.model_yaml does not exist: {model_yaml}")
        with open(model_yaml, "r", encoding="utf-8") as handle:
            model_cfg = yaml.safe_load(handle) or {}
        mp = model_cfg.get("model_param", {})
    model_name = cfg["model"].get("model_name")
    if model_name == "Foundation_Model" and "platform_conditioning" in mp:
        pcfg = mp.get("platform_conditioning", {})
        if not pcfg.get("enabled", False):
            raise ValueError(
                "Foundation_Model spectral specialized training expects "
                "model_param.platform_conditioning.enabled: True"
            )
        if pcfg.get("encoder", "spectral") != "spectral":
            raise ValueError(
                "spectral_specialized branch expects "
                "model_param.platform_conditioning.encoder: spectral"
            )
    stage = mp.get("stage", 1)
    input_dim = mp.get("input_dim", 6)
    output_dim = mp.get("output_dim", 3)
    if output_dim != 3:
        raise ValueError(f"output_dim must be 3 (xyz velocity), got {output_dim}")
    expected = STAGE_INPUT_DIMS.get(stage)
    if expected is None:
        raise ValueError(f"Unknown stage {stage}, expected one of {sorted(STAGE_INPUT_DIMS)}")
    if input_dim != expected:
        raise ValueError(
            f"Stage {stage} requires input_dim={expected}, got {input_dim}"
        )
    return True
