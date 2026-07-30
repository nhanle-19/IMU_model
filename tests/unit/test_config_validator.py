"""Unit 3b: startup guardrails for config validation.

Adds split-dir existence checks and stage-aware model dim consistency to the
existing validator, so misconfigured runs fail fast with a clear message
instead of crashing deep inside data loading or the model forward pass.
"""
import pytest

from tartan_imu.utils.config_validator import validate_split_dirs, validate_model_dims


def _base_cfg(tmp_path):
    return {
        "data": {
            "dataset": "HumanoidPostProcessed",
            "data_path": {"human": str(tmp_path)},
            "train_dir": "train",
            "validation_dir": "val",
            "test_dir": "test",
        },
        "model": {
            "model_yaml": "x.yaml",
            "model_param": {"input_dim": 6, "output_dim": 3},
        },
    }


def test_missing_split_dir_raises(tmp_path):
    cfg = _base_cfg(tmp_path)  # no train/val/test subdirs created
    with pytest.raises(ValueError, match="split dir"):
        validate_split_dirs(cfg)


def test_present_split_dirs_pass(tmp_path):
    for d in ("train", "val", "test"):
        (tmp_path / d).mkdir()
    cfg = _base_cfg(tmp_path)
    assert validate_split_dirs(cfg) is True


def test_stage1_requires_input_dim_6(tmp_path):
    cfg = _base_cfg(tmp_path)
    cfg["model"]["model_param"]["input_dim"] = 35  # wrong for stage 1
    cfg["model"]["model_param"]["stage"] = 1
    with pytest.raises(ValueError, match="input_dim"):
        validate_model_dims(cfg)


def test_bad_output_dim_raises(tmp_path):
    cfg = _base_cfg(tmp_path)
    cfg["model"]["model_param"]["stage"] = 1
    cfg["model"]["model_param"]["output_dim"] = 7  # must be 3 (xyz velocity)
    with pytest.raises(ValueError, match="output_dim"):
        validate_model_dims(cfg)


def test_valid_dims_pass(tmp_path):
    cfg = _base_cfg(tmp_path)
    cfg["model"]["model_param"]["stage"] = 1
    assert validate_model_dims(cfg) is True


def test_stage2_requires_input_dim_35(tmp_path):
    """Stage 2 fuses the 29 joint angles -> IMU 6 + 29 = 35."""
    cfg = _base_cfg(tmp_path)
    cfg["model"]["model_param"]["stage"] = 2
    cfg["model"]["model_param"]["input_dim"] = 35
    assert validate_model_dims(cfg) is True


def test_stage2_wrong_input_dim_raises(tmp_path):
    cfg = _base_cfg(tmp_path)
    cfg["model"]["model_param"]["stage"] = 2
    cfg["model"]["model_param"]["input_dim"] = 64  # that's stage 3's dim
    with pytest.raises(ValueError, match="input_dim"):
        validate_model_dims(cfg)


def test_stage3_requires_input_dim_64(tmp_path):
    """Stage 3 masked-attention fuses 29 joints x (q, dq) -> IMU 6 + 58 = 64."""
    cfg = _base_cfg(tmp_path)
    cfg["model"]["model_param"]["stage"] = 3
    cfg["model"]["model_param"]["input_dim"] = 64
    assert validate_model_dims(cfg) is True


def test_stage3_wrong_input_dim_raises(tmp_path):
    cfg = _base_cfg(tmp_path)
    cfg["model"]["model_param"]["stage"] = 3
    cfg["model"]["model_param"]["input_dim"] = 35  # forgot the dq channels
    with pytest.raises(ValueError, match="input_dim"):
        validate_model_dims(cfg)
