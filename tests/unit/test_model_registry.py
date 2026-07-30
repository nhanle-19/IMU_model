# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Model registry characterization tests (LSTM-only release).

The registry must build the Foundation_Model backbone IDENTICALLY to the legacy
in-line construction in config/configer.build_model (param count + per-parameter
shapes). The Transformer backbone interface stays registered, but its core is
not shipped in this release, so building it must raise a clear NotImplementedError.
"""
import pytest
import torch

from tartan_imu.config import configer


def _load_cfg(yaml_path: str) -> dict:
    """Load a dataset config and merge its model_yaml exactly as build_model does,
    so cfg carries the top-level model_param the builders read."""
    import yaml as _yaml

    cfg = configer.load_config(yaml_path)
    with open(cfg["model"]["model_yaml"], "r") as f:
        cfg_special = _yaml.load(f, Loader=_yaml.Loader)
    configer.update_recursive(cfg, cfg_special)
    return cfg


def _legacy_build_foundation(cfg: dict):
    """Reconstruct the legacy in-line Foundation_Model object (the if/elif body
    of build_model) WITHOUT the device/multi-GPU tail, to compare against the
    registry."""
    import torch.nn as nn
    from tartan_imu.model.lstm import model_lstm

    if cfg["model"].get("cross_xyz", False):
        trunk = model_lstm.Crossxy_LSTM_Model(cfg)
    else:
        trunk = model_lstm.ResNetLSTMSeqNet(cfg)
    heads = nn.ModuleDict({
        "dog": model_lstm.OutputHead(cfg, "dog"),
        "human": model_lstm.OutputHead(cfg, "human"),
        "car": model_lstm.OutputHead(cfg, "car"),
        "drone": model_lstm.OutputHead(cfg, "drone"),
    })
    return model_lstm.FoundationModel(cfg=cfg, trunk=trunk, heads=heads)


def _assert_param_equivalence(m_legacy, m_reg):
    legacy_params = list(m_legacy.named_parameters())
    reg_params = list(m_reg.named_parameters())
    assert len(legacy_params) == len(reg_params), "different number of parameters"
    assert sum(p.numel() for _, p in legacy_params) == \
        sum(p.numel() for _, p in reg_params), "different total param count"
    for (n_l, p_l), (n_r, p_r) in zip(legacy_params, reg_params):
        assert n_l == n_r, f"param name mismatch: {n_l} != {n_r}"
        assert p_l.shape == p_r.shape, f"shape mismatch at {n_l}"


def test_registry_foundation_matches_legacy():
    from tartan_imu.model.registry import build_backbone

    cfg = _load_cfg("config/datasets/tartanimu/tartan_imu_multihead_smoke.yaml")

    torch.manual_seed(0)
    legacy = _legacy_build_foundation(cfg)
    torch.manual_seed(0)
    reg = build_backbone("Foundation_Model", cfg)

    _assert_param_equivalence(legacy, reg)


def test_registry_unknown_key_lists_available():
    from tartan_imu.model.registry import build_backbone, available

    with pytest.raises(ValueError) as exc:
        build_backbone("NoSuchModel", {"model": {"model_name": "NoSuchModel"}})
    msg = str(exc.value)
    assert "NoSuchModel" in msg
    # Error must enumerate the registered keys to aid debugging.
    for key in available():
        assert key in msg


def test_foundation_model_is_registered():
    from tartan_imu.model.registry import available

    assert "Foundation_Model" in available()


def test_transformer_interface_kept_but_core_not_shipped():
    """The Transformer backbone interface stays registered so the core can be
    dropped back in later with zero wiring changes; until then, building it
    raises a clear NotImplementedError (not a raw ImportError)."""
    from tartan_imu.model.registry import available, build_backbone

    assert "Transformer" in available(), "interface should stay registered"

    with pytest.raises(NotImplementedError):
        build_backbone("Transformer", {"model": {"model_name": "Transformer"}})
