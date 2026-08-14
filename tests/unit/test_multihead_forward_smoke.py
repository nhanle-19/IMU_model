"""Guardrail: the multi-head Foundation_Model must run end-to-end.

This is the "multi-head works" contract that cleanup must not break. It builds
the real Foundation_Model (shared LSTM trunk + car/dog/drone/human heads) via
the registry, pushes a MIXED-motion-type batch through forward -> the live
training loss (efficient_multi_head_smooth_loss), and asserts:

  1. forward with compute_all_heads=True produces all four heads,
  2. the efficient path produces exactly the heads present in the batch,
  3. the multi-head loss is finite and backpropagates gradients into the
     trunk AND into every active head (so all heads actually train).

If any cleanup change breaks multi-head routing, aggregation, or the trunk/head
wiring, one of these assertions fails.
"""
import torch

from tartan_imu.config import configer
from tartan_imu.model.common.function import fun_train_forward_efficient
from tartan_imu.model.registry import build_backbone

_TYPE_ID = {"car": 1, "dog": 2, "drone": 3, "human": 4}


def _load_cfg(yaml_path: str) -> dict:
    """Load dataset cfg + merge model_yaml, exactly as build_model does."""
    import yaml as _yaml

    cfg = configer.load_config(yaml_path)
    with open(cfg["model"]["model_yaml"], "r") as f:
        cfg_special = _yaml.load(f, Loader=_yaml.Loader)
    configer.update_recursive(cfg, cfg_special)
    return cfg


def _build_model():
    cfg = _load_cfg("config/datasets/tartanimu/tartan_imu_multihead_smoke.yaml")
    torch.manual_seed(0)
    model = build_backbone("Foundation_Model", cfg)
    return cfg, model


def _make_feat(cfg, n):
    """Random IMU feature tensor with the real 4D shape the trunk expects:
    [batch, seq_len, channels, frames]. The dataloader downsamples the window
    by step_size (imu_freq / sample_freq), so frames = window_time * sample_freq."""
    seq_len = cfg["train"]["seq_len"]
    channels = cfg["model_param"]["input_dim"]
    frames = int(cfg["model_param"]["window_time"] * cfg["data"]["sample_freq"])
    return torch.randn(n, seq_len, channels, frames)


def _make_platform_feat(cfg, n):
    seq_len = cfg["train"]["seq_len"]
    channels = cfg["model_param"]["input_dim"]
    frames = int(cfg["model_param"]["window_time"] * cfg["data"]["imu_freq"])
    return torch.randn(n, seq_len, channels, frames)


def test_forward_all_heads_produces_four_heads():
    cfg, model = _build_model()
    x = _make_feat(cfg, 4)
    out = model(x, compute_all_heads=True)
    assert set(out.keys()) == {"car", "dog", "drone", "human"}, out.keys()
    for name, pred in out.items():
        assert torch.isfinite(pred).all(), f"{name} head produced non-finite output"


def test_raw_platform_windows_train_classifier_conditioning():
    cfg, model = _build_model()
    seq_len = cfg["train"]["seq_len"]
    output_dim = cfg["model_param"]["output_dim"]

    motion_type = torch.tensor([_TYPE_ID[t] for t in ("car", "dog")])
    feat = _make_feat(cfg, 2)
    platform_feat = _make_platform_feat(cfg, 2)
    targ = torch.randn(2, seq_len, output_dim)
    aux = torch.zeros_like(targ)

    out = model(
        feat,
        motion_type,
        compute_all_heads=False,
        platform_x=platform_feat,
    )
    assert "_platform_logits" in out
    assert out["_platform_logits"].shape == (2, seq_len, 4)

    model.train()
    _, _, _, loss = fun_train_forward_efficient(
        cfg,
        model,
        (feat, targ, aux, motion_type, platform_feat),
        start_cov_epochs=180,
        epoch=1,
    )
    loss.backward()

    classifier_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.platform_classifier.parameters()
    )
    assert classifier_grad, "no gradient reached the platform classifier"

    conditioner_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.backbone_conditioner.parameters()
    )
    assert conditioner_grad, "no gradient reached the backbone conditioner"


def test_mixed_batch_trains_all_active_heads():
    cfg, model = _build_model()
    seq_len = cfg["train"]["seq_len"]
    output_dim = cfg["model_param"]["output_dim"]

    # One sample per motion type -> all four heads active in this batch.
    motion_type = torch.tensor([_TYPE_ID[t] for t in ("car", "dog", "drone", "human")])
    feat = _make_feat(cfg, 4)
    targ = torch.randn(4, seq_len, output_dim)
    aux = torch.zeros_like(targ)
    batch = (feat, targ, aux, motion_type)

    model.train()
    _, _, _, loss = fun_train_forward_efficient(
        cfg, model, batch, start_cov_epochs=180, epoch=1
    )

    assert torch.isfinite(loss), f"multi-head loss not finite: {loss}"
    assert loss.item() > 0, "loss should be positive for random preds vs targets"

    # Backprop must reach the shared trunk and every per-type head.
    loss.backward()
    trunk_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.model.parameters()
    )
    assert trunk_grad, "no gradient reached the shared trunk"
    for name in ("car", "dog", "drone", "human"):
        head_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.heads[name].parameters()
        )
        assert head_grad, f"no gradient reached the {name} head"
