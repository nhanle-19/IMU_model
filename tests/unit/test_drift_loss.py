import torch

from tartan_imu.model.common.losses import (
    sequence_velocity_bias_loss,
    single_head_mask_loss,
    single_head_velocity_loss,
)


def test_sequence_velocity_bias_loss_penalizes_signed_sequence_bias():
    pred = torch.ones(2, 10, 3)
    targ = torch.zeros_like(pred)
    mask = torch.tensor([1, 0], dtype=torch.float32)

    loss = sequence_velocity_bias_loss(pred, targ, mask)

    assert torch.allclose(loss, torch.tensor(1.0))


def test_single_head_mask_loss_unchanged_when_drift_disabled():
    torch.manual_seed(0)
    pred = torch.randn(4, 10, 3)
    targ = torch.randn(4, 10, 3)
    cov = torch.zeros_like(pred)
    mask = torch.ones(4)

    loss = single_head_mask_loss(
        pred,
        cov,
        targ,
        epoch=1,
        mask=mask,
        start_cov_epoch=180,
        use_local_coord=True,
        drift_loss_cfg={"enabled": False, "weight": 2.0},
    )
    expected = single_head_velocity_loss(pred, cov, targ)["loss"].mean()

    assert torch.allclose(loss, expected, atol=1e-6)


def test_single_head_mask_loss_adds_drift_penalty_when_enabled():
    pred = torch.ones(2, 10, 3)
    targ = torch.zeros_like(pred)
    cov = torch.zeros_like(pred)
    mask = torch.ones(2)

    base = single_head_mask_loss(
        pred,
        cov,
        targ,
        epoch=1,
        mask=mask,
        start_cov_epoch=180,
        use_local_coord=True,
    )
    with_drift = single_head_mask_loss(
        pred,
        cov,
        targ,
        epoch=1,
        mask=mask,
        start_cov_epoch=180,
        use_local_coord=True,
        drift_loss_cfg={"enabled": True, "weight": 2.0},
    )

    assert torch.allclose(with_drift, base + torch.tensor(2.0))
