"""Train/eval loss must be the SAME metric so the curves are comparable.

Bug R1: training used L1 (weight=20) via single_head_velocity_loss, but the
eval path (get_sequence_smooth_loss) used plain MSE before start_cov. That made
test_loss incomparable to train_loss (test_loss sat ~0.095 across all stages
regardless of training signal).

The eval velocity loss (pre-covariance) must equal the training velocity loss
on identical inputs.
"""
import torch

from tartan_imu.model.common.losses import get_sequence_smooth_loss, single_head_velocity_loss


def _train_vel_loss(pred, targ):
    out = single_head_velocity_loss(pred, torch.zeros_like(pred), targ)
    return out["loss"].mean()


def test_eval_loss_equals_train_l1_loss():
    torch.manual_seed(0)
    pred = torch.randn(8, 3)
    targ = torch.randn(8, 3)
    # epoch <= start_cov_epoch -> velocity (non-covariance) branch
    eval_loss = get_sequence_smooth_loss(
        pred, torch.zeros_like(pred), targ, epoch=1, start_cov_epoch=180
    )
    train_loss = _train_vel_loss(pred, targ)
    assert torch.allclose(eval_loss, train_loss, atol=1e-6), (
        f"eval {eval_loss.item():.6f} != train {train_loss.item():.6f}"
    )
