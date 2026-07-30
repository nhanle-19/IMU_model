"""Multi-head loss must weight each head by ITS OWN samples, not batch share.

Bug: single_head_mask_loss masked the per-element loss then took torch.mean
over ALL elements (B*T*3), so a head that owns few samples in a mixed batch
got its loss divided by the whole batch size -- diluting its gradient in
proportion to how under-represented its motion type was. With multi-head
training (car/human/dog/drone) this systematically down-weights rare types.

Fix: normalize a head's masked loss by its active element count. Two invariants:
1. Baseline preserved: an all-True mask (single-head human training) must give
   exactly torch.mean(loss) -- i.e. the verified stage1/2/3 ATE is unchanged.
2. Composition-independent: a head's loss over a subset of samples must equal
   the loss computed on just those samples (all-True mask).
"""
import torch

from tartan_imu.model.common.losses import single_head_mask_loss, single_head_velocity_loss

_KW = dict(epoch=1, start_cov_epoch=180, use_local_coord=True)


def test_all_true_mask_equals_plain_mean_baseline():
    # Single-head training: all samples active -> must equal plain mean.
    torch.manual_seed(1)
    pred = torch.randn(5, 10, 3)
    targ = torch.randn(5, 10, 3)
    cov = torch.zeros_like(pred)
    out = single_head_mask_loss(pred, cov, targ, mask=torch.ones(5), **_KW)
    expected = single_head_velocity_loss(pred, cov, targ)["loss"].mean()
    assert torch.allclose(out, expected, atol=1e-6), (
        f"baseline changed: {out.item():.6f} != {expected.item():.6f}"
    )


def test_masked_loss_independent_of_batch_composition():
    # A head's masked loss over the first 3 of 8 samples must equal the loss
    # computed on just those 3 samples with an all-True mask.
    torch.manual_seed(0)
    pred = torch.randn(8, 10, 3)
    targ = torch.randn(8, 10, 3)
    cov = torch.zeros_like(pred)
    mask = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.float32)

    full = single_head_mask_loss(pred, cov, targ, mask=mask, **_KW)
    sub = single_head_mask_loss(pred[:3], cov[:3], targ[:3], mask=torch.ones(3), **_KW)
    assert torch.allclose(full, sub, atol=1e-6), (
        f"head loss depends on batch composition: {full.item():.6f} != {sub.item():.6f}"
    )
