"""Locks partition_data + calculate_velocity_from_poses behaviour across the
extraction into dataloader/_common.py. Values captured from the pre-extraction
implementation; any numeric change breaks these."""
import numpy as np


def _make_velocity_inputs():
    n = 50
    ts = np.arange(n) * 0.05
    pos = np.cumsum(np.random.RandomState(0).randn(n, 3) * 0.01, axis=0)
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))  # identity xyzw
    return ts, pos, quat


def test_calculate_velocity_matches_reference():
    from tartan_imu.dataloader._common import calculate_velocity_from_poses
    ts, pos, quat = _make_velocity_inputs()
    vg, vb = calculate_velocity_from_poses(ts, pos, quat)
    assert vg.shape == (49, 3) and vb.shape == (49, 3)
    np.testing.assert_allclose(vg, vb, atol=1e-9)  # identity rotation -> body==global
    np.testing.assert_allclose(vg[0], (pos[1] - pos[0]) / 0.05, atol=1e-6)


def test_partition_data_split_is_deterministic():
    from tartan_imu.dataloader._common import partition_data
    index_map = [[[i, 0, 4]] for i in range(20)]
    valid_samples = {i: 1 for i in range(20)}
    train, val = partition_data(
        list(index_map), valid_samples=valid_samples, valid_all_samples=20,
        valuation_rate=0.1, data_rate=1.0, shuffle=False,
    )
    assert len(val) == 2
    # Canonical partition_data excludes the break-index sequence from both
    # splits, so the total is one short of the flattened input (locks the
    # pre-extraction behaviour, not an invariant we want to "fix").
    flat_total = len([x for sub in index_map for x in sub])
    assert len(train) + len(val) == flat_total - 1 == 19


def test_partition_data_no_overlap():
    from tartan_imu.dataloader._common import partition_data
    index_map = [[[i, 0, 4]] for i in range(30)]
    valid_samples = {i: 1 for i in range(30)}
    train, val = partition_data(
        list(index_map), valid_samples=valid_samples, valid_all_samples=30,
        valuation_rate=0.2, data_rate=1.0, shuffle=False,
    )
    train_ids = {t[0] for t in train}
    val_ids = {v[0] for v in val}
    assert train_ids.isdisjoint(val_ids)
