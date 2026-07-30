"""Cache key must include every param that changes the cached output (R2).

The cached loader keyed only on (source, mtime, size, stage). But the produced
features/targets also depend on use_local_coord (velocity vs displacement
targets), window_size (interval), and stage3_zero_dq (whether the dq channel is
zeroed). Two runs differing only in those would silently share a stale cache
entry and load wrong-variant data. The key must separate them.
"""
import os
import tempfile

from tartan_imu.dataloader.dataset_HumanoidPostProcessedCached import _cache_key


def _f():
    fd, p = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    return p


def test_same_variant_same_key():
    p = _f()
    k1 = _cache_key(p, stage=3, use_local_coord=True, window_size=10, zero_dq=False)
    k2 = _cache_key(p, stage=3, use_local_coord=True, window_size=10, zero_dq=False)
    assert k1 == k2


def test_zero_dq_changes_key():
    p = _f()
    k_off = _cache_key(p, stage=3, use_local_coord=True, window_size=10, zero_dq=False)
    k_on = _cache_key(p, stage=3, use_local_coord=True, window_size=10, zero_dq=True)
    assert k_off != k_on


def test_use_local_coord_changes_key():
    p = _f()
    k_t = _cache_key(p, stage=2, use_local_coord=True, window_size=10, zero_dq=False)
    k_f = _cache_key(p, stage=2, use_local_coord=False, window_size=10, zero_dq=False)
    assert k_t != k_f


def test_window_size_changes_key():
    p = _f()
    k10 = _cache_key(p, stage=2, use_local_coord=True, window_size=10, zero_dq=False)
    k20 = _cache_key(p, stage=2, use_local_coord=True, window_size=20, zero_dq=False)
    assert k10 != k20


def test_stage_changes_key():
    p = _f()
    assert _cache_key(p, stage=1, use_local_coord=True, window_size=10, zero_dq=False) \
        != _cache_key(p, stage=2, use_local_coord=True, window_size=10, zero_dq=False)
