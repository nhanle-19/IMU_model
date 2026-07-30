"""Characterization tests for the dataset_Humanoid / dataset_AirLab refactor
(gravity_aligner removal + dead-code/bug cleanup).

These assert the loaders produce identical (features, targets, aux) on synthetic
data for every path that worked before the refactor, and that the AirLab
local-coordinate path no longer crashes when window_size != 200 (former Bug #2:
hardcoded `pos_gt[:-200]`).
"""
import logging
import os
import tempfile

import numpy as np
import pytest

from tests.unit import _dataloader_fixtures as F

logging.disable(logging.CRITICAL)


@pytest.fixture(scope="module")
def synth_paths():
    tmp = tempfile.mkdtemp(prefix="dl_char_")
    return {
        "humanoid": F.build_humanoid_npz(os.path.join(tmp, "human_synth.npz")),
        "airlab": F.build_airlab_npz(os.path.join(tmp, "human_synth_airlab.npz")),
    }


# Expected sums captured from the pre-refactor implementation. Any change to the
# loaders' numeric output on these paths will break these assertions.
EXPECTED = {
    "humanoid__global__features": (1499, 6, 1670.706964),
    "humanoid__global__targets": (1449, 3, 14.484961),
    "humanoid__global__aux": (1499, 12, 15779.445388),
    "humanoid__local__features": (1499, 6, 14740.938395),
    "humanoid__local__targets": (1449, 3, 37.839061),
    "humanoid__local__aux": (1499, 12, 15779.445388),
    "airlab__global__features": (1499, 6, 1575.139668),
    "airlab__global__targets": (1449, 3, 29.943809),
    "airlab__global__aux": (1499, 12, 15162.074495),
    "airlab__local_ws200__features": (1499, 6, 14641.518220),
    "airlab__local_ws200__targets": (1299, 3, 37.798268),
    "airlab__local_ws200__aux": (1499, 12, 15162.074495),
}


def _check(out, prefix):
    for field, arr in out.items():
        rows, cols, total = EXPECTED[f"{prefix}__{field}"]
        assert arr.shape == (rows, cols), f"{prefix}__{field} shape {arr.shape}"
        assert np.isfinite(arr).all(), f"{prefix}__{field} has non-finite values"
        assert abs(arr.sum() - total) < 1e-4, (
            f"{prefix}__{field} sum {arr.sum():.6f} != expected {total:.6f}"
        )


def test_humanoid_global_unchanged(synth_paths):
    _check(F.run_humanoid_sequence(synth_paths["humanoid"], False, window_size=50),
           "humanoid__global")


def test_humanoid_local_unchanged(synth_paths):
    _check(F.run_humanoid_sequence(synth_paths["humanoid"], True, window_size=50),
           "humanoid__local")


def test_airlab_global_unchanged(synth_paths):
    _check(F.run_airlab_sequence(synth_paths["airlab"], False, window_size=50),
           "airlab__global")


def test_airlab_local_ws200_unchanged(synth_paths):
    # The only AirLab local-coord case that worked before the refactor.
    _check(F.run_airlab_sequence(synth_paths["airlab"], True, window_size=200),
           "airlab__local_ws200")


def test_airlab_local_non200_no_longer_crashes(synth_paths):
    # Before the refactor this raised ValueError from the hardcoded pos_gt[:-200].
    out = F.run_airlab_sequence(synth_paths["airlab"], True, window_size=50)
    assert out["targets"].shape[1] == 3
    assert np.isfinite(out["targets"]).all()
