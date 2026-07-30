"""Stage-aware feature assembly in HumanoidPostProcessedSequence.

Bug found by the Stage 1 smoke run: the loader unconditionally concatenated
joint_angles, always producing 35-dim features (IMU 6 + joints 29). Stage 1 is
IMU-only and the model builds bn_input for 6 dims, so training crashed with
"running_mean should contain 35 elements not 6". The loader must gate the
joint_angles by stage:
  stage 1 -> 6  (pelvis gyro 3 + pelvis acc 3)
  stage 2 -> 35 (+ joint_angles 29)
"""
import os

import numpy as np

from tests.conftest import NFS_DATA_DIR, requires_nfs
from tartan_imu.dataloader.dataset_HumanoidPostProcessed import HumanoidPostProcessedSequence


def _a_may23_file():
    files = sorted(
        f for f in os.listdir(NFS_DATA_DIR)
        if f.endswith(".npz") and "may23-2026" in f
    )
    return os.path.join(NFS_DATA_DIR, files[0])


@requires_nfs
def test_stage1_features_are_imu_only_6dim():
    seq = HumanoidPostProcessedSequence(
        data_path=_a_may23_file(),
        imu_freq=200.0,
        window_size=20,
        verbose=False,
        use_local_coord=True,
        mode="test",
        stage=1,
    )
    assert seq.valid
    feat = seq.get_feature()
    assert feat.shape[1] == 6, f"stage 1 must be IMU-only 6-dim, got {feat.shape[1]}"
    assert np.isfinite(feat).all()


@requires_nfs
def test_stage2_features_include_joints_35dim():
    seq = HumanoidPostProcessedSequence(
        data_path=_a_may23_file(),
        imu_freq=200.0,
        window_size=20,
        verbose=False,
        use_local_coord=True,
        mode="test",
        stage=2,
    )
    assert seq.valid
    feat = seq.get_feature()
    assert feat.shape[1] == 35, f"stage 2 must be IMU+joints 35-dim, got {feat.shape[1]}"
    assert np.isfinite(feat).all()


@requires_nfs
def test_default_stage_is_backward_compatible_35dim():
    """No stage arg -> preserve the historical 35-dim behavior (stage 2 default),
    so existing stage-2 configs keep working unchanged."""
    seq = HumanoidPostProcessedSequence(
        data_path=_a_may23_file(),
        imu_freq=200.0,
        window_size=20,
        verbose=False,
        use_local_coord=True,
        mode="test",
    )
    assert seq.valid
    assert seq.get_feature().shape[1] == 35
