"""Stage 3 (masked-attention) feature assembly in HumanoidPostProcessedSequence.

Stage 3's TransformerOdoModel expects input_dim=64: IMU(6) + 29 joints x 2,
where the joint block is reshaped via `view(B*T, 29, 2)` -> each joint carries
(q, dq). The may23 NPZ only stores joint_angles [T,29,1] (angle q, no velocity),
so dq must be derived by finite-difference of q: dq[t]=(q[t]-q[t-1])/dt, dq[0]=0.

Contract this file pins down:
  stage 1 -> 6   (pelvis gyro 3 + pelvis acc 3)              [unchanged]
  stage 2 -> 35  (+ 29 joint angles)                          [unchanged]
  stage 3 -> 64  (IMU 6 + 29 joints x (q, dq) interleaved)    [new]
"""
import os
import tempfile

import numpy as np
import pytest

from tartan_imu.dataloader.dataset_HumanoidPostProcessed import HumanoidPostProcessedSequence

IMU_FREQ = 20.0
DT = 1.0 / IMU_FREQ
T = 400  # must be >= imu_freq*10 = 200


def _build_synth_npz(path):
    """Minimal valid Post_Processed NPZ with 3 IMUs (pelvis at idx 1) and 29 joints.

    Timestamps use a realistic absolute-ns epoch (~1.78e18, like may23 data) so
    the loader's ns->s conversion path (abs > 1e12) is exercised; otherwise dt
    would be left in nanoseconds and dq would be off by 1e9.
    """
    epoch_ns = 1_778_000_000_000_000_000  # ~ real may23 epoch
    ts = (epoch_ns + np.arange(T) * DT * 1e9).astype(np.int64)  # ns, strictly increasing
    time_ns = np.tile(ts.reshape(1, T, 1, 1), (1, 1, 3, 1))

    rng = np.random.RandomState(0)
    imu_acc = rng.randn(1, T, 3, 3).astype(np.float64)
    imu_gyro = rng.randn(1, T, 3, 3).astype(np.float64)
    imu_velocity = rng.randn(1, T, 3, 3).astype(np.float32)

    poses = np.zeros((1, T, 3, 7), dtype=np.float32)
    poses[..., :3] = rng.randn(1, T, 3, 3).astype(np.float32)  # positions
    poses[..., 6] = 1.0  # unit quaternion (xyzw -> w=1)

    # Deterministic, smoothly varying joint angles so finite-difference dq is meaningful.
    t_axis = np.arange(T).reshape(T, 1)
    joint_idx = np.arange(29).reshape(1, 29)
    q = np.sin(0.01 * t_axis + 0.1 * joint_idx).astype(np.float32)  # [T, 29]
    joint_angles = q.reshape(1, T, 29, 1)

    np.savez(
        path,
        time_ns=time_ns,
        imu_acc_m_s2=imu_acc,
        imu_gyro_rad_s=imu_gyro,
        imu_poses=poses,
        imu_velocity=imu_velocity,
        joint_angles=joint_angles,
        meta_imu_names=np.array(["head", "pelvis", "foot"], dtype=object),
    )
    return path, q


@pytest.fixture(scope="module")
def synth():
    tmp = tempfile.mkdtemp(prefix="stage3_")
    path, q = _build_synth_npz(os.path.join(tmp, "synth_may23.npz"))
    return path, q


def _seq(path, stage):
    return HumanoidPostProcessedSequence(
        data_path=path, imu_freq=IMU_FREQ, window_size=20,
        verbose=False, use_local_coord=True, mode="test", stage=stage,
    )


def test_stage1_unchanged_6dim(synth):
    feat = _seq(synth[0], 1).get_feature()
    assert feat.shape[1] == 6


def test_stage2_unchanged_35dim(synth):
    feat = _seq(synth[0], 2).get_feature()
    assert feat.shape[1] == 35


def test_stage3_features_are_64dim(synth):
    seq = _seq(synth[0], 3)
    assert seq.valid
    feat = seq.get_feature()
    assert feat.shape[1] == 64, f"stage 3 must be 64-dim, got {feat.shape[1]}"
    assert np.isfinite(feat).all()


def test_stage3_joint_block_is_q_dq_interleaved_per_joint(synth):
    """The 58 joint dims must reshape to [29, 2] = (q, dq) per joint, matching the
    model's `view(B*T, 29, 2)`. So joints[:, :, 0] == q (up to the [:-1] trim)."""
    path, q = synth
    seq = _seq(path, 3)
    feat = seq.get_feature()  # [N, 64], N = T-1 (loader trims last frame)
    joints = feat[:, 6:].reshape(feat.shape[0], 29, 2)
    n = joints.shape[0]
    np.testing.assert_allclose(joints[:, :, 0], q[:n], rtol=1e-5, atol=1e-5)


def test_stage3_dq_is_finite_difference_of_q(synth):
    path, q = synth
    seq = _seq(path, 3)
    feat = seq.get_feature()
    joints = feat[:, 6:].reshape(feat.shape[0], 29, 2)
    dq = joints[:, :, 1]
    n = dq.shape[0]
    expected = np.zeros_like(q[:n])
    expected[1:] = (q[1:n] - q[: n - 1]) / DT
    np.testing.assert_allclose(dq, expected, rtol=1e-4, atol=1e-4)


def test_stage3_zero_dq_zeros_velocity_channel_but_keeps_64dim(synth):
    """zero_dq=True keeps the 64-dim shape (so the model is unchanged) but the dq
    channel is all zeros -- an ablation isolating whether the finite-difference dq
    noise is what hurts Stage 3."""
    path, q = synth
    seq = HumanoidPostProcessedSequence(
        data_path=path, imu_freq=IMU_FREQ, window_size=20,
        verbose=False, use_local_coord=True, mode="test", stage=3, zero_dq=True,
    )
    assert seq.valid
    feat = seq.get_feature()
    assert feat.shape[1] == 64
    joints = feat[:, 6:].reshape(feat.shape[0], 29, 2)
    n = joints.shape[0]
    # q preserved, dq forced to 0
    np.testing.assert_allclose(joints[:, :, 0], q[:n], rtol=1e-5, atol=1e-5)
    assert np.all(joints[:, :, 1] == 0.0)
