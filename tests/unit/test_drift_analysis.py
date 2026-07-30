"""Tests for long-horizon drift analysis (tools/drift_analysis.py).

The analyzer quantifies integration drift from a trajectory.txt
([ts, pred_xyz, gt_xyz, cov_xyz]): final position error per axis, per-axis
mean velocity bias, and the bias*duration vs measured-drift agreement that
distinguishes systematic-bias drift from random noise.
"""
import numpy as np

from tools.drift_analysis import analyze_trajectory_drift, DriftReport


def _write_traj(tmp_path, ts, pred, gt):
    cov = np.zeros_like(pred)
    data = np.concatenate([ts.reshape(-1, 1), pred, gt, cov], axis=1)
    p = tmp_path / "trajectory.txt"
    np.savetxt(str(p), data, delimiter=",")
    return str(p)


def test_constant_z_velocity_bias_produces_linear_drift(tmp_path):
    # GT stays put; pred drifts in Z at a constant 0.04 m/s over 100s.
    n = 1001
    ts = np.linspace(0, 100, n)
    gt = np.zeros((n, 3))
    bias = 0.04  # m/s in Z
    pred = np.zeros((n, 3))
    pred[:, 2] = bias * ts  # integral of a constant velocity bias
    path = _write_traj(tmp_path, ts, pred, gt)

    rep = analyze_trajectory_drift(path)
    assert isinstance(rep, DriftReport)
    # Final drift is dominated by Z and ~ bias * duration.
    assert abs(rep.final_error_xyz[2] - bias * 100) < 1e-6
    assert rep.dominant_axis == "Z"
    # Velocity bias recovered.
    assert abs(rep.vel_bias_xyz[2] - bias) < 1e-3
    # bias*duration explains the measured drift -> systematic, not random.
    assert rep.bias_explains_fraction > 0.9


def test_no_drift_when_pred_matches_gt(tmp_path):
    n = 501
    ts = np.linspace(0, 50, n)
    gt = np.cumsum(np.ones((n, 3)) * 0.01, axis=0)
    path = _write_traj(tmp_path, ts, gt.copy(), gt)
    rep = analyze_trajectory_drift(path)
    assert rep.final_error_norm < 1e-6
    assert rep.vel_bias_norm < 1e-6


def test_duration_and_path_length(tmp_path):
    n = 101
    ts = np.linspace(0, 10, n)
    gt = np.zeros((n, 3))
    gt[:, 0] = np.linspace(0, 5, n)  # walk 5m in X
    path = _write_traj(tmp_path, ts, gt.copy(), gt)
    rep = analyze_trajectory_drift(path)
    assert abs(rep.duration_s - 10.0) < 1e-6
    assert abs(rep.gt_path_length - 5.0) < 1e-3
