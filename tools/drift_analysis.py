"""Long-horizon drift analysis for inertial-odometry trajectories.

Quantifies *integration drift* from a saved trajectory.txt with columns
[ts, pred_xyz, gt_xyz, cov_xyz]. Drift from a small constant velocity bias
grows LINEARLY with time (drift ~ bias * duration), whereas random noise grows
~sqrt(t); the `bias_explains_fraction` field separates the two.

Motivated by the Stage 2 baseline: Traj_2 (470s) drifted 17.7m, 99% in Z, with
a -0.037 m/s Z velocity bias whose integral (0.037*470 ~ 17.6m) almost exactly
matched the measured drift.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DriftReport:
    duration_s: float
    gt_path_length: float
    final_error_xyz: np.ndarray      # pred - gt at the last sample, per axis
    final_error_norm: float
    dominant_axis: str               # "X" | "Y" | "Z" — axis with largest |final error|
    vel_bias_xyz: np.ndarray         # mean(pred_vel - gt_vel) per axis
    vel_bias_norm: float
    bias_explains_fraction: float    # |bias|*duration / measured drift  (clipped to [0,1] in spirit)
    mid_vs_final_ratio: float        # |err(mid)| / |err(final)| — ~0.5 => linear accumulation


def analyze_trajectory_drift(traj_txt_path: str) -> DriftReport:
    d = np.loadtxt(traj_txt_path, delimiter=",")
    if d.ndim == 1:
        d = d.reshape(1, -1)
    ts = d[:, 0]
    # ts may be in nanoseconds (epoch) or seconds; normalize to seconds span.
    span = ts[-1] - ts[0]
    if span > 1e7:  # clearly nanoseconds
        ts = (ts - ts[0]) / 1e9
    else:
        ts = ts - ts[0]
    pred = d[:, 1:4]
    gt = d[:, 4:7]

    duration = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    gt_path = float(np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1))) if len(gt) > 1 else 0.0

    err = pred - gt
    final_err = err[-1]
    final_norm = float(np.linalg.norm(final_err))
    axis_idx = int(np.argmax(np.abs(final_err)))
    dominant = "XYZ"[axis_idx]

    if len(ts) > 1:
        dt = np.diff(ts)
        dt[dt == 0] = np.finfo(float).eps
        vp = np.diff(pred, axis=0) / dt[:, None]
        vg = np.diff(gt, axis=0) / dt[:, None]
        vbias = np.mean(vp - vg, axis=0)
    else:
        vbias = np.zeros(3)
    vbias_norm = float(np.linalg.norm(vbias))

    # How much of the measured drift does a constant velocity bias explain?
    expected_drift = vbias_norm * duration
    bias_frac = float(expected_drift / final_norm) if final_norm > 1e-9 else 0.0

    mid = len(err) // 2
    mid_ratio = float(np.linalg.norm(err[mid]) / final_norm) if final_norm > 1e-9 else 0.0

    return DriftReport(
        duration_s=duration,
        gt_path_length=gt_path,
        final_error_xyz=final_err,
        final_error_norm=final_norm,
        dominant_axis=dominant,
        vel_bias_xyz=vbias,
        vel_bias_norm=vbias_norm,
        bias_explains_fraction=bias_frac,
        mid_vs_final_ratio=mid_ratio,
    )
