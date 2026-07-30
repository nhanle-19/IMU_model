"""TartanIMU / IMUNet Challenge — Kaggle custom scoring metric: macro-averaged
20 m-segment ATE.

Primary metric: the macro-average, over the four platforms, of the mean 20 m-segment
Absolute Trajectory Error (lower is better).

Per test trajectory the ground-truth path is cut into consecutive segments of ~20 m of
travelled distance. Within each segment the participant's per-window body-frame
velocities are integrated into a path (rotated to the world frame with ground-truth
orientation, used ONLY in this scoring step, never as a model input), SE(3)-aligned to
the ground-truth path over that segment (Umeyama, rotation + translation, no scale), and
the RMS position error is the segment's ATE. A trajectory's ATE is the mean over its
segments, a platform's is the mean over its trajectories, and the reported score is the
equal-weight mean of the four per-platform values, so no platform dominates.

Why segments rather than whole trajectories: a whole-trajectory ATE rewards predicting
nothing. An all-zero submission integrates to a single point, whose aligned error is just
the ground-truth path's radius of gyration — on spatially compact trajectories (a person
pacing indoors, a drone circling) that beats an honest model which accumulates drift.
Fixing the segment length to 20 m of travel removes that degenerate strategy: every
scored segment covers the same ground, so standing still is never cheap.

Segment length is 20 m to keep the alignment well conditioned on every platform. The
platforms differ by an order of magnitude in speed (~0.2 m/s human to ~2 m/s drone), and
at 5 m the fastest one is down to ~4 one-second windows per segment, where a nearly
straight flight leaves the Umeyama rotation poorly constrained. At 20 m every platform
keeps a double-digit median window count per segment.

Submission columns: window_id, vx, vy, vz                       (one body-frame velocity per window)
Solution columns  : window_id, traj_id, win_idx, platform,
                    qx, qy, qz, qw, gx, gy, gz, dt, Usage        (integration ingredients, host-side)
"""
import numpy as np
import pandas as pd

SEGMENT_LENGTH_M = 20.0
MIN_SEGMENT_POINTS = 3          # Umeyama needs three points to pin down a rotation


class ParticipantVisibleError(Exception):
    """Message shown to the participant on the leaderboard."""


def _quat_to_R(q):
    """(M,4) quaternion (x, y, z, w) -> (M,3,3) rotation matrices."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    n[n == 0] = 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _umeyama_align(P, Q):
    """Least-squares SE(3) (R, t), no scale, aligning P onto Q. P,Q are (M,3)."""
    muP, muQ = P.mean(0), Q.mean(0)
    Pc, Qc = P - muP, Q - muQ
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = muQ - R @ muP
    return R, t


def _segment_bounds(Q, seg_len=SEGMENT_LENGTH_M, min_pts=MIN_SEGMENT_POINTS):
    """Cut a ground-truth path into ~seg_len-metre pieces.

    Returns (start, end) index pairs, end inclusive. Boundaries fall where the cumulative
    travelled distance crosses another seg_len; the tail is kept as its own segment. Any
    piece with fewer than min_pts points is merged into its neighbour, so every returned
    segment can support an SE(3) alignment. A trajectory shorter than seg_len yields one
    segment covering all of it.
    """
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    cuts, reached = [], 0.0
    for i, c in enumerate(cum):
        if c >= reached + seg_len:
            cuts.append(i)
            reached = c
    if not cuts or cuts[-1] != len(Q) - 1:
        cuts.append(len(Q) - 1)

    segs, start = [], 0
    for end in cuts:
        if end > start:
            segs.append([start, end])
            start = end
    if not segs:
        return []

    merged = [segs[0]]
    for s, e in segs[1:]:
        if e - s + 1 < min_pts:                       # short tail -> extend the previous
            merged[-1][1] = e
        else:
            merged.append([s, e])
    if merged[0][1] - merged[0][0] + 1 < min_pts and len(merged) > 1:
        merged[1][0] = merged[0][0]                   # short head -> fold into the next
        merged.pop(0)
    return [(s, e) for s, e in merged if e - s + 1 >= min_pts]


def _ate_segment(R, v, dt, Q):
    """RMS position error of one aligned segment."""
    disp = np.einsum("mij,mj->mi", R, v) * dt
    P = np.cumsum(disp, axis=0)
    Rr, t = _umeyama_align(P, Q)
    return float(np.sqrt(np.mean(np.sum((P @ Rr.T + t - Q) ** 2, axis=1))))


def _ate_traj(g):
    """Mean 20 m-segment ATE for one trajectory dataframe (already time-ordered)."""
    Q = g[["gx", "gy", "gz"]].to_numpy(float)
    bounds = _segment_bounds(Q)
    if not bounds:
        return np.nan
    R = _quat_to_R(g[["qx", "qy", "qz", "qw"]].to_numpy(float))
    v = g[["vx_pred", "vy_pred", "vz_pred"]].to_numpy(float)
    dt = g["dt"].to_numpy(float)[:, None]
    return float(np.mean([_ate_segment(R[s:e + 1], v[s:e + 1], dt[s:e + 1], Q[s:e + 1])
                          for s, e in bounds]))


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """Macro-averaged 20 m-segment Absolute Trajectory Error (ATE), lower is better.

    Cuts each test trajectory into ~20 m pieces of travelled ground truth, integrates the
    predicted per-window body-frame velocities within each piece using ground-truth
    orientation (scoring-only), SE(3)-aligns it to the ground-truth path (Umeyama, no
    scale), and takes the RMS position error. Segment errors are averaged per trajectory,
    trajectories per platform, and the four platform values equally.

    Args:
        solution: window_id, traj_id, win_idx, platform, qx, qy, qz, qw, gx, gy, gz, dt
            (Usage already removed by Kaggle).
        submission: window_id, vx, vy, vz.
        row_id_column_name: name of the id column ("window_id").

    Returns:
        Macro-averaged 20 m-segment ATE in metres as a finite float.
    """
    for col in (row_id_column_name, "vx", "vy", "vz"):
        if col not in submission.columns:
            raise ParticipantVisibleError(f"Submission must contain column '{col}'.")

    sub = submission[[row_id_column_name, "vx", "vy", "vz"]].copy()
    for c in ("vx", "vy", "vz"):
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    if sub[["vx", "vy", "vz"]].isnull().any().any():
        raise ParticipantVisibleError("Submission contains non-numeric or missing vx/vy/vz values.")
    if not np.isfinite(sub[["vx", "vy", "vz"]].to_numpy(float)).all():
        raise ParticipantVisibleError("Submission contains infinite vx/vy/vz values.")

    need = {"traj_id", "win_idx", "platform", "qx", "qy", "qz", "qw", "gx", "gy", "gz", "dt"}
    if not need.issubset(solution.columns):
        raise ParticipantVisibleError("Solution file is missing required integration columns.")

    m = solution.merge(sub.rename(columns={"vx": "vx_pred", "vy": "vy_pred", "vz": "vz_pred"}),
                       on=row_id_column_name, how="left")
    if m[["vx_pred", "vy_pred", "vz_pred"]].isnull().any().any():
        raise ParticipantVisibleError("Submission is missing one or more required window_id rows.")

    plat_ate = {}
    for _, g in m.groupby("traj_id", sort=False):
        g = g.sort_values("win_idx")
        ate = _ate_traj(g)
        if np.isfinite(ate):
            plat_ate.setdefault(g["platform"].iloc[0], []).append(ate)

    if not plat_ate:
        raise ParticipantVisibleError("No scorable trajectories found.")
    result = float(np.mean([np.mean(v) for v in plat_ate.values()]))
    if not np.isfinite(result):
        raise ParticipantVisibleError("Score is not finite; check for extreme velocity values.")
    return result
