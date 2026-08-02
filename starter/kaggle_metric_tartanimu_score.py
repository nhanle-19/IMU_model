"""TartanIMU / IMUNet Challenge — Kaggle custom scoring metric: **TartanIMU Score**.

    TartanIMU Score = 0.6 * (macro AVE / AVE_REF)  +  0.4 * (macro ATE20 / ATE_REF)

Lower is better. The score is dimensionless: each component is divided by the value that
the all-zero submission attains on the full test set, so the two terms — one in m/s, one
in metres — are put on a common scale before they are combined. Two consequences worth
stating plainly:

* **The all-zero submission scores exactly 1.000.** Anything above 1.0 is worse than
  submitting nothing at all.
* **The declared 0.6 / 0.4 weights are the weights that actually act.** Without
  normalisation a raw `0.6*AVE + 0.4*ATE` would be dominated by ATE, because ATE spans
  roughly 0.15 - 3.5 m while AVE spans only 0.0 - 0.75 m/s; the nominal 60 % term would
  carry about 25 % of the discriminative power. After normalisation the two terms
  contribute 59 % / 41 % of the spread across our reference submissions, matching intent.

The two components measure genuinely different failure modes, which is why both are kept:

* **AVE — Absolute Velocity Error** (m/s). The mean, over a trajectory's windows, of the
  Euclidean norm ||v_pred - v_gt||. This is instantaneous, per-window accuracy: it
  rewards getting the speed and the direction of motion right at each moment, and it is
  the quantity a downstream state estimator actually consumes.
* **ATE20 — 20 m-segment Absolute Trajectory Error** (m). The predicted body-frame
  velocities are integrated into a path and compared with the ground-truth path over
  fixed 20 m pieces after an SE(3) alignment. This is temporal consistency: it punishes
  correlated bias and drift that a per-window average error hides.

A model can be good at one and bad at the other. Low AVE with high ATE means small but
systematically signed errors that accumulate; low ATE with high AVE means noisy
predictions whose errors happen to cancel. Requiring both is what makes the benchmark
about deployable inertial odometry rather than about either statistic alone.

Both components use the same hierarchical averaging: per window -> mean over a
trajectory's windows/segments -> mean over a platform's trajectories -> equal-weight mean
over the four platforms (macro-average), so no platform dominates and each contributes
exactly 25 % of each component.

ATE20 details. Per test trajectory the ground-truth path is cut into consecutive segments
of ~20 m of travelled distance. Within each segment the participant's per-window
body-frame velocities are rotated to the world frame with ground-truth orientation (used
ONLY in this scoring step, never as a model input, because global heading is not
observable from an IMU alone), multiplied by the window duration, accumulated into a path,
SE(3)-aligned to the ground-truth path over that segment (Umeyama, rotation + translation,
no scale), and the RMS position error is the segment's ATE.

Why segments rather than whole trajectories: a whole-trajectory ATE rewards predicting
nothing. An all-zero submission integrates to a single point, whose aligned error is just
the ground-truth path's radius of gyration — on spatially compact trajectories (a person
pacing indoors, a drone circling) that beats an honest model which accumulates drift.
Fixing the segment length to 20 m of travel removes that degenerate strategy: every scored
segment covers the same ground, so standing still is never cheap. The length is 20 m
rather than 5 m to keep the alignment well conditioned on every platform: the platforms
differ by an order of magnitude in speed, and at 5 m the fastest one is down to ~4
one-second windows per segment, where a nearly straight flight leaves the Umeyama rotation
poorly constrained.

Reference scores (organizers' own submissions, full test set):

    submission                       ATE20     AVE   TartanIMU Score
    ground-truth velocities (floor)  0.151   0.000            0.019
    organizers' unified baseline     1.261   0.461            0.538
    all-zeros (sample_submission)    3.116   0.736            1.000
    per-platform mean velocity       3.496   0.749            1.060

The metric is shrink-proof: scaling the baseline's predictions by any factor away from 1.0
strictly worsens the score (verified over 0.0 - 1.5), so submitting deliberately damped
velocities is never advantageous.

Submission columns: window_id, vx, vy, vz            (one body-frame velocity per window)
Solution columns  : window_id, traj_id, win_idx, platform, qx, qy, qz, qw,
                    gx, gy, gz, dt, vx_gt, vy_gt, vz_gt, Usage
"""
import numpy as np
import pandas as pd

SEGMENT_LENGTH_M = 20.0
MIN_SEGMENT_POINTS = 3          # Umeyama needs three points to pin down a rotation

W_AVE = 0.6                     # weight of the velocity-accuracy term
W_ATE = 0.4                     # weight of the trajectory-consistency term

# Value each component attains for the all-zero submission on the full test set. Fixed,
# published constants — they put m/s and m on a common scale and pin the all-zero
# submission to exactly 1.000. They are properties of the test set, not tunables.
AVE_REF = 0.7356384388          # m/s
ATE_REF = 3.1160277267          # m


class ParticipantVisibleError(Exception):
    """Message shown to the participant on the leaderboard."""


def _quat_to_R(q: np.ndarray) -> np.ndarray:
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


def _umeyama_align(P: np.ndarray, Q: np.ndarray) -> tuple:
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


def _segment_bounds(Q: np.ndarray, seg_len: float = SEGMENT_LENGTH_M,
                    min_pts: int = MIN_SEGMENT_POINTS) -> list:
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


def _ate_segment(R: np.ndarray, v: np.ndarray, dt: np.ndarray, Q: np.ndarray) -> float:
    """RMS position error of one aligned segment."""
    disp = np.einsum("mij,mj->mi", R, v) * dt
    P = np.cumsum(disp, axis=0)
    Rr, t = _umeyama_align(P, Q)
    return float(np.sqrt(np.mean(np.sum((P @ Rr.T + t - Q) ** 2, axis=1))))


def _ate_traj(g: pd.DataFrame) -> float:
    """Mean 20 m-segment ATE for one trajectory dataframe (already time-ordered)."""
    Q = g[["gx", "gy", "gz"]].to_numpy(float)
    bounds = _segment_bounds(Q)
    if not bounds:
        return float("nan")
    R = _quat_to_R(g[["qx", "qy", "qz", "qw"]].to_numpy(float))
    v = g[["vx_pred", "vy_pred", "vz_pred"]].to_numpy(float)
    dt = g["dt"].to_numpy(float)[:, None]
    return float(np.mean([_ate_segment(R[s:e + 1], v[s:e + 1], dt[s:e + 1], Q[s:e + 1])
                          for s, e in bounds]))


def _ave_traj(g: pd.DataFrame) -> float:
    """Mean per-window Euclidean velocity error (m/s) for one trajectory."""
    err = (g[["vx_pred", "vy_pred", "vz_pred"]].to_numpy(float)
           - g[["vx_gt", "vy_gt", "vz_gt"]].to_numpy(float))
    return float(np.mean(np.linalg.norm(err, axis=1)))


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """TartanIMU Score = 0.6*(macro AVE / AVE_REF) + 0.4*(macro ATE20 / ATE_REF).

    Both components are macro-averaged over the four platforms, so each platform carries
    exactly 25 % of each term. AVE is the mean per-window Euclidean velocity error; ATE20
    integrates the predicted body-frame velocities over fixed 20 m pieces of the
    ground-truth path and takes the RMS position error after an SE(3) alignment. Each is
    divided by the value the all-zero submission attains on the full test set, which makes
    the score dimensionless and pins the all-zero submission to 1.000.

    Args:
        solution: window_id, traj_id, win_idx, platform, qx, qy, qz, qw, gx, gy, gz, dt,
            vx_gt, vy_gt, vz_gt (Usage already removed by Kaggle).
        submission: window_id, vx, vy, vz.
        row_id_column_name: name of the id column ("window_id").

    Returns:
        The TartanIMU Score as a finite float; lower is better.
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
    if sub[row_id_column_name].duplicated().any():
        raise ParticipantVisibleError(f"Submission contains duplicate '{row_id_column_name}' values.")

    need = {"traj_id", "win_idx", "platform", "qx", "qy", "qz", "qw",
            "gx", "gy", "gz", "dt", "vx_gt", "vy_gt", "vz_gt"}
    if not need.issubset(solution.columns):
        raise ParticipantVisibleError("Solution file is missing required scoring columns.")

    m = solution.merge(sub.rename(columns={"vx": "vx_pred", "vy": "vy_pred", "vz": "vz_pred"}),
                       on=row_id_column_name, how="left")
    if m[["vx_pred", "vy_pred", "vz_pred"]].isnull().any().any():
        raise ParticipantVisibleError("Submission is missing one or more required window_id rows.")

    per_platform = {}
    for _, g in m.groupby("traj_id", sort=False):
        g = g.sort_values("win_idx")
        ate, ave = _ate_traj(g), _ave_traj(g)
        if np.isfinite(ate) and np.isfinite(ave):
            per_platform.setdefault(g["platform"].iloc[0], []).append((ate, ave))

    if not per_platform:
        raise ParticipantVisibleError("No scorable trajectories found.")

    macro_ate = float(np.mean([np.mean([x[0] for x in v]) for v in per_platform.values()]))
    macro_ave = float(np.mean([np.mean([x[1] for x in v]) for v in per_platform.values()]))

    result = W_AVE * (macro_ave / AVE_REF) + W_ATE * (macro_ate / ATE_REF)
    if not np.isfinite(result):
        raise ParticipantVisibleError("Score is not finite; check for extreme velocity values.")
    return result
