# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Trajectory segmentation helpers.

Splits full trajectories into fixed-distance (e.g. 5 m) chunks for local
accuracy evaluation, computes per-segment ATE/RTE/accuracy metrics, and
performs sanity checks on segment alignment.

Migrated from the monolithic top-level ``test.py``.
"""

import logging

import numpy as np
from tartan_imu.evaluation.metrics import compute_accruacy_metrics, compute_ate_rte

from tartan_imu.evaluation.io import save_segment_data


def segment_trajectory_5m(
    traj_attr_dict: dict, segment_length: float = 5.0
) -> list:
    """Segment a trajectory into ~``segment_length``-meter chunks.

    Segment trajectory into 5-meter chunks for local accuracy evaluation.

    Args:
        traj_attr_dict: Dictionary containing trajectory data
        segment_length: Length of each segment in meters (default: 5.0)

    Returns:
        List of trajectory segments, each containing data for ~5m of travel
    """
    segments = []

    # Get ground truth positions
    pos_gt = traj_attr_dict["pos_gt"]

    # Calculate cumulative distance from start
    cumulative_distance = np.zeros(len(pos_gt))
    for i in range(1, len(pos_gt)):
        step_distance = np.linalg.norm(pos_gt[i] - pos_gt[i - 1])
        cumulative_distance[i] = cumulative_distance[i - 1] + step_distance

    # Find segment boundaries
    segment_boundaries = []
    current_distance = 0.0

    for i in range(len(cumulative_distance)):
        if cumulative_distance[i] >= current_distance + segment_length:
            segment_boundaries.append(i)
            current_distance = cumulative_distance[i]

    # Add final boundary if not already included
    if len(segment_boundaries) == 0 or segment_boundaries[-1] != len(pos_gt) - 1:
        segment_boundaries.append(len(pos_gt) - 1)

    # Create segments
    start_idx = 0
    for end_idx in segment_boundaries:
        if end_idx > start_idx:  # Ensure segment has at least 2 points
            segment = {}
            for key in traj_attr_dict.keys():
                if isinstance(traj_attr_dict[key], np.ndarray):
                    segment[key] = traj_attr_dict[key][start_idx : end_idx + 1]
                else:
                    segment[key] = traj_attr_dict[key]

            # Calculate actual segment length
            segment_actual_length = (
                cumulative_distance[end_idx] - cumulative_distance[start_idx]
            )
            segment["segment_length"] = segment_actual_length
            segment["start_idx"] = start_idx
            segment["end_idx"] = end_idx

            segments.append(segment)
            start_idx = end_idx

    return segments


def segment_trajectory_by_distance_gt(
    gt_positions: np.ndarray,
    gt_timestamps: np.ndarray,
    imu_timestamps: np.ndarray,
    segment_length: float = 5.0,
) -> list:
    """Segment by GT distance and map each segment to IMU index ranges.

    Segment trajectory into 5-meter chunks based on ground truth odometry.
    Find corresponding IMU data indices for each segment.

    Args:
        gt_positions: Ground truth positions (N, 3)
        gt_timestamps: Ground truth timestamps (N,)
        imu_timestamps: IMU timestamps (M,)
        segment_length: Length of each segment in meters (default: 5.0)

    Returns:
        List of segment dictionaries containing:
        - gt_start_idx, gt_end_idx: Ground truth indices
        - imu_start_idx, imu_end_idx: IMU indices
        - initial_pose: First pose of the segment
        - segment_length: Actual segment length
    """
    segments = []

    # Calculate cumulative distance from start
    cumulative_distance = np.zeros(len(gt_positions))
    for i in range(1, len(gt_positions)):
        step_distance = np.linalg.norm(gt_positions[i] - gt_positions[i - 1])
        cumulative_distance[i] = cumulative_distance[i - 1] + step_distance

    # Find segment boundaries
    segment_boundaries = []
    current_distance = 0.0

    for i in range(len(cumulative_distance)):
        if cumulative_distance[i] >= current_distance + segment_length:
            segment_boundaries.append(i)
            current_distance = cumulative_distance[i]

    # Add final boundary if not already included
    if len(segment_boundaries) == 0 or segment_boundaries[-1] != len(gt_positions) - 1:
        segment_boundaries.append(len(gt_positions) - 1)

    # Create segments
    start_idx = 0
    for end_idx in segment_boundaries:
        if end_idx > start_idx:  # Ensure segment has at least 2 points
            # Get ground truth timestamps for this segment
            segment_gt_start_time = gt_timestamps[start_idx]
            segment_gt_end_time = gt_timestamps[end_idx]

            # Find corresponding IMU indices using timestamp overlap
            imu_start_idx = np.searchsorted(
                imu_timestamps, segment_gt_start_time, side="left"
            )
            imu_end_idx = np.searchsorted(
                imu_timestamps, segment_gt_end_time, side="right"
            )

            # Ensure we have valid IMU data
            if imu_start_idx < len(imu_timestamps) and imu_end_idx > imu_start_idx:
                # Calculate actual segment length
                segment_actual_length = (
                    cumulative_distance[end_idx] - cumulative_distance[start_idx]
                )

                # Get initial pose (first pose of the segment)
                initial_pose = gt_positions[start_idx]

                segment = {
                    "gt_start_idx": start_idx,
                    "gt_end_idx": end_idx,
                    "imu_start_idx": imu_start_idx,
                    "imu_end_idx": imu_end_idx,
                    "gt_start_time": segment_gt_start_time,
                    "gt_end_time": segment_gt_end_time,
                    "initial_pose": initial_pose,
                    "segment_length": segment_actual_length,
                    "cumulative_distance_start": cumulative_distance[start_idx],
                    "cumulative_distance_end": cumulative_distance[end_idx],
                }

                segments.append(segment)

            start_idx = end_idx

    return segments


def compute_segment_metrics(segment: dict, cfg: dict) -> dict:
    """Compute ATE/RTE and accuracy metrics for a single segment.

    Compute metrics for a single trajectory segment.

    Args:
        segment: Trajectory segment dictionary
        cfg: Configuration dictionary

    Returns:
        Dictionary containing all computed metrics
    """
    ate, t_rte, d_rte = compute_ate_rte(
        segment["pos_pred"], segment["pos_gt"], int(cfg["data"]["imu_freq"] * 1)
    )

    ATE, AVE, P_RMSE, V_RMSE, X_ATE, Y_ATE, Z_ATE, X_AVE, Y_AVE, Z_AVE = (
        compute_accruacy_metrics(
            segment, int(cfg["data"]["imu_freq"] * 1), cfg["data"]["use_local_coord"]
        )
    )

    return {
        "ate": ate,
        "t_rte": t_rte,
        "d_rte": d_rte,
        "ATE": ATE,
        "AVE": AVE,
        "P_RMSE": P_RMSE,
        "V_RMSE": V_RMSE,
        "X_ATE": X_ATE,
        "Y_ATE": Y_ATE,
        "Z_ATE": Z_ATE,
        "X_AVE": X_AVE,
        "Y_AVE": Y_AVE,
        "Z_AVE": Z_AVE,
    }


def process_trajectory_segments(
    trajectory_segments: list,
    outdir: str,
    cfg: dict,
    trajectory_info: dict = None,
) -> list:
    """Compute and persist metrics for every segment of a trajectory.

    Process all trajectory segments and compute metrics.

    Args:
        trajectory_segments: List of trajectory segments
        outdir: Output directory
        cfg: Configuration dictionary
        trajectory_info: Dictionary with trajectory metadata (data_name, robot_type, trajectory_id)

    Returns:
        List of segment metrics
    """
    segment_metrics = []

    for seg_idx, segment in enumerate(trajectory_segments):
        logging.info(
            f"Processing segment {seg_idx + 1}/{len(trajectory_segments)} (length: {segment['segment_length']:.2f}m)"
        )

        # Compute metrics for this segment
        metrics = compute_segment_metrics(segment, cfg)

        # Add segment metadata
        segment_metric = {
            "segment_id": seg_idx,
            "segment_length": segment["segment_length"],
            "start_idx": segment["start_idx"],
            "end_idx": segment["end_idx"],
            **metrics,
        }

        # Add trajectory information if provided
        if trajectory_info:
            segment_metric.update(trajectory_info)

        segment_metrics.append(segment_metric)

        # Save segment data
        save_segment_data(segment, segment_metric, outdir, seg_idx)

    return segment_metrics


def sanity_check_segment_alignment(segment: dict, seg_idx: int) -> bool:
    """Verify a segment's predicted start matches its GT start position.

    Sanity check to ensure segment inference starts from ground truth initial position.

    Args:
        segment: Trajectory segment data
        seg_idx: Segment index

    Returns:
        bool: True if alignment is correct, False otherwise
    """
    pos_pred = segment["pos_pred"]
    pos_gt = segment["pos_gt"]

    # Check initial position alignment
    gt_initial = pos_gt[0]
    pred_initial = pos_pred[0]
    initial_diff = np.linalg.norm(pred_initial - gt_initial)

    # Tolerance for floating point precision
    tolerance = 1e-6

    return initial_diff <= tolerance
