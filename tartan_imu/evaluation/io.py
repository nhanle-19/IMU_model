# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Evaluation output/saving helpers.

Creates output directory trees and writes per-segment and full-trajectory data,
JSON metric files, and the comprehensive per-run CSV summaries.

Migrated from the monolithic top-level ``test.py``.
"""

import csv
import json
import os
from os import path as osp

import numpy as np
from tartan_imu.utils.rich_logging import info, success

from tartan_imu.evaluation.plots import create_segment_plot


def create_output_directories(
    out_dir: str, subfolder_name: str, data_name: str
) -> str:
    """Create (if needed) and return the per-trajectory output directory.

    Create output directories for test results.

    Args:
        out_dir: Base output directory
        subfolder_name: Subfolder name
        data_name: Data name

    Returns:
        outdir: Path to the output directory
    """
    sub_outdir = osp.join(out_dir, subfolder_name)
    if not osp.exists(sub_outdir):
        os.mkdir(sub_outdir)

    outdir = osp.join(sub_outdir, data_name)
    if not osp.exists(outdir):
        os.mkdir(outdir)

    return outdir


def save_segment_data(
    segment: dict, segment_metric: dict, outdir: str, seg_idx: int
) -> None:
    """Write a segment's trajectory, metrics JSON, and plot to disk.

    Save segment trajectory and metrics to files.

    Args:
        segment: Trajectory segment data
        segment_metric: Segment metrics
        outdir: Output directory
        seg_idx: Segment index
    """
    # Create segment directory
    segment_outdir = osp.join(outdir, f"segment_{seg_idx:03d}")
    if not osp.exists(segment_outdir):
        os.makedirs(segment_outdir)

    # Save segment trajectory data
    segment_trajectory_data = np.concatenate(
        [
            segment["ts"].reshape(-1, 1),
            segment["pos_pred"],
            segment["pos_gt"],
            segment["cov_pred"],
        ],
        axis=1,
    )
    segment_traj_file = osp.join(segment_outdir, "trajectory.txt")
    np.savetxt(segment_traj_file, segment_trajectory_data, delimiter=",")

    # Save segment metrics
    segment_metrics_file = osp.join(segment_outdir, "metrics.json")
    with open(segment_metrics_file, "w") as f:
        json.dump(segment_metric, f, indent=1)

    # Create plots for this segment
    create_segment_plot(segment, segment_metric, outdir, seg_idx)


def save_full_trajectory_data(
    traj_attr_dict: dict, outdir: str, epoch_num: int, plot_dict: dict
) -> None:
    """Write the full trajectory and est/gt pose files to disk.

    Save full trajectory data and metrics.

    Args:
        traj_attr_dict: Full trajectory data
        outdir: Output directory
        epoch_num: Epoch number
        plot_dict: Plot dictionary
    """
    # Save full trajectory data
    outfile = osp.join(outdir, "trajectory.txt")
    trajectory_data = np.concatenate(
        [
            traj_attr_dict["ts"].reshape(-1, 1),
            traj_attr_dict["pos_pred"],
            traj_attr_dict["pos_gt"],
            traj_attr_dict["cov_pred"],
        ],
        axis=1,
    )
    np.savetxt(outfile, trajectory_data, delimiter=",")

    # Save network outputs
    est_pose_file = osp.join(outdir, f"est_pose_{epoch_num}.txt")
    gt_pose_file = osp.join(outdir, f"gt_pose_{epoch_num}.txt")

    est_pose = plot_dict["pos_pred"]
    gt_pose = plot_dict["pos_gt"]
    np.savetxt(gt_pose_file, gt_pose, delimiter=",")
    np.savetxt(est_pose_file, est_pose, delimiter=",")


def save_metrics_files(
    all_metrics: dict, outdir: str, data_name: str, segment_metrics: list
) -> None:
    """Write the main metrics JSON and the segment-metrics summary JSON.

    Save all metrics to JSON files.

    Args:
        all_metrics: All metrics dictionary
        outdir: Output directory
        data_name: Data name
        segment_metrics: List of segment metrics
    """
    # Save main metrics
    with open(outdir + "/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=1)

    # Save segment metrics summary
    segment_summary = {
        "trajectory": data_name,
        "num_segments": len(segment_metrics),
        "segment_metrics": segment_metrics,
    }
    with open(outdir + "/segment_metrics_summary.json", "w") as f:
        json.dump(segment_summary, f, indent=1)


def save_comprehensive_csv_files(
    all_trajectory_results: list,
    overall_stats: dict,
    segment_metrics_all: list,
    out_dir: str,
) -> None:
    """Write the full suite of per-run summary/analysis CSV files.

    Save comprehensive CSV files with all metrics organized by category.

    Args:
        all_trajectory_results: List of all trajectory results
        overall_stats: Overall statistics dictionary
        segment_metrics_all: List of all segment metrics
        out_dir: Output directory
    """

    # 1. Overall Summary CSV
    summary_file = osp.join(out_dir, "overall_summary.csv")
    with open(summary_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Category", "Metric", "Value", "Description"])

        # Basic statistics
        writer.writerow(
            [
                "Basic",
                "Total Trajectories",
                len(all_trajectory_results),
                "Number of test trajectories",
            ]
        )
        writer.writerow(
            [
                "Basic",
                "Total Segments",
                overall_stats.get("total_segments", 0),
                "Total number of 5m segments",
            ]
        )

        # Overall aggregated metrics
        for key, value in overall_stats.items():
            if key not in ["total_segments", "segment_statistics"]:
                writer.writerow(["Overall", key, f"{value:.6f}", f"Overall {key}"])

        # Segment statistics
        if "segment_statistics" in overall_stats:
            seg_stats = overall_stats["segment_statistics"]
            writer.writerow(
                [
                    "Segments",
                    "Average Segment Length",
                    f"{seg_stats.get('avg_segment_length', 0):.4f}",
                    "Average length of segments",
                ]
            )

            ate_stats = seg_stats.get("segment_ate_stats", {})
            writer.writerow(
                [
                    "Segments",
                    "ATE Mean",
                    f"{ate_stats.get('mean', 0):.6f}",
                    "Average ATE across all segments",
                ]
            )
            writer.writerow(
                [
                    "Segments",
                    "ATE Std",
                    f"{ate_stats.get('std', 0):.6f}",
                    "Standard deviation of segment ATE",
                ]
            )
            writer.writerow(
                [
                    "Segments",
                    "ATE Min",
                    f"{ate_stats.get('min', 0):.6f}",
                    "Minimum ATE across segments",
                ]
            )
            writer.writerow(
                [
                    "Segments",
                    "ATE Max",
                    f"{ate_stats.get('max', 0):.6f}",
                    "Maximum ATE across segments",
                ]
            )

            rmse_stats = seg_stats.get("segment_rmse_stats", {})
            writer.writerow(
                [
                    "Segments",
                    "RMSE Mean",
                    f"{rmse_stats.get('mean', 0):.6f}",
                    "Average RMSE across all segments",
                ]
            )
            writer.writerow(
                [
                    "Segments",
                    "RMSE Std",
                    f"{rmse_stats.get('std', 0):.6f}",
                    "Standard deviation of segment RMSE",
                ]
            )
            writer.writerow(
                [
                    "Segments",
                    "RMSE Min",
                    f"{rmse_stats.get('min', 0):.6f}",
                    "Minimum RMSE across segments",
                ]
            )
            writer.writerow(
                [
                    "Segments",
                    "RMSE Max",
                    f"{rmse_stats.get('max', 0):.6f}",
                    "Maximum RMSE across segments",
                ]
            )

    # 2. Trajectory-wise Metrics CSV
    traj_file = osp.join(out_dir, "trajectory_metrics.csv")
    with open(traj_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Trajectory_ID",
                "Data_Name",
                "Robot_Type",
                "Num_Segments",
                "Avg_ATE",
                "Avg_T_RTE",
                "Avg_D_RTE",
                "Avg_ATE_metric",
                "Avg_AVE",
                "Avg_P_RMSE",
                "Avg_V_RMSE",
                "Avg_X_ATE",
                "Avg_Y_ATE",
                "Avg_Z_ATE",
                "Avg_X_AVE",
                "Avg_Y_AVE",
                "Avg_Z_AVE",
                "Full_ATE",
                "Full_T_RTE",
                "Full_D_RTE",
                "Full_ATE_metric",
                "Full_AVE",
                "Full_P_RMSE",
                "Full_V_RMSE",
                "Full_X_ATE",
                "Full_Y_ATE",
                "Full_Z_ATE",
                "Full_X_AVE",
                "Full_Y_AVE",
                "Full_Z_AVE",
            ]
        )

        for i, traj_result in enumerate(all_trajectory_results):
            full_traj = traj_result.get("full_trajectory", {})
            row = [
                f"Traj_{i+1}",
                traj_result.get("data", "N/A"),
                traj_result.get("robot_type", "N/A"),
                traj_result.get("num_segments", 0),
                traj_result.get("avg_ate", 0),
                traj_result.get("avg_t_rte", 0),
                traj_result.get("avg_d_rte", 0),
                traj_result.get("avg_ATE", 0),
                traj_result.get("avg_AVE", 0),
                traj_result.get("avg_P_RMSE", 0),
                traj_result.get("avg_V_RMSE", 0),
                traj_result.get("avg_X_ATE", 0),
                traj_result.get("avg_Y_ATE", 0),
                traj_result.get("avg_Z_ATE", 0),
                traj_result.get("avg_X_AVE", 0),
                traj_result.get("avg_Y_AVE", 0),
                traj_result.get("avg_Z_AVE", 0),
                full_traj.get("ate", 0),
                full_traj.get("t_rte", 0),
                full_traj.get("d_rte", 0),
                full_traj.get("ATE", 0),
                full_traj.get("AVE", 0),
                full_traj.get("P_RMSE", 0),
                full_traj.get("V_RMSE", 0),
                full_traj.get("X_ATE", 0),
                full_traj.get("Y_ATE", 0),
                full_traj.get("Z_ATE", 0),
                full_traj.get("X_AVE", 0),
                full_traj.get("Y_AVE", 0),
                full_traj.get("Z_AVE", 0),
            ]
            writer.writerow(row)

    # 3. Segment-level Metrics CSV
    if segment_metrics_all:
        seg_file = osp.join(out_dir, "segment_metrics.csv")
        with open(seg_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Segment_ID",
                    "Trajectory_ID",
                    "Data_Name",
                    "Segment_Length",
                    "Start_Index",
                    "End_Index",
                    "Num_Points",
                    "ATE",
                    "T_RTE",
                    "D_RTE",
                    "ATE_metric",
                    "AVE",
                    "P_RMSE",
                    "V_RMSE",
                    "X_ATE",
                    "Y_ATE",
                    "Z_ATE",
                    "X_AVE",
                    "Y_AVE",
                    "Z_AVE",
                ]
            )

            for i, seg_metric in enumerate(segment_metrics_all):
                row = [
                    f"Seg_{i+1:03d}",
                    f"Traj_{seg_metric.get('trajectory_id', 'N/A')}",
                    seg_metric.get("data_name", "N/A"),
                    seg_metric.get("segment_length", 0),
                    seg_metric.get("start_idx", 0),
                    seg_metric.get("end_idx", 0),
                    seg_metric.get("end_idx", 0) - seg_metric.get("start_idx", 0) + 1,
                    seg_metric.get("ate", 0),
                    seg_metric.get("t_rte", 0),
                    seg_metric.get("d_rte", 0),
                    seg_metric.get("ATE", 0),
                    seg_metric.get("AVE", 0),
                    seg_metric.get("P_RMSE", 0),
                    seg_metric.get("V_RMSE", 0),
                    seg_metric.get("X_ATE", 0),
                    seg_metric.get("Y_ATE", 0),
                    seg_metric.get("Z_ATE", 0),
                    seg_metric.get("X_AVE", 0),
                    seg_metric.get("Y_AVE", 0),
                    seg_metric.get("Z_AVE", 0),
                ]
                writer.writerow(row)

    # 4. Statistical Summary CSV
    stats_file = osp.join(out_dir, "statistical_summary.csv")
    with open(stats_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Metric_Category",
                "Metric_Name",
                "Mean",
                "Std",
                "Min",
                "Max",
                "Median",
                "Count",
            ]
        )

        if segment_metrics_all:
            # Calculate statistics for all metrics
            metrics_to_analyze = [
                "ate",
                "P_RMSE",
                "V_RMSE",
                "t_rte",
                "d_rte",
                "ATE",
                "AVE",
                "X_ATE",
                "Y_ATE",
                "Z_ATE",
                "X_AVE",
                "Y_AVE",
                "Z_AVE",
            ]

            for metric in metrics_to_analyze:
                values = [
                    seg.get(metric, 0) for seg in segment_metrics_all if metric in seg
                ]
                if values:
                    writer.writerow(
                        [
                            "Segment_Level",
                            metric.upper(),
                            f"{np.mean(values):.6f}",
                            f"{np.std(values):.6f}",
                            f"{np.min(values):.6f}",
                            f"{np.max(values):.6f}",
                            f"{np.median(values):.6f}",
                            len(values),
                        ]
                    )

    # 5. Performance Comparison CSV
    comp_file = osp.join(out_dir, "performance_comparison.csv")
    with open(comp_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Comparison_Type",
                "Metric",
                "Segment_Average",
                "Full_Trajectory_Average",
                "Difference",
                "Percentage_Difference",
            ]
        )

        # Compare segment averages vs full trajectory averages
        seg_avg_ate = (
            np.mean([seg.get("ate", 0) for seg in segment_metrics_all])
            if segment_metrics_all
            else 0
        )
        full_avg_ate = np.mean(
            [
                traj.get("full_trajectory", {}).get("ate", 0)
                for traj in all_trajectory_results
            ]
        )

        diff_ate = full_avg_ate - seg_avg_ate
        pct_diff_ate = (diff_ate / seg_avg_ate * 100) if seg_avg_ate != 0 else 0

        writer.writerow(
            [
                "ATE_Comparison",
                "ATE",
                f"{seg_avg_ate:.6f}",
                f"{full_avg_ate:.6f}",
                f"{diff_ate:.6f}",
                f"{pct_diff_ate:.2f}%",
            ]
        )

        # Add more comparisons as needed
        seg_avg_rmse = (
            np.mean([seg.get("P_RMSE", 0) for seg in segment_metrics_all])
            if segment_metrics_all
            else 0
        )
        full_avg_rmse = np.mean(
            [
                traj.get("full_trajectory", {}).get("P_RMSE", 0)
                for traj in all_trajectory_results
            ]
        )

        diff_rmse = full_avg_rmse - seg_avg_rmse
        pct_diff_rmse = (diff_rmse / seg_avg_rmse * 100) if seg_avg_rmse != 0 else 0

        writer.writerow(
            [
                "RMSE_Comparison",
                "P_RMSE",
                f"{seg_avg_rmse:.6f}",
                f"{full_avg_rmse:.6f}",
                f"{diff_rmse:.6f}",
                f"{pct_diff_rmse:.2f}%",
            ]
        )

    success(f"✅ Comprehensive CSV files saved to {out_dir}:")
    info("   📊 overall_summary.csv - Overall test statistics")
    info("   🚀 trajectory_metrics.csv - Individual trajectory results")
    info("   📏 segment_metrics.csv - Segment-level detailed metrics")
    info("   📈 statistical_summary.csv - Statistical analysis of all metrics")
    info("   🔍 performance_comparison.csv - Segment vs Full trajectory comparison")
