# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Evaluation plotting module.

Migrated verbatim from the monolithic top-level ``test.py``. Contains the
per-segment, per-trajectory, and aggregate summary plotting/reporting helpers
used during trajectory evaluation. Logic is unchanged; only type hints and
docstrings were added during the migration.
"""

import logging
import os
from os import path as osp

import numpy as np


def create_segment_plot(
    segment: dict, segment_metric: dict, outdir: str, seg_idx: int
) -> None:
    """
    Create plots for a single trajectory segment.

    Args:
        segment: Trajectory segment data
        segment_metric: Segment metrics
        outdir: Output directory
        seg_idx: Segment index
    """
    import matplotlib.pyplot as plt

    # ``sanity_check_segment_alignment`` still lives in the top-level test.py;
    # import it lazily here to avoid a circular import at module load time.
    from test import sanity_check_segment_alignment

    # Create segment directory
    segment_outdir = osp.join(outdir, f"segment_{seg_idx:03d}")
    if not osp.exists(segment_outdir):
        os.makedirs(segment_outdir)

    # SANITY CHECK: Verify that inference starts from ground truth initial position
    alignment_correct = sanity_check_segment_alignment(segment, seg_idx)

    # Extract data
    pos_pred = segment["pos_pred"]
    pos_gt = segment["pos_gt"]
    ts = segment["ts"]

    # FIX: Align segment to start from origin (0,0,0) for consistent visualization
    # This makes all segments start from the same reference point for easy comparison
    gt_initial = pos_gt[0]
    pred_initial = pos_pred[0]

    # Align both trajectories to start from origin (0,0,0)
    pos_pred_aligned = pos_pred - gt_initial  # Start from origin
    pos_gt_aligned = pos_gt - gt_initial  # Start from origin

    # Create 2D trajectory plot
    plt.figure(figsize=(12, 8))

    # Main trajectory plot
    plt.subplot(2, 2, 1)
    plt.plot(
        pos_pred_aligned[:, 0],
        pos_pred_aligned[:, 1],
        "b-",
        linewidth=2,
        label="Predicted",
    )
    plt.plot(
        pos_gt_aligned[:, 0],
        pos_gt_aligned[:, 1],
        "r-",
        linewidth=2,
        label="Ground Truth",
    )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"Segment {seg_idx} - 2D Trajectory (Origin Aligned)")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)

    # Mark the start point
    plt.plot(0, 0, "ko", markersize=8, label="Start (0,0)")

    # 3D trajectory plot
    ax = plt.subplot(2, 2, 2, projection="3d")
    ax.plot(
        pos_pred_aligned[:, 0],
        pos_pred_aligned[:, 1],
        pos_pred_aligned[:, 2],
        "b-",
        linewidth=2,
        label="Predicted",
    )
    ax.plot(
        pos_gt_aligned[:, 0],
        pos_gt_aligned[:, 1],
        pos_gt_aligned[:, 2],
        "r-",
        linewidth=2,
        label="Ground Truth",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Segment {seg_idx} - 3D Trajectory (Origin Aligned)")
    ax.legend()
    ax.grid(True)

    # Mark the start point in 3D
    ax.scatter([0], [0], [0], c="black", s=100, marker="o", label="Start (0,0,0)")

    # Position error over time (using aligned positions for proper error calculation)
    plt.subplot(2, 2, 3)
    pos_error = np.linalg.norm(pos_pred_aligned - pos_gt_aligned, axis=1)
    plt.plot(ts, pos_error, "g-", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (m)")
    plt.title(f"Segment {seg_idx} - Position Error (Origin Aligned)")
    plt.grid(True)

    # Metrics summary
    plt.subplot(2, 2, 4)
    alignment_status = "CORRECT" if alignment_correct else "INCORRECT"
    metrics_text = f"""Segment {seg_idx} Metrics (Origin Aligned):

    ATE: {segment_metric['ate']:.4f} m
    T_RTE: {segment_metric['t_rte']:.4f} m
    D_RTE: {segment_metric['d_rte']:.4f} m
    P_RMSE: {segment_metric['P_RMSE']:.4f} m
    V_RMSE: {segment_metric['V_RMSE']:.4f} m/s
    Segment Length: {segment_metric['segment_length']:.2f} m
    Points: {segment_metric['end_idx'] - segment_metric['start_idx'] + 1}

    ORIGINAL POSITIONS:
    GT Start: {gt_initial}
    Pred Start: {pred_initial}

    ALIGNED POSITIONS:
    GT Start: (0, 0, 0)
    Pred Start: {pred_initial - gt_initial}

    Alignment: {alignment_status}
    Status: {'Inference starts from GT position' if alignment_correct else 'WARNING: Check inference logic'}"""

    plt.text(
        0.1,
        0.5,
        metrics_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
    )
    plt.axis("off")
    plt.title(f"Segment {seg_idx} - Metrics Summary")

    plt.tight_layout()
    plt.savefig(
        osp.join(segment_outdir, f"segment_{seg_idx:03d}_plot.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Create additional detailed plots
    create_detailed_segment_plots(segment, segment_metric, segment_outdir, seg_idx)


def create_detailed_segment_plots(
    segment: dict, segment_metric: dict, segment_outdir: str, seg_idx: int
) -> None:
    """
    Create detailed plots for a segment including velocity and acceleration.

    Args:
        segment: Trajectory segment data
        segment_metric: Segment metrics
        segment_outdir: Segment output directory
        seg_idx: Segment index
    """
    import matplotlib.pyplot as plt

    pos_pred = segment["pos_pred"]
    pos_gt = segment["pos_gt"]
    ts = segment["ts"]

    # Align segment to start from origin (0,0,0) for consistent visualization (same as in create_segment_plot)
    gt_initial = pos_gt[0]
    pos_pred_aligned = pos_pred - gt_initial  # Start from origin
    pos_gt_aligned = pos_gt - gt_initial  # Start from origin

    # Use direct velocity predictions if available, otherwise calculate from position
    if (
        "vel_pred" in segment
        and "vel_gt" in segment
        and segment["vel_pred"] is not None
    ):
        # Use direct velocity predictions from model
        vel_pred = segment["vel_pred"]
        vel_gt = segment["vel_gt"]
        vel_ts = ts  # Use same timestamps as position
        logging.info(
            f"Using direct velocity predictions - vel_pred shape: {vel_pred.shape}"
        )
    else:
        # Fallback: Calculate velocities (simple finite difference) using aligned positions
        dt = np.diff(ts)
        vel_pred = np.diff(pos_pred_aligned, axis=0) / dt[:, np.newaxis]
        vel_gt = np.diff(pos_gt_aligned, axis=0) / dt[:, np.newaxis]
        vel_ts = ts[:-1]  # Time stamps for velocity
        logging.info(
            f"Using finite difference velocity calculation - vel_pred shape: {vel_pred.shape}"
        )


    # Velocity plot
    plt.figure(figsize=(15, 10))

    # Velocity components
    plt.subplot(3, 3, 1)
    plt.plot(vel_ts, vel_pred[:, 0], "b-", label="Pred X")
    plt.plot(vel_ts, vel_gt[:, 0], "r-", label="GT X")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity X (m/s)")
    plt.title("Velocity X Component")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 3, 2)
    plt.plot(vel_ts, vel_pred[:, 1], "b-", label="Pred Y")
    plt.plot(vel_ts, vel_gt[:, 1], "r-", label="GT Y")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity Y (m/s)")
    plt.title("Velocity Y Component")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 3, 3)
    plt.plot(vel_ts, vel_pred[:, 2], "b-", label="Pred Z")
    plt.plot(vel_ts, vel_gt[:, 2], "r-", label="GT Z")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity Z (m/s)")
    plt.title("Velocity Z Component")
    plt.legend()
    plt.grid(True)

    # Velocity magnitude
    plt.subplot(3, 3, 4)
    vel_pred_mag = np.linalg.norm(vel_pred, axis=1)
    vel_gt_mag = np.linalg.norm(vel_gt, axis=1)
    plt.plot(vel_ts, vel_pred_mag, "b-", label="Predicted")
    plt.plot(vel_ts, vel_gt_mag, "r-", label="Ground Truth")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity Magnitude (m/s)")
    plt.title("Velocity Magnitude")
    plt.legend()
    plt.grid(True)

    # Position components
    plt.subplot(3, 3, 5)
    plt.plot(ts, pos_pred[:, 0], "b-", label="Pred X")
    plt.plot(ts, pos_gt[:, 0], "r-", label="GT X")
    plt.xlabel("Time (s)")
    plt.ylabel("Position X (m)")
    plt.title("Position X Component")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 3, 6)
    plt.plot(ts, pos_pred[:, 1], "b-", label="Pred Y")
    plt.plot(ts, pos_gt[:, 1], "r-", label="GT Y")
    plt.xlabel("Time (s)")
    plt.ylabel("Position Y (m)")
    plt.title("Position Y Component")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 3, 7)
    plt.plot(ts, pos_pred[:, 2], "b-", label="Pred Z")
    plt.plot(ts, pos_gt[:, 2], "r-", label="GT Z")
    plt.xlabel("Time (s)")
    plt.ylabel("Position Z (m)")
    plt.title("Position Z Component")
    plt.legend()
    plt.grid(True)

    # Error analysis
    plt.subplot(3, 3, 8)
    pos_error_components = np.abs(pos_pred - pos_gt)
    plt.plot(ts, pos_error_components[:, 0], "g-", label="X Error")
    plt.plot(ts, pos_error_components[:, 1], "m-", label="Y Error")
    plt.plot(ts, pos_error_components[:, 2], "c-", label="Z Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (m)")
    plt.title("Position Error Components")
    plt.legend()
    plt.grid(True)

    # Cumulative distance
    plt.subplot(3, 3, 9)
    cum_dist_pred = np.cumsum(np.linalg.norm(np.diff(pos_pred, axis=0), axis=1))
    cum_dist_gt = np.cumsum(np.linalg.norm(np.diff(pos_gt, axis=0), axis=1))
    plt.plot(ts[1:], cum_dist_pred, "b-", label="Predicted")
    plt.plot(ts[1:], cum_dist_gt, "r-", label="Ground Truth")
    plt.xlabel("Time (s)")
    plt.ylabel("Cumulative Distance (m)")
    plt.title("Cumulative Distance")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(
        osp.join(segment_outdir, f"segment_{seg_idx:03d}_detailed.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_3d_segments_summary(all_trajectory_results: list, out_dir: str) -> None:
    """
    Create a 3D summary plot showing all segments aligned to origin for easy comparison.
    Also includes the full trajectory with ground truth position correction.

    Args:
        all_trajectory_results: List of all trajectory results
        out_dir: Output directory
    """
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    # Create figure with 3D projection
    fig = plt.figure(figsize=(24, 18))

    # Main 3D plot showing all segments
    ax1 = fig.add_subplot(3, 4, 1, projection="3d")

    # Collect all segment data and align to origin
    all_segments_data = []
    colors = cm.tab20(np.linspace(0, 1, 20))  # Use 20 different colors

    for traj_idx, traj_result in enumerate(all_trajectory_results):
        for seg_idx, seg_metric in enumerate(traj_result["segment_metrics"]):
            # Load segment data from saved files
            segment_dir = osp.join(out_dir, f"segment_{seg_idx:03d}")
            trajectory_file = osp.join(segment_dir, "trajectory.txt")

            if osp.exists(trajectory_file):
                # Load trajectory data: [ts, pos_pred_x, pos_pred_y, pos_pred_z, pos_gt_x, pos_gt_y, pos_gt_z, cov_pred_x, cov_pred_y, cov_pred_z]
                traj_data = np.loadtxt(trajectory_file, delimiter=",")

                # Extract positions
                pos_pred = traj_data[:, 1:4]  # predicted positions
                pos_gt = traj_data[:, 4:7]  # ground truth positions

                # Align to origin
                gt_initial = pos_gt[0]
                pos_pred_aligned = pos_pred - gt_initial
                pos_gt_aligned = pos_gt - gt_initial

                all_segments_data.append(
                    {
                        "traj_idx": traj_idx,
                        "seg_idx": seg_idx,
                        "pos_pred": pos_pred_aligned,
                        "pos_gt": pos_gt_aligned,
                        "ate": seg_metric["ate"],
                        "color": colors[seg_idx % len(colors)],
                    }
                )

    # Plot all segments in 3D
    for seg_data in all_segments_data:
        color = seg_data["color"]
        alpha = 0.7

        # Plot ground truth (solid line)
        ax1.plot(
            seg_data["pos_gt"][:, 0],
            seg_data["pos_gt"][:, 1],
            seg_data["pos_gt"][:, 2],
            color=color,
            linewidth=2,
            alpha=alpha,
            label=f'GT Seg {seg_data["seg_idx"]}' if seg_data["seg_idx"] < 5 else "",
        )

        # Plot predicted (dashed line)
        ax1.plot(
            seg_data["pos_pred"][:, 0],
            seg_data["pos_pred"][:, 1],
            seg_data["pos_pred"][:, 2],
            color=color,
            linewidth=2,
            alpha=alpha,
            linestyle="--",
            label=f'Pred Seg {seg_data["seg_idx"]}' if seg_data["seg_idx"] < 5 else "",
        )

        # Mark start points
        ax1.scatter([0], [0], [0], c="black", s=50, marker="o", alpha=0.8)

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title(
        "All Segments Aligned to Origin (0,0,0)\nGround Truth: Solid, Predicted: Dashed"
    )
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Full trajectory with drift correction
    ax2 = fig.add_subplot(3, 4, 2, projection="3d")

    # Load full trajectory data
    full_trajectory_file = osp.join(out_dir, "trajectory.txt")
    full_pos_pred = None
    full_pos_gt = None
    full_gt_initial = None
    full_pred_initial = None
    full_pos_pred_corrected = None

    if osp.exists(full_trajectory_file):
        full_traj_data = np.loadtxt(full_trajectory_file, delimiter=",")

        # Extract full trajectory positions
        full_pos_pred = full_traj_data[:, 1:4]  # predicted positions
        full_pos_gt = full_traj_data[:, 4:7]  # ground truth positions

        # Apply drift correction to full trajectory (same as in process_single_trajectory)
        full_gt_initial = full_pos_gt[0]
        full_pred_initial = full_pos_pred[0]
        full_pos_pred_corrected = full_pos_pred - full_pred_initial + full_gt_initial

        # Plot full trajectory
        ax2.plot(
            full_pos_gt[:, 0],
            full_pos_gt[:, 1],
            full_pos_gt[:, 2],
            color="red",
            linewidth=3,
            label="Full GT Trajectory",
        )
        ax2.plot(
            full_pos_pred_corrected[:, 0],
            full_pos_pred_corrected[:, 1],
            full_pos_pred_corrected[:, 2],
            color="blue",
            linewidth=3,
            linestyle="--",
            label="Full Pred Trajectory (Drift Corrected)",
        )

        # Mark start and end points
        ax2.scatter(
            [full_gt_initial[0]],
            [full_gt_initial[1]],
            [full_gt_initial[2]],
            c="green",
            s=100,
            marker="o",
            label="Start",
        )
        ax2.scatter(
            [full_pos_gt[-1, 0]],
            [full_pos_gt[-1, 1]],
            [full_pos_gt[-1, 2]],
            c="red",
            s=100,
            marker="s",
            label="End",
        )

    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.set_title(
        "Full Trajectory with Drift Correction\nGT: Red Solid, Pred: Blue Dashed"
    )
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: ATE vs Segment Index
    ax3 = fig.add_subplot(3, 4, 3)

    if all_segments_data:  # Check if segments exist
        segment_indices = [seg_data["seg_idx"] for seg_data in all_segments_data]
        ates = [seg_data["ate"] for seg_data in all_segments_data]
        colors_ate = [seg_data["color"] for seg_data in all_segments_data]

        ax3.scatter(segment_indices, ates, c=colors_ate, alpha=0.7, s=50)
        ax3.set_xlabel("Segment Index")
        ax3.set_ylabel("ATE (m)")
        ax3.set_title("ATE vs Segment Index")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(
            0.5,
            0.5,
            "No segments found",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
        ax3.set_title("ATE vs Segment Index (No Data)")

    # Plot 4: Segment length distribution
    ax4 = fig.add_subplot(3, 4, 4)
    segment_lengths = [
        seg_metric["segment_length"]
        for traj_result in all_trajectory_results
        for seg_metric in traj_result["segment_metrics"]
    ]

    if segment_lengths:  # Check if segment lengths exist
        ax4.hist(
            segment_lengths, bins=15, alpha=0.7, color="lightgreen", edgecolor="black"
        )
        ax4.set_xlabel("Segment Length (m)")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Segment Length Distribution")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(
            0.5,
            0.5,
            "No segments found",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )
        ax4.set_title("Segment Length Distribution (No Data)")

    # Plot 5: 2D projection (X-Y) of all segments
    ax5 = fig.add_subplot(3, 4, 5)
    for seg_data in all_segments_data:
        color = seg_data["color"]
        alpha = 0.7

        # Plot ground truth (solid line)
        ax5.plot(
            seg_data["pos_gt"][:, 0],
            seg_data["pos_gt"][:, 1],
            color=color,
            linewidth=2,
            alpha=alpha,
        )

        # Plot predicted (dashed line)
        ax5.plot(
            seg_data["pos_pred"][:, 0],
            seg_data["pos_pred"][:, 1],
            color=color,
            linewidth=2,
            alpha=alpha,
            linestyle="--",
        )

    ax5.set_xlabel("X (m)")
    ax5.set_ylabel("Y (m)")
    ax5.set_title("2D Projection (X-Y) of All Segments")
    ax5.grid(True, alpha=0.3)
    ax5.axis("equal")

    # Mark origin
    ax5.plot(0, 0, "ko", markersize=8, label="Origin (0,0)")
    ax5.legend()

    # Plot 6: 2D projection (X-Z) of all segments
    ax6 = fig.add_subplot(3, 4, 6)
    for seg_data in all_segments_data:
        color = seg_data["color"]
        alpha = 0.7

        # Plot ground truth (solid line)
        ax6.plot(
            seg_data["pos_gt"][:, 0],
            seg_data["pos_gt"][:, 2],
            color=color,
            linewidth=2,
            alpha=alpha,
        )

        # Plot predicted (dashed line)
        ax6.plot(
            seg_data["pos_pred"][:, 0],
            seg_data["pos_pred"][:, 2],
            color=color,
            linewidth=2,
            alpha=alpha,
            linestyle="--",
        )

    ax6.set_xlabel("X (m)")
    ax6.set_ylabel("Z (m)")
    ax6.set_title("2D Projection (X-Z) of All Segments")
    ax6.grid(True, alpha=0.3)

    # Mark origin
    ax6.plot(0, 0, "ko", markersize=8, label="Origin (0,0)")
    ax6.legend()

    # Plot 7: 2D projection (X-Y) of full trajectory
    ax7 = fig.add_subplot(3, 4, 7)
    if full_pos_gt is not None and full_pos_pred_corrected is not None:
        ax7.plot(
            full_pos_gt[:, 0],
            full_pos_gt[:, 1],
            color="red",
            linewidth=3,
            label="Full GT Trajectory",
        )
        ax7.plot(
            full_pos_pred_corrected[:, 0],
            full_pos_pred_corrected[:, 1],
            color="blue",
            linewidth=3,
            linestyle="--",
            label="Full Pred Trajectory (Drift Corrected)",
        )

        # Mark start and end points
        ax7.plot(
            full_gt_initial[0], full_gt_initial[1], "go", markersize=10, label="Start"
        )
        ax7.plot(
            full_pos_gt[-1, 0], full_pos_gt[-1, 1], "rs", markersize=10, label="End"
        )

    ax7.set_xlabel("X (m)")
    ax7.set_ylabel("Y (m)")
    ax7.set_title("2D Projection (X-Y) of Full Trajectory")
    ax7.grid(True, alpha=0.3)
    ax7.axis("equal")
    ax7.legend()

    # Plot 8: 2D projection (X-Z) of full trajectory
    ax8 = fig.add_subplot(3, 4, 8)
    if full_pos_gt is not None and full_pos_pred_corrected is not None:
        ax8.plot(
            full_pos_gt[:, 0],
            full_pos_gt[:, 2],
            color="red",
            linewidth=3,
            label="Full GT Trajectory",
        )
        ax8.plot(
            full_pos_pred_corrected[:, 0],
            full_pos_pred_corrected[:, 2],
            color="blue",
            linewidth=3,
            linestyle="--",
            label="Full Pred Trajectory (Drift Corrected)",
        )

        # Mark start and end points
        ax8.plot(
            full_gt_initial[0], full_gt_initial[2], "go", markersize=10, label="Start"
        )
        ax8.plot(
            full_pos_gt[-1, 0], full_pos_gt[-1, 2], "rs", markersize=10, label="End"
        )

    ax8.set_xlabel("X (m)")
    ax8.set_ylabel("Z (m)")
    ax8.set_title("2D Projection (X-Z) of Full Trajectory")
    ax8.grid(True, alpha=0.3)
    ax8.legend()

    # Plot 9: Comparison of full trajectory vs segments
    ax9 = fig.add_subplot(3, 4, 9)

    # Calculate full trajectory ATE
    if (
        full_pos_gt is not None
        and full_pos_pred_corrected is not None
        and all_segments_data
    ):
        full_ate = np.mean(
            np.linalg.norm(full_pos_pred_corrected - full_pos_gt, axis=1)
        )
        segment_avg_ate = np.mean(ates)

        comparison_data = ["Full Trajectory", "Average Segments"]
        comparison_values = [full_ate, segment_avg_ate]
        colors_comp = ["red", "blue"]

        bars = ax9.bar(comparison_data, comparison_values, color=colors_comp, alpha=0.7)
        ax9.set_ylabel("ATE (m)")
        ax9.set_title("Full Trajectory vs Segment Average ATE")
        ax9.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, comparison_values):
            ax9.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
    else:
        ax9.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax9.transAxes,
        )
        ax9.set_title("Full Trajectory vs Segment Average ATE (No Data)")

    # Plot 10: Trajectory length comparison
    ax10 = fig.add_subplot(3, 4, 10)
    if (
        full_pos_gt is not None
        and full_pos_pred_corrected is not None
        and segment_lengths
    ):
        # Calculate trajectory lengths
        full_gt_length = np.sum(np.linalg.norm(np.diff(full_pos_gt, axis=0), axis=1))
        full_pred_length = np.sum(
            np.linalg.norm(np.diff(full_pos_pred_corrected, axis=0), axis=1)
        )
        segment_avg_length = np.mean(segment_lengths)

        length_data = ["Full GT", "Full Pred", "Avg Segment"]
        length_values = [full_gt_length, full_pred_length, segment_avg_length]
        colors_length = ["red", "blue", "green"]

        bars = ax10.bar(length_data, length_values, color=colors_length, alpha=0.7)
        ax10.set_ylabel("Length (m)")
        ax10.set_title("Trajectory Length Comparison")
        ax10.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, length_values):
            ax10.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
    else:
        ax10.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax10.transAxes,
        )
        ax10.set_title("Trajectory Length Comparison (No Data)")

    # Plot 11: Summary statistics
    ax11 = fig.add_subplot(3, 4, 11)
    ax11.axis("off")

    # Calculate summary statistics
    total_segments = len(all_segments_data)

    if all_segments_data:  # Check if segments exist
        avg_ate = np.mean(ates)
        std_ate = np.std(ates)
        min_ate = np.min(ates)
        max_ate = np.max(ates)
    else:
        avg_ate = "N/A"
        std_ate = "N/A"
        min_ate = "N/A"
        max_ate = "N/A"

    if full_pos_gt is not None and full_pos_pred_corrected is not None:
        full_ate = np.mean(
            np.linalg.norm(full_pos_pred_corrected - full_pos_gt, axis=1)
        )
        full_gt_length = np.sum(np.linalg.norm(np.diff(full_pos_gt, axis=0), axis=1))
    else:
        full_ate = "N/A"
        full_gt_length = "N/A"

    # Format ATE values based on their type
    if isinstance(avg_ate, (int, float)):
        avg_ate_str = f"{avg_ate:.4f}"
    else:
        avg_ate_str = str(avg_ate)

    if isinstance(std_ate, (int, float)):
        std_ate_str = f"{std_ate:.4f}"
    else:
        std_ate_str = str(std_ate)

    if isinstance(min_ate, (int, float)):
        min_ate_str = f"{min_ate:.4f}"
    else:
        min_ate_str = str(min_ate)

    if isinstance(max_ate, (int, float)):
        max_ate_str = f"{max_ate:.4f}"
    else:
        max_ate_str = str(max_ate)

    summary_text = f"""COMPREHENSIVE SUMMARY

SEGMENTS:
Total Segments: {total_segments}
Average ATE: {avg_ate_str} m
ATE Std Dev: {std_ate_str} m
Min ATE: {min_ate_str} m
Max ATE: {max_ate_str} m

FULL TRAJECTORY:
Full Trajectory ATE: {full_ate} m
Full Trajectory Length: {full_gt_length} m

VISUALIZATION:
- Segments aligned to origin (0,0,0)
- Full trajectory with drift correction
- Ground Truth: Solid lines
- Predicted: Dashed lines"""

    ax11.text(
        0.1,
        0.5,
        summary_text,
        transform=ax11.transAxes,
        fontsize=10,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
    )

    # Plot 12: Drift correction visualization
    ax12 = fig.add_subplot(3, 4, 12)
    if (
        full_pos_gt is not None
        and full_pos_pred is not None
        and full_pos_pred_corrected is not None
    ):
        # Show original vs corrected prediction
        ax12.plot(
            full_pos_gt[:, 0],
            full_pos_gt[:, 1],
            color="red",
            linewidth=3,
            label="Ground Truth",
        )
        ax12.plot(
            full_pos_pred[:, 0],
            full_pos_pred[:, 1],
            color="orange",
            linewidth=2,
            linestyle=":",
            label="Original Prediction",
        )
        ax12.plot(
            full_pos_pred_corrected[:, 0],
            full_pos_pred_corrected[:, 1],
            color="blue",
            linewidth=2,
            linestyle="--",
            label="Drift Corrected Prediction",
        )

        # Mark start points
        ax12.plot(
            full_gt_initial[0], full_gt_initial[1], "go", markersize=10, label="Start"
        )
        ax12.plot(
            full_pos_pred[0, 0],
            full_pos_pred[0, 1],
            "mo",
            markersize=8,
            label="Original Start",
        )
        ax12.plot(
            full_pos_pred_corrected[0, 0],
            full_pos_pred_corrected[0, 1],
            "co",
            markersize=8,
            label="Corrected Start",
        )

    ax12.set_xlabel("X (m)")
    ax12.set_ylabel("Y (m)")
    ax12.set_title("Drift Correction Effect")
    ax12.grid(True, alpha=0.3)
    ax12.axis("equal")
    ax12.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(
        osp.join(out_dir, "3d_comprehensive_summary.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def display_rich_metrics_tables(
    all_trajectory_results: list, overall_stats: dict, segment_metrics_all: list
) -> None:
    """
    Display comprehensive metrics using rich tables.

    Args:
        all_trajectory_results: List of all trajectory results
        overall_stats: Overall statistics dictionary
        segment_metrics_all: List of all segment metrics
    """
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # 1. Overall Summary Table
    summary_table = Table(
        title="📊 Overall Test Results Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    summary_table.add_column("Metric", style="cyan", no_wrap=True)
    summary_table.add_column("Value", style="green", justify="right")
    summary_table.add_column("Description", style="yellow")

    summary_table.add_row(
        "Total Trajectories",
        str(len(all_trajectory_results)),
        "Number of test trajectories",
    )
    summary_table.add_row(
        "Total Segments",
        str(overall_stats.get("total_segments", 0)),
        "Total number of 5m segments",
    )

    if "segment_statistics" in overall_stats:
        seg_stats = overall_stats["segment_statistics"]
        summary_table.add_row(
            "Avg Segment Length",
            f"{seg_stats.get('avg_segment_length', 0):.2f} m",
            "Average length of segments",
        )
        summary_table.add_row(
            "Avg Segment ATE",
            f"{seg_stats['segment_ate_stats']['mean']:.4f} m",
            "Average ATE across all segments",
        )
        summary_table.add_row(
            "Segment ATE Std",
            f"{seg_stats['segment_ate_stats']['std']:.4f} m",
            "Standard deviation of segment ATE",
        )
        summary_table.add_row(
            "Segment ATE Range",
            f"{seg_stats['segment_ate_stats']['min']:.4f} - {seg_stats['segment_ate_stats']['max']:.4f} m",
            "Min-Max ATE across segments",
        )

    console.print(summary_table)
    console.print()

    # 2. Trajectory-wise Metrics Table
    traj_table = Table(
        title="🚀 Individual Trajectory Results",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold blue",
    )
    traj_table.add_column("Trajectory", style="cyan", no_wrap=True)
    traj_table.add_column("Data", style="magenta")
    traj_table.add_column("Segments", style="green", justify="right")
    traj_table.add_column("Avg ATE (m)", style="yellow", justify="right")
    traj_table.add_column("Avg RMSE (m)", style="red", justify="right")
    traj_table.add_column("T_RTE (m)", style="blue", justify="right")
    traj_table.add_column("D_RTE (m)", style="cyan", justify="right")

    for i, traj_result in enumerate(all_trajectory_results):
        traj_table.add_row(
            f"Traj_{i+1}",
            traj_result.get("data", "N/A"),
            str(traj_result.get("num_segments", 0)),
            f"{traj_result.get('avg_ate', 0):.4f}",
            f"{traj_result.get('avg_P_RMSE', 0):.4f}",
            f"{traj_result.get('avg_t_rte', 0):.4f}",
            f"{traj_result.get('avg_d_rte', 0):.4f}",
        )

    console.print(traj_table)
    console.print()

    # 3. Segment Statistics Table
    if segment_metrics_all:
        seg_table = Table(
            title="📏 Segment-level Statistics",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold green",
        )
        seg_table.add_column("Metric", style="cyan", no_wrap=True)
        seg_table.add_column("Mean", style="green", justify="right")
        seg_table.add_column("Std", style="yellow", justify="right")
        seg_table.add_column("Min", style="red", justify="right")
        seg_table.add_column("Max", style="blue", justify="right")

        # Calculate statistics for key metrics
        metrics_to_show = ["ate", "P_RMSE", "V_RMSE", "t_rte", "d_rte", "ATE", "AVE"]
        for metric in metrics_to_show:
            values = [
                seg.get(metric, 0) for seg in segment_metrics_all if metric in seg
            ]
            if values:
                seg_table.add_row(
                    metric.upper(),
                    f"{np.mean(values):.4f}",
                    f"{np.std(values):.4f}",
                    f"{np.min(values):.4f}",
                    f"{np.max(values):.4f}",
                )

        console.print(seg_table)
        console.print()

    # 4. Full Trajectory Metrics Table
    full_traj_table = Table(
        title="🎯 Full Trajectory Metrics",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold red",
    )
    full_traj_table.add_column("Trajectory", style="cyan", no_wrap=True)
    full_traj_table.add_column("ATE (m)", style="green", justify="right")
    full_traj_table.add_column("T_RTE (m)", style="yellow", justify="right")
    full_traj_table.add_column("D_RTE (m)", style="red", justify="right")
    full_traj_table.add_column("P_RMSE (m)", style="blue", justify="right")
    full_traj_table.add_column("V_RMSE (m/s)", style="magenta", justify="right")

    for i, traj_result in enumerate(all_trajectory_results):
        full_traj = traj_result.get("full_trajectory", {})
        full_traj_table.add_row(
            f"Traj_{i+1}",
            f"{full_traj.get('ate', 0):.4f}",
            f"{full_traj.get('t_rte', 0):.4f}",
            f"{full_traj.get('d_rte', 0):.4f}",
            f"{full_traj.get('P_RMSE', 0):.4f}",
            f"{full_traj.get('V_RMSE', 0):.4f}",
        )

    console.print(full_traj_table)
    console.print()


def create_segments_summary_plot(all_trajectory_results: list, out_dir: str) -> None:
    """
    Create a summary plot showing all trajectory segments.

    Args:
        all_trajectory_results: List of all trajectory results
        out_dir: Output directory
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 12))

    # Plot 1: Segment ATE distribution
    plt.subplot(2, 3, 1)
    all_ate = []
    segment_labels = []
    for i, traj_result in enumerate(all_trajectory_results):
        for j, seg_metric in enumerate(traj_result["segment_metrics"]):
            all_ate.append(seg_metric["ate"])
            segment_labels.append(f"T{i}_S{j}")

    plt.hist(all_ate, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
    plt.xlabel("ATE (m)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Segment ATE")
    plt.grid(True, alpha=0.3)

    # Plot 2: Segment RMSE distribution
    plt.subplot(2, 3, 2)
    all_rmse = [
        seg_metric["P_RMSE"]
        for traj_result in all_trajectory_results
        for seg_metric in traj_result["segment_metrics"]
    ]
    plt.hist(all_rmse, bins=20, alpha=0.7, color="lightcoral", edgecolor="black")
    plt.xlabel("P_RMSE (m)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Segment P_RMSE")
    plt.grid(True, alpha=0.3)

    # Plot 3: Segment length vs ATE
    plt.subplot(2, 3, 3)
    segment_lengths = [
        seg_metric["segment_length"]
        for traj_result in all_trajectory_results
        for seg_metric in traj_result["segment_metrics"]
    ]
    plt.scatter(segment_lengths, all_ate, alpha=0.6, color="green")
    plt.xlabel("Segment Length (m)")
    plt.ylabel("ATE (m)")
    plt.title("Segment Length vs ATE")
    plt.grid(True, alpha=0.3)

    # Plot 4: Trajectory-wise average ATE
    plt.subplot(2, 3, 4)
    traj_avg_ate = [
        np.mean([seg["ate"] for seg in traj["segment_metrics"]])
        for traj in all_trajectory_results
    ]
    traj_names = [f"Traj_{i}" for i in range(len(all_trajectory_results))]
    plt.bar(traj_names, traj_avg_ate, color="orange", alpha=0.7)
    plt.xlabel("Trajectory")
    plt.ylabel("Average ATE (m)")
    plt.title("Average ATE per Trajectory")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Plot 5: Metrics comparison
    plt.subplot(2, 3, 5)
    metrics = ["ate", "t_rte", "d_rte", "P_RMSE", "V_RMSE"]
    metric_names = ["ATE", "T_RTE", "D_RTE", "P_RMSE", "V_RMSE"]
    avg_metrics = []

    for metric in metrics:
        values = [
            seg_metric[metric]
            for traj_result in all_trajectory_results
            for seg_metric in traj_result["segment_metrics"]
        ]
        avg_metrics.append(np.mean(values))

    plt.bar(
        metric_names,
        avg_metrics,
        color=["red", "blue", "green", "purple", "orange"],
        alpha=0.7,
    )
    plt.ylabel("Average Value")
    plt.title("Average Metrics Across All Segments")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Plot 6: Segment count per trajectory
    plt.subplot(2, 3, 6)
    segment_counts = [len(traj["segment_metrics"]) for traj in all_trajectory_results]
    plt.bar(traj_names, segment_counts, color="lightblue", alpha=0.7)
    plt.xlabel("Trajectory")
    plt.ylabel("Number of Segments")
    plt.title("Segment Count per Trajectory")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        osp.join(out_dir, "segments_summary_plot.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Create additional summary statistics
    create_segments_summary_statistics(all_trajectory_results, out_dir)

    # Create 3D summary plot showing all segments aligned to origin
    create_3d_segments_summary(all_trajectory_results, out_dir)


def create_segments_summary_statistics(
    all_trajectory_results: list, out_dir: str
) -> None:
    """
    Create summary statistics for all segments.

    Args:
        all_trajectory_results: List of all trajectory results
        out_dir: Output directory
    """
    import matplotlib.pyplot as plt

    # Collect all segment metrics
    all_segments = []
    for traj_result in all_trajectory_results:
        all_segments.extend(traj_result["segment_metrics"])

    if not all_segments:
        return

    # Create statistics summary
    metrics_to_analyze = ["ate", "t_rte", "d_rte", "P_RMSE", "V_RMSE", "segment_length"]
    metric_names = ["ATE", "T_RTE", "D_RTE", "P_RMSE", "V_RMSE", "Segment Length"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, (metric, name) in enumerate(zip(metrics_to_analyze, metric_names)):
        values = [seg[metric] for seg in all_segments]

        # Box plot
        axes[i].boxplot(
            values, patch_artist=True, boxprops=dict(facecolor="lightblue", alpha=0.7)
        )
        axes[i].set_title(f"{name} Distribution")
        axes[i].set_ylabel(name)
        axes[i].grid(True, alpha=0.3)

        # Add statistics text
        mean_val = np.mean(values)
        std_val = np.std(values)
        median_val = np.median(values)
        min_val = np.min(values)
        max_val = np.max(values)

        stats_text = f"Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMedian: {median_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}"
        axes[i].text(
            0.02,
            0.98,
            stats_text,
            transform=axes[i].transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(
        osp.join(out_dir, "segments_statistics.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()
