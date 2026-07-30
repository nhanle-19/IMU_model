#!/usr/bin/env python3
"""Plot position and quaternion time series from an aligned_gt.csv file."""
import argparse
import csv
import os
from statistics import median

import matplotlib.pyplot as plt


def detect_time_unit_and_normalize(timestamps):
    """Normalize timestamps to seconds-from-start, inferring the source unit."""
    if len(timestamps) < 2:
        t0 = timestamps[0] if timestamps else 0.0
        return [0.0 for _ in timestamps], "unknown"

    diffs = [b - a for a, b in zip(timestamps[:-1], timestamps[1:]) if b - a > 0]
    if not diffs:
        return [float(i) for i in range(len(timestamps))], "index"

    med_dt = median(diffs)
    t0 = timestamps[0]

    # Heuristic: nanoseconds > 1e7, microseconds > 1e4, else seconds
    if med_dt > 1e7:
        scale = 1e9
        unit = "s (from ns)"
    elif med_dt > 1e4:
        scale = 1e6
        unit = "s (from us)"
    else:
        scale = 1.0
        unit = "s"

    return [(t - t0) / scale for t in timestamps], unit


def read_aligned_gt_csv(path):
    """Read timestamp/position/quaternion columns from an aligned_gt.csv file."""
    timestamps = []
    x_list, y_list, z_list = [], [], []
    qx_list, qy_list, qz_list, qw_list = [], [], [], []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        # Accept either x,y,z or tx,ty,tz (legacy)
        fieldnames = reader.fieldnames or []
        if "timestamp" not in fieldnames:
            raise ValueError(f"CSV missing 'timestamp' column. Found: {fieldnames}")

        pos_cols = {
            "x": "x" if "x" in fieldnames else ("tx" if "tx" in fieldnames else None),
            "y": "y" if "y" in fieldnames else ("ty" if "ty" in fieldnames else None),
            "z": "z" if "z" in fieldnames else ("tz" if "tz" in fieldnames else None),
        }
        quat_required = ["qx", "qy", "qz", "qw"]
        missing_quat = [c for c in quat_required if c not in fieldnames]
        if any(v is None for v in pos_cols.values()) or missing_quat:
            raise ValueError(
                "CSV missing required columns. Need timestamp, (x,y,z or tx,ty,tz), qx,qy,qz,qw. "
                f"Found: {fieldnames}"
            )

        for row in reader:
            try:
                timestamps.append(float(row["timestamp"]))
                x_list.append(float(row[pos_cols["x"]]))
                y_list.append(float(row[pos_cols["y"]]))
                z_list.append(float(row[pos_cols["z"]]))
                qx_list.append(float(row["qx"]))
                qy_list.append(float(row["qy"]))
                qz_list.append(float(row["qz"]))
                qw_list.append(float(row["qw"]))
            except (TypeError, ValueError):
                # Skip malformed rows quietly
                continue

    return {
        "timestamp": timestamps,
        "x": x_list,
        "y": y_list,
        "z": z_list,
        "qx": qx_list,
        "qy": qy_list,
        "qz": qz_list,
        "qw": qw_list,
    }


def plot_timeseries(data, output_path):
    """Render position and quaternion vs time to output_path."""
    t_norm, unit = detect_time_unit_and_normalize(data["timestamp"])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Position
    axes[0].plot(t_norm, data["x"], label="x")
    axes[0].plot(t_norm, data["y"], label="y")
    axes[0].plot(t_norm, data["z"], label="z")
    axes[0].set_ylabel("Position [m]")
    axes[0].set_title("Position vs Time")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="best")

    # Quaternion components
    axes[1].plot(t_norm, data["qx"], label="qx")
    axes[1].plot(t_norm, data["qy"], label="qy")
    axes[1].plot(t_norm, data["qz"], label="qz")
    axes[1].plot(t_norm, data["qw"], label="qw")
    axes[1].set_ylabel("Quaternion")
    axes[1].set_xlabel(f"Time [{unit}]")
    axes[1].set_title("Quaternion vs Time")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="best")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    """CLI entry: read the CSV and write the time-series plot."""
    parser = argparse.ArgumentParser(description="Plot time-series from aligned_gt.csv")
    parser.add_argument("--input", required=True, help="Path to aligned_gt.csv")
    parser.add_argument("--out", default=None, help="Output image path (PNG)")
    args = parser.parse_args()

    data = read_aligned_gt_csv(args.input)

    if args.out is not None:
        out_path = args.out
    else:
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_dir = os.path.dirname(os.path.abspath(args.input))
        out_path = os.path.join(out_dir, f"{base}_timeseries.png")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plot_timeseries(data, out_path)
    print(out_path)


if __name__ == "__main__":
    main()


