#!/usr/bin/env python3
"""Plot IMU, position, and quaternion time series from a retargetted NPZ file."""
import argparse
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np


def normalize_time_seconds(ts: np.ndarray):
    """Normalize a timestamp array to seconds-from-start, inferring the unit."""
    if ts.size < 2:
        return np.zeros_like(ts, dtype=float), "s"
    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return np.arange(ts.size, dtype=float), "index"
    med_dt = float(median(diffs.tolist()))
    if med_dt > 1e7:
        scale = 1e9
        unit = "s (from ns)"
    elif med_dt > 1e4:
        scale = 1e6
        unit = "s (from us)"
    else:
        scale = 1.0
        unit = "s"
    t0 = float(ts[0])
    return (ts - t0) / scale, unit


def plot_npz(npz_path: Path, out_path: Path | None = None) -> Path:
    """Plot IMU/pos/quat from the NPZ to out_path and return the written path."""
    data = np.load(npz_path)
    ts = data["retargetted_ts"]
    imu = data["retargetted_imu"]  # Nx6 [ax,ay,az,gx,gy,gz]
    pos = data["retargetted_pos"]  # Nx3 [x,y,z]
    quat = data["retargetted_quat"]  # Nx4 [qx,qy,qz,qw]

    t, unit = normalize_time_seconds(ts.astype(float))

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # Accel
    axes[0].plot(t, imu[:, 0], label="ax")
    axes[0].plot(t, imu[:, 1], label="ay")
    axes[0].plot(t, imu[:, 2], label="az")
    axes[0].set_ylabel("Accel [m/s^2]")
    axes[0].set_title("IMU Acceleration")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(loc="best")

    # Gyro
    axes[1].plot(t, imu[:, 3], label="gx")
    axes[1].plot(t, imu[:, 4], label="gy")
    axes[1].plot(t, imu[:, 5], label="gz")
    axes[1].set_ylabel("Gyro [rad/s]")
    axes[1].set_title("IMU Gyroscope")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(loc="best")

    # Position
    axes[2].plot(t, pos[:, 0], label="x")
    axes[2].plot(t, pos[:, 1], label="y")
    axes[2].plot(t, pos[:, 2], label="z")
    axes[2].set_ylabel("Position [m]")
    axes[2].set_title("Position")
    axes[2].grid(True, linestyle=":", alpha=0.5)
    axes[2].legend(loc="best")

    # Quaternion
    axes[3].plot(t, quat[:, 0], label="qx")
    axes[3].plot(t, quat[:, 1], label="qy")
    axes[3].plot(t, quat[:, 2], label="qz")
    axes[3].plot(t, quat[:, 3], label="qw")
    axes[3].set_ylabel("Quaternion")
    axes[3].set_xlabel(f"Time [{unit}]")
    axes[3].set_title("Orientation (xyzw)")
    axes[3].grid(True, linestyle=":", alpha=0.5)
    axes[3].legend(loc="best")

    plt.tight_layout()

    if out_path is None:
        out_path = npz_path.with_name(npz_path.stem + "_timeseries.png")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    """CLI entry: plot the given NPZ and print the output path."""
    ap = argparse.ArgumentParser(description="Plot time series from NPZ (ts, imu, pos, quat)")
    ap.add_argument("--input", required=True, help="Path to NPZ file")
    ap.add_argument("--out", help="Optional PNG output path")
    args = ap.parse_args()

    npz_path = Path(args.input)
    out_path = Path(args.out) if args.out else None
    out = plot_npz(npz_path, out_path)
    print(out)


if __name__ == "__main__":
    main()
















