#!/usr/bin/env python3
"""Convert aligned IMU/GT CSV sequences into the retargetted_* NPZ format."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys


def verify_npz_file(npz_path: Path) -> None:
    """Assert the saved NPZ has the expected imu/pos/quat shapes."""
    data = np.load(npz_path)
    ts = data["retargetted_ts"]
    imu = data["retargetted_imu"]
    pos = data["retargetted_pos"]
    quat = data["retargetted_quat"]
    n = len(ts)
    assert imu.shape == (n, 6), f"imu shape {imu.shape} != ({n},6)"
    assert pos.shape == (n, 3), f"pos shape {pos.shape} != ({n},3)"
    assert quat.shape == (n, 4), f"quat shape {quat.shape} != ({n},4)"


def convert_csv_to_npz(sequence_dir, output_dir):
    """
    Convert aligned CSV files to NPZ format with specified structure

    NPZ keys:
      retargetted_ts: timestamps
      retargetted_imu: Nx6 array [ax,ay,az,gx,gy,gz]
      retargetted_pos: Nx3 array [x,y,z]
      retargetted_quat: Nx4 array [qx,qy,qz,qw] (xyzw)
    """
    sequence_dir = Path(sequence_dir)
    output_dir = Path(output_dir)

    imu_file = sequence_dir / 'aligned_imu.csv'
    gt_file = sequence_dir / 'aligned_gt.csv'

    if not imu_file.exists():
        raise FileNotFoundError(f"IMU file not found: {imu_file}")
    if not gt_file.exists():
        raise FileNotFoundError(f"GT file not found: {gt_file}")

    imu_data = pd.read_csv(imu_file)
    gt_data = pd.read_csv(gt_file)

    # timestamps
    ts = imu_data['timestamp'].values

    # choose accel columns (prefer gravity-aligned)
    accel_candidates = [
        ('accel_x_ga','accel_y_ga','accel_z_ga'),
        ('accel_x','accel_y','accel_z'),
        ('ax','ay','az')
    ]
    gyro_candidates = [
        ('gyro_x_ga','gyro_y_ga','gyro_z_ga'),
        ('gyro_x','gyro_y','gyro_z'),
        ('gx','gy','gz')
    ]

    def _find_cols(df, candidates):
        for cand in candidates:
            if all(c in df.columns for c in cand):
                return list(cand)
        return []

    accel_cols = _find_cols(imu_data, accel_candidates)
    gyro_cols = _find_cols(imu_data, gyro_candidates)

    if not accel_cols or not gyro_cols:
        raise ValueError('Required IMU accel/gyro columns not found in aligned_imu.csv')

    imu_accel = imu_data[accel_cols].values
    imu_gyro = imu_data[gyro_cols].values
    imu = np.hstack([imu_accel, imu_gyro])

    # position
    if all(c in gt_data.columns for c in ('x','y','z')):
        pos = gt_data[['x','y','z']].values
    else:
        raise ValueError('GT position columns x,y,z not found')

    # quaternion: prefer qx,qy,qz,qw (xyzw)
    if all(c in gt_data.columns for c in ('qx','qy','qz','qw')):
        quat = gt_data[['qx','qy','qz','qw']].values
    elif all(c in gt_data.columns for c in ('w','x','y','z')):
        # convert from wxyz to xyzw
        q = gt_data[['w','x','y','z']].values
        quat = q[:,[1,2,3,0]]
    else:
        # fallback identity
        quat = np.zeros((len(pos),4), dtype=float)
        quat[:,3] = 1.0

    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_name = sequence_dir.name
    npz_file = output_dir / f"{sequence_name}.npz"

    np.savez_compressed(
        npz_file,
        retargetted_ts=ts,
        retargetted_imu=imu,
        retargetted_pos=pos,
        retargetted_quat=quat
    )

    print(f"Saved NPZ file: {npz_file}")
    print(f"  - Timestamps: {ts.shape}")
    print(f"  - IMU data: {imu.shape} (ax,ay,az,gx,gy,gz)")
    print(f"  - Position: {pos.shape} (x,y,z)")
    print(f"  - Quaternion: {quat.shape} (x,y,z,w)")

    verify_npz_file(npz_file)


def find_sequences(root: Path):
    """Yield sequence dirs under root that contain both aligned IMU and GT CSVs."""
    for aligned_imu in root.rglob('aligned_imu.csv'):
        seq_dir = aligned_imu.parent
        if (seq_dir / 'aligned_gt.csv').exists():
            yield seq_dir


def main():
    """CLI entry: convert a single sequence or bulk-convert under a root dir."""
    parser = argparse.ArgumentParser(description='Convert aligned CSVs to NPZ files')
    parser.add_argument('--sequence_dir', help='Path containing aligned_imu.csv and aligned_gt.csv')
    parser.add_argument('--output_dir', required=True, help='Directory to write NPZ file(s)')
    parser.add_argument('--bulk_root', help='If set, search under this root for sequences and convert all')
    args = parser.parse_args()

    out = Path(args.output_dir)

    if args.bulk_root:
        root = Path(args.bulk_root)
        any_found = False
        for seq in find_sequences(root):
            any_found = True
            try:
                convert_csv_to_npz(seq, out / seq.relative_to(root))
            except Exception as e:
                print(f"[WARN] Skipped {seq}: {e}", file=sys.stderr)
        if not any_found:
            print(f"No sequences found under {root}")
        return

    if not args.sequence_dir:
        print('Either --sequence_dir or --bulk_root must be provided', file=sys.stderr)
        sys.exit(1)

    convert_csv_to_npz(args.sequence_dir, out)


if __name__ == '__main__':
    main()
















