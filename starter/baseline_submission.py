#!/usr/bin/env python3
# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.
"""Write a guaranteed-valid baseline submission for the Tartan IMU Challenge.

The all-zero baseline is the fastest way to confirm your submission pipeline and
the leaderboard work end-to-end: it has the exact ``window_id`` id space and the
required ``window_id, vx, vy, vz`` columns. Optionally emit a constant velocity
for every window instead of zeros.

Usage:
    python baseline_submission.py \
        --sample_submission /path/to/sample_submission.csv \
        --out submission_zero.csv

    # constant forward speed instead of zeros:
    python baseline_submission.py --sample_submission sample_submission.csv \
        --out sub_const.csv --vx 0.5
"""
from __future__ import annotations

import argparse
import csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero/constant baseline submission")
    parser.add_argument("--sample_submission", required=True,
                        help="Kaggle sample_submission.csv (defines the window_id space)")
    parser.add_argument("--out", required=True, help="output submission CSV path")
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--vz", type=float, default=0.0)
    args = parser.parse_args()

    with open(args.sample_submission, newline="") as handle:
        window_ids = [row["window_id"] for row in csv.DictReader(handle)]

    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["window_id", "vx", "vy", "vz"])
        for wid in window_ids:
            writer.writerow([wid, args.vx, args.vy, args.vz])

    print(f"wrote {args.out} with {len(window_ids)} rows "
          f"(vx={args.vx}, vy={args.vy}, vz={args.vz})")


if __name__ == "__main__":
    main()
