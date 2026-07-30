"""Generate a standard experiment-log entry from a training output directory.

Reads the eval artifacts (trajectory_metrics.csv, statistical_summary.csv),
the config snapshot (default.yaml / model.yaml copied into the output dir), and
the current git/data-split state, then emits a markdown entry ready to paste
into docs/EXPERIMENT_LOG.md or the Notion Experiments DB.

Captures the four improvements requested:
  1. Auto-generated (no hand transcription).
  2. Reproducibility anchors: git commit, split_manifest hash, config snapshot path.
  3. Per-trajectory ATE distribution (not just the average).
  4. A failure/outlier section flagging trajectories that blow up.

Usage:
    python tools/gen_experiment_entry.py \
        --exp_dir ../exp_result/transformer/humanoid_nfs_stage2 \
        --id E002 --title "Transformer Stage 2 (IMU + joints)" \
        --manifest ../dataset/humanoid_nfs/may23-2026/split_manifest.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess


def parse_trajectory_metrics(path: str) -> list[dict]:
    """Parse trajectory_metrics.csv into a list of row dicts (strings)."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_statistical_summary(path: str) -> dict[str, dict]:
    """Parse statistical_summary.csv into {metric_name: {Mean,Std,Min,Max,Median,Count}}.

    The file can list a metric name more than once (e.g. ATE at two metric
    categories); the first occurrence wins.
    """
    out: dict[str, dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["Metric_Name"]
            if name in out:
                continue
            out[name] = {
                "Mean": float(row["Mean"]),
                "Std": float(row["Std"]),
                "Min": float(row["Min"]),
                "Max": float(row["Max"]),
                "Median": float(row["Median"]),
                "Count": int(row["Count"]),
            }
    return out


def flag_outlier_trajectories(rows: list[dict], metric: str = "Full_ATE",
                              factor: float = 3.0) -> list[dict]:
    """Return trajectories whose `metric` exceeds factor * median of the rest.

    Catches the failure mode the averages hide (e.g. one trajectory's ATE
    blowing up to 10x). Empty when the distribution is uniform.
    """
    vals = [float(r[metric]) for r in rows]
    if len(vals) < 3:
        return []
    srt = sorted(vals)
    median = srt[len(srt) // 2]
    if median <= 0:
        return []
    return [r for r in rows if float(r[metric]) > factor * median]


def _sha1_of_file(path: str) -> str:
    if not os.path.isfile(path):
        return "(missing)"
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:12]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "(unknown)"


def _git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        return False


def render_entry(exp_dir: str, exp_id: str, title: str, manifest: str | None) -> str:
    """Render the full markdown experiment-log entry for one run directory."""
    traj = parse_trajectory_metrics(os.path.join(exp_dir, "trajectory_metrics.csv"))
    stats = parse_statistical_summary(os.path.join(exp_dir, "statistical_summary.csv"))
    outliers = flag_outlier_trajectories(traj)

    commit = _git_commit()
    dirty = " (dirty tree)" if _git_dirty() else ""
    manifest_hash = _sha1_of_file(manifest) if manifest else "(n/a)"
    cfg_default = os.path.join(exp_dir, "default.yaml")
    cfg_model = os.path.join(exp_dir, "model.yaml")

    lines = [f"## {exp_id} — {title}", ""]

    # Run info / reproducibility anchors
    lines += [
        "### Run info (复现锚点)",
        f"- exp_dir: `{exp_dir}`",
        f"- git commit: `{commit}`{dirty}",
        f"- split_manifest sha1: `{manifest_hash}`" + (f" (`{manifest}`)" if manifest else ""),
        f"- config snapshot: `{cfg_default}`, `{cfg_model}`"
        + ("" if os.path.isfile(cfg_default) else "  ⚠️ default.yaml 缺失"),
        "",
    ]

    # Aggregate results
    lines += ["### 结果 (汇总, segment-level)", "", "| 指标 | Mean | Std | Min | Max | Median | N |",
              "|---|---|---|---|---|---|---|"]
    for m in ("ATE", "T_RTE", "D_RTE", "P_RMSE", "V_RMSE", "AVE"):
        if m in stats:
            s = stats[m]
            lines.append(
                f"| {m} | {s['Mean']:.4f} | {s['Std']:.4f} | {s['Min']:.4f} | "
                f"{s['Max']:.4f} | {s['Median']:.4f} | {s['Count']} |"
            )
    lines.append("")

    # Per-trajectory distribution
    lines += ["### 逐轨迹分布", "", "| Trajectory | Episode | Segs | Avg_ATE | Full_ATE | Avg_T_RTE |",
              "|---|---|---|---|---|---|"]
    for r in traj:
        ep = r.get("Data_Name", "?")
        lines.append(
            f"| {r['Trajectory_ID']} | {ep} | {r.get('Num_Segments','?')} | "
            f"{float(r.get('Avg_ATE',0)):.4f} | {float(r.get('Full_ATE',0)):.4f} | "
            f"{float(r.get('Avg_T_RTE',0)):.4f} |"
        )
    lines.append("")

    # Failure / outlier section
    lines += ["### 失败 / 异常轨迹"]
    if outliers:
        lines.append("⚠️ 以下轨迹 Full_ATE 远超中位数（>3×），汇总平均掩盖了它们：")
        for r in outliers:
            lines.append(
                f"- **{r['Trajectory_ID']}** ({r.get('Data_Name','?')}): "
                f"Full_ATE={float(r['Full_ATE']):.4f} m, segments={r.get('Num_Segments','?')}"
            )
    else:
        lines.append("- 无异常轨迹（分布均匀）。")
    lines.append("")

    # Long-horizon drift diagnostic (per-axis + velocity-bias attribution)
    drift_rows = _collect_drift(exp_dir)
    if drift_rows:
        lines += [
            "### 长程漂移诊断 (分轴 + 速度偏置归因)",
            "",
            "末端漂移、主导轴、各轴速度偏置；`bias解释占比` = |速度偏置|×时长 / 实测漂移"
            "（接近 100% 说明漂移由系统性速度偏置积分造成，而非随机噪声）。",
            "",
            "| Episode | 时长(s) | 末端漂移(m) | 主导轴 | Vbias X/Y/Z (m/s) | bias解释占比 | mid/final |",
            "|---|---|---|---|---|---|---|",
        ]
        for ep, r in drift_rows:
            lines.append(
                f"| {ep} | {r.duration_s:.0f} | {r.final_error_norm:.2f} | {r.dominant_axis} | "
                f"{r.vel_bias_xyz[0]:+.4f} / {r.vel_bias_xyz[1]:+.4f} / {r.vel_bias_xyz[2]:+.4f} | "
                f"{r.bias_explains_fraction:.0%} | {r.mid_vs_final_ratio:.2f} |"
            )
        lines.append("")
    return "\n".join(lines)


def _collect_drift(exp_dir: str):
    """Find per-trajectory trajectory.txt files and run drift analysis on each."""
    import glob
    import sys
    # Make sibling module importable whether run as a script or as tools.*.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from tools.drift_analysis import analyze_trajectory_drift
    except ModuleNotFoundError:
        from drift_analysis import analyze_trajectory_drift

    rows = []
    pattern = os.path.join(exp_dir, "**", "trajectory.txt")
    for f in sorted(glob.glob(pattern, recursive=True)):
        # Only the per-episode top-level trajectory.txt (skip per-segment ones).
        if os.sep + "segment_" in f:
            continue
        try:
            r = analyze_trajectory_drift(f)
        except Exception:
            continue
        ep = os.path.basename(os.path.dirname(f)).replace("humanoid_", "")[:34]
        rows.append((ep, r))
    return rows


def main() -> None:
    """CLI entry: render an entry and print it or append it to --out."""
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--out", default=None, help="append the entry to this file instead of stdout")
    args = p.parse_args()
    entry = render_entry(args.exp_dir, args.id, args.title, args.manifest)
    if args.out:
        with open(args.out, "a") as f:
            f.write("\n" + entry + "\n")
        print(f"Appended {args.id} to {args.out}")
    else:
        print(entry)


if __name__ == "__main__":
    main()
