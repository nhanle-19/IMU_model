"""Tests for the experiment-entry generator (tools/gen_experiment_entry.py).

The generator turns an experiment output dir + config + git state into a
standard markdown entry with: run-info anchors (git commit, manifest hash,
config snapshot), per-trajectory ATE distribution, and aggregate metrics. Pure
parsing/formatting is import-testable without a real training run.
"""
import textwrap


from tools.gen_experiment_entry import (
    parse_trajectory_metrics,
    parse_statistical_summary,
    flag_outlier_trajectories,
)


TRAJ_CSV = textwrap.dedent("""\
    Trajectory_ID,Data_Name,Robot_Type,Num_Segments,Avg_ATE,Avg_T_RTE,Avg_D_RTE,Full_ATE
    Traj_1,epA.npz,human,1,0.1866,0.0839,0.3428,0.1866
    Traj_2,epB.npz,human,12,1.1620,0.1321,0.1946,10.6060
    Traj_3,epC.npz,human,1,0.2724,0.1617,0.3306,0.2724
    Traj_4,epD.npz,human,1,1.9062,0.3565,0.6815,1.9062
""")

STAT_CSV = textwrap.dedent("""\
    Metric_Category,Metric_Name,Mean,Std,Min,Max,Median,Count
    Segment_Level,ATE,1.087265,0.479392,0.186614,1.906204,1.183347,15
    Segment_Level,T_RTE,0.145814,0.069404,0.077078,0.356513,0.119421,15
""")


def test_parse_trajectory_metrics(tmp_path):
    p = tmp_path / "trajectory_metrics.csv"
    p.write_text(TRAJ_CSV)
    rows = parse_trajectory_metrics(str(p))
    assert len(rows) == 4
    assert rows[0]["Trajectory_ID"] == "Traj_1"
    assert rows[0]["Data_Name"] == "epA.npz"
    assert abs(float(rows[1]["Full_ATE"]) - 10.6060) < 1e-6


def test_parse_statistical_summary(tmp_path):
    p = tmp_path / "statistical_summary.csv"
    p.write_text(STAT_CSV)
    stats = parse_statistical_summary(str(p))
    # keyed by metric name; first ATE row wins (segment-level)
    assert abs(stats["ATE"]["Mean"] - 1.087265) < 1e-6
    assert stats["ATE"]["Count"] == 15
    assert abs(stats["T_RTE"]["Median"] - 0.119421) < 1e-6


def test_flag_outlier_trajectories(tmp_path):
    p = tmp_path / "trajectory_metrics.csv"
    p.write_text(TRAJ_CSV)
    rows = parse_trajectory_metrics(str(p))
    # Traj_2 Full_ATE 10.6 is a gross outlier vs the others (~0.2-1.9).
    outliers = flag_outlier_trajectories(rows, metric="Full_ATE", factor=3.0)
    ids = {o["Trajectory_ID"] for o in outliers}
    assert "Traj_2" in ids
    assert "Traj_1" not in ids


def test_flag_outliers_empty_when_uniform(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(
        "Trajectory_ID,Data_Name,Full_ATE\n"
        "Traj_1,a,1.0\nTraj_2,b,1.1\nTraj_3,c,0.9\n"
    )
    rows = parse_trajectory_metrics(str(p))
    assert flag_outlier_trajectories(rows, metric="Full_ATE", factor=3.0) == []
