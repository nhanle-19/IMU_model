# TartanIMU Challenge — Starter Kit

Everything you need to make a valid submission to the **TartanIMU Challenge:
Multi-Platform Inertial Odometry** on Kaggle, plus the released pretrained
baseline.

The task: from a **1.0 s window of 6-axis IMU** (accelerometer + gyroscope,
200 Hz, body frame, gravity retained), predict the sensor's **3-D body-frame
velocity** `(vx, vy, vz)` — the mean velocity over the window. A single model
must handle four embodiments: **car / dog (legged) / drone / human**.

## Data you download from Kaggle

| File | Columns / keys | Notes |
| --- | --- | --- |
| `train/<platform>/*.npz` | `imu (N,6)`, `ts`, `pos`, `quat`, `vel_body (N,3)`, `platform_id`, `fs` | labelled training trajectories |
| `val/<platform>/*.npz` | same as train | validation trajectories |
| `test/test_0000.npz` … `test_0088.npz` | `imu (N,6)`, `ts`, `fs` only | **anonymized** — no pose, no platform anywhere |
| `index/{train,val}_windows.csv` | `window_id, platform, traj_id, win_idx` | window → trajectory map |
| `index/{train,val}_targets.csv` | `window_id, vx, vy, vz` | supervision (body-frame velocity) |
| `index/test_windows.csv` | `window_id, traj_id, win_idx` | **no platform column** (anonymized) |
| `sample_submission.csv` | `window_id, vx, vy, vz` | all-zero template, the exact 30,644 rows to fill |

IMU channel order in every `.npz` is `[ax, ay, az, gx, gy, gz]` (accel m/s²,
gyro rad/s). A window is `1.0 s = 200 frames`, non-overlapping: window `k` of a
trajectory is `imu[k*200:(k+1)*200]`.

> ⚠️ `platform_id` in train/val is **0-based** (`car=0, dog=1, drone=2, human=3`).
> The released TartanIMU model's `motion_type` API is **1-based** — add 1 when
> passing a dataset `platform_id` into that model.

## Submission format

A CSV with exactly these columns, one row per test `window_id` (30,644 rows):

```
window_id,vx,vy,vz
1,0.0,0.0,0.0
2,0.0,0.0,0.0
...
```

`vx, vy, vz` is the predicted mean **body-frame** velocity (m/s) of that window.

## Scoring — macro-averaged 20 m-segment ATE (lower is better)

The leaderboard metric is **macro-averaged 20 m-segment Absolute Trajectory
Error**, in metres:

1. Each test trajectory's ground-truth path is cut into consecutive segments of
   about **20 m of travelled distance**.
2. Within a segment, your per-window velocities are rotated to the world frame
   with ground-truth orientation (used *only* in scoring, never available as a
   model input), multiplied by the window duration, and accumulated into an
   estimated path.
3. The estimated segment is **SE(3)-aligned** to the ground-truth segment
   (Umeyama, rotation + translation, no scale); its error is the RMS position
   error after alignment.
4. Trajectory error = mean over its segments; platform error = mean over its
   trajectories; the **score is the equal-weight mean of the four per-platform
   values**, so no platform (or window count) dominates.

Segments rather than whole trajectories, because a whole-trajectory ATE rewards
predicting nothing: an all-zero submission integrates to a single point whose
aligned error is merely the path's radius of gyration. With fixed-length
segments, standing still loses on **every** platform.

The exact scoring code is `starter/kaggle_metric_ate20.py` — byte-identical to
the metric running on the leaderboard. You cannot score the test set locally
(the pose is withheld), but you can **self-score on the labelled `val` split**
while iterating.

Reference points (Public / Private): ground-truth velocities **0.162 / 0.147**
(the floor), released unified baseline **1.587 / 1.048**, all-zeros
**3.058 / 3.103**, per-platform mean velocity **3.154 / 3.870**.

## Quick start

```bash
# 0. install the library
git clone https://github.com/superxslam/TartanIMU && cd TartanIMU
pip install -e .
pip install huggingface_hub          # for the pretrained baseline

# 1. a guaranteed-valid all-zero submission (verifies your pipeline)
python starter/baseline_submission.py \
    --sample_submission /path/to/sample_submission.csv \
    --out submission_zero.csv

# 2. the released pretrained baseline on the labelled val split
#    (heads route automatically from the CSV's platform column)
python starter/tartanimu_submission.py \
    --test_root  /path/to/val \
    --windows    /path/to/index/val_windows.csv \
    --out        submission_val.csv

# 3. the same model on the anonymized test split — no platform column exists,
#    so a head must be named (or supply your own routing via --routing)
python starter/tartanimu_submission.py \
    --test_root  /path/to/test \
    --windows    /path/to/index/test_windows.csv \
    --head       human \
    --out        submission_tartanimu.csv
    # optional: --checkpoint <local .pt>  --config <local yaml>  --device cpu
```

Upload the resulting CSV on the competition's **Submit Predictions** page.

> **The rules in one line:** predictions must come from a **single model with
> one shared set of weights**. The test set is anonymized precisely so windows
> cannot be routed to four per-platform experts; a single model that adapts
> internally (learned conditioning, mixture-of-experts inside one network) is
> allowed and encouraged.

## Files in this kit

| File | Purpose |
| --- | --- |
| `README.md` | this guide |
| `baseline_submission.py` | writes an all-zero (or constant) valid submission |
| `tartanimu_submission.py` | released pretrained baseline → submission |
| `kaggle_metric_ate20.py` | the exact leaderboard metric (for reference / val self-scoring) |
| `starter.ipynb` | notebook walking through data → prediction → submission |

## Pretrained model

The released baseline weights live on the Hugging Face Hub:
**[`Tartan-IMU/TartanIMU`](https://huggingface.co/Tartan-IMU/TartanIMU)**
(unified 4-head model + per-platform experts, Apache License 2.0; drone-derived
weights are research / non-commercial use). See the model card for
per-platform accuracy and known limitations.
