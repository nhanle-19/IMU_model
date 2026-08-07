# IMU Velocity Diffusion

This branch trains a diffusion-assisted base-velocity estimator from IMU windows.

The deployed pipeline is:

1. **Diffusion candidate generator**: samples multiple plausible base velocities
   from an IMU window.
2. **Velocity selector policy**: uses the same IMU window plus those sampled
   candidate velocities to select or lightly correct the final velocity.

The point is not to force the diffusion model to pick one answer. It proposes a
set of velocity hypotheses, then the policy learns how to use that set.

## Data Format

Real data can be added later as NPZ trajectories. By default the loader expects:

```text
data/
  train/*.npz
  val/*.npz
  test/*.npz
```

Each NPZ should contain:

```text
imu       [T, C] or [1, T, ...]
velocity  [T, 3] or [1, T, 3]
```

Keys, split names, stride, window size, and optional columns are configured in
`configs/default.yaml`.

## Install

```bash
pip install -e ".[dev]"
```

## 1. Train Diffusion Candidates

The default config uses a small synthetic dataset so the code can be
smoke-tested before real data is present.

```bash
python train_diffusion.py --config configs/default.yaml
```

Outputs:

```text
runs/velocity_diffusion/diffusion_last.pt
runs/velocity_diffusion/diffusion_best.pt
```

The validation metric `candidate_min_rmse` measures the oracle error of the
closest sampled candidate.

## 2. Train Selector Policy

```bash
python train_policy.py \
  --config configs/default.yaml \
  --diffusion-checkpoint runs/velocity_diffusion/diffusion_best.pt
```

Outputs:

```text
runs/velocity_diffusion/policy_last.pt
runs/velocity_diffusion/policy_best.pt
```

During policy training, the diffusion model is frozen. It samples candidate
velocities first, then the policy receives:

```text
IMU window + [K, 3] candidate velocities
```

The policy loss combines final velocity MSE, a ranking loss toward the closest
candidate, and a small residual penalty so the selector does not ignore the
candidate set.

## 3. Run Deployment Inference

```bash
python infer.py \
  --config configs/default.yaml \
  --diffusion-checkpoint runs/velocity_diffusion/diffusion_best.pt \
  --policy-checkpoint runs/velocity_diffusion/policy_best.pt \
  --input-npz path/to/trajectory.npz \
  --output-npz runs/velocity_diffusion/predictions.npz
```

The output NPZ contains:

```text
window_starts
candidate_velocities  [N, K, 3]
candidate_weights     [N, K]
selected_velocity     [N, 3]
```

## Real Data Switch

Edit `configs/default.yaml`:

```yaml
data:
  synthetic: false
  root: /path/to/data
  imu_key: imu
  velocity_key: velocity
  window_size: 200
  stride: 10
```

If your NPZ arrays use different names or include extra channels, change
`imu_key`, `velocity_key`, `imu_columns`, and `velocity_columns`.

For real training, increase:

```yaml
diffusion:
  steps: 50
policy:
  num_candidates: 16
train:
  epochs: 100
```
