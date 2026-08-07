# IMU Velocity Diffusion

This branch trains a diffusion-assisted base-velocity estimator from IMU windows.

The deployed pipeline is:

1. **Diffusion candidate generator**: samples multiple plausible base velocities
   from an IMU window.
2. **Velocity distribution refiner**: uses the same IMU window plus the sampled
   velocity distribution to produce one usable velocity estimate.

The point is not to force the diffusion model to pick one answer. It proposes a
set of velocity hypotheses. The second network then aggregates and refines that
sampled distribution: it can use the sample mean as the likely velocity, sample
variance as uncertainty, multimodal structure as a cue that several motions are
plausible, and IMU features to reject inconsistent samples.

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

## 2. Train Distribution Refiner

```bash
python train_policy.py \
  --config configs/default.yaml \
  --diffusion-checkpoint runs/velocity_diffusion/diffusion_best.pt
```

Outputs:

```text
runs/velocity_diffusion/refiner_last.pt
runs/velocity_diffusion/refiner_best.pt
```

During refiner training, the diffusion model is frozen. It samples candidate
velocities first, then the refiner receives:

```text
IMU window + [K, 3] candidate velocities + sample statistics
```

The refiner uses sample mean, variance, min/max range, per-candidate normalized
offsets, and IMU context. Its loss is final velocity MSE plus a small residual
penalty, so the model learns to convert the distribution into one usable
velocity estimate rather than merely picking a single candidate.

## 3. Run Deployment Inference

```bash
python infer.py \
  --config configs/default.yaml \
  --diffusion-checkpoint runs/velocity_diffusion/diffusion_best.pt \
  --refiner-checkpoint runs/velocity_diffusion/refiner_best.pt \
  --input-npz path/to/trajectory.npz \
  --output-npz runs/velocity_diffusion/predictions.npz
```

The output NPZ contains:

```text
window_starts
candidate_velocities  [N, K, 3]
candidate_weights     [N, K]
refined_velocity      [N, 3]
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
