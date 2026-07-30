"""Velocity-outlier filtering for TartanIMU GT labels.

TartanIMU car/dog GT positions have occasional jumps that differentiate into
physically-impossible velocity spikes (car peaks at 210 m/s, p50 only 0.56).
These wreck long-horizon ATE: one 210 m/s frame at 40 Hz jumps position ~5 m.
The filter clips per-frame speed to a physical ceiling, leaving normal motion
(e.g. human, max 1.44 m/s) untouched.
"""
import numpy as np

from tartan_imu.dataloader._common import clip_velocity_outliers


def test_normal_velocities_unchanged():
    # All speeds well under the ceiling -> returned verbatim.
    rng = np.random.RandomState(0)
    vel = rng.uniform(-1.0, 1.0, size=(100, 3))  # |v| <= ~1.7 m/s
    out = clip_velocity_outliers(vel, max_speed=15.0)
    assert np.allclose(out, vel), "normal velocities must be untouched"


def test_spike_is_clipped_to_ceiling():
    # One frame is a 210 m/s spike; it must be scaled down to the ceiling,
    # keeping its direction.
    vel = np.zeros((5, 3))
    vel[2] = [210.0, 0.0, 0.0]
    out = clip_velocity_outliers(vel, max_speed=15.0)
    assert np.linalg.norm(out[2]) <= 15.0 + 1e-6, "spike must be clipped"
    assert out[2, 0] > 0, "direction must be preserved"
    # non-spike frames untouched
    assert np.allclose(out[[0, 1, 3, 4]], vel[[0, 1, 3, 4]])


def test_multi_axis_spike_direction_preserved():
    vel = np.zeros((3, 3))
    vel[1] = [30.0, 40.0, 0.0]  # norm 50 -> clip to 15, keep 3:4 ratio
    out = clip_velocity_outliers(vel, max_speed=15.0)
    n = np.linalg.norm(out[1])
    assert abs(n - 15.0) < 1e-6
    # 3:4:0 direction preserved
    assert abs(out[1, 0] / out[1, 1] - 0.75) < 1e-6
