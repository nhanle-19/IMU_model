"""Synthetic-data builders + loader drivers for the dataloader refactor
characterization tests.

The two NPZSequence classes (`HumanoidNPZSequence`, `AirLabNPZSequence`) own all
the gravity-compensation / target-construction logic touched by the refactor, and
their `load()` paths are fully deterministic (no RNG). So we can build a synthetic
npz, run the loader, and snapshot (features, targets, aux) to assert the refactor
changes nothing numerically.
"""

import numpy as np


def _smooth_quat_series(n, seed=0):
    """A slowly-varying unit-quaternion (xyzw) series of length n."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2.0, n)
    # small rotation about a fixed tilted axis so orientation actually varies
    axis = np.array([0.3, 0.6, 0.74])
    axis = axis / np.linalg.norm(axis)
    angle = 0.4 * np.sin(t) + 0.05 * t
    half = angle / 2.0
    xyz = axis[None, :] * np.sin(half)[:, None]
    w = np.cos(half)[:, None]
    quat = np.concatenate([xyz, w], axis=1)  # xyzw
    quat += 1e-6 * rng.randn(n, 4)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return quat.astype(np.float64)


def build_humanoid_npz(path, n=1500, imu_freq=100.0, seed=1):
    """Write a synthetic npz matching HumanoidNPZSequence's preferred schema
    (`time` / `z_pelvis` / `gt_translation` / `gt_orientation`), stage-1 fields."""
    rng = np.random.RandomState(seed)
    ts = (np.arange(n) / imu_freq).astype(np.float64)
    accel = (0.5 * rng.randn(n, 3) + np.array([0.0, 0.0, 9.81])).astype(np.float64)
    gyro = (0.2 * rng.randn(n, 3)).astype(np.float64)
    z_pelvis = np.concatenate([accel, gyro], axis=1)  # [:,0:3]=acc, [:,3:6]=gyro
    pos = np.cumsum(0.01 * rng.randn(n, 3), axis=0).astype(np.float64)
    quat = _smooth_quat_series(n, seed=seed)
    np.savez(
        path,
        time=ts,
        z_pelvis=z_pelvis,
        gt_translation=pos,
        gt_orientation=quat,
    )
    return path


def build_airlab_npz(path, n=1500, imu_freq=100.0, seed=2):
    """Write a synthetic npz matching AirLabNPZSequence's schema
    (`retargetted_ts/imu/pos/quat`). Path includes 'human' for motion_type."""
    rng = np.random.RandomState(seed)
    ts = (np.arange(n) / imu_freq).astype(np.float64)
    accel = (0.5 * rng.randn(n, 3) + np.array([0.0, 0.0, 9.81])).astype(np.float64)
    gyro = (0.2 * rng.randn(n, 3)).astype(np.float64)
    imu = np.concatenate([accel, gyro], axis=1)  # [:,:3]=acc, [:,3:]=gyro
    pos = np.cumsum(0.01 * rng.randn(n, 3), axis=0).astype(np.float64)
    quat = _smooth_quat_series(n, seed=seed)
    np.savez(
        path,
        retargetted_ts=ts,
        retargetted_imu=imu,
        retargetted_pos=pos,
        retargetted_quat=quat,
    )
    return path


def run_humanoid_sequence(npz_path, use_local_coord, imu_freq=100.0, window_size=50):
    from tartan_imu.dataloader.dataset_Humanoid import HumanoidNPZSequence

    seq = HumanoidNPZSequence(
        npz_path,
        imu_freq,
        window_size,
        verbose=False,
        use_local_coord=use_local_coord,
        mode="test",
        stage=1,
    )
    assert seq.valid, "synthetic humanoid sequence failed to load"
    return {
        "features": np.asarray(seq.get_feature(), dtype=np.float64),
        "targets": np.asarray(seq.get_target(), dtype=np.float64),
        "aux": np.asarray(seq.get_aux(), dtype=np.float64),
    }


def run_airlab_sequence(npz_path, use_local_coord, imu_freq=100.0, window_size=50):
    from tartan_imu.dataloader.dataset_AirLab import AirLabNPZSequence

    seq = AirLabNPZSequence(
        npz_path,
        imu_freq,
        window_size,
        verbose=False,
        use_local_coord=use_local_coord,
        mode="test",
    )
    assert seq.valid, "synthetic airlab sequence failed to load"
    return {
        "features": np.asarray(seq.get_feature(), dtype=np.float64),
        "targets": np.asarray(seq.get_target(), dtype=np.float64),
        "aux": np.asarray(seq.get_aux(), dtype=np.float64),
    }
