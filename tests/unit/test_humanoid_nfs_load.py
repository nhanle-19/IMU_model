import os

import numpy as np

from tests.conftest import NFS_DATA_DIR, requires_nfs
from tartan_imu.dataloader.dataset_HumanoidPostProcessed import HumanoidPostProcessedSequence


def _a_may23_file():
    files = sorted(
        f for f in os.listdir(NFS_DATA_DIR)
        if f.endswith(".npz") and "may23-2026" in f
    )
    return os.path.join(NFS_DATA_DIR, files[0])


@requires_nfs
def test_may23_npz_has_expected_keys():
    d = np.load(_a_may23_file(), allow_pickle=True)
    for key in ("time_ns", "imu_acc_m_s2", "imu_gyro_rad_s",
                "imu_poses", "imu_velocity", "joint_angles"):
        assert key in d.files, f"missing key {key}"
    assert d["joint_angles"].shape[-2:] == (29, 1)
    assert d["imu_acc_m_s2"].shape[-2:] == (3, 3)


@requires_nfs
def test_sequence_loads_and_is_finite():
    # HumanoidPostProcessedSequence is a plain loader object (not a torch
    # Dataset): __init__ parses the npz in .load(), sets .valid, and exposes
    # the parsed arrays via get_feature()/get_target()/get_aux(). There is no
    # __len__/__getitem__, so this characterization test exercises that real
    # API instead of the dict/tuple-of-tensors interface in the task stub.
    seq = HumanoidPostProcessedSequence(
        data_path=_a_may23_file(),
        imu_freq=200.0,
        window_size=20,
        verbose=True,
        use_local_coord=True,
        mode="test",
    )
    assert seq.valid, "loader rejected a known-good may23 file"
    assert seq.data_valid

    feature = seq.get_feature()
    target = seq.get_target()
    aux = seq.get_aux()

    # features = pelvis gyro(3) + pelvis acc(3) + joint_angles(29) = 35 cols
    assert feature.ndim == 2 and feature.shape[1] == 35
    assert feature.shape[0] > 0
    assert target.shape[0] > 0
    assert aux.shape[0] > 0

    for name, arr in (("feature", feature), ("target", target), ("aux", aux)):
        a = np.asarray(arr)
        if np.issubdtype(a.dtype, np.floating):
            assert np.isfinite(a).all(), f"non-finite values in {name}"
