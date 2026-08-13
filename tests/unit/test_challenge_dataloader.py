from __future__ import annotations

import numpy as np
import yaml

from tartan_imu.config import configer
from tartan_imu.dataloader import dataset_TartanIMUChallenge as challenge


def _write_challenge_npz(path, n=2600):
    rng = np.random.RandomState(0)
    imu = rng.randn(n, 6).astype(np.float32)
    vel_body = rng.randn(n, 3).astype(np.float32)
    np.savez(path, imu=imu, vel_body=vel_body)


def _load_cfg():
    cfg = configer.load_config(
        "config/datasets/tartanimu/tartan_imu_multihead_smoke.yaml"
    )
    with open(cfg["model"]["model_yaml"], encoding="utf-8") as handle:
        configer.update_recursive(cfg, yaml.load(handle, Loader=yaml.Loader))
    cfg["train"]["seq_len"] = 3
    cfg["data"]["sample_freq"] = 40
    return cfg


def test_challenge_dataset_emits_spectral_training_batch(tmp_path):
    data_dir = tmp_path / "data" / "train" / "car"
    data_dir.mkdir(parents=True)
    npz_path = data_dir / "car_train_0000.npz"
    _write_challenge_npz(npz_path)

    cfg = _load_cfg()
    basic = challenge.BasicSequenceData(cfg, [str(npz_path)], mode="train")
    dataset = challenge.SeqToSeqDataset(
        cfg, basic, basic.get_merged_index_map(), mode="train"
    )

    seq_feat, target, orientation, motion_type, platform_feat = dataset[0]

    assert seq_feat.shape == (3, 6, 40)
    assert target.shape == (3, 3)
    assert orientation.shape == (600, 4)
    assert int(motion_type) == 1
    assert platform_feat.shape == (3, 6, 200)
