# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""TartanIMU challenge NPZ dataloader for the main spectral training stack.

The challenge data is organized as ``data/{train,val}/<platform>/*.npz`` with
``imu`` in accel-then-gyro order and ``vel_body`` body-frame velocity targets.
The foundation model expects gyro-then-accel features, so this adapter performs
that conversion and otherwise follows the same batch contract as AirLab.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from tartan_imu.dataloader.dataset_AirLab import ResNetLSTMSeqToSeqDataset


_MOTION_TYPE_BY_PLATFORM = {
    "car": 1,
    "dog": 2,
    "drone": 3,
    "human": 4,
}


def _flatten_time_series(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim >= 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim < 2:
        raise ValueError(f"Expected [T, C...] array, got shape {arr.shape}")
    return arr.reshape(arr.shape[0], -1).astype(np.float32)


def _platform_from_path(path: str | Path) -> str | None:
    parts = Path(path).parts
    for platform in _MOTION_TYPE_BY_PLATFORM:
        if platform in parts:
            return platform
    return None


def _rolling_window_mean(values: np.ndarray, window_size: int) -> np.ndarray:
    if len(values) < window_size:
        return np.empty((0, values.shape[1]), dtype=np.float32)
    padded = np.concatenate(
        [np.zeros((1, values.shape[1]), dtype=np.float64), values.astype(np.float64)],
        axis=0,
    )
    cumsum = np.cumsum(padded, axis=0)
    means = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
    return means.astype(np.float32)


class ChallengeNPZSequence:
    def __init__(
        self,
        data_path,
        imu_freq,
        window_size,
        verbose=True,
        use_local_coord=True,
        mode="train",
    ):
        del imu_freq, use_local_coord
        self.interval = int(window_size)
        self.mode = mode
        self.valid = False
        self.data_valid = False
        self.sum_duration = 0.0
        self.motion_type = torch.tensor(0, dtype=torch.long)
        self.ts = None
        self.features = None
        self.targets = None
        self.orientations = None
        self.pos_gt = None
        self.gt_ori = None
        self.valid = self.load(data_path, verbose=verbose)

    def load(self, data_path, verbose=False) -> bool:
        try:
            with np.load(data_path, allow_pickle=True) as npz:
                if "imu" not in npz:
                    raise KeyError("imu")
                imu = _flatten_time_series(npz["imu"])
                velocity = (
                    _flatten_time_series(npz["vel_body"])
                    if "vel_body" in npz
                    else None
                )
                ts = np.asarray(npz["ts"], dtype=np.float64) if "ts" in npz else None
        except (EOFError, OSError, ValueError, KeyError) as exc:
            if verbose:
                logging.warning("Failed to load challenge file %s: %s", data_path, exc)
            return False

        if imu.shape[1] < 6:
            if verbose:
                logging.warning("%s: expected at least 6 IMU channels", data_path)
            return False
        if imu.shape[0] < self.interval:
            return False

        platform = _platform_from_path(data_path)
        self.motion_type = torch.tensor(
            _MOTION_TYPE_BY_PLATFORM.get(platform or "", 0), dtype=torch.long
        )

        # Challenge IMU is [accel, gyro]; the foundation model uses [gyro, accel].
        self.features = np.concatenate([imu[:, 3:6], imu[:, 0:3]], axis=1).astype(
            np.float32
        )
        n = int(self.features.shape[0])

        if velocity is not None:
            velocity = velocity[:n, :3]
            self.targets = _rolling_window_mean(velocity, self.interval)
        else:
            # Hidden test split has no labels. Keep shape valid for inference-only
            # paths; training configs should normally set data.test_dir: null.
            self.targets = np.zeros((n - self.interval + 1, 3), dtype=np.float32)

        if len(self.targets) <= 0:
            return False

        if ts is None or len(ts) < n:
            ts = np.arange(n, dtype=np.float64) / 200.0
        ts = ts[:n]
        self.ts = ts[:, None]
        self.orientations = np.zeros((n, 4), dtype=np.float32)
        self.orientations[:, 3] = 1.0
        self.pos_gt = np.zeros((n, 3), dtype=np.float32)
        self.gt_ori = self.orientations.copy()
        self.sum_duration = float(ts[-1] - ts[0]) if n > 1 else 0.0
        self.data_valid = True
        return True

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_data_valid(self):
        return self.data_valid

    def get_aux(self):
        return np.concatenate(
            [self.ts.reshape(-1, 1), self.orientations, self.pos_gt, self.gt_ori],
            axis=1,
        )


class BasicSequenceData:
    def __init__(self, cfg, source_folder, verbose=False, **kwargs):
        self.window_size = int(
            cfg["model_param"]["window_time"] * cfg["data"]["imu_freq"]
        )
        self.past_data_size = int(
            cfg["model_param"]["past_time"] * cfg["data"]["imu_freq"]
        )
        self.future_data_size = int(
            cfg["model_param"]["future_time"] * cfg["data"]["imu_freq"]
        )
        self.step_size = int(cfg["data"]["imu_freq"] / cfg["data"]["sample_freq"])
        self.seq_len = cfg["train"]["seq_len"]
        self.add_noise = cfg["train"]["add_noise"]
        self.index_map = []
        self.ts, self.gt_pos, self.gt_ori = [], [], []
        self.features, self.targets = [], []
        self.valid_samples = []
        self.data_paths = []
        self.use_local_coord = cfg["data"]["use_local_coord"]
        self.mode = kwargs.get("mode", "train")
        self.length = kwargs.get("current_frame")
        self.valid_all_samples = 0
        max_v_norm = float(cfg["data"].get("max_velocity_norm", 20.0))
        valid_i = 0

        for data_path in source_folder:
            try:
                seq = ChallengeNPZSequence(
                    data_path,
                    cfg["data"]["imu_freq"],
                    self.window_size,
                    verbose=verbose,
                    use_local_coord=self.use_local_coord,
                    mode=self.mode,
                )
                if not seq.valid:
                    continue
            except Exception as exc:
                if verbose:
                    logging.warning("Failed to process %s: %s", data_path, exc)
                continue

            feat, targ, aux, motion_type = (
                seq.get_feature(),
                seq.get_target(),
                seq.get_aux(),
                seq.motion_type,
            )
            if self.length is not None:
                feat = feat[: self.length, :]
                targ = targ[: max(self.length - self.window_size + 1, 0), :]
                aux = aux[: self.length, :]

            index_map = []
            valid_samples = 0
            stop = targ.shape[0] - self.future_data_size - (
                self.seq_len - 1
            ) * self.window_size
            for j in range(self.past_data_size, stop, self.step_size):
                outlier = False
                for k in range(self.seq_len):
                    index = j + k * self.window_size
                    if (
                        index >= targ.shape[0]
                        or np.linalg.norm(targ[index]) > max_v_norm
                    ):
                        outlier = True
                        break
                if not outlier:
                    index_map.append([valid_i, j, motion_type])
                    self.valid_all_samples += 1
                    valid_samples += 1

            if index_map:
                self.data_paths.append(data_path)
                self.index_map.append(index_map)
                self.features.append(feat)
                self.targets.append(targ)
                self.ts.append(aux[:, 0])
                self.gt_pos.append(aux[:, 5:8])
                self.gt_ori.append(aux[:, 8:12])
                self.valid_samples.append(valid_samples)
                self.motion_type = motion_type
                valid_i += 1

        if not self.data_paths:
            raise ValueError(
                "No valid challenge data files found. Expected NPZ files with "
                "'imu' and, for train/val, 'vel_body'."
            )

    def get_data(self):
        return (
            self.features,
            self.targets,
            self.ts,
            self.gt_pos,
            self.gt_ori,
            self.motion_type,
        )

    def get_index_map(self):
        return self.index_map

    def get_merged_index_map(self):
        index_map = []
        for sequence_map in self.index_map:
            index_map += sequence_map
        return index_map


def SeqToSeqDataset(cfg, basic_data: BasicSequenceData, index_map, **kwargs):
    return ResNetLSTMSeqToSeqDataset(cfg, basic_data, index_map, **kwargs)
