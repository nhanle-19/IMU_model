# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Post-processed Humanoid sequence dataloader (pre-aligned IMU/GT)."""
import logging
import random

import numpy as np
import torch
from torch.utils.data import Dataset



DEFAULT_META_IMU_NAMES = ("livox", "pelvis", "torso")
PELVIS_IMU_NAME = "pelvis"


def _moving_average_lowpass(feat: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply a simple edge-padded moving average along the time axis."""
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return feat
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    padded = np.pad(feat, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    filtered = np.empty_like(feat, dtype=np.float32)
    for channel in range(feat.shape[1]):
        filtered[:, channel] = np.convolve(
            padded[:, channel], kernel, mode="valid"
        )
    return filtered


def _squeeze_batch_axis(array: np.ndarray, expected_ndim: int, name: str) -> np.ndarray:
    """Convert [1, ...] array to [...] and validate rank."""
    arr = np.asarray(array)
    if arr.ndim != expected_ndim:
        raise ValueError(
            f"{name} rank mismatch, expected {expected_ndim} dims and got {arr.ndim}."
        )
    if arr.shape[0] != 1:
        raise ValueError(f"{name} expected batch axis size 1, got shape {arr.shape}.")
    return arr[0]


def _to_imu_names(raw_names) -> tuple:
    """Normalize stored meta_imu_names into a python tuple."""
    if raw_names is None:
        return DEFAULT_META_IMU_NAMES

    names = raw_names
    if isinstance(names, np.ndarray):
        if names.ndim == 0:
            names = names.item()
        else:
            names = names.tolist()

    if isinstance(names, str):
        names = [names]

    if not isinstance(names, (list, tuple)):
        return DEFAULT_META_IMU_NAMES

    return tuple(str(x) for x in names)


def _normalize_xyzw_quaternion(quat: np.ndarray) -> np.ndarray:
    """Normalize quaternion [qx, qy, qz, qw] with safe fallback."""
    quat = quat.astype(np.float64)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    valid = norm[:, 0] > 1e-12
    if not np.any(valid):
        out = np.zeros_like(quat)
        out[:, 3] = 1.0
        return out

    out = np.zeros_like(quat)
    out[valid] = quat[valid] / norm[valid]
    out[~valid, 3] = 1.0
    return out


class HumanoidPostProcessedSequence(object):
    """
    Loader for Post_Processed_Data format.

    Expected keys:
      - time_ns: [1, T]
      - imu_acc_m_s2: [1, T, 3, 3]
      - imu_gyro_rad_s: [1, T, 3, 3]
      - imu_poses: [1, T, 3, 7], world frame, quaternion xyzw
      - imu_velocity: [1, T, 3, 3], body frame of each IMU
      - joint_angles: [1, T, 29, 1]
      - meta_imu_names: optional
    """

    def __init__(
        self,
        data_path,
        imu_freq,
        window_size,
        verbose=True,
        use_local_coord=True,
        mode="test",
        stage=2,
        zero_dq=False,
    ):
        super().__init__()
        (
            self.ts,
            self.features,
            self.targets,
            self.orientations,
            self.pos_gt,
            self.gt_ori,
        ) = (None, None, None, None, None, None)
        self.imu_freq = imu_freq
        self.interval = window_size
        self.data_valid = False
        self.sum_duration = 0
        self.valid = False
        self.mode = mode
        self.stage = stage
        self.zero_dq = zero_dq  # Stage 3 ablation: force the dq channel to 0
        self.use_local_coord = use_local_coord
        self.motion_type = torch.tensor(4, dtype=torch.long)  # human
        self.data_path = data_path
        self.get_gt = True

        if data_path is not None:
            self.valid = self.load(data_path, verbose=verbose)

    def load(self, data_path, verbose=False):
        try:
            all_data = np.load(data_path, allow_pickle=True)
        except (EOFError, OSError, ValueError) as e:
            if verbose:
                logging.warning(f"Failed to load file {data_path}: {e}")
            return False

        required_keys = [
            "time_ns",
            "imu_acc_m_s2",
            "imu_gyro_rad_s",
            "imu_poses",
            "imu_velocity",
            "joint_angles",
        ]
        missing = [k for k in required_keys if k not in all_data]
        if missing:
            if verbose:
                logging.warning(f"Missing keys in {data_path}: {missing}")
            return False

        try:
            time_ns_raw = np.asarray(all_data["time_ns"])
            imu_acc = _squeeze_batch_axis(
                all_data["imu_acc_m_s2"], expected_ndim=4, name="imu_acc_m_s2"
            )
            imu_gyro = _squeeze_batch_axis(
                all_data["imu_gyro_rad_s"], expected_ndim=4, name="imu_gyro_rad_s"
            )
            imu_poses = _squeeze_batch_axis(
                all_data["imu_poses"], expected_ndim=4, name="imu_poses"
            )
            imu_velocity = _squeeze_batch_axis(
                all_data["imu_velocity"], expected_ndim=4, name="imu_velocity"
            )
            joint_angles = _squeeze_batch_axis(
                all_data["joint_angles"], expected_ndim=4, name="joint_angles"
            )
        except ValueError as e:
            if verbose:
                logging.warning(f"Shape validation failed for {data_path}: {e}")
            return False

        if (
            imu_acc.shape[1:] != (3, 3)
            or imu_gyro.shape[1:] != (3, 3)
            or imu_poses.shape[1:] != (3, 7)
            or imu_velocity.shape[1:] != (3, 3)
            or joint_angles.shape[1:] != (29, 1)
        ):
            if verbose:
                logging.warning(
                    f"Unexpected tensor shape in {data_path}: "
                    f"imu_acc={imu_acc.shape}, imu_gyro={imu_gyro.shape}, "
                    f"imu_poses={imu_poses.shape}, imu_velocity={imu_velocity.shape}, "
                    f"joint_angles={joint_angles.shape}"
                )
            return False

        T = imu_acc.shape[0]
        if not (
            imu_acc.shape[0]
            == imu_gyro.shape[0]
            == imu_poses.shape[0]
            == imu_velocity.shape[0]
            == joint_angles.shape[0]
        ):
            if verbose:
                logging.warning(f"Length mismatch in {data_path}.")
            return False

        imu_names = _to_imu_names(all_data.get("meta_imu_names", None))
        if PELVIS_IMU_NAME in imu_names:
            pelvis_idx = imu_names.index(PELVIS_IMU_NAME)
        else:
            pelvis_idx = 1
            if verbose:
                logging.warning(
                    f"No '{PELVIS_IMU_NAME}' in meta_imu_names={imu_names}, using index 1."
                )

        if pelvis_idx >= 3:
            if verbose:
                logging.warning(f"Pelvis index {pelvis_idx} is out of bounds for IMU count=3.")
            return False

        if time_ns_raw.ndim == 1:
            time_ns = time_ns_raw
        elif time_ns_raw.ndim == 2 and time_ns_raw.shape[0] == 1:
            time_ns = time_ns_raw[0]
        elif time_ns_raw.ndim == 4 and time_ns_raw.shape[0] == 1 and time_ns_raw.shape[-1] == 1:
            # Supports [1, T, 3, 1] time format aligned with IMU axis.
            time_ns = time_ns_raw[0, :, pelvis_idx, 0]
        else:
            if verbose:
                logging.warning(f"Unsupported time_ns shape in {data_path}: {time_ns_raw.shape}")
            return False

        if time_ns.shape[0] != T:
            if verbose:
                logging.warning(
                    f"time_ns length mismatch in {data_path}: "
                    f"time={time_ns.shape[0]}, data={T}"
                )
            return False

        ts = time_ns.astype(np.float64)
        if np.nanmax(np.abs(ts)) > 1e12:
            ts = ts * 1e-9  # ns -> s

        if len(ts) < 2:
            return False

        if np.any(np.diff(ts) <= 0):
            if verbose:
                logging.warning(f"Timestamps are not strictly increasing in {data_path}.")
            return False

        pelvis_acc = imu_acc[:, pelvis_idx, :].astype(np.float64)
        pelvis_gyro = imu_gyro[:, pelvis_idx, :].astype(np.float64)
        pelvis_pose = imu_poses[:, pelvis_idx, :].astype(np.float64)  # [T, 7]
        pelvis_velocity = imu_velocity[:, pelvis_idx, :].astype(np.float64)
        joint_q = joint_angles[:, :, 0].astype(np.float64)  # [T, 29]

        pos_world = pelvis_pose[:, :3]
        quat_xyzw = _normalize_xyzw_quaternion(pelvis_pose[:, 3:7])

        self.seq_len = T
        if self.seq_len < self.imu_freq * 10:
            if verbose:
                logging.warning(
                    f"Sequence too short in {data_path}: {self.seq_len} < {int(self.imu_freq * 10)}"
                )
            return False

        pos_gt = pos_world - pos_world[0]
        # Stage-gated feature assembly. The model builds bn_input to match
        # input_dim, so this MUST agree with the configured stage:
        #   stage 1 -> 6   (pelvis gyro 3 + pelvis acc 3)
        #   stage 2 -> 35  (+ 29 joint angles q)
        #   stage 3 -> 64  (IMU 6 + 29 joints x (q, dq) interleaved)
        # Stage 3's masked-attention model reshapes the joint block via
        # view(B*T, 29, 2), so each joint must carry (q, dq) contiguously. The NPZ
        # stores only q, so dq is the finite-difference joint angular velocity
        # dq[t] = (q[t] - q[t-1]) / dt (dq[0] = 0), matching the dataset_Humanoid
        # loader's z_*_link dq semantics.
        if self.stage == 1:
            features = np.concatenate([pelvis_gyro, pelvis_acc], axis=1)  # [T, 6]
        elif self.stage == 3:
            joint_dq = np.zeros_like(joint_q)
            if not self.zero_dq:
                dt = float(np.median(np.diff(ts)))
                joint_dq[1:] = (joint_q[1:] - joint_q[:-1]) / dt
            # Interleave to [T, 29, 2] = (q, dq) per joint, then flatten to [T, 58].
            joint_qdq = np.stack([joint_q, joint_dq], axis=2).reshape(joint_q.shape[0], -1)
            features = np.concatenate([pelvis_gyro, pelvis_acc, joint_qdq], axis=1)  # [T, 64]
        else:
            features = np.concatenate([pelvis_gyro, pelvis_acc, joint_q], axis=1)  # [T, 35]

        if self.interval >= len(pos_gt):
            return False

        self.sum_duration = ts[-1] - ts[0]
        self.data_valid = True
        self.dt = ts[1] - ts[0]

        self.ts = ts[:, np.newaxis][:-1]
        self.features = features[:-1]
        self.orientations = quat_xyzw[:-1]
        self.pos_gt = pos_gt[:-1]
        self.gt_ori = quat_xyzw[:-1]
        pelvis_velocity = pelvis_velocity[:-1]

        interval_int = int(self.interval)
        if self.use_local_coord:
            if pelvis_velocity.shape[0] >= interval_int:
                num_windows = int(pelvis_velocity.shape[0] - interval_int + 1)
                vel_targets = np.zeros((num_windows, pelvis_velocity.shape[1]))
                for i in range(num_windows):
                    window = pelvis_velocity[i : i + interval_int]
                    vel_targets[i] = np.mean(window, axis=0)
                self.targets = vel_targets
            else:
                return False
        else:
            if len(self.pos_gt) > interval_int:
                self.targets = (
                    self.pos_gt[interval_int:] - self.pos_gt[:-interval_int]
                )
            else:
                return False

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


class BasicSequenceData(object):
    """Basic sequence manager for post-processed humanoid NPZ files."""

    def __init__(self, cfg, source_folder, verbose=False, **kwargs):
        super(BasicSequenceData, self).__init__()
        self.window_size = int(cfg["model_param"]["window_time"] * cfg["data"]["imu_freq"])
        self.past_data_size = int(cfg["model_param"]["past_time"] * cfg["data"]["imu_freq"])
        self.future_data_size = int(cfg["model_param"]["future_time"] * cfg["data"]["imu_freq"])
        self.step_size = int(cfg["data"]["imu_freq"] / cfg["data"]["sample_freq"])
        self.seq_len = cfg["train"]["seq_len"]
        self.add_noise = cfg["train"]["add_noise"]
        self.index_map = []
        self.ts, self.gt_pos, self.gt_ori = [], [], []
        self.features, self.targets = [], []
        self.valid_t, self.valid_samples = [], []
        self.data_paths = []
        self.valid_continue_good_time = 0.1
        self.use_local_coord = cfg["data"]["use_local_coord"]
        self.mode = kwargs.get("mode", "train")
        self.is_transformer = cfg["model"]["model_name"] == "Transformer"

        sum_t = 0
        self.length = kwargs.get("current_frame")
        self.valid_sum_t = 0
        self.valid_all_samples = 0
        valid_i = 0
        data_list = source_folder

        if verbose:
            logging.info(
                f"Processing {len(data_list)} post-processed files for {self.mode} mode"
            )

        for i in range(len(data_list)):
            try:
                seq = HumanoidPostProcessedSequence(
                    data_list[i],
                    cfg["data"]["imu_freq"],
                    self.window_size,
                    verbose=verbose,
                    use_local_coord=self.use_local_coord,
                    mode=self.mode,
                    stage=cfg.get("model_param", {}).get("stage", 2),
                    zero_dq=cfg.get("model_param", {}).get("stage3_zero_dq", False),
                )
                if seq.valid is False:
                    continue
            except Exception as e:
                if verbose:
                    logging.warning(f"Failed to process {data_list[i]}: {e}")
                continue

            feat, targ, aux, motion_type = (
                seq.get_feature(),
                seq.get_target(),
                seq.get_aux(),
                seq.motion_type,
            )

            if self.length is not None:
                feat = feat[: self.length, :]
                targ = targ[: (self.length - int(cfg["data"]["imu_freq"])), :]
                aux = aux[: self.length, :]

            sum_t += seq.sum_duration
            valid_samples = 0
            index_map = []
            step_size = self.step_size

            start_search = self.past_data_size
            end_search = targ.shape[0] - self.future_data_size - (self.seq_len - 1) * self.window_size

            if start_search >= end_search:
                continue

            for j in range(start_search, end_search, step_size):
                index_map.append([valid_i, j, motion_type])
                self.valid_all_samples += 1
                valid_samples += 1

            if len(index_map) > 0:
                self.data_paths.append(data_list[i])
                self.index_map.append(index_map)
                self.features.append(feat)
                self.targets.append(targ)
                self.ts.append(aux[:, 0])
                self.gt_pos.append(aux[:, 5:8])
                self.gt_ori.append(aux[:, 8:12])
                self.valid_samples.append(valid_samples)
                valid_i += 1
                self.motion_type = motion_type

        if verbose:
            logging.info(f"post-processed datasets sum time {sum_t:.2f}s")

        if len(self.data_paths) == 0:
            raise ValueError("No valid post-processed data files found.")

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
        for i in range(len(self.index_map)):
            index_map += self.index_map[i]
        return index_map


class ResNetLSTMSeqToSeqDataset(Dataset):
    """PyTorch Dataset for post-processed humanoid sequences."""

    def __init__(self, cfg, basic_data: BasicSequenceData, index_map, **kwargs):
        super(ResNetLSTMSeqToSeqDataset, self).__init__()
        self.window_size = basic_data.window_size
        self.past_data_size = basic_data.past_data_size
        self.future_data_size = basic_data.future_data_size
        self.step_size = basic_data.step_size
        self.seq_len = basic_data.seq_len

        self.add_bias_noise = cfg["augment"]["add_bias_noise"]
        self.accel_bias_range = cfg["augment"]["accel_bias_range"]
        self.gyro_bias_range = cfg["augment"]["gyro_bias_range"]
        if self.add_bias_noise is False:
            self.accel_bias_range = 0.0
            self.gyro_bias_range = 0.0
        self.add_gravity_noise = cfg["augment"]["add_gravity_noise"]
        self.gravity_noise_theta_range = cfg["augment"]["gravity_noise_theta_range"]

        self.feat_acc_sigma = cfg["augment"]["feat_acc_sigma"]
        self.feat_gyr_sigma = cfg["augment"]["feat_gyr_sigma"]
        self.use_local_coord = cfg["data"]["use_local_coord"]
        self.is_transformer = basic_data.is_transformer
        pcfg = cfg.get("model_param", {}).get("platform_conditioning", {})
        lp_cfg = cfg["data"].get("low_pass_filter", {})
        self.return_platform_windows = bool(pcfg.get("enabled", False))
        self.low_pass_before_downsample = bool(
            lp_cfg.get("enabled", self.return_platform_windows)
        )
        self.low_pass_kernel_size = int(lp_cfg.get("kernel_size", self.step_size))

        self.mode = kwargs.get("mode", "train")
        self.shuffle, self.transform, self.gauss = False, False, False
        if self.mode == "train":
            self.shuffle = True
            self.transform = True
            self.gauss = True
        elif self.mode == "val":
            self.shuffle = True
        elif self.mode == "test":
            self.shuffle = False
        self.window_stride = self.window_size if self.is_transformer else self.step_size
        self.window_total = (
            self.past_data_size + self.window_size + self.future_data_size
        )
        self.window_offsets = np.arange(
            0, self.window_total, self.step_size, dtype=np.int64
        )
        self.platform_window_offsets = np.arange(
            0, self.window_total, dtype=np.int64
        )
        self.window_starts = (
            np.arange(self.seq_len, dtype=np.int64) * self.window_stride
        )

        (
            self.features,
            self.targets,
            self.ts,
            self.gt_pos,
            self.gt_ori,
            self.motion_type,
        ) = basic_data.get_data()
        self.index_map = index_map
        if self.shuffle:
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id, label = (
            self.index_map[item][0],
            self.index_map[item][1],
            self.index_map[item][2],
        )

        feat = self.features[seq_id][
            frame_id
            - self.past_data_size : frame_id
            + (self.seq_len - 1) * self.window_stride
            + self.window_size
            + self.future_data_size
        ]
        targ = self.targets[seq_id][
            frame_id : frame_id + self.seq_len * self.window_stride : self.window_stride
        ]
        ori = self.gt_ori[seq_id][
            frame_id : frame_id + self.seq_len * self.window_stride
        ]

        if self.mode in ["train"]:
            targ_aug = np.copy(targ)
            feat_aug = np.copy(feat)
            if self.transform:
                angle = np.random.random() * (2 * np.pi)
                rm = np.array(
                    [[np.cos(angle), -(np.sin(angle))], [np.sin(angle), np.cos(angle)]]
                )
                feat_aug[:, 0:2] = np.matmul(rm, feat_aug[:, 0:2].T).T
                feat_aug[:, 3:5] = np.matmul(rm, feat_aug[:, 3:5].T).T
                targ_aug[:, 0:2] = np.matmul(rm, targ_aug[:, 0:2].T).T

            if self.add_bias_noise:
                random_bias = np.random.random((1, 6))
                random_bias[:, 0:3] = (
                    (random_bias[:, 0:3] - 0.5) * self.gyro_bias_range / 0.5
                )
                random_bias[:, 3:6] = (
                    (random_bias[:, 3:6] - 0.5) * self.accel_bias_range / 0.5
                )
                feat_aug[:, :6] += random_bias

            if self.add_gravity_noise:
                from scipy.spatial.transform import Rotation as R

                angle_rand = random.random() * np.pi * 2
                vec_rand = np.array([np.cos(angle_rand), np.sin(angle_rand), 0])
                theta_rand = (
                    random.random() * np.pi * self.gravity_noise_theta_range / 180.0
                )
                rvec = theta_rand * vec_rand
                r = R.from_rotvec(rvec)
                r_mat = r.as_matrix()

                feat_aug[:, 0:3] = np.matmul(r_mat, feat_aug[:, 0:3].T).T
                feat_aug[:, 3:6] = np.matmul(r_mat, feat_aug[:, 3:6].T).T

                if hasattr(self, "use_local_coord") and self.use_local_coord:
                    targ_aug = np.matmul(r_mat, targ_aug.T).T

            if self.gauss:
                from numpy.random import normal as gen_normal

                if self.feat_gyr_sigma > 0:
                    feat_aug[:, 0:3] += gen_normal(
                        loc=0.0,
                        scale=self.feat_gyr_sigma,
                        size=(len(feat_aug[:, 0]), 3),
                    )
                if self.feat_acc_sigma > 0:
                    feat_aug[:, 3:6] += gen_normal(
                        loc=0.0,
                        scale=self.feat_acc_sigma,
                        size=(len(feat_aug[:, 0]), 3),
                    )

            feat = feat_aug
            targ = targ_aug

        velocity_feat_source = (
            _moving_average_lowpass(feat, self.low_pass_kernel_size)
            if self.low_pass_before_downsample
            else feat
        )
        sample_indices = self.window_starts[:, None] + self.window_offsets[None, :]
        seq_feat = velocity_feat_source[sample_indices]
        seq_feat = np.transpose(seq_feat, (0, 2, 1))
        batch = (
            seq_feat.astype(np.float32),
            targ.astype(np.float32),
            ori.astype(np.float32),
            label,
        )
        if self.return_platform_windows:
            platform_indices = (
                self.window_starts[:, None] + self.platform_window_offsets[None, :]
            )
            platform_feat = feat[platform_indices]
            platform_feat = np.transpose(platform_feat, (0, 2, 1))
            batch = batch + (platform_feat.astype(np.float32),)
        return batch

    def __len__(self):
        return len(self.index_map)


def SeqToSeqDataset(cfg, basic_data: BasicSequenceData, index_map, **kwargs):
    return ResNetLSTMSeqToSeqDataset(cfg, basic_data, index_map, **kwargs)
