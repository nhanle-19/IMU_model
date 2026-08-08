# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Humanoid (G1) inertial-odometry sequence dataloader."""
import logging
import os
import random

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from tartan_imu.dataloader._common import calculate_velocity_from_poses
from tartan_imu.utils.constants import GRAVITY

# G1 29-DOF joint names in URDF order (from Locomotiondataset.py)
G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def get_g1_adjacency_matrix():
    """
    Generates the adjacency matrix for the G1 humanoid based on G1_JOINT_NAMES.
    Returns a boolean tensor of shape [num_joints, num_joints].
    """
    num_joints = len(G1_JOINT_NAMES)
    adj = np.eye(num_joints, dtype=bool)

    # Define parent-child relationships based on G1 kinematic chains
    # (Simplified: connect consecutive joints in standard chains)
    connections = [
        # Left Leg: hip_pitch -> roll -> yaw -> knee -> ankle_pitch -> roll
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        # Right Leg: hip_pitch -> roll -> yaw -> knee -> ankle_pitch -> roll
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        # Waist: yaw -> roll -> pitch
        (12, 13),
        (13, 14),
        # Left Arm: shoulder_pitch -> roll -> yaw -> elbow -> wrist_roll -> pitch -> yaw
        (15, 16),
        (16, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (20, 21),
        # Right Arm: shoulder_pitch -> roll -> yaw -> elbow -> wrist_roll -> pitch -> yaw
        (22, 23),
        (23, 24),
        (24, 25),
        (25, 26),
        (26, 27),
        (27, 28),
    ]

    for i, j in connections:
        adj[i, j] = True
        adj[j, i] = True  # Bi-directional attention

    return torch.from_numpy(adj)


class HumanoidNPZSequence(object):
    """
    Load humanoid NPZ sequence created by extract_network_input.py.
    
    Supports Stage 1 (IMU only) and Stage 2 (IMU + joints).
    """
    
    def __init__(
        self,
        data_path,
        imu_freq,
        window_size,
        verbose=True,
        use_local_coord=False,
        mode="test",
        plot=False,
        stage=1,  # 1: IMU only, 2: IMU + joints
    ):
        super().__init__()
        (
            self.ts,
            self.features,
            self.targets,
            self.orientations,
            self.gt_pos,
            self.gt_ori,
        ) = (None, None, None, None, None, None)
        self.imu_freq = imu_freq
        self.interval = window_size
        self.data_valid = False
        self.sum_duration = 0
        self.valid = False
        self.plot = plot
        self.add_noise = False
        self.mode = mode
        self.use_local_coord = use_local_coord
        self.motion_type = torch.tensor(4, dtype=torch.long)  # human
        self.stage = stage  # Stage 1 or 2
        self.data_path = data_path

        if data_path is not None:
            self.valid = self.load(data_path, verbose=verbose)

    def _auto_align_imu(self, accel, gyro, ts, quat, verbose=False):
        """
        Automatically find the best 3D rotation and time shift to align
        IMU with Ground Truth orientation using Procrustes analysis.
        """
        from scipy.spatial.transform import Rotation as R
        import numpy as np

        T = len(ts)
        # 1. Identify a segment with significant motion for robust alignment
        window = 1000
        best_segment_start = 0
        max_var = -1
        for start in range(0, T - window, window):
            curr_var = np.var(gyro[start : start + window])
            if curr_var > max_var:
                max_var = curr_var
                best_segment_start = start

        start, end = best_segment_start, best_segment_start + window
        
        # 2. Calculate Ground Truth Angular Velocity
        dt_seg = np.diff(ts[start : end + 1])
        r_gt_seg = R.from_quat(quat[start : end + 1])
        dq_seg = r_gt_seg[:-1].inv() * r_gt_seg[1:]
        gt_gyro_seg = dq_seg.as_rotvec() / dt_seg[:, np.newaxis]
        imu_gyro_seg = gyro[start : end]

        # 3. Find Optimal Time Sync (Shift) using Magnitude
        gt_mag = np.linalg.norm(gt_gyro_seg, axis=1)
        imu_mag = np.linalg.norm(imu_gyro_seg, axis=1)
        
        best_sync_corr = -1
        best_shift = 0
        # Search range +/- 100ms (20 frames)
        for s in range(-20, 21):
            if s == 0:
                c = np.corrcoef(gt_mag, imu_mag)[0, 1]
            elif s > 0:
                c = np.corrcoef(gt_mag[s:], imu_mag[:-s])[0, 1]
            else:
                s_abs = abs(s)
                c = np.corrcoef(gt_mag[:-s_abs], imu_mag[s_abs:])[0, 1]
            if c > best_sync_corr:
                best_sync_corr = c
                best_shift = s

        # 4. Solve for Optimal Rotation Matrix (Procrustes/Wahba's Problem)
        if best_shift > 0:
            y = gt_gyro_seg[best_shift:]
            x = imu_gyro_seg[:-best_shift]
        elif best_shift < 0:
            s_abs = abs(best_shift)
            y = gt_gyro_seg[:-s_abs]
            x = imu_gyro_seg[s_abs:]
        else:
            y, x = gt_gyro_seg, imu_gyro_seg

        B = np.dot(y.T, x)
        U, S, Vt = np.linalg.svd(B)
        Rot_matrix = np.dot(U, Vt)
        if np.linalg.det(Rot_matrix) < 0:
            Vt[2, :] *= -1
            Rot_matrix = np.dot(U, Vt)
        
        r_align = R.from_matrix(Rot_matrix)
        
        # 5. Evaluate Correlation
        x_rot = r_align.apply(x)
        corrs = [np.corrcoef(y[:, i], x_rot[:, i])[0, 1] for i in range(3)]
        avg_corr = np.mean(corrs)

        # 6. Save Correlation to file
        log_dir = "debug_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "alignment_stats.txt")
        with open(log_file, "a") as f:
            f.write(f"File: {os.path.basename(self.data_path)}\n")
            f.write(f"  Corr: {avg_corr:.4f} (X:{corrs[0]:.4f}, Y:{corrs[1]:.4f}, Z:{corrs[2]:.4f})\n")
            f.write(f"  Shift: {best_shift*5} ms\n")
            f.write(f"  Euler: {r_align.as_euler('xyz', degrees=True)}\n")
            f.write("-" * 30 + "\n")

        if avg_corr > 0.6:
            if verbose:
                logging.info(f"✨ PRECISE ALIGNMENT: corr={avg_corr:.4f}, shift={best_shift*5}ms, Euler={r_align.as_euler('xyz', degrees=True)}")
            
            if best_shift > 0:
                accel = np.pad(accel[:-best_shift], ((best_shift, 0), (0, 0)), mode='edge')
                gyro = np.pad(gyro[:-best_shift], ((best_shift, 0), (0, 0)), mode='edge')
            elif best_shift < 0:
                s_abs = abs(best_shift)
                accel = np.pad(accel[s_abs:], ((0, s_abs), (0, 0)), mode='edge')
                gyro = np.pad(gyro[s_abs:], ((0, s_abs), (0, 0)), mode='edge')
            
            return r_align.apply(accel), r_align.apply(gyro)
        else:
            logging.warning(f"⚠️ Precise alignment failed (corr={avg_corr:.3f}). Using raw.")
            return accel, gyro

    def load(self, data_path, verbose=False):
        """Load NPZ file created by extract_network_input.py."""
        try:
            all_data = np.load(data_path, allow_pickle=True)
            if verbose:
                logging.info(
                    f"Successfully loaded {data_path} with keys: {list(all_data.keys())}"
                )
        except (EOFError, OSError, ValueError) as e:
            if verbose:
                logging.warning(
                    f"Failed to load corrupted or empty file: {data_path}, Error: {e}"
                )
            return False

        # Extract timestamps
        if "time" in all_data:
            ts = all_data["time"]
        elif "retargetted_ts" in all_data:
            ts = all_data["retargetted_ts"]
        else:
            if verbose:
                logging.warning(f"No timestamp key found in {data_path}")
            return False

        # Extract pelvis IMU data
        if "z_pelvis" in all_data:
            pelvis_features = all_data["z_pelvis"]
            accel_raw = pelvis_features[:, 0:3].astype(np.float64)
            gyro_raw = pelvis_features[:, 3:6].astype(np.float64)
        elif "retargetted_imu" in all_data:
            imu = all_data["retargetted_imu"]
            accel_raw = imu[:, 0:3].astype(np.float64)
            gyro_raw = imu[:, 3:6].astype(np.float64)
        else:
            if verbose:
                logging.warning(f"No IMU key found in {data_path}")
            return False

        T = accel_raw.shape[0]

        # Extract GT poses
        if "gt_translation" in all_data:
            pos = all_data["gt_translation"].astype(np.float64)
            quat = all_data["gt_orientation"].astype(np.float64)
        elif "retargetted_pos" in all_data:
            pos = all_data["retargetted_pos"].astype(np.float64)
            quat = all_data["retargetted_quat"].astype(np.float64)
        else:
            if verbose:
                logging.warning(f"No GT poses found in {data_path}, using zeros")
            pos = np.zeros((T, 3), dtype=np.float64)
            quat = np.zeros((T, 4), dtype=np.float64)
            quat[:, 3] = 1.0

        # PERFORM PRECISE ALIGNMENT AND SYNC
        accel_body, angular_vel_body = self._auto_align_imu(
            accel_raw, gyro_raw, ts, quat, verbose=verbose
        )

        # Calculate velocity from aligned IMU and poses
        velocity_global, velocity_body = calculate_velocity_from_poses(ts, pos, quat)

        # Process GT data
        pos_gt = pos - pos[0]
        quat_gt = Rotation.from_quat(quat)
        self.quat_gt = quat_gt

        # Gravity compensation.
        # Global frame: rotate body IMU into world and subtract gravity.
        # Local frame: keep body-frame IMU as-is (gravity left in).
        gravity = GRAVITY
        if not self.use_local_coord:
            accel_est = quat_gt.apply(accel_body)
            accel_est -= np.array([0, 0, gravity])
            angular_vel_est = quat_gt.apply(angular_vel_body)
        else:
            accel_est = accel_body
            angular_vel_est = angular_vel_body

        # Extract joint angles for Stage 2 & 3
        self.joint_features = None
        if self.stage >= 2:
            joint_features_list = []
            # Map joint names to link names as stored in NPZ keys
            joint_to_link = {name: name.replace('_joint', '_link') for name in G1_JOINT_NAMES}
            # Special case for waist_pitch which connects to torso_link in URDF
            joint_to_link["waist_pitch_joint"] = "torso_link"
            
            for j_name in G1_JOINT_NAMES:
                link_name = joint_to_link.get(j_name)
                key = f"z_{link_name}"
                if key in all_data:
                    # z_data contains [acc(3), gyro(3), has_imu(1), pos(3), q(1), dq(1), contact(1)]
                    # q is at index 10, dq is at index 11
                    q = all_data[key][:, 10:11].astype(np.float64)
                    dq = all_data[key][:, 11:12].astype(np.float64)
                    joint_features_list.append(np.concatenate([q, dq], axis=1))
                else:
                    if verbose:
                        logging.warning(f"  Missing joint feature for {j_name} (expected key: {key})")
                    # Fill with zeros if missing to maintain dimension
                    joint_features_list.append(np.zeros((T, 2), dtype=np.float64))
            
            if joint_features_list:
                self.joint_features = np.concatenate(joint_features_list, axis=1) # [T, 2 * num_joints]

        # Check interval
        if self.interval >= len(pos_gt):
            return False

        self.seq_len = ts.shape[0]
        if self.seq_len < self.imu_freq * 10:
            return False
        self.dt = ts[1] - ts[0] if len(ts) > 1 else 1.0 / self.imu_freq

        self.get_gt = True
        self.data_valid = True
        self.sum_duration = ts[-1] - ts[0] if len(ts) > 1 else 0.0

        # Prepare features
        new_ts = ts[:, np.newaxis]
        self.ts = new_ts

        if self.stage == 1:
            self.features = np.concatenate([angular_vel_est, accel_est], axis=1)
        elif self.stage >= 2:
            if self.joint_features is not None:
                imu_features = np.concatenate([angular_vel_est, accel_est], axis=1)
                self.features = np.concatenate([imu_features, self.joint_features], axis=1)
            else:
                if verbose:
                    logging.warning(f"Stage {self.stage} requested but joint_features not loaded for {data_path}")
                return False

        self.orientations = quat_gt.as_quat()
        self.pos_gt = pos_gt
        self.gt_ori = quat_gt.as_quat()

        # Trim and Calculate targets
        self.ts = self.ts[:-1]
        self.features = self.features[:-1]
        self.orientations = self.orientations[:-1]
        self.pos_gt = self.pos_gt[:-1]
        self.gt_ori = self.gt_ori[:-1]
        velocity_body = velocity_body[:-1]
        
        interval_int = int(self.interval)
        if not self.use_local_coord:
            if len(self.pos_gt) > interval_int:
                gt_disp = self.pos_gt[interval_int :] - self.pos_gt[: -interval_int]
                self.targets = gt_disp
            else:
                return False
        else:
            if velocity_body.shape[0] >= interval_int:
                num_windows = int(velocity_body.shape[0] - interval_int + 1)
                vel_body_targets = np.zeros((num_windows, velocity_body.shape[1]))
                for i in range(num_windows):
                    window = velocity_body[i : i + interval_int]
                    vel_body_targets[i] = np.mean(window, axis=0)
                self.targets = vel_body_targets
            else:
                return False
        
        self.velocity_body = velocity_body
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
    """
    Basic sequence data manager for humanoid dataset.
    Similar to AirLab's BasicSequenceData but for humanoid NPZ files.
    """
    
    def __init__(self, cfg, source_folder, verbose=False, **kwargs):
        super(BasicSequenceData, self).__init__()
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
        self.valid_t, self.valid_samples = [], []
        self.data_paths = []
        self.valid_continue_good_time = 0.1
        self.use_local_coord = cfg["data"]["use_local_coord"]
        self.mode = kwargs.get("mode", "train")
        self.is_transformer = cfg["model"]["model_name"] == "Transformer"
        
        # Get stage from model config
        self.stage = cfg["model_param"].get("stage", 1)

        sum_t = 0
        self.length = kwargs.get("current_frame")
        self.valid_sum_t = 0
        self.valid_all_samples = 0
        valid_i = 0

        # Parse data list from source folder
        data_list = source_folder

        if verbose:
            logging.info(f"Processing {len(data_list)} files for {self.mode} mode")

        for i in range(len(data_list)):
            try:
                seq = HumanoidNPZSequence(
                    data_list[i],
                    cfg["data"]["imu_freq"],
                    self.window_size,
                    verbose=verbose,
                    use_local_coord=self.use_local_coord,
                    mode=self.mode,
                    stage=self.stage,
                )

                if seq.valid is False:
                    if verbose:
                        logging.warning(f"File marked invalid by HumanoidNPZSequence: {data_list[i]}")
                    continue
            except Exception as e:
                if verbose:
                    logging.warning(f"Failed to process file {data_list[i]}: {e}")
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

            # Humanoid sequences always have ground truth (seq.get_gt is True), so
            # the AirLab-style velocity-outlier filtering does not apply here; we
            # simply enumerate every valid window.
            start_search = self.past_data_size
            end_search = targ.shape[0] - self.future_data_size - (self.seq_len - 1) * self.window_size

            if start_search >= end_search:
                if verbose:
                    logging.warning(f"File {data_list[i]} too short for requested sequence length: "
                                    f"past={self.past_data_size}, future={self.future_data_size}, "
                                    f"seq_len={self.seq_len}, win={self.window_size}, total={targ.shape[0]}")

            for j in range(
                start_search,
                end_search,
                step_size,
            ):
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
            logging.info(f"datasets sum time {sum_t:.2f}s")

        if len(self.data_paths) == 0:
            raise ValueError("No valid data files found.")

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
    """
    PyTorch Dataset for humanoid sequences.
    Reuses the same structure as AirLab's ResNetLSTMSeqToSeqDataset.
    """
    
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

        # Determine stride between windows in a sequence
        if self.is_transformer:
            # For Transformers, windows must be contiguous (no overlap) to maintain global timeline
            window_stride = self.window_size
        else:
            # For LSTM, windows can overlap (sliding window style)
            window_stride = self.step_size

        feat = self.features[seq_id][
            frame_id
            - self.past_data_size : frame_id
            + (self.seq_len - 1) * window_stride
            + self.window_size
            + self.future_data_size
        ]

        targ = self.targets[seq_id][
            frame_id : frame_id + self.seq_len * window_stride : window_stride
        ]
        ori = self.gt_ori[seq_id][
            frame_id : frame_id + self.seq_len * window_stride
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
                # Create a 6-dim bias for IMU (gyro + accel)
                random_bias = np.random.random((1, 6))
                random_bias[:, 0:3] = (
                    (random_bias[:, 0:3] - 0.5) * self.gyro_bias_range / 0.5
                )
                random_bias[:, 3:6] = (
                    (random_bias[:, 3:6] - 0.5) * self.accel_bias_range / 0.5
                )
                # Apply bias ONLY to the IMU part of the features (first 6 columns)
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
                R_mat = r.as_matrix()

                feat_aug[:, 0:3] = np.matmul(R_mat, feat_aug[:, 0:3].T).T
                feat_aug[:, 3:6] = np.matmul(R_mat, feat_aug[:, 3:6].T).T

                if hasattr(self, "use_local_coord") and self.use_local_coord:
                    targ_aug = np.matmul(R_mat, targ_aug.T).T

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

        seq_feat = []
        # Re-calculate stride for the loop
        window_stride = self.window_size if self.is_transformer else self.step_size
        
        for i in range(self.seq_len):
            start_idx = i * window_stride
            end_idx = start_idx + self.window_size
            
            window_feat = feat[
                start_idx : self.past_data_size + end_idx + self.future_data_size,
                :,
            ]

            if self.step_size > 1:
                # Downsample window if step_size > 1
                window_feat = window_feat[::self.step_size, :]

            seq_feat.append(window_feat.T)

        try:
            seq_feat = np.array(seq_feat)
        except ValueError as e:
            print("ValueError:", e)

        return (
            seq_feat.astype(np.float32),
            targ.astype(np.float32),
            ori.astype(np.float32),
            label,
        )

    def __len__(self):
        return len(self.index_map)


def SeqToSeqDataset(cfg, basic_data: BasicSequenceData, index_map, **kwargs):
    return ResNetLSTMSeqToSeqDataset(cfg, basic_data, index_map, **kwargs)
