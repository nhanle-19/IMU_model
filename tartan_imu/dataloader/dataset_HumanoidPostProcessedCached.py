# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Disk-cached variant of the post-processed Humanoid sequence dataloader."""
import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from torch.utils.data import Dataset

from tartan_imu.dataloader.dataset_HumanoidPostProcessed import (
    HumanoidPostProcessedSequence,
    _moving_average_lowpass,
)


@dataclass(frozen=True)
class CachedSequenceRef:
    source_path: str
    cache_dir: str
    features_path: str
    targets_path: str
    gt_ori_path: str
    feature_len: int
    target_len: int
    sum_duration: float
    motion_type: int


def _resolve_window_time(cfg) -> float:
    if "model_param" in cfg and "window_time" in cfg["model_param"]:
        return float(cfg["model_param"]["window_time"])
    model_yaml_path = cfg.get("model", {}).get("model_yaml")
    if model_yaml_path is None:
        raise KeyError("Missing model_param.window_time and model.model_yaml")
    with open(model_yaml_path, "r", encoding="utf-8") as fh:
        model_cfg = yaml.safe_load(fh)
    return float(model_cfg["model_param"]["window_time"])


def _cache_key(
    source_path: str,
    stage: int = 2,
    use_local_coord: bool = True,
    window_size: int = 0,
    zero_dq: bool = False,
) -> str:
    source = Path(source_path)
    stat = source.stat()
    # Every param that changes the cached features/targets must be in the
    # fingerprint, else two runs differing only in one of them silently share a
    # stale entry: stage (6/35/64-dim), use_local_coord (velocity vs
    # displacement targets), window_size (interval), zero_dq (Stage 3 dq=0).
    fingerprint = (
        f"{source.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"
        f"::stage{stage}::loc{int(use_local_coord)}::win{int(window_size)}::zdq{int(zero_dq)}"
    )
    return hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:20]


def _cache_variant(cfg) -> dict:
    """Extract the output-affecting variant params from a config."""
    mp = cfg.get("model_param", {})
    return {
        "stage": mp.get("stage", 2),
        "use_local_coord": cfg["data"]["use_local_coord"],
        "window_size": int(_resolve_window_time(cfg) * cfg["data"]["imu_freq"]),
        "zero_dq": mp.get("stage3_zero_dq", False),
    }


def _cache_dir_for(cache_root: Path, source_path: str, **variant) -> Path:
    return cache_root / _cache_key(source_path, **variant)


def _build_cache_from_npz(
    cfg,
    source_path: str,
    cache_root: Path,
    mode: str,
    cache_fp16: bool,
    verbose: bool,
):
    window_size = int(_resolve_window_time(cfg) * cfg["data"]["imu_freq"])
    seq = HumanoidPostProcessedSequence(
        source_path,
        cfg["data"]["imu_freq"],
        window_size,
        verbose=verbose,
        use_local_coord=cfg["data"]["use_local_coord"],
        mode=mode,
        stage=cfg.get("model_param", {}).get("stage", 2),
        zero_dq=cfg.get("model_param", {}).get("stage3_zero_dq", False),
    )
    if not seq.valid:
        return None

    cache_dir = _cache_dir_for(cache_root, source_path, **_cache_variant(cfg))
    cache_dir.mkdir(parents=True, exist_ok=True)

    features = seq.features.astype(np.float16 if cache_fp16 else np.float32, copy=False)
    targets = seq.targets.astype(np.float32, copy=False)
    gt_ori = seq.gt_ori.astype(np.float32, copy=False)

    features_path = cache_dir / "features.npy"
    targets_path = cache_dir / "targets.npy"
    gt_ori_path = cache_dir / "gt_ori.npy"
    meta_path = cache_dir / "meta.json"

    np.save(features_path, features)
    np.save(targets_path, targets)
    np.save(gt_ori_path, gt_ori)

    meta = {
        "source_path": str(Path(source_path).resolve()),
        "feature_len": int(features.shape[0]),
        "target_len": int(targets.shape[0]),
        "sum_duration": float(seq.sum_duration),
        "motion_type": int(seq.motion_type.item()) if hasattr(seq.motion_type, "item") else int(seq.motion_type),
        "cache_fp16": bool(cache_fp16),
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta


def _load_or_build_cache(cfg, source_path: str, cache_root: Path, mode: str, cache_fp16: bool, verbose: bool):
    cache_dir = _cache_dir_for(cache_root, source_path, **_cache_variant(cfg))
    meta_path = cache_dir / "meta.json"
    features_path = cache_dir / "features.npy"
    targets_path = cache_dir / "targets.npy"
    gt_ori_path = cache_dir / "gt_ori.npy"

    meta = None
    if meta_path.exists() and features_path.exists() and targets_path.exists() and gt_ori_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = None

    if meta is None:
        meta = _build_cache_from_npz(cfg, source_path, cache_root, mode, cache_fp16, verbose)
        if meta is None:
            return None
    return CachedSequenceRef(
        source_path=source_path,
        cache_dir=str(cache_dir),
        features_path=str(features_path),
        targets_path=str(targets_path),
        gt_ori_path=str(gt_ori_path),
        feature_len=int(meta["feature_len"]),
        target_len=int(meta["target_len"]),
        sum_duration=float(meta["sum_duration"]),
        motion_type=int(meta["motion_type"]),
    )


class BasicSequenceData:
    """Cached sequence manager using mmap-ready arrays."""

    def __init__(self, cfg, source_folder, verbose=False, **kwargs):
        self.window_size = int(_resolve_window_time(cfg) * cfg["data"]["imu_freq"])
        self.past_data_size = int(cfg["model_param"]["past_time"] * cfg["data"]["imu_freq"])
        self.future_data_size = int(cfg["model_param"]["future_time"] * cfg["data"]["imu_freq"])
        self.step_size = int(cfg["data"]["imu_freq"] / cfg["data"]["sample_freq"])
        self.seq_len = cfg["train"]["seq_len"]
        self.use_local_coord = cfg["data"]["use_local_coord"]
        self.mode = kwargs.get("mode", "train")
        self.is_transformer = cfg["model"]["model_name"] == "Transformer"

        self.valid_samples = []
        self.valid_all_samples = 0
        self.index_map = []
        self.sequence_refs = []
        self.data_paths = []

        cache_root = Path(
            cfg["data"].get(
                "cache_dir",
                os.path.join(cfg["train"]["out_dir"], "data_cache", "humanoid_postprocessed"),
            )
        )
        cache_fp16 = bool(cfg["data"].get("cache_fp16", True))
        cache_root.mkdir(parents=True, exist_ok=True)

        valid_i = 0
        sum_t = 0.0
        for source_path in source_folder:
            try:
                seq_ref = _load_or_build_cache(
                    cfg=cfg,
                    source_path=source_path,
                    cache_root=cache_root,
                    mode=self.mode,
                    cache_fp16=cache_fp16,
                    verbose=verbose,
                )
            except Exception as exc:
                if verbose:
                    logging.warning("Cache build/load failed for %s: %s", source_path, exc)
                continue

            if seq_ref is None:
                continue

            start_search = self.past_data_size
            end_search = (
                seq_ref.target_len
                - self.future_data_size
                - (self.seq_len - 1) * self.window_size
            )
            if start_search >= end_search:
                continue

            sample_index_map = []
            for frame_id in range(start_search, end_search, self.step_size):
                sample_index_map.append([valid_i, frame_id, seq_ref.motion_type])
                self.valid_all_samples += 1

            if not sample_index_map:
                continue

            self.sequence_refs.append(seq_ref)
            self.data_paths.append(source_path)
            self.index_map.append(sample_index_map)
            self.valid_samples.append(len(sample_index_map))
            sum_t += seq_ref.sum_duration
            valid_i += 1

        if verbose:
            logging.info("Cached postprocessed dataset total time: %.2fs", sum_t)

        if len(self.sequence_refs) == 0:
            raise ValueError("No valid cached post-processed data files found.")

    def get_index_map(self):
        return self.index_map

    def get_merged_index_map(self):
        merged = []
        for index_list in self.index_map:
            merged += index_list
        return merged

    def get_sequence_ref(self, seq_id: int) -> CachedSequenceRef:
        return self.sequence_refs[seq_id]


class ResNetLSTMSeqToSeqDataset(Dataset):
    def __init__(self, cfg, basic_data: BasicSequenceData, index_map, **kwargs):
        self.window_size = basic_data.window_size
        self.past_data_size = basic_data.past_data_size
        self.future_data_size = basic_data.future_data_size
        self.step_size = basic_data.step_size
        self.seq_len = basic_data.seq_len
        self.use_local_coord = basic_data.use_local_coord
        self.is_transformer = basic_data.is_transformer
        self.basic_data = basic_data

        self.add_bias_noise = cfg["augment"]["add_bias_noise"]
        self.accel_bias_range = cfg["augment"]["accel_bias_range"]
        self.gyro_bias_range = cfg["augment"]["gyro_bias_range"]
        if not self.add_bias_noise:
            self.accel_bias_range = 0.0
            self.gyro_bias_range = 0.0
        self.add_gravity_noise = cfg["augment"]["add_gravity_noise"]
        self.gravity_noise_theta_range = cfg["augment"]["gravity_noise_theta_range"]
        self.feat_acc_sigma = cfg["augment"]["feat_acc_sigma"]
        self.feat_gyr_sigma = cfg["augment"]["feat_gyr_sigma"]
        pcfg = cfg.get("model_param", {}).get("platform_conditioning", {})
        lp_cfg = cfg["data"].get("low_pass_filter", {})
        self.return_platform_windows = bool(pcfg.get("enabled", False))
        self.low_pass_before_downsample = bool(
            lp_cfg.get("enabled", self.return_platform_windows)
        )
        self.low_pass_kernel_size = int(lp_cfg.get("kernel_size", self.step_size))

        self.mode = kwargs.get("mode", "train")
        self.shuffle = self.mode in ["train", "val"]
        self.transform = self.mode == "train"
        self.gauss = self.mode == "train"
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

        self.index_map = index_map
        if self.shuffle:
            random.shuffle(self.index_map)

        self._array_cache = {}

    def _get_arrays(self, seq_id: int):
        if seq_id not in self._array_cache:
            seq_ref = self.basic_data.get_sequence_ref(seq_id)
            self._array_cache[seq_id] = {
                "features": np.load(seq_ref.features_path, mmap_mode="r"),
                "targets": np.load(seq_ref.targets_path, mmap_mode="r"),
                "gt_ori": np.load(seq_ref.gt_ori_path, mmap_mode="r"),
            }
        return self._array_cache[seq_id]

    def __getitem__(self, item):
        seq_id, frame_id, label = (
            self.index_map[item][0],
            self.index_map[item][1],
            self.index_map[item][2],
        )
        arrays = self._get_arrays(seq_id)
        features = arrays["features"]
        targets = arrays["targets"]
        gt_ori = arrays["gt_ori"]

        feat = np.asarray(
            features[
                frame_id
                - self.past_data_size : frame_id
                + (self.seq_len - 1) * self.window_stride
                + self.window_size
                + self.future_data_size
            ],
            dtype=np.float32,
        )
        targ = np.asarray(
            targets[
                frame_id : frame_id + self.seq_len * self.window_stride : self.window_stride
            ],
            dtype=np.float32,
        )
        ori = np.asarray(
            gt_ori[
                frame_id : frame_id + self.seq_len * self.window_stride
            ],
            dtype=np.float32,
        )

        if self.mode == "train":
            targ_aug = np.copy(targ)
            feat_aug = np.copy(feat)
            if self.transform:
                angle = np.random.random() * (2 * np.pi)
                rm = np.array(
                    [[np.cos(angle), -(np.sin(angle))], [np.sin(angle), np.cos(angle)]],
                    dtype=np.float32,
                )
                feat_aug[:, 0:2] = np.matmul(rm, feat_aug[:, 0:2].T).T
                feat_aug[:, 3:5] = np.matmul(rm, feat_aug[:, 3:5].T).T
                targ_aug[:, 0:2] = np.matmul(rm, targ_aug[:, 0:2].T).T

            if self.add_bias_noise:
                random_bias = np.random.random((1, 6)).astype(np.float32)
                random_bias[:, 0:3] = (
                    (random_bias[:, 0:3] - 0.5) * self.gyro_bias_range / 0.5
                )
                random_bias[:, 3:6] = (
                    (random_bias[:, 3:6] - 0.5) * self.accel_bias_range / 0.5
                )
                feat_aug[:, :6] += random_bias

            if self.gauss:
                from numpy.random import normal as gen_normal

                if self.feat_gyr_sigma > 0:
                    feat_aug[:, 0:3] += gen_normal(
                        loc=0.0,
                        scale=self.feat_gyr_sigma,
                        size=(len(feat_aug[:, 0]), 3),
                    ).astype(np.float32)
                if self.feat_acc_sigma > 0:
                    feat_aug[:, 3:6] += gen_normal(
                        loc=0.0,
                        scale=self.feat_acc_sigma,
                        size=(len(feat_aug[:, 0]), 3),
                    ).astype(np.float32)

            feat = feat_aug
            targ = targ_aug

        velocity_feat_source = (
            _moving_average_lowpass(feat, self.low_pass_kernel_size)
            if self.low_pass_before_downsample
            else feat
        )
        # Vectorized window extraction removes per-step Python-loop overhead.
        sample_indices = self.window_starts[:, None] + self.window_offsets[None, :]
        seq_feat = velocity_feat_source[sample_indices]
        seq_feat = np.transpose(seq_feat, (0, 2, 1)).astype(np.float32, copy=False)
        batch = (
            seq_feat,
            targ.astype(np.float32),
            ori.astype(np.float32),
            label,
        )
        if self.return_platform_windows:
            platform_indices = (
                self.window_starts[:, None] + self.platform_window_offsets[None, :]
            )
            platform_feat = feat[platform_indices]
            platform_feat = np.transpose(platform_feat, (0, 2, 1)).astype(
                np.float32, copy=False
            )
            batch = batch + (platform_feat,)
        return batch

    def __len__(self):
        return len(self.index_map)


def SeqToSeqDataset(cfg, basic_data: BasicSequenceData, index_map, **kwargs):
    return ResNetLSTMSeqToSeqDataset(cfg, basic_data, index_map, **kwargs)
