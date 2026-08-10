"""Dataset adapters for IMU windows and base velocity targets."""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SequenceMeta:
    path: Path
    length: int
    input_channels: int
    num_windows: int


def _squeeze_optional_batch_axis(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim >= 3 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _flatten_time_series(array: np.ndarray) -> np.ndarray:
    arr = _squeeze_optional_batch_axis(array)
    if arr.ndim < 2:
        raise ValueError(f"Expected [T, C...] array, got shape {arr.shape}")
    return arr.reshape(arr.shape[0], -1).astype(np.float32)


def _select_columns(array: np.ndarray, columns: Iterable[int] | None) -> np.ndarray:
    if columns is None:
        return array
    return array[:, list(columns)]


def _time_and_channels(array: np.ndarray) -> tuple[int, int]:
    arr = np.asarray(array)
    if arr.ndim >= 3 and arr.shape[0] == 1:
        return int(arr.shape[1]), int(np.prod(arr.shape[2:]))
    if arr.ndim < 2:
        raise ValueError(f"Expected [T, C...] array, got shape {arr.shape}")
    return int(arr.shape[0]), int(np.prod(arr.shape[1:]))


def resolve_downsample_step(data_cfg: dict) -> int:
    if data_cfg.get("imu_downsample_step") is not None:
        step = int(data_cfg["imu_downsample_step"])
    elif (
        data_cfg.get("imu_freq") is not None
        and data_cfg.get("sample_freq") is not None
    ):
        ratio = float(data_cfg["imu_freq"]) / float(data_cfg["sample_freq"])
        step = int(round(ratio))
        if step < 1 or not np.isclose(ratio, step):
            raise ValueError(
                "data.imu_freq / data.sample_freq must be a positive integer"
            )
    else:
        step = 1
    if step < 1:
        raise ValueError("data.imu_downsample_step must be >= 1")
    return step


def model_window_size(data_cfg: dict) -> int:
    window_size = int(data_cfg["window_size"])
    step = resolve_downsample_step(data_cfg)
    return len(range(0, window_size, step))


def require_average_velocity_targets(cfg: dict) -> None:
    target_at = str(cfg["data"].get("target_at", "end"))
    if target_at != "mean":
        raise ValueError(
            "Diffusion candidates must represent average velocity over the "
            "actual window; set data.target_at: mean."
        )


def assert_compatible_velocity_contract(current_cfg: dict, diffusion_cfg: dict) -> None:
    current = current_cfg["data"]
    diffusion = diffusion_cfg["data"]
    fields = (
        "window_size",
        "stride",
        "target_at",
        "imu_key",
        "velocity_key",
        "imu_columns",
        "velocity_columns",
    )
    mismatches = [
        name
        for name in fields
        if current.get(name) != diffusion.get(name)
    ]
    if mismatches:
        details = ", ".join(mismatches)
        raise ValueError(
            "Current training config does not match the diffusion checkpoint "
            f"velocity contract: {details}."
        )


assert_compatible_data_config = assert_compatible_velocity_contract


class NPZVelocityWindowDataset(Dataset):
    """Sliding-window dataset over NPZ files.

    Expected arrays are configurable, but the default structure is:

    - ``imu``: ``[T, C]`` or ``[1, T, ...]`` IMU features.
    - ``velocity``: ``[T, 3]`` or ``[1, T, 3]`` base velocity labels.

    Each item returns ``{"imu": [C, window], "velocity": [3]}``.
    """

    def __init__(self, cfg: dict, split: str):
        data_cfg = cfg["data"]
        self.window_size = int(data_cfg["window_size"])
        self.stride = int(data_cfg.get("stride", 1))
        self.imu_downsample_step = resolve_downsample_step(data_cfg)
        self.target_at = str(data_cfg.get("target_at", "end"))
        self.imu_key = str(data_cfg.get("imu_key", "imu"))
        self.velocity_key = str(data_cfg.get("velocity_key", "velocity"))
        self.imu_columns = data_cfg.get("imu_columns")
        self.velocity_columns = data_cfg.get("velocity_columns")
        self.sequence_cache_size = int(data_cfg.get("sequence_cache_size", 2))

        split_dirs = data_cfg.get("split_dirs", {})
        split_name = split_dirs.get(split, split)
        root = Path(data_cfg["root"]).expanduser()
        split_root = root / split_name
        file_glob = str(data_cfg.get("file_glob", "*.npz"))
        max_files = data_cfg.get("max_files")

        files = sorted(split_root.rglob(file_glob))
        if max_files is not None:
            files = files[: int(max_files)]
        if not files:
            raise FileNotFoundError(f"No NPZ files found under {split_root}")

        self.sequences: list[SequenceMeta] = []
        self.cumulative_windows: list[int] = []
        self._array_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = (
            OrderedDict()
        )

        for file_path in files:
            with np.load(file_path, allow_pickle=True) as npz:
                if self.imu_key not in npz or self.velocity_key not in npz:
                    continue
                imu_len, imu_channels = _time_and_channels(npz[self.imu_key])
                vel_len, vel_channels = _time_and_channels(npz[self.velocity_key])

            if self.imu_columns is not None:
                imu_channels = len(self.imu_columns)
            if self.velocity_columns is not None:
                vel_channels = len(self.velocity_columns)
            if vel_channels != 3:
                raise ValueError(
                    f"{file_path}: velocity target must have 3 columns, got "
                    f"{vel_channels}"
                )
            length = min(imu_len, vel_len)
            if length < self.window_size:
                continue
            num_windows = ((length - self.window_size) // self.stride) + 1
            self.sequences.append(
                SequenceMeta(
                    path=file_path,
                    length=length,
                    input_channels=imu_channels,
                    num_windows=num_windows,
                )
            )
            total = num_windows
            if self.cumulative_windows:
                total += self.cumulative_windows[-1]
            self.cumulative_windows.append(total)

        if not self.sequences:
            raise ValueError("No valid IMU windows were created from the dataset.")

        self.input_channels = int(self.sequences[0].input_channels)

    def __len__(self) -> int:
        return int(self.cumulative_windows[-1])

    def _target_velocity(self, velocity: np.ndarray, start: int) -> np.ndarray:
        end = start + self.window_size
        if self.target_at == "center":
            return velocity[start + self.window_size // 2]
        if self.target_at == "mean":
            return velocity[start:end].mean(axis=0)
        if self.target_at != "end":
            raise ValueError("data.target_at must be one of: end, center, mean")
        return velocity[end - 1]

    def _locate(self, item: int) -> tuple[int, int]:
        if item < 0:
            item += len(self)
        if item < 0 or item >= len(self):
            raise IndexError(item)
        seq_id = bisect_right(self.cumulative_windows, item)
        prev = 0 if seq_id == 0 else self.cumulative_windows[seq_id - 1]
        local_window = item - prev
        return seq_id, local_window * self.stride

    def _load_sequence(self, seq_id: int) -> tuple[np.ndarray, np.ndarray]:
        if seq_id in self._array_cache:
            arrays = self._array_cache.pop(seq_id)
            self._array_cache[seq_id] = arrays
            return arrays

        meta = self.sequences[seq_id]
        with np.load(meta.path, allow_pickle=True) as npz:
            imu = _select_columns(
                _flatten_time_series(npz[self.imu_key]), self.imu_columns
            )
            velocity = _select_columns(
                _flatten_time_series(npz[self.velocity_key]), self.velocity_columns
            )
        imu = imu[: meta.length]
        velocity = velocity[: meta.length]

        if self.sequence_cache_size > 0:
            self._array_cache[seq_id] = (imu, velocity)
            while len(self._array_cache) > self.sequence_cache_size:
                self._array_cache.popitem(last=False)
        return imu, velocity

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        seq_id, start = self._locate(item)
        imu_seq, velocity_seq = self._load_sequence(seq_id)
        imu = imu_seq[start : start + self.window_size : self.imu_downsample_step]
        vel = self._target_velocity(velocity_seq, start)
        return {
            "imu": torch.from_numpy(imu.T.copy()),
            "velocity": torch.from_numpy(vel.astype(np.float32, copy=False)),
            "platform": self.sequences[seq_id].path.parent.name,
        }


class SyntheticVelocityWindowDataset(Dataset):
    """Small deterministic dataset used for smoke tests and dry runs."""

    def __init__(self, n: int = 256, window_size: int = 64, input_channels: int = 6):
        generator = torch.Generator().manual_seed(7)
        self.imu = torch.randn(n, input_channels, window_size, generator=generator)
        summary = self.imu.mean(dim=-1)
        self.velocity = torch.stack(
            [
                0.7 * summary[:, 0] - 0.2 * summary[:, 3],
                -0.4 * summary[:, 1] + 0.3 * summary[:, 4],
                0.5 * summary[:, 2] + 0.1 * summary[:, 5],
            ],
            dim=-1,
        )

    @property
    def input_channels(self) -> int:
        return int(self.imu.shape[1])

    def __len__(self) -> int:
        return int(self.imu.shape[0])

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        return {
            "imu": self.imu[item],
            "velocity": self.velocity[item],
            "platform": "synthetic",
        }


def build_dataset(cfg: dict, split: str) -> Dataset:
    if cfg["data"].get("synthetic", False):
        return SyntheticVelocityWindowDataset(
            n=int(cfg["data"].get(f"{split}_synthetic_size", 256)),
            window_size=int(cfg["data"].get("window_size", 64)),
            input_channels=int(cfg["data"].get("input_channels", 6)),
        )
    return NPZVelocityWindowDataset(cfg, split)
