"""Dataset adapters for IMU windows and base velocity targets."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowIndex:
    seq_id: int
    start: int


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
        self.target_at = str(data_cfg.get("target_at", "end"))
        self.imu_key = str(data_cfg.get("imu_key", "imu"))
        self.velocity_key = str(data_cfg.get("velocity_key", "velocity"))
        self.imu_columns = data_cfg.get("imu_columns")
        self.velocity_columns = data_cfg.get("velocity_columns")

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

        self.imu_sequences: list[np.ndarray] = []
        self.velocity_sequences: list[np.ndarray] = []
        self.index: list[WindowIndex] = []

        for file_path in files:
            with np.load(file_path, allow_pickle=True) as npz:
                if self.imu_key not in npz or self.velocity_key not in npz:
                    continue
                imu = _select_columns(
                    _flatten_time_series(npz[self.imu_key]), self.imu_columns
                )
                vel = _select_columns(
                    _flatten_time_series(npz[self.velocity_key]), self.velocity_columns
                )
            if vel.shape[1] != 3:
                raise ValueError(
                    f"{file_path}: velocity target must have 3 columns, got {vel.shape}"
                )
            length = min(len(imu), len(vel))
            if length < self.window_size:
                continue
            seq_id = len(self.imu_sequences)
            self.imu_sequences.append(imu[:length])
            self.velocity_sequences.append(vel[:length])
            for start in range(0, length - self.window_size + 1, self.stride):
                self.index.append(WindowIndex(seq_id=seq_id, start=start))

        if not self.index:
            raise ValueError("No valid IMU windows were created from the dataset.")

        self.input_channels = int(self.imu_sequences[0].shape[1])

    def __len__(self) -> int:
        return len(self.index)

    def _target_velocity(self, velocity: np.ndarray, start: int) -> np.ndarray:
        end = start + self.window_size
        if self.target_at == "center":
            return velocity[start + self.window_size // 2]
        if self.target_at == "mean":
            return velocity[start:end].mean(axis=0)
        if self.target_at != "end":
            raise ValueError("data.target_at must be one of: end, center, mean")
        return velocity[end - 1]

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        wi = self.index[item]
        imu = self.imu_sequences[wi.seq_id][wi.start : wi.start + self.window_size]
        vel = self._target_velocity(self.velocity_sequences[wi.seq_id], wi.start)
        return {
            "imu": torch.from_numpy(imu.T.copy()),
            "velocity": torch.from_numpy(vel.astype(np.float32, copy=False)),
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
        return {"imu": self.imu[item], "velocity": self.velocity[item]}


def build_dataset(cfg: dict, split: str) -> Dataset:
    if cfg["data"].get("synthetic", False):
        return SyntheticVelocityWindowDataset(
            n=int(cfg["data"].get(f"{split}_synthetic_size", 256)),
            window_size=int(cfg["data"].get("window_size", 64)),
            input_channels=int(cfg["data"].get("input_channels", 6)),
        )
    return NPZVelocityWindowDataset(cfg, split)
