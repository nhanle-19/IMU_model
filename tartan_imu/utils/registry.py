# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Runtime registries for dataset and model family dispatch."""

from __future__ import annotations

from importlib import import_module


# Reader keys -> fully-qualified module paths. Only the readers shipped in this
# release are listed; `main_net.py` resolves `data.dataset` through here, so an
# entry that is not importable would fail at run time rather than at import.
DATASET_MODULE_REGISTRY = {
    "AirLab": "tartan_imu.dataloader.dataset_AirLab",
    "TartanIMUChallenge": "tartan_imu.dataloader.dataset_TartanIMUChallenge",
    "Humanoid": "tartan_imu.dataloader.dataset_Humanoid",
    "HumanoidPostProcessed": "tartan_imu.dataloader.dataset_HumanoidPostProcessed",
    "HumanoidPostProcessedCached": "tartan_imu.dataloader.dataset_HumanoidPostProcessedCached",
}


def load_dataset_module(dataset_name: str):
    """Load the dataset reader module registered under ``dataset_name``.

    Args:
        dataset_name: Value of ``data.dataset`` in the experiment YAML.

    Returns:
        The imported reader module.

    Raises:
        ValueError: If ``dataset_name`` is not a registered reader.
    """
    if dataset_name not in DATASET_MODULE_REGISTRY:
        raise ValueError(
            f"Unknown data.dataset '{dataset_name}'. "
            f"Available readers: {sorted(DATASET_MODULE_REGISTRY)}"
        )
    return import_module(DATASET_MODULE_REGISTRY[dataset_name])
