#!/usr/bin/env python3
"""Prebuild cache files for HumanoidPostProcessedCached dataset."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tartan_imu.dataloader.dataset_HumanoidPostProcessedCached import (  # noqa: E402  (needs sys.path setup above)
    _load_or_build_cache,
)


def _collect_npz_files(root: Path):
    """Return .npz paths directly under root, or in its immediate subdirs."""
    if not root.exists():
        return []
    files = sorted(root.glob("*.npz"))
    if files:
        return [str(p) for p in files]
    nested = []
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            nested.extend(sorted(sub.glob("*.npz")))
    return [str(p) for p in nested]


def main():
    """Prebuild the mmap cache for all split files referenced by a dataset config."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./config/datasets/tartanimu/humanoid_locoformer_all_overfitting.yaml",
        help="NIT dataset config yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    data_cfg = cfg["data"]
    human_root = Path(data_cfg["data_path"]["human"])
    splits = [
        data_cfg.get("train_dir"),
        data_cfg.get("validation_dir"),
        data_cfg.get("test_dir"),
    ]
    all_paths = []
    for split in splits:
        if split:
            all_paths.extend(_collect_npz_files(human_root / split))

    cache_root = Path(
        data_cfg.get("cache_dir", "../dataset/superodometry_overfiting/.cache_hpp_mmap")
    )
    cache_fp16 = bool(data_cfg.get("cache_fp16", True))

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info("Building cache for %d files into %s", len(all_paths), cache_root)
    built = 0
    for path in all_paths:
        ref = _load_or_build_cache(
            cfg=cfg,
            source_path=path,
            cache_root=cache_root,
            mode="train",
            cache_fp16=cache_fp16,
            verbose=False,
        )
        if ref is not None:
            built += 1
    logging.info("Cache ready for %d/%d files", built, len(all_paths))


if __name__ == "__main__":
    main()
