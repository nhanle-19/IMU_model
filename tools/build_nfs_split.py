"""Build a deterministic episode-level train/val/test split of the NFS humanoid
data by symlinking .npz files into a local workspace. Does NOT mutate the NFS mount.

Usage:
    python tools/build_nfs_split.py \
        --src /nfs/data/Processed_Humanoid_Data \
        --dst ../dataset/humanoid_nfs \
        --variant may23-2026 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    seed: int = 42


def assign_splits(files: list[str], cfg: SplitConfig) -> dict[str, list[str]]:
    """Assign whole episodes to train/val/test. Deterministic for a fixed seed.

    Episodes are sorted first, then shuffled with a seeded RNG so the result is
    reproducible and independent of input ordering. Train gets a floor count and
    val gets a rounded count (so e.g. 24 episodes at 70/15/15 give 16/4/4 rather
    than the 16/3/5 that flooring both would produce); the remainder goes to test
    so every file is assigned exactly once.
    """
    ordered = sorted(files)
    rng = random.Random(cfg.seed)
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * cfg.train)
    n_val = round(n * cfg.val)
    train = sorted(shuffled[:n_train])
    val = sorted(shuffled[n_train:n_train + n_val])
    test = sorted(shuffled[n_train + n_val:])
    return {"train": train, "val": val, "test": test}


def build_workspace(src: str, dst: str, variant: str, cfg: SplitConfig) -> dict:
    """Create dst/{train,val,test} with symlinks to the variant's .npz files in src."""
    files = [f for f in os.listdir(src) if f.endswith(".npz") and variant in f]
    if not files:
        raise FileNotFoundError(f"No '{variant}' .npz files found in {src}")
    assignment = assign_splits(files, cfg)
    for split_name, split_files in assignment.items():
        split_dir = os.path.join(dst, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for fname in split_files:
            link = os.path.join(split_dir, fname)
            target = os.path.join(src, fname)
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(target, link)
    manifest = {"variant": variant, "seed": cfg.seed, "src": src, "splits": assignment}
    with open(os.path.join(dst, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def main() -> None:
    """CLI entry: build the symlinked split workspace and print split counts."""
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/nfs/data/Processed_Humanoid_Data")
    p.add_argument("--dst", default="../dataset/humanoid_nfs")
    p.add_argument("--variant", default="may23-2026")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    cfg = SplitConfig(seed=args.seed)
    dst = os.path.join(args.dst, args.variant)
    manifest = build_workspace(args.src, dst, args.variant, cfg)
    counts = {k: len(v) for k, v in manifest["splits"].items()}
    print(f"Built split at {dst}: {counts}")


if __name__ == "__main__":
    main()
