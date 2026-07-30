import json
from tools.build_nfs_split import assign_splits, SplitConfig


def _fake_episodes(n):
    return [f"humanoid_ep{i:02d}_may23-2026.npz" for i in range(n)]


def test_split_is_deterministic_and_disjoint():
    files = _fake_episodes(24)
    cfg = SplitConfig(train=0.70, val=0.15, test=0.15, seed=42)
    a = assign_splits(files, cfg)
    b = assign_splits(files, cfg)
    assert a == b, "split must be deterministic for a fixed seed"
    train, val, test = set(a["train"]), set(a["val"]), set(a["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(files), "every episode assigned exactly once"


def test_split_ratio_counts_24_episodes():
    files = _fake_episodes(24)
    cfg = SplitConfig(train=0.70, val=0.15, test=0.15, seed=42)
    a = assign_splits(files, cfg)
    assert len(a["train"]) == 16
    assert len(a["val"]) == 4
    assert len(a["test"]) == 4


def test_manifest_roundtrip(tmp_path):
    files = _fake_episodes(24)
    cfg = SplitConfig(train=0.70, val=0.15, test=0.15, seed=42)
    a = assign_splits(files, cfg)
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(json.dumps(a, indent=2, sort_keys=True))
    loaded = json.loads(manifest_path.read_text())
    assert loaded == a
