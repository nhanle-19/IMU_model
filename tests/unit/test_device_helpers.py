"""Unit tests for tartan_imu.utils.device (CPU/CUDA portability helpers)."""

import torch

from tartan_imu.utils.device import get_device, sync_if_cuda


def test_get_device_returns_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device(0) == torch.device("cpu")


def test_get_device_pins_cuda_when_available(monkeypatch):
    pinned = {}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda idx: pinned.setdefault("idx", idx))
    assert get_device(0) == torch.device("cuda:0")
    assert pinned["idx"] == 0


def test_sync_if_cuda_is_noop_on_cpu(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.__setitem__("n", calls["n"] + 1))
    sync_if_cuda()
    assert calls["n"] == 0


def test_sync_if_cuda_syncs_when_available(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.__setitem__("n", calls["n"] + 1))
    sync_if_cuda()
    assert calls["n"] == 1
