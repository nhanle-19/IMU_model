"""Unit tests for tartan_imu.config.resume.resolve_resume.

Guards the [RESUME-FIX]: an explicit --resume_from must flip
cfg train.use_pretrain_model on so build_trainer actually restores the
checkpoint instead of restarting from epoch 0.
"""

from tartan_imu.config.resume import resolve_resume


def test_resolve_resume_enables_flag_for_existing_checkpoint(tmp_path):
    ckpt = tmp_path / "checkpoint_epoch_5.pt"
    ckpt.write_bytes(b"stub")
    cfg = {"train": {"use_pretrain_model": False}}
    assert resolve_resume(cfg, str(ckpt)) is True
    assert cfg["train"]["use_pretrain_model"] is True


def test_resolve_resume_noop_when_path_is_none():
    cfg = {"train": {"use_pretrain_model": False}}
    assert resolve_resume(cfg, None) is False
    assert cfg["train"]["use_pretrain_model"] is False


def test_resolve_resume_noop_when_path_missing(tmp_path):
    cfg = {"train": {"use_pretrain_model": False}}
    missing = str(tmp_path / "does_not_exist.pt")
    assert resolve_resume(cfg, missing) is False
    assert cfg["train"]["use_pretrain_model"] is False
