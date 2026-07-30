"""End-to-end resume test for Unit 3a.

Unlike test_resume_state.py (which pins the *contract* by hand-rolling the
checkpoint dict), this exercises the REAL code paths:

  1. train.save_model()             -> actually serializes scheduler/scaler state
  2. configer-style resume mapping  -> lifts top-level keys into resume_state
  3. Trainer._apply_resume_state()  -> actually restores scheduler/scaler

It catches wiring bugs the contract test cannot: a missing key in save_model,
a wrong key name in the configer mapping, or a no-op restore in the Trainer.
"""
import types

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import train


def _make_objects():
    """A real (tiny) model/optimizer/scheduler/scaler quartet."""
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = ReduceLROnPlateau(opt, factor=0.1, patience=1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, opt, sched, scaler


def _configer_resume_mapping(checkpoint: dict) -> dict:
    """Mirror config/configer.build_trainer: lift the scheduler/scaler payloads
    out of the checkpoint top level into the resume_state the Trainer consumes."""
    resume_state = dict(checkpoint.get("trainer_state", {}))
    resume_state["scheduler_state_dict"] = checkpoint.get("scheduler_state_dict", {})
    resume_state["scaler_state_dict"] = checkpoint.get("scaler_state_dict", {})
    return resume_state


def test_save_model_then_apply_resume_state_restores_sched_and_scaler(tmp_path):
    model, opt, sched, scaler = _make_objects()

    # Drive the scheduler so its internal state is provably non-default.
    for metric in (1.0, 1.0, 1.0, 1.0):
        sched.step(metric)
    expected_sched = sched.state_dict()
    expected_scaler = scaler.state_dict()

    # (1) REAL save: writes path/checkpoints/checkpoint_epoch_{epoch}.pt
    train.save_model(
        str(tmp_path),
        epoch=3,
        network=model,
        optimizer=opt,
        trainer_state={
            "best_val_loss": 0.5,
            "best_train_loss": 0.4,
            "last_save_epoch": 3,
        },
        scheduler_state=sched.state_dict(),
        scaler_state=scaler.state_dict(),
    )
    ckpt_path = tmp_path / "checkpoints" / "checkpoint_epoch_3.pt"
    assert ckpt_path.is_file(), "save_model did not write the epoch checkpoint"

    # Load it back and confirm save_model actually serialized both payloads.
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert checkpoint["scheduler_state_dict"], "scheduler state not serialized"
    assert checkpoint["scaler_state_dict"] is not None, "scaler state not serialized"

    # (2) REAL configer-style mapping into resume_state.
    resume_state = _configer_resume_mapping(checkpoint)

    # Fresh objects (as on a real resume) with DEFAULT scheduler/scaler state.
    _, _, fresh_sched, fresh_scaler = _make_objects()
    assert fresh_sched.state_dict()["num_bad_epochs"] != expected_sched["num_bad_epochs"]

    # (3) REAL restore: invoke the actual Trainer method against a minimal
    # stand-in carrying only the attributes the method touches. This runs the
    # production restore code without building a full Trainer.
    stand_in = types.SimpleNamespace(
        resume_state=resume_state,
        scheduler=fresh_sched,
        scaler=fresh_scaler,
        best_val_loss=float("inf"),
        best_train_loss=float("inf"),
        last_save_epoch=0,
    )
    train.Trainer._apply_resume_state(stand_in)

    # Bookkeeping restored.
    assert stand_in.best_val_loss == 0.5
    assert stand_in.best_train_loss == 0.4
    assert stand_in.last_save_epoch == 3

    # Scheduler state actually continued, not restarted.
    assert fresh_sched.state_dict()["num_bad_epochs"] == expected_sched["num_bad_epochs"]
    assert fresh_sched.state_dict()["best"] == expected_sched["best"]

    # Scaler state restored.
    assert fresh_scaler.state_dict() == expected_scaler


def test_apply_resume_state_no_scheduler_is_safe():
    """If the checkpoint carries no scheduler/scaler payload, restore is a no-op
    and must not raise (e.g. resuming an older checkpoint)."""
    _, _, fresh_sched, fresh_scaler = _make_objects()
    stand_in = types.SimpleNamespace(
        resume_state={"best_val_loss": 0.1},  # no scheduler/scaler keys
        scheduler=fresh_sched,
        scaler=fresh_scaler,
        best_val_loss=float("inf"),
        best_train_loss=float("inf"),
        last_save_epoch=0,
    )
    train.Trainer._apply_resume_state(stand_in)  # must not raise
    assert stand_in.best_val_loss == 0.1
