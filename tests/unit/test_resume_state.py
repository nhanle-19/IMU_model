import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def test_scheduler_and_scaler_state_roundtrip(tmp_path):
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = ReduceLROnPlateau(opt, factor=0.1, patience=1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    for metric in (1.0, 1.0, 1.0, 1.0):
        sched.step(metric)
    sched_state = sched.state_dict()
    scaler_state = scaler.state_dict()

    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched_state,
        "scaler_state_dict": scaler_state,
        "trainer_state": {"best_val_loss": 0.5, "best_train_loss": 0.4, "last_save_epoch": 3},
        "epoch": 3,
    }
    path = tmp_path / "ckpt.pt"
    torch.save(ckpt, path)

    model2 = torch.nn.Linear(4, 2)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    sched2 = ReduceLROnPlateau(opt2, factor=0.1, patience=1)
    scaler2 = torch.amp.GradScaler("cuda", enabled=False)

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert "scheduler_state_dict" in loaded
    assert "scaler_state_dict" in loaded
    sched2.load_state_dict(loaded["scheduler_state_dict"])
    scaler2.load_state_dict(loaded["scaler_state_dict"])
    assert sched2.state_dict()["num_bad_epochs"] == sched_state["num_bad_epochs"]
    assert sched2.state_dict()["best"] == sched_state["best"]
