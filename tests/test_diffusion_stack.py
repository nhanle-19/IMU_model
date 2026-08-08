import numpy as np
import torch
import torch.nn.functional as F
import yaml

from imu_velocity_diffusion.data import (
    NPZVelocityWindowDataset,
    SyntheticVelocityWindowDataset,
)
from imu_velocity_diffusion.diffusion import DiffusionSchedule
from imu_velocity_diffusion.models import (
    VelocityDiffusionModel,
    VelocityDistributionRefiner,
)


def test_diffusion_samples_velocity_candidates():
    dataset = SyntheticVelocityWindowDataset(n=8, window_size=32, input_channels=6)
    batch = {
        "imu": torch.stack([dataset[i]["imu"] for i in range(4)]),
        "velocity": torch.stack([dataset[i]["velocity"] for i in range(4)]),
    }
    model = VelocityDiffusionModel(input_channels=6, hidden_dim=32, time_dim=16)
    schedule = DiffusionSchedule(steps=4)

    timesteps = schedule.sample_timesteps(batch["velocity"].shape[0])
    noisy_velocity, noise = schedule.add_noise(batch["velocity"], timesteps)
    predicted_noise = model(batch["imu"], noisy_velocity, timesteps)
    loss = F.mse_loss(predicted_noise, noise)
    loss.backward()

    candidates = schedule.sample(model, batch["imu"], num_candidates=5)
    assert candidates.shape == (4, 5, 3)
    assert torch.isfinite(candidates).all()


def test_refiner_uses_candidate_distribution():
    imu = torch.randn(4, 6, 32)
    candidates = torch.randn(4, 7, 3)
    refiner = VelocityDistributionRefiner(
        input_channels=6, hidden_dim=32, candidate_dim=16
    )

    outputs = refiner(imu, candidates)
    assert outputs["velocity"].shape == (4, 3)
    assert outputs["weights"].shape == (4, 7)
    assert outputs["corrected_candidates"].shape == (4, 7, 3)
    assert outputs["distribution_mean"].shape == (4, 3)
    assert outputs["distribution_variance"].shape == (4, 3)
    torch.testing.assert_close(outputs["weights"].sum(dim=1), torch.ones(4))


def test_npz_dataset_defaults_to_mean_velocity_target(tmp_path):
    with open("configs/default.yaml", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["data"]["root"] = str(tmp_path)
    cfg["data"]["split_dirs"] = {"train": "train"}
    cfg["data"]["window_size"] = 4
    cfg["data"]["stride"] = 2

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    imu = torch.arange(36, dtype=torch.float32).reshape(6, 6).numpy()
    velocity = torch.arange(18, dtype=torch.float32).reshape(6, 3).numpy()
    npz_path = train_dir / "sequence.npz"

    np.savez(npz_path, imu=imu, vel_body=velocity)

    dataset = NPZVelocityWindowDataset(cfg, "train")

    torch.testing.assert_close(dataset[0]["velocity"], torch.tensor([4.5, 5.5, 6.5]))
    torch.testing.assert_close(dataset[1]["velocity"], torch.tensor([10.5, 11.5, 12.5]))
