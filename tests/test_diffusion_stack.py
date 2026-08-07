import torch
import torch.nn.functional as F

from imu_velocity_diffusion.data import SyntheticVelocityWindowDataset
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
