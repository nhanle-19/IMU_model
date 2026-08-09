import numpy as np
import torch
import torch.nn.functional as F
import yaml

from imu_velocity_diffusion.data import (
    NPZVelocityWindowDataset,
    SyntheticVelocityWindowDataset,
    assert_compatible_velocity_contract,
    model_window_size,
    require_average_velocity_targets,
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
    cfg["data"]["imu_downsample_step"] = 2

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    imu = torch.arange(36, dtype=torch.float32).reshape(6, 6).numpy()
    velocity = torch.arange(18, dtype=torch.float32).reshape(6, 3).numpy()
    npz_path = train_dir / "sequence.npz"

    np.savez(npz_path, imu=imu, vel_body=velocity)

    dataset = NPZVelocityWindowDataset(cfg, "train")

    assert dataset[0]["imu"].shape == (6, 2)
    torch.testing.assert_close(
        dataset[0]["imu"],
        torch.tensor(
            [
                [0.0, 12.0],
                [1.0, 13.0],
                [2.0, 14.0],
                [3.0, 15.0],
                [4.0, 16.0],
                [5.0, 17.0],
            ]
        ),
    )
    torch.testing.assert_close(dataset[0]["velocity"], torch.tensor([4.5, 5.5, 6.5]))
    torch.testing.assert_close(dataset[1]["velocity"], torch.tensor([10.5, 11.5, 12.5]))


def test_diffusion_data_contract_matches_actual_average_window(tmp_path):
    with open("configs/default.yaml", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["data"]["root"] = str(tmp_path)
    cfg["data"]["split_dirs"] = {"train": "train"}

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    imu = torch.arange(2400, dtype=torch.float32).reshape(400, 6).numpy()
    velocity = torch.arange(1200, dtype=torch.float32).reshape(400, 3).numpy()
    np.savez(train_dir / "sequence.npz", imu=imu, vel_body=velocity)

    require_average_velocity_targets(cfg)
    assert model_window_size(cfg["data"]) == 100
    dataset = NPZVelocityWindowDataset(cfg, "train")

    first = dataset[0]
    assert first["imu"].shape == (6, 100)
    torch.testing.assert_close(
        first["velocity"], torch.from_numpy(velocity[:200].mean(axis=0))
    )

    batch_imu = first["imu"].unsqueeze(0)
    batch_velocity = first["velocity"].unsqueeze(0)
    model = VelocityDiffusionModel(input_channels=6, hidden_dim=32, time_dim=16)
    schedule = DiffusionSchedule(steps=4)
    candidates = schedule.sample(model, batch_imu, num_candidates=7)
    refiner = VelocityDistributionRefiner(
        input_channels=6, hidden_dim=32, candidate_dim=16
    )
    outputs = refiner(batch_imu, candidates)

    assert candidates.shape == (1, 7, 3)
    assert outputs["velocity"].shape == batch_velocity.shape


def test_refiner_training_rejects_mismatched_diffusion_window():
    with open("configs/default.yaml", encoding="utf-8") as handle:
        current = yaml.safe_load(handle)
    diffusion = yaml.safe_load(yaml.dump(current))
    diffusion["data"]["window_size"] = 100

    try:
        assert_compatible_velocity_contract(current, diffusion)
    except ValueError as exc:
        assert "window_size" in str(exc)
    else:
        raise AssertionError("mismatched diffusion data config should fail")


def test_actual_40hz_model_can_consume_100hz_diffusion_velocities():
    with open("configs/default.yaml", encoding="utf-8") as handle:
        diffusion = yaml.safe_load(handle)
    actual = yaml.safe_load(yaml.dump(diffusion))
    actual["data"]["sample_freq"] = 40

    assert model_window_size(diffusion["data"]) == 100
    assert model_window_size(actual["data"]) == 40
    assert_compatible_velocity_contract(actual, diffusion)


def test_diffusion_training_requires_average_velocity_targets():
    with open("configs/default.yaml", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["data"]["target_at"] = "end"

    try:
        require_average_velocity_targets(cfg)
    except ValueError as exc:
        assert "average velocity" in str(exc)
    else:
        raise AssertionError("diffusion should require average velocity targets")
