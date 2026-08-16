"""DDPM schedule and sampling utilities for 3D velocity candidates."""

from __future__ import annotations

import torch


class DiffusionSchedule:
    """Fixed beta schedule for conditional velocity diffusion."""

    def __init__(
        self,
        steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | str = "cpu",
    ):
        self.steps = int(steps)
        self.device = torch.device(device)
        self.betas = torch.linspace(
            beta_start, beta_end, self.steps, device=self.device
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device | str) -> DiffusionSchedule:
        return DiffusionSchedule(
            steps=self.steps,
            beta_start=float(self.betas[0].detach().cpu()),
            beta_end=float(self.betas[-1].detach().cpu()),
            device=device,
        )

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.steps, (batch_size,), device=self.device)

    def add_noise(
        self, clean_velocity: torch.Tensor, timesteps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean_velocity)
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1)
        noisy = alpha_bar.sqrt() * clean_velocity + (1.0 - alpha_bar).sqrt() * noise
        return noisy, noise

    @torch.no_grad()
    def sample(
        self,
        model,
        imu: torch.Tensor,
        num_candidates: int,
        velocity_dim: int = 3,
        max_sample_batch_size: int | None = None,
    ) -> torch.Tensor:
        """Generate ``[B, K, 3]`` velocity candidates conditioned on IMU."""
        model.eval()
        batch_size = imu.shape[0]
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if max_sample_batch_size is not None and max_sample_batch_size > 0:
            candidates_per_chunk = max(1, max_sample_batch_size // max(batch_size, 1))
            if candidates_per_chunk < num_candidates:
                chunks = []
                for start in range(0, num_candidates, candidates_per_chunk):
                    chunk_candidates = min(candidates_per_chunk, num_candidates - start)
                    chunks.append(
                        self.sample(
                            model,
                            imu,
                            num_candidates=chunk_candidates,
                            velocity_dim=velocity_dim,
                            max_sample_batch_size=None,
                        )
                    )
                return torch.cat(chunks, dim=1)

        imu_rep = imu.repeat_interleave(num_candidates, dim=0)
        x = torch.randn(
            batch_size * num_candidates,
            velocity_dim,
            device=imu.device,
            dtype=imu.dtype,
        )

        for step in reversed(range(self.steps)):
            t = torch.full((x.shape[0],), step, device=imu.device, dtype=torch.long)
            predicted_noise = model(imu_rep, x, t)
            beta = self.betas[step]
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            x = (
                x - ((1.0 - alpha) / (1.0 - alpha_bar).sqrt()) * predicted_noise
            ) / alpha.sqrt()
            if step > 0:
                x = x + beta.sqrt() * torch.randn_like(x)

        return x.view(batch_size, num_candidates, velocity_dim)
