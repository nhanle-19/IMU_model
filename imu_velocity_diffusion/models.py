"""Neural modules for velocity candidate generation and distribution refinement."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("SinusoidalEmbedding dim must be even")
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = timesteps.device
        scale = math.log(10000.0) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=device) * -scale)
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class IMUEncoder(nn.Module):
    """Encode an IMU window ``[B, C, T]`` into a context vector."""

    def __init__(self, input_channels: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=7, padding=3),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        x = self.net(imu)
        x = x.transpose(1, 2)
        _, hidden = self.gru(x)
        return self.proj(hidden[-1])


class VelocityDiffusionModel(nn.Module):
    """Predict diffusion noise for a noisy 3D velocity conditioned on IMU."""

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int = 128,
        time_dim: int = 64,
        velocity_dim: int = 3,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.velocity_dim = velocity_dim
        self.imu_encoder = IMUEncoder(input_channels, hidden_dim)
        self.time_embedding = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.velocity_embedding = nn.Sequential(
            nn.Linear(velocity_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.denoiser = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, velocity_dim),
        )

    def forward(
        self, imu: torch.Tensor, noisy_velocity: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        context = self.imu_encoder(imu)
        t_embed = self.time_embedding(timesteps)
        v_embed = self.velocity_embedding(noisy_velocity)
        return self.denoiser(torch.cat([context, t_embed, v_embed], dim=-1))


class VelocityDistributionRefiner(nn.Module):
    """Refine a sampled velocity distribution into one usable estimate.

    The diffusion model produces a set of candidate velocities. This module does
    not treat that set as a menu where exactly one candidate must be selected.
    It consumes both per-sample candidates and global sample statistics
    (mean/variance/range) so it can aggregate unimodal distributions, preserve
    uncertainty cues, and use IMU context to reject inconsistent samples.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int = 128,
        candidate_dim: int = 64,
        velocity_dim: int = 3,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.velocity_dim = velocity_dim
        self.imu_encoder = IMUEncoder(input_channels, hidden_dim)
        stats_dim = velocity_dim * 4
        self.stats_encoder = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(velocity_dim * 3, candidate_dim),
            nn.SiLU(),
            nn.Linear(candidate_dim, candidate_dim),
            nn.SiLU(),
        )
        fused_dim = hidden_dim * 2 + candidate_dim
        self.attention_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.candidate_refine_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, velocity_dim),
        )
        self.global_refine_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + velocity_dim * 2),
            nn.Linear(hidden_dim * 2 + velocity_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, velocity_dim),
        )

    def _distribution_stats(self, candidates: torch.Tensor) -> dict[str, torch.Tensor]:
        mean = candidates.mean(dim=1)
        variance = candidates.var(dim=1, unbiased=False)
        minimum = candidates.min(dim=1).values
        maximum = candidates.max(dim=1).values
        return {
            "mean": mean,
            "variance": variance,
            "minimum": minimum,
            "maximum": maximum,
        }

    def forward(
        self, imu: torch.Tensor, candidates: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return distribution statistics, aggregation weights, and velocity."""
        batch_size, num_candidates, velocity_dim = candidates.shape
        if velocity_dim != self.velocity_dim:
            raise ValueError(
                f"Expected candidate dim {self.velocity_dim}, got {velocity_dim}"
            )

        stats = self._distribution_stats(candidates)
        distribution_features = torch.cat(
            [stats["mean"], stats["variance"], stats["minimum"], stats["maximum"]],
            dim=-1,
        )
        imu_context = self.imu_encoder(imu)
        stats_context = self.stats_encoder(distribution_features)

        centered = candidates - stats["mean"][:, None, :]
        normalized = centered / torch.sqrt(stats["variance"][:, None, :] + 1e-6)
        candidate_features = torch.cat([candidates, centered, normalized], dim=-1)
        candidate_context = self.candidate_encoder(candidate_features)

        imu_context_expanded = imu_context[:, None, :].expand(
            batch_size, num_candidates, -1
        )
        stats_context_expanded = stats_context[:, None, :].expand(
            batch_size, num_candidates, -1
        )
        fused = torch.cat(
            [imu_context_expanded, stats_context_expanded, candidate_context], dim=-1
        )
        logits = self.attention_head(fused).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)

        candidate_delta = self.candidate_refine_head(fused)
        corrected = candidates + candidate_delta
        weighted_velocity = torch.sum(weights.unsqueeze(-1) * corrected, dim=1)
        global_delta = self.global_refine_head(
            torch.cat(
                [imu_context, stats_context, stats["mean"], stats["variance"]],
                dim=-1,
            )
        )
        velocity = weighted_velocity + global_delta
        return {
            "logits": logits,
            "weights": weights,
            "corrected_candidates": corrected,
            "distribution_mean": stats["mean"],
            "distribution_variance": stats["variance"],
            "distribution_minimum": stats["minimum"],
            "distribution_maximum": stats["maximum"],
            "weighted_velocity": weighted_velocity,
            "velocity": velocity,
        }
