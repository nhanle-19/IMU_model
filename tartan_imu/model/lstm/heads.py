# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Per-motion-type output heads and the multi-head FoundationModel.

``OutputHead`` maps shared trunk features to a per-motion-type velocity (and
optional covariance) prediction. ``FoundationModel`` runs a trunk once and
dispatches its features to the relevant heads (all heads, or only those present
in the batch's motion types).
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from tartan_imu.model.common.blocks import FcBlock
from tartan_imu.model.lstm.trunks import SpectralPlatformLSTMEncoder


def _platform_conditioning_cfg(cfg):
    return cfg.get("model_param", {}).get("platform_conditioning", {})


def _platform_condition_dim(cfg) -> int:
    pcfg = _platform_conditioning_cfg(cfg)
    if not pcfg.get("enabled", False):
        return 0
    mode = pcfg.get("condition_mode", "latent")
    if mode == "latent":
        return int(pcfg.get("latent_dim", 16))
    return int(pcfg.get("num_platforms", 4))


class OutputHead(nn.Module):
    """
    General purpose output head for a neural network model.

    Attributes:
        lstm_size (int): Size of the LSTM layer.
        output_dim (int): Dimension of the output.
        drop_ratio (float): Dropout ratio.
        compute_type (str): Type of computation to perform.
        batch_size (int): Batch size.
        seq_len (int): Sequence length.
    """

    def __init__(self, cfg, motion_type: str):
        super(OutputHead, self).__init__()

        self.lstm_size = cfg["model_param"]["lstm_size"]  # 256
        self.input_size = self.lstm_size + _platform_condition_dim(cfg)
        self.lstm_dropout = cfg["model_param"]["lstm_dropout"]  # 0.0
        self.output_dim = cfg["model_param"]["output_dim"]  # 3
        self.drop_ratio = cfg["model_param"]["drop_ratio"]  # 0.5 original:0.2
        self.split_z = cfg["model_param"]["split_z"]  # True/False
        # Output module
        self.output_block1 = FcBlock(
            self.input_size, self.output_dim, dropout=self.drop_ratio, cfg=cfg
        )
        self.output_block2 = FcBlock(
            self.input_size, self.output_dim, dropout=self.drop_ratio, cfg=cfg
        )
        if self.split_z:
            self.output_block1 = FcBlock(
                self.input_size, self.output_dim - 1, dropout=self.drop_ratio, cfg=cfg
            )  # dp mean
            self.output_block2 = FcBlock(
                self.input_size, self.output_dim, dropout=self.drop_ratio, cfg=cfg
            )  # dp cov
            self.output_block1_z = FcBlock(
                self.input_size, 1, dropout=self.drop_ratio, cfg=cfg
            )  # dp cov
        self.motion_type = motion_type

        # Add learnable scale factor for velocity prediction
        self.velocity_scale = nn.Parameter(torch.ones(1, self.output_dim))
        self.use_velocity_scale = cfg.get("model", {}).get("pred_velocity", False)

    def forward(self, out, batch_size, seq_len, predict_cov=False):
        """
        Forward pass of the module.

        Args:
            out: Input tensor.

        Returns:
            Tensor: Output after forward pass.
        """
        if self.split_z:
            x = self.output_block1(out)  # mean  10*3 Linear layer converts 10*256->10*3
            z = self.output_block1_z(out)
            x1 = torch.cat((x, z), dim=1)
            x1 = x1.view(batch_size, seq_len, -1)
        else:
            x1 = self.output_block1(out)
            x1 = x1.view(batch_size, seq_len, -1)

        # Apply learnable scale factor if velocity prediction is enabled
        if self.use_velocity_scale:
            x1 = x1 * self.velocity_scale

        if predict_cov:
            x2 = self.output_block2(out)
            x2 = x2.view(batch_size, seq_len, -1)
            return x1, x2
        else:
            return x1


class FoundationModel(nn.Module):
    """Shared trunk + per-motion-type output heads.

    Runs ``trunk`` once and dispatches its features to the relevant ``heads``
    (all heads, or only those present in the batch's motion types).
    """

    def __init__(self, cfg, trunk: nn.Module, heads: nn.ModuleDict):
        super(FoundationModel, self).__init__()
        self.model = trunk
        self.heads = heads
        self.types = ["car", "dog", "drone", "human"]
        self.lstm_size = cfg["model_param"]["lstm_size"]  # 256
        self.lstm_dropout = cfg["model_param"]["lstm_dropout"]  # 0.0
        self.output_dim = cfg["model_param"]["output_dim"]  # 3
        self.drop_ratio = cfg["model_param"]["drop_ratio"]  # 0.5 original:0.2
        self.platform_cfg = _platform_conditioning_cfg(cfg)
        self.use_platform_conditioning = self.platform_cfg.get("enabled", False)
        self.num_platforms = int(self.platform_cfg.get("num_platforms", 4))
        self.platform_condition_mode = self.platform_cfg.get("condition_mode", "latent")
        self.platform_condition_dim = _platform_condition_dim(cfg)

        if self.use_platform_conditioning:
            platform_encoder_cfg = deepcopy(cfg)
            classifier_sample_freq = float(
                self.platform_cfg.get(
                    "classifier_sample_freq", cfg["data"]["imu_freq"]
                )
            )
            platform_encoder_cfg["data"]["sample_freq"] = classifier_sample_freq
            self.platform_encoder = SpectralPlatformLSTMEncoder(platform_encoder_cfg)
            self.platform_classifier = nn.Linear(self.lstm_size, self.num_platforms)
            if self.platform_condition_mode == "latent":
                self.platform_condition_proj = nn.Sequential(
                    nn.Linear(self.lstm_size, self.platform_condition_dim),
                    nn.ReLU(inplace=True),
                )
            else:
                self.platform_condition_proj = None
            self.platform_window_frames = int(
                cfg["model_param"]["window_time"] * classifier_sample_freq
            )
            self.platform_hard_eval = bool(
                self.platform_cfg.get("hard_index_at_eval", False)
            )

    def forward(
        self,
        x,
        motion_type=None,
        predict_cov=False,
        compute_all_heads=True,
        platform_x=None,
    ):
        """
        Forward pass with optional efficient computation.

        Args:
            x: Input tensor
            motion_type: Motion type tensor or None
            predict_cov: Whether to predict covariance
            compute_all_heads: If True, compute all heads. If False, compute only needed heads.
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Forward pass through shared backbone
        model_output = self.model(x)

        # Handle case where trunk model returns a tuple (output, hidden_states)
        if isinstance(model_output, tuple):
            model_output = model_output[0]  # Take the first element (output tensor)

        platform_aux = {}
        if self.use_platform_conditioning:
            has_raw_platform_windows = platform_x is not None
            condition, platform_aux = self._compute_platform_condition(x, platform_x)
            if not has_raw_platform_windows and not self.platform_cfg.get(
                "emit_aux_for_fallback", False
            ):
                platform_aux = {}
            model_output = torch.cat((model_output, condition), dim=1)

        if compute_all_heads:
            # Original behavior - compute all heads
            outputs = self._compute_all_heads(
                model_output, batch_size, seq_len, predict_cov
            )
        else:
            # Efficient behavior - compute only needed heads
            outputs = self._compute_needed_heads(
                model_output, batch_size, seq_len, predict_cov, motion_type
            )
        return self._attach_platform_aux(outputs, platform_aux, predict_cov)

    def _compute_platform_condition(self, velocity_x, platform_x):
        """Predict platform from raw 200 Hz windows and build head conditioning."""
        if platform_x is None:
            platform_x = self._upsample_velocity_windows_for_platform(velocity_x)

        batch_size = platform_x.size(0)
        seq_len = platform_x.size(1)
        platform_features = self.platform_encoder(platform_x)
        if isinstance(platform_features, tuple):
            platform_features = platform_features[0]

        logits_flat = self.platform_classifier(platform_features)
        logits = logits_flat.view(batch_size, seq_len, self.num_platforms)

        if self.platform_condition_mode == "latent":
            condition = self.platform_condition_proj(platform_features)
        else:
            probs = F.softmax(logits_flat, dim=-1)
            if (
                self.platform_condition_mode == "index"
                and self.platform_hard_eval
                and not self.training
            ):
                idx = torch.argmax(probs, dim=-1)
                probs = F.one_hot(idx, num_classes=self.num_platforms).to(probs.dtype)
            condition = probs

        return condition, {
            "_platform_logits": logits,
            "_platform_probs": F.softmax(logits, dim=-1),
        }

    def _upsample_velocity_windows_for_platform(self, x):
        """Compatibility fallback for callers that only provide 40 Hz windows."""
        batch_size, seq_len, channels, frames = x.shape
        flat = x.reshape(batch_size * seq_len, channels, frames)
        upsampled = F.interpolate(
            flat,
            size=self.platform_window_frames,
            mode="linear",
            align_corners=False,
        )
        return upsampled.view(
            batch_size, seq_len, channels, self.platform_window_frames
        )

    def _attach_platform_aux(self, outputs, platform_aux, predict_cov):
        if not platform_aux:
            return outputs
        if predict_cov:
            pred, cov = outputs
            pred.update(platform_aux)
            return pred, cov
        outputs.update(platform_aux)
        return outputs

    def _compute_all_heads(self, model_output, batch_size, seq_len, predict_cov):
        """Original implementation - compute all heads."""
        multi_head = {}
        multi_head_cov = {}

        if predict_cov:
            car_head, car_cov = self.heads["car"](
                model_output, batch_size, seq_len, predict_cov
            )
            dog_head, dog_cov = self.heads["dog"](
                model_output, batch_size, seq_len, predict_cov
            )
            drone_head, drone_cov = self.heads["drone"](
                model_output, batch_size, seq_len, predict_cov
            )
            human_head, human_cov = self.heads["human"](
                model_output, batch_size, seq_len, predict_cov
            )

            multi_head["car"] = car_head
            multi_head["dog"] = dog_head
            multi_head["drone"] = drone_head
            multi_head["human"] = human_head

            multi_head_cov["car"] = car_cov
            multi_head_cov["dog"] = dog_cov
            multi_head_cov["drone"] = drone_cov
            multi_head_cov["human"] = human_cov

            return multi_head, multi_head_cov
        else:
            car_head = self.heads["car"](model_output, batch_size, seq_len)
            dog_head = self.heads["dog"](model_output, batch_size, seq_len)
            drone_head = self.heads["drone"](model_output, batch_size, seq_len)
            human_head = self.heads["human"](model_output, batch_size, seq_len)

            multi_head["car"] = car_head
            multi_head["dog"] = dog_head
            multi_head["drone"] = drone_head
            multi_head["human"] = human_head

            return multi_head

    def _compute_needed_heads(
        self, model_output, batch_size, seq_len, predict_cov, motion_type
    ):
        """Efficient computation - only compute heads that have data."""
        if motion_type is None:
            # Fallback to all heads if no motion type specified
            return self._compute_all_heads(
                model_output, batch_size, seq_len, predict_cov
            )

        # Determine which heads are needed based on motion type
        needed_heads = self._get_needed_heads(motion_type)

        multi_head = {}
        multi_head_cov = {}

        for motion_type_name in needed_heads:
            if predict_cov:
                pred, cov = self.heads[motion_type_name](
                    model_output, batch_size, seq_len, predict_cov
                )
                multi_head[motion_type_name] = pred
                multi_head_cov[motion_type_name] = cov
            else:
                pred = self.heads[motion_type_name](model_output, batch_size, seq_len)
                multi_head[motion_type_name] = pred

        if predict_cov:
            return multi_head, multi_head_cov
        else:
            return multi_head

    def _get_needed_heads(self, motion_type):
        """Determine which heads are needed based on motion type in batch."""
        if isinstance(motion_type, torch.Tensor):
            # Get unique motion types in batch
            unique_types = torch.unique(motion_type).tolist()
        else:
            unique_types = [motion_type]

        # Map motion type IDs to head names
        motion_types = {1: "car", 2: "dog", 3: "drone", 4: "human"}
        needed_heads = []

        for motion_id in unique_types:
            if motion_id in motion_types:
                needed_heads.append(motion_types[motion_id])

        return needed_heads

    def get_num_params(self):
        """Get the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
