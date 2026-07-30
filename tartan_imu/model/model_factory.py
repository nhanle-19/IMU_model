# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""
Simplified model factory for neural inertial tracking system.
Provides centralized model creation and configuration.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
from tartan_imu.utils.error_handling import ModelCreationError
from tartan_imu.utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """Factory for creating neural inertial tracking models."""

    @staticmethod
    def create_model(model_config: Dict[str, Any], device: torch.device) -> nn.Module:
        """Create model based on configuration."""
        model_name = model_config.get("model_name", "Foundation_Model")

        if model_name == "Foundation_Model":
            return ModelFactory._create_foundation_model(model_config, device)
        elif model_name == "Transformer":
            return ModelFactory._create_transformer_odo(model_config, device)
        else:
            raise ModelCreationError(f"Unknown model name: {model_name}")

    @staticmethod
    def _create_foundation_model(
        model_config: Dict[str, Any], device: torch.device
    ) -> nn.Module:
        """Create Foundation Model with multi-head architecture."""
        from tartan_imu.model.lstm.model_lstm import FoundationModel

        # Create model
        model = FoundationModel(model_config)
        model = model.to(device)

        logger.info(
            f"Created Foundation Model with {sum(p.numel() for p in model.parameters()):,} parameters"
        )
        return model

    @staticmethod
    def _create_transformer_odo(
        model_config: Dict[str, Any], device: torch.device
    ) -> nn.Module:
        """Create hybrid transformer odometry model with support for 3 stages."""
        try:
            from .transformer_odom import TransformerOdoModel
        except ImportError as e:
            # LSTM-only builds drop the transformer backbone; surface a clear,
            # actionable error instead of a raw ImportError.
            raise ModelCreationError(
                "Transformer backbone is not available in this build "
                "(tartan_imu.model.transformer.transformer_odom missing). Use "
                "model_name='Foundation_Model', or install the transformer extra."
            ) from e

        mp = model_config.get("model_param", {})
        data_cfg = model_config.get("data", {})

        input_dim = mp.get("input_dim", 6)
        d_model = mp.get("d_model", 128)
        n_heads = mp.get("n_heads", 4)
        n_layers = mp.get("n_layers", 3)
        output_dim = mp.get("output_dim", 3)
        dropout = mp.get("dropout", 0.1)  # Default to 0.1 for better regularization
        
        # Stage configuration
        stage = mp.get("stage", 1)  # 1: IMU only, 2: IMU+joints, 3: Masked attention
        use_hybrid = mp.get("use_hybrid", True)  # Use hybrid architecture by default
        
        # derive seq_len from window_time * imu_freq if present
        window_time = model_config.get("model_param", {}).get("window_time", 1.0)
        imu_freq = data_cfg.get("imu_freq", 100)
        seq_len = int(window_time * imu_freq)
        max_seq_len = mp.get("max_seq_len", 2000)

        # Stage 3: Masked attention parameters
        adjacency_matrix = mp.get("adjacency_matrix", None)
        urdf_path = mp.get("urdf_path", None)
        joint_order = mp.get("joint_order", None)
        
        if stage == 3:
            if adjacency_matrix is None and urdf_path is None:
                logger.warning("Stage 3 requires urdf_path or adjacency_matrix, falling back to stage 1")
                stage = 1
            elif urdf_path is not None and joint_order is None:
                logger.warning("Stage 3 requires joint_order when using urdf_path, falling back to stage 1")
                stage = 1

        model = TransformerOdoModel(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            seq_len=seq_len,
            max_seq_len=max_seq_len,
            output_dim=output_dim,
            dropout=dropout,
            stage=stage,
            use_hybrid=use_hybrid,
            adjacency_matrix=adjacency_matrix,
            urdf_path=urdf_path,
            joint_order=joint_order,
        ).to(device)

        stage_names = {1: "IMU only", 2: "IMU+Joints", 3: "Masked Attention"}
        arch_type = "Hybrid" if use_hybrid else "LIMU-BERT"
        logger.info(
            f"Created {arch_type} Transformer_Odo - Stage {stage} ({stage_names.get(stage, 'Unknown')}) "
            f"with {sum(p.numel() for p in model.parameters()):,} parameters"
        )
        return model

    @staticmethod
    def create_optimizer(
        model: nn.Module, optimizer_config: Dict[str, Any]
    ) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        method = optimizer_config.get("method", "Adam")
        learning_rate = optimizer_config.get("learning_rate", 0.0005)
        weight_decay = optimizer_config.get("weight_decay", 0.01)

        if method.lower() == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif method.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif method.lower() == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        else:
            raise ModelCreationError(f"Unknown optimizer method: {method}")

        logger.info(
            f"Created {method} optimizer with lr={learning_rate}, weight_decay={weight_decay}"
        )
        return optimizer

    @staticmethod
    def create_scheduler(
        optimizer: torch.optim.Optimizer, scheduler_config: Dict[str, Any]
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        scheduler_type = scheduler_config.get("type", "ReduceLROnPlateau")

        if scheduler_type == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=scheduler_config.get("factor", 0.1),
                patience=scheduler_config.get("patience", 5),
                verbose=True,
                eps=1e-6,
            )
        elif scheduler_type == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=scheduler_config.get("T_0", 10),
                T_mult=scheduler_config.get("T_mult", 2),
                eta_min=scheduler_config.get("eta_min", 1e-6),
            )
        else:
            raise ModelCreationError(f"Unknown scheduler type: {scheduler_type}")

        logger.info(f"Created {scheduler_type} scheduler")
        return scheduler

    @staticmethod
    def freeze_backbone(model: nn.Module) -> nn.Module:
        """Freeze backbone parameters for fine-tuning."""
        # Freeze main model components
        components_to_freeze = [
            "model.input_block",
            "model.residual_groups",
            "model.resnet_post_pro",
            "model.lstm",
        ]

        for component in components_to_freeze:
            try:
                for param in getattr(model, component).parameters():
                    param.requires_grad = False
            except AttributeError:
                logger.warning(f"Component {component} not found in model")

        # Freeze batch normalization layers
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                for param in module.parameters():
                    param.requires_grad = False

        logger.info("Backbone parameters frozen for fine-tuning")
        return model

    @staticmethod
    def enable_batch_norm_training(model: nn.Module) -> nn.Module:
        """Enable training for batch normalization layers."""
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                for param in module.parameters():
                    param.requires_grad = True

        logger.info("Batch normalization parameters enabled for training")
        return model

    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count trainable and total parameters."""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        return {
            "trainable": trainable_params,
            "total": total_params,
            "frozen": total_params - trainable_params,
        }

    @staticmethod
    def print_model_summary(model: nn.Module):
        """Print model parameter summary."""
        param_counts = ModelFactory.count_parameters(model)
        trainable_percent = 100 * param_counts["trainable"] / param_counts["total"]

        logger.info(
            f"Model Parameters: "
            f"Trainable: {param_counts['trainable']:,} "
            f"({trainable_percent:.1f}%), "
            f"Frozen: {param_counts['frozen']:,}, "
            f"Total: {param_counts['total']:,}"
        )

    @staticmethod
    def print_trainable_parameters(model: nn.Module):
        """Print the number of trainable parameters in the model."""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / all_params

        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_params:,} || "
            f"Trainable: {trainable_percent:.1f}%"
        )


def create_model_from_config(
    model_config: Dict[str, Any], device: torch.device
) -> nn.Module:
    """Convenience function to create model from config."""
    return ModelFactory.create_model(model_config, device)


def create_optimizer_from_config(
    model: nn.Module, optimizer_config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """Convenience function to create optimizer from config."""
    return ModelFactory.create_optimizer(model, optimizer_config)


def create_scheduler_from_config(
    optimizer: torch.optim.Optimizer, scheduler_config: Dict[str, Any]
) -> torch.optim.lr_scheduler._LRScheduler:
    """Convenience function to create scheduler from config."""
    return ModelFactory.create_scheduler(optimizer, scheduler_config)
