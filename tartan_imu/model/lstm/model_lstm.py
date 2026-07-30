# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""ResNet+LSTM backbones, output heads, and the multi-head FoundationModel.

This module defines the "Foundation_Model" family: a shared ResNet/LSTM trunk
(``ResNetLSTMSeqNet`` and lighter/cross-axis variants), per-motion-type
``OutputHead`` blocks, and ``FoundationModel`` which dispatches the trunk
features to the relevant heads.

The implementation now lives in focused submodules; this module re-exports the
public names so existing ``from ...model.model_lstm import X`` and
``model_lstm.X`` accesses keep working unchanged.
"""
from tartan_imu.model.common.blocks import (  # noqa: F401
    EinOpsRearrange,
    FcBlock,
    IMU_instantiate_trunk,
    MultiheadAttention,
    ResBlock,
    SimpleTransformer,
)
from tartan_imu.model.lstm.heads import FoundationModel, OutputHead  # noqa: F401
from tartan_imu.model.lstm.trunks import (  # noqa: F401
    Crossxy_LSTM_Model,
    ResNetLSTMSeqNet,
    ResNetLSTMSeqNet_Light,
)

__all__ = [
    "EinOpsRearrange",
    "FcBlock",
    "IMU_instantiate_trunk",
    "MultiheadAttention",
    "ResBlock",
    "SimpleTransformer",
    "FoundationModel",
    "OutputHead",
    "Crossxy_LSTM_Model",
    "ResNetLSTMSeqNet",
    "ResNetLSTMSeqNet_Light",
]
