# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""LSTM multi-head ("Foundation_Model") backbone builder.

Extracted verbatim from config/configer.build_model's `Foundation_Model`
branch. Head ordering preserved (dog, human, car, drone) so the constructed
ModuleDict matches the legacy model parameter-for-parameter.
"""
import torch.nn as nn

from tartan_imu.model.registry import register


@register("Foundation_Model")
def build_foundation_model(cfg: dict):
    from tartan_imu.model.lstm import model_lstm

    if cfg["model"].get("cross_xyz", False):
        trunk = model_lstm.Crossxy_LSTM_Model(cfg)
    else:
        trunk = model_lstm.ResNetLSTMSeqNet(cfg)
    heads = nn.ModuleDict({
        "dog": model_lstm.OutputHead(cfg, "dog"),
        "human": model_lstm.OutputHead(cfg, "human"),
        "car": model_lstm.OutputHead(cfg, "car"),
        "drone": model_lstm.OutputHead(cfg, "drone"),
    })
    return model_lstm.FoundationModel(cfg=cfg, trunk=trunk, heads=heads)
