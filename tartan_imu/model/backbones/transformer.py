# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Transformer backbone builder (interface placeholder).

NOTE: This is the LSTM-only public release. The Transformer core
(``tartan_imu/model/transformer/``) is not included in this release. The builder
interface stays registered so that adding the core later restores the backbone
with zero wiring changes; until then, selecting model_name="Transformer" raises
a clear NotImplementedError.
"""
import logging

from tartan_imu.model.registry import register


@register("Transformer")
def build_transformer(cfg: dict):
    try:
        from tartan_imu.model.transformer.transformer_odom import TransformerOdoModel
    except ImportError as e:
        raise NotImplementedError(
            "The Transformer backbone is not included in this LSTM-only release "
            "(tartan_imu.model.transformer is not included). "
            "Use model_name='Foundation_Model'."
        ) from e

    mp = cfg.get("model_param", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})

    # For transformer: seq_len in train config is number of windows.
    # Total timesteps = num_windows * samples_per_window, where
    # samples_per_window = window_time * sample_freq (use sample_freq, not imu_freq).
    num_windows = train_cfg.get("seq_len", 10)
    window_time = mp.get("window_time", 1.0)
    sample_freq = data_cfg.get("sample_freq", 20)
    samples_per_window = int(window_time * sample_freq)
    max_seq_len = num_windows * samples_per_window

    logging.info(
        f"Transformer config: {num_windows} windows × {samples_per_window} samples/window "
        f"= {max_seq_len} total timesteps (at {sample_freq}Hz)"
    )
    if max_seq_len > 500:
        logging.warning(
            f" Transformer with {max_seq_len} timesteps will need ~"
            f"{(128 * 4 * max_seq_len * max_seq_len * 4) / 1024**3:.2f}GB "
            f"just for attention scores (batch=128)! "
            f"Consider reducing sample_freq (current: {sample_freq}Hz) or seq_len (current: {num_windows})"
        )

    return TransformerOdoModel(
        input_dim=mp.get("input_dim", 6),
        d_model=mp.get("d_model", 128),
        n_heads=mp.get("n_heads", 4),
        n_layers=mp.get("n_layers", 3),
        seq_len=num_windows,
        max_seq_len=max_seq_len,
        output_dim=mp.get("output_dim", 3),
        dropout=mp.get("dropout", 0.0),
        share_params=mp.get("share_params", True),
        stage=mp.get("stage", 1),
        use_hybrid=mp.get("use_hybrid", True),
        use_kv_cache=mp.get("use_kv_cache", False),
    )
