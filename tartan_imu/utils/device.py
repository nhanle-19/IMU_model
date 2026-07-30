# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Device-selection helpers shared by the training and evaluation entry points.

These wrappers keep the CLI portable between CUDA hosts (training) and
CPU-only hosts (for example a shared CPU partition running test-only
evaluation), where the bare ``torch.cuda.*`` calls would raise.
"""

import torch


def get_device(index: int = 0) -> torch.device:
    """Return the compute device, pinning a CUDA device when one is available.

    On a CUDA host this selects ``cuda:index`` and makes it the active device,
    so subsequent tensor allocations land there. On a CPU-only host it returns
    ``torch.device("cpu")`` without touching the CUDA runtime.

    Args:
      index: CUDA device ordinal to select when CUDA is available.

    Returns:
      The selected :class:`torch.device`.
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def sync_if_cuda() -> None:
    """Block until the current CUDA stream is done, or no-op on a CPU-only host.

    ``torch.cuda.synchronize`` raises on a machine without a CUDA device, so
    call sites that only need the barrier for GPU ordering/timing use this guard
    to stay portable to CPU-only evaluation.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
