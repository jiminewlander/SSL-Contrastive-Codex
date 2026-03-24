"""
Runtime helpers for selecting the best available accelerator and keeping
backend-specific behavior in one place.
"""
from contextlib import nullcontext
import os
import platform

import torch


def get_best_device() -> torch.device:
    """Prefer CUDA when available, then MPS on Apple Silicon, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def configure_torch_runtime(device: torch.device) -> None:
    """Apply low-risk runtime defaults for the active backend."""
    try:
        torch.set_float32_matmul_precision("high")
    except RuntimeError:
        pass

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def resolve_num_workers(value) -> int:
    """
    Use a conservative default on macOS, where PIL-based loading with spawned
    workers is often less stable than a single-process loader.
    """
    if value not in (None, "", "None", "none", "null", "Null"):
        return int(value)

    cpu_count = os.cpu_count() or 1
    if platform.system() == "Darwin":
        return 0
    return cpu_count


def should_pin_memory(device: torch.device) -> bool:
    """Pinned memory only helps CUDA host-to-device transfers."""
    return device.type == "cuda"


def autocast_context(device: torch.device):
    """Keep mixed precision enabled only on CUDA."""
    if device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def grad_scaler(device: torch.device):
    """Use GradScaler only on CUDA where AMP is active."""
    return torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
