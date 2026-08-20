"""Torch device selection shared by training and evaluation entry points."""

from __future__ import annotations

import torch


DEVICE_PREFERENCE = ("cuda", "mps", "cpu")


def _is_available(kind: str) -> bool:
    if kind == "cuda":
        return torch.cuda.is_available()
    if kind == "mps":
        backend = getattr(torch.backends, "mps", None)
        return backend is not None and bool(backend.is_available())
    return True


def resolve_device(device: str | None = "auto") -> torch.device:
    """Resolve an explicit device or select CUDA, then MPS, then CPU."""
    if device is None or device == "auto":
        return torch.device(
            next(kind for kind in DEVICE_PREFERENCE if _is_available(kind))
        )
    result = torch.device(device)
    if not _is_available(result.type):
        raise RuntimeError(
            f"{result.type.upper()} was requested, but it is not available"
        )
    return result


def resolve_device_name(device: str | None = "auto") -> str:
    """Resolve a device and return its string representation."""
    return str(resolve_device(device))
