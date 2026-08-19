"""Torch device selection shared by the training and evaluation entry points."""

from __future__ import annotations

import torch


# Preference order used when the caller asks for automatic selection.
DEVICE_PREFERENCE = ("cuda", "mps", "cpu")


def _is_available(kind: str) -> bool:
    if kind == "cuda":
        return torch.cuda.is_available()
    if kind == "mps":
        backend = getattr(torch.backends, "mps", None)
        return backend is not None and bool(backend.is_available())
    # CPU is always usable; anything more exotic is left to torch to reject.
    return True


def resolve_device(device: str | None = "auto") -> torch.device:
    """Return the device requested by ``device``.

    ``None`` and ``"auto"`` pick the first available backend from
    :data:`DEVICE_PREFERENCE`, i.e. CUDA, then MPS, then CPU. An explicit
    request for an unavailable backend raises ``RuntimeError`` instead of
    silently falling back.
    """
    if device is None or device == "auto":
        return torch.device(next(k for k in DEVICE_PREFERENCE if _is_available(k)))
    result = torch.device(device)
    if not _is_available(result.type):
        raise RuntimeError(
            f"{result.type.upper()} was requested, but it is not available"
        )
    return result


def resolve_device_name(device: str | None = "auto") -> str:
    """Same as :func:`resolve_device`, but as a string for string-typed APIs."""
    return str(resolve_device(device))
