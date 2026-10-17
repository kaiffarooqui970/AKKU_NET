"""Choosing the compute device: NVIDIA GPU (CUDA), Apple GPU (MPS), or CPU."""


def get_device(preference: str = "auto") -> str:
    """Return a device string Ultralytics understands.

    "auto" tries CUDA first, then Apple MPS, then falls back to CPU.
    Any other value (e.g. "cpu") is returned unchanged, so you can force a device.
    """
    if preference != "auto":
        return preference

    # torch is imported here, not at the top, so scripts that don't need it
    # (like frame extraction) start quickly.
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
