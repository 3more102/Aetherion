"""Optional Intel XPU detection; CPU-only PyTorch remains supported."""
import torch


def xpu_available():
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def resolve_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if xpu_available():
            return "xpu"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    device = torch.device(name)
    if device.type == "xpu":
        if not xpu_available():
            raise ValueError("XPU unavailable; install XPU-enabled PyTorch and compatible Intel GPU drivers")
        if device.index is not None and device.index >= torch.xpu.device_count():
            raise ValueError("XPU device index out of range")
    return name
