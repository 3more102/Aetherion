"""M34: Check a live accelerator before starting an approved 125M pilot.

This is a point-in-time admission check, not a guarantee against OOM.
Self-reported benchmark metadata is not authenticated.
"""
from __future__ import annotations

import math
import re

_HEADROOM_FACTOR = 1.25
_DEVICE_RE = re.compile(r"^(cuda|xpu):([0-9]+)$")


def _nonnegative_integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"Live accelerator {name} must be a nonnegative integer")
    return value


def check_live_accelerator(benchmark: dict, *, device: str, torch_backend=None) -> dict:
    """Bind a shape-validated benchmark summary to the active device.

    Require matching index/name, usable accelerator and sampled free memory
    >= 125% of the reported synthetic peak reserved bytes. Device availability
    can change immediately afterwards. A user can forge the benchmark itself.
    """
    if type(benchmark) is not dict:
        raise ValueError("Live accelerator requires validated benchmark summary")
    match = _DEVICE_RE.fullmatch(device) if isinstance(device, str) else None
    if match is None or benchmark.get("device") != device:
        raise ValueError("Live accelerator requires matching indexed CUDA/XPU device")
    kind, index = match.group(1), int(match.group(2))
    if torch_backend is None:
        import torch
        torch_backend = torch
    backend = getattr(torch_backend, kind, None)
    if backend is None or not all(callable(getattr(backend, x, None)) for x in (
            "is_available", "device_count", "get_device_name",
            "get_device_properties", "mem_get_info")):
        raise ValueError(f"Live accelerator {kind} inspection APIs are unavailable")
    try:
        available = backend.is_available()
        devices = backend.device_count()
    except (RuntimeError, AssertionError, OSError) as exc:
        raise ValueError("Unable to inspect live accelerator availability") from exc
    if type(available) is not bool or not available or type(devices) is not int or not 0 <= index < devices:
        raise ValueError("Requested live accelerator is not available")
    try:
        current_name = backend.get_device_name(index)
        props = backend.get_device_properties(index)
        free, reported_total = backend.mem_get_info(index)
    except (RuntimeError, AssertionError, OSError, TypeError) as exc:
        raise ValueError("Unable to inspect live accelerator identity or free memory") from exc
    expected_name = benchmark.get("hardware")
    if not isinstance(current_name, str) or not current_name or current_name != expected_name:
        raise ValueError("Live accelerator identity differs from benchmark hardware")
    physical_total = _nonnegative_integer(getattr(props, "total_memory", None), "device capacity")
    free = _nonnegative_integer(free, "free bytes")
    reported_total = _nonnegative_integer(reported_total, "reported total bytes")
    if physical_total == 0 or reported_total == 0 or not 0 <= free <= reported_total:
        raise ValueError("Live accelerator memory accounting is invalid")
    # APIs may report different totals due to runtime/driver accounting.
    if reported_total > physical_total:
        raise ValueError("Live accelerator reported total exceeds device capacity")
    reserved = _nonnegative_integer(benchmark.get("sampled_peak_reserved_bytes"),
                                    "benchmark peak reserved bytes")
    allocated = _nonnegative_integer(benchmark.get("sampled_peak_allocated_bytes"),
                                     "benchmark peak allocated bytes")
    if reserved <= 0 or allocated <= 0 or reserved < allocated or reserved > reported_total:
        raise ValueError("Benchmark memory demand exceeds live accelerator capacity")
    required = math.ceil(reserved * _HEADROOM_FACTOR)
    if free < required:
        raise ValueError("Insufficient free accelerator memory for measured peak plus 25% headroom")
    return {
        "schema_version": 1,
        "kind": "aetherion-live-accelerator-preflight",
        "device": device,
        "hardware": current_name,
        "live_free_bytes": free,
        "live_total_bytes": reported_total,
        "device_capacity_bytes": physical_total,
        "synthetic_peak_reserved_bytes": reserved,
        "required_free_bytes": required,
        "headroom_factor": _HEADROOM_FACTOR,
        "result": "point_in_time_preflight_passed",
        "limitations": (
            "A matching live device and a free-memory snapshot cannot authenticate a user-supplied "
            "benchmark or guarantee long-run training memory, availability or stability. "
            "This preflight does not train or benchmark the model."
        ),
    }
