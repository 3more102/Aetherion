"""M28: Validate self-reported synthetic accelerator benchmark before a 125M pilot.

The report is *not* authenticated evidence of actual hardware performance.
It supplies a conservative local execution gate for explicitly requested runs.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import re

from .presets import get_preset, parameter_breakdown


_MAX_REPORT_BYTES = 64 * 1024
_EXPLICIT_ACCELERATOR = re.compile(r"^(cuda|xpu):([0-9]+)$")


def _positive_finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Benchmark {name} must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Benchmark {name} must be a finite positive number")
    return float(value)


def _nonnegative_integer(value, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Benchmark {name} must be a nonnegative integer")
    return value


def validate_pilot_benchmark(
    path: str | Path,
    *,
    context: int,
    batch_size: int,
    accumulation: int,
    precision: str,
    device: str,
    max_peak_reserved_gib: float,
) -> dict:
    """Require an exact-shape measured accelerator run under a declared budget.

    The report can be edited by its author. SHA-256 provides only identity,
    not authenticity. Peak reserved is a sampled tool-reported metric, not
    a promise about future long training runs or whole-device RAM.
    """
    if not isinstance(device, str) or not _EXPLICIT_ACCELERATOR.fullmatch(device):
        raise ValueError("Execution requires explicit accelerator --device cuda:N or xpu:N")
    budget_gib = _positive_finite(max_peak_reserved_gib, "max-peak-reserved-gib")
    if budget_gib > 1024:
        raise ValueError("Benchmark peak reserved budget exceeds 1024 GiB")

    file = Path(path)
    if not file.is_file() or file.stat().st_size < 2 or file.stat().st_size > _MAX_REPORT_BYTES:
        raise ValueError("Benchmark report is missing, empty or exceeds 64 KiB")
    raw = file.read_bytes()
    def reject_constant(s: str):
        raise ValueError(f"Benchmark report contains non-finite {s}")
    try:
        report = json.loads(raw, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid benchmark JSON report") from exc
    if not isinstance(report, dict) or report.get("schema_version") != 1 or (
        report.get("kind") != "aetherion-synthetic-optimizer-hardware-benchmark"
    ):
        raise ValueError("Unsupported training benchmark schema or kind")
    if report.get("synthetic") is not True or report.get("preset") != "125m":
        raise ValueError("Require synthetic optimizer benchmark for exact 125m preset")
    expected_parameters = parameter_breakdown(get_preset("125m"))["total"]
    if report.get("parameters") != expected_parameters or type(report.get("parameters")) is not int:
        raise ValueError("Benchmark parameter count differs from 125m preset")
    required = {"context": context, "batch_size": batch_size,
                "accumulation": accumulation, "device": device, "precision": precision}
    for key, expected in required.items():
        observed = report.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(f"Benchmark {key} differs from requested training configuration")
    hardware = report.get("hardware")
    if not isinstance(hardware, str) or not hardware.strip() or len(hardware) > 512:
        raise ValueError("Benchmark hardware identity is missing or invalid")
    warmups = _nonnegative_integer(report.get("warmup_updates"), "warmup_updates")
    steps = _nonnegative_integer(report.get("measured_updates"), "measured_updates")
    if warmups < 1 or steps < 3:
        raise ValueError("Benchmark needs >=1 warmup and >=3 measured optimizer updates")
    expected_tokens = steps * context * batch_size * accumulation
    if _nonnegative_integer(report.get("tokens"), "tokens") != expected_tokens:
        raise ValueError("Benchmark token accounting mismatch")
    duration = _positive_finite(report.get("seconds"), "seconds")
    rate = _positive_finite(report.get("tokens_per_second"), "tokens_per_second")
    if not math.isclose(rate, expected_tokens / duration, rel_tol=0.01):
        raise ValueError("Benchmark tokens/second differs from recorded duration")
    _positive_finite(report.get("last_loss"), "last_loss")
    kind = device.split(":", 1)[0]
    allocated = _nonnegative_integer(report.get(f"{kind}_peak_allocated_bytes"),
                                     f"{kind}_peak_allocated_bytes")
    reserved = _nonnegative_integer(report.get(f"{kind}_peak_reserved_bytes"),
                                    f"{kind}_peak_reserved_bytes")
    if allocated <= 0 or reserved < allocated:
        raise ValueError("Benchmark accelerator peak memory measurements are invalid")
    if reserved > budget_gib * (1024 ** 3):
        raise ValueError("Measured peak reserved memory exceeds declared execution budget")
    return {
        "schema_version": 1,
        "benchmark_sha256": sha256(raw).hexdigest(),
        "preset": "125m",
        "device": device,
        "hardware": hardware,
        "precision": precision,
        "measured_updates": steps,
        "sampled_peak_reserved_bytes": reserved,
        "sampled_peak_allocated_bytes": allocated,
        "declared_max_peak_reserved_gib": budget_gib,
        "synthetic_tokens_per_second": rate,
        "limitations": (
            "User-supplied report is not authenticated; a sampled synthetic benchmark "
            "cannot guarantee available RAM, thermal stability or real-corpus throughput."
        ),
    }
