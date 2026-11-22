"""M22: locally compare provenance-verified M22 phone inference benchmark reports.

Reports may contain browser/device hints. This tool never sends data anywhere
and deliberately omits user agents and timestamps from its summary.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import median

_FORMATS = {
    "fp32_full_window": ("full_window", 1),
    "int4_full_window": ("full_window", 1),
    "fp32_kv_cache": ("cached", 2),
    "int4_kv_cache": ("cached", 3),
}
_PROTOCOL = "hi_warmup2_16byte_greedy_v1"
_SHA = re.compile(r"^[0-9a-f]{64}$")


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _positive(value: object) -> bool:
    return (type(value) in (float, int) and
            math.isfinite(value) and value > 0)


def validate_report(data: dict) -> dict:
    """Reject partial, mixed-protocol and untraceable benchmark inputs."""
    _check(isinstance(data, dict), "Benchmark must be a JSON object")
    _check(data.get("schema_version") == 3 and
           data.get("kind") == "aetherion-mobile-local-browser-inference-benchmark",
           "Expected a complete M22 mobile benchmark report (schema v3)")
    _check(data.get("benchmark_protocol") == _PROTOCOL, "Benchmark protocol mismatch")
    _check(data.get("scope") == "offline-local-single-thread-wasm-byte-token-inference",
           "Benchmark execution scope mismatch")
    fmt = data.get("inference_format")
    _check(fmt in _FORMATS, "Unsupported inference format")
    family, schema = _FORMATS[fmt]
    _check(data.get("model_manifest_schema") == schema, "Manifest schema/format mismatch")
    for field in ("model_sha256", "source_model_sha256"):
        _check(isinstance(data.get(field), str) and bool(_SHA.fullmatch(data[field])),
               f"Missing or invalid {field}")
    if fmt.startswith("fp32_"):
        _check(data["model_sha256"] == data["source_model_sha256"],
               "FP32 report provenance does not match loaded graph")
    _check(type(data.get("window_byte_tokens")) is int and data["window_byte_tokens"] >= 2,
           "Invalid context window")
    _check(type(data.get("model_file_bytes")) is int and data["model_file_bytes"] > 0,
           "Invalid ONNX graph byte count")
    _check(data.get("interrupted") is False and
           data.get("warmup_steps_completed") == 2 and
           data.get("measured_steps") == 16,
           "Only completed 2-warmup/16-measured reports are comparable")
    _check(_positive(data.get("byte_tokens_per_inference_second")),
           "Invalid byte-token throughput")
    elapsed = data.get("elapsed_inference_and_feeds_seconds")
    _check(_positive(elapsed), "Invalid measured inference elapsed time")
    # Both values are rounded in the browser; allow rounding and timer noise.
    _check(math.isclose(data["byte_tokens_per_inference_second"],
                        16 / elapsed, rel_tol=0.05),
           "Throughput disagrees with measured time")
    phases = data.get("latency_ms")
    _check(isinstance(phases, dict), "Missing phase latency statistics")
    total = 0
    for phase in ("prefill", "decode", "full_window", "overall"):
        stats = phases.get(phase)
        if stats is None:
            continue
        _check(isinstance(stats, dict) and
               type(stats.get("samples")) is int and stats["samples"] > 0,
               f"Invalid {phase} sample count")
        for name in ("min_ms", "mean_ms", "median_ms", "p95_ms", "max_ms"):
            _check(_positive(stats.get(name)), f"Invalid {phase} {name}")
        _check(stats["min_ms"] <= stats["median_ms"] <= stats["max_ms"] and
               stats["min_ms"] <= stats["p95_ms"] <= stats["max_ms"],
               f"Inconsistent {phase} latency order")
        if phase != "overall":
            total += stats["samples"]
    _check(phases.get("overall") is not None and
           phases["overall"]["samples"] == total == 16,
           "Latency phase sample counts do not total 16")
    if family == "cached":
        _check(phases.get("full_window") is None and
               (phases.get("prefill") is not None or phases.get("decode") is not None),
               "Cached report has incompatible phase labels")
    else:
        _check(phases.get("prefill") is None and phases.get("decode") is None and
               phases.get("full_window") is not None,
               "Full-window report has incompatible phase labels")
    device = data.get("device")
    _check(isinstance(device, dict) and
           isinstance(device.get("user_agent"), str) and bool(device["user_agent"]) and
           type(device.get("installed_standalone")) is bool,
           "Missing browser environment information")
    return data


def load_reports(paths: list[str | Path]) -> list[dict]:
    _check(len(paths) >= 3, "At least three independent runs are required per format")
    _check(len({str(Path(p).resolve()) for p in paths}) == len(paths),
           "Duplicate benchmark report paths are not independent trials")
    reports = []
    for path in paths:
        p = Path(path)
        _check(p.stat().st_size <= 1_000_000, "Benchmark JSON exceeds the 1 MB limit")
        try:
            report = json.loads(p.read_text(encoding="utf-8"))
            reports.append(validate_report(report))
        except (ValueError, UnicodeError) as error:
            raise ValueError(f"{p.name}: {error}") from error
    return reports


def _device_contract(item: dict) -> tuple:
    device = item["device"]
    return (device["user_agent"], device.get("hardware_concurrency_reported"),
            device["installed_standalone"])


def _cohort(reports: list[dict], label: str) -> dict:
    first = reports[0]
    for item in reports[1:]:
        for key in ("benchmark_protocol", "inference_format", "model_sha256",
                    "source_model_sha256", "window_byte_tokens", "model_file_bytes"):
            _check(item[key] == first[key], f"{label} trials mix {key}")
        _check(_device_contract(item) == _device_contract(first),
               f"{label} trials mix browser environment settings")
    return {
        "format": first["inference_format"],
        "model_sha256_prefix": first["model_sha256"][:12],
        "trials": len(reports),
        "model_bytes": first["model_file_bytes"],
        "median_byte_tokens_per_second": round(
            median(r["byte_tokens_per_inference_second"] for r in reports), 3),
        "median_overall_p95_ms": round(
            median(r["latency_ms"]["overall"]["p95_ms"] for r in reports), 3),
    }


def compare_reports(baseline: list[dict], candidate: list[dict]) -> dict:
    _check(len(baseline) >= 3 and len(candidate) >= 3,
           "At least three complete trials per format are required")
    for report in baseline + candidate:
        validate_report(report)
    left = _cohort(baseline, "Baseline")
    right = _cohort(candidate, "Candidate")
    a, b = baseline[0], candidate[0]
    _check(a["source_model_sha256"] == b["source_model_sha256"],
           "Comparison refused: checkpoints/export origins differ")
    _check(a["window_byte_tokens"] == b["window_byte_tokens"],
           "Comparison refused: token context windows differ")
    _check(_FORMATs_family(a) == _FORMATs_family(b),
           "Comparison refused: cached and full-window modes differ")
    _check(_device_contract(a) == _device_contract(b),
           "Comparison refused: browser/device hints differ")
    return {
        "status": "comparable_benchmarks_not_statistical_significance",
        "source_model_sha256_prefix": a["source_model_sha256"][:12],
        "benchmark_protocol": _PROTOCOL,
        "window_byte_tokens": a["window_byte_tokens"],
        "baseline": left,
        "candidate": right,
        "candidate_over_baseline_throughput_ratio": round(
            right["median_byte_tokens_per_second"] /
            left["median_byte_tokens_per_second"], 4),
        "candidate_over_baseline_model_size_ratio": round(
            right["model_bytes"] / left["model_bytes"], 4),
        "caveat": ("Same browser user-agent does not prove the same physical device. "
                   "Verify phone, software version, temperature and power conditions "
                   "manually. No quality, energy or significance claims."),
    }


def _FORMATs_family(report: dict) -> str:
    return _FORMATS[report["inference_format"]][0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare private M22 local phone benchmark reports without uploading data")
    parser.add_argument("--baseline", nargs="+", required=True, metavar="JSON")
    parser.add_argument("--candidate", nargs="+", required=True, metavar="JSON")
    parser.add_argument("--json", action="store_true", help="Print sanitized JSON summary")
    args = parser.parse_args()
    try:
        result = compare_reports(
            load_reports(args.baseline), load_reports(args.candidate))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Comparison rejected: {error}\n")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("M22 comparison: matching model origin, protocol, window and browser hints")
        for name in ("baseline", "candidate"):
            row = result[name]
            print(f"{name}: {row['format']} — {row['trials']} trials, "
                  f"median {row['median_byte_tokens_per_second']:.3f} byte-tokens/s, "
                  f"median p95 {row['median_overall_p95_ms']:.3f} ms")
        print(f"Candidate throughput ratio: "
              f"{result['candidate_over_baseline_throughput_ratio']:.4f}x")
        print(f"Candidate model size ratio: "
              f"{result['candidate_over_baseline_model_size_ratio']:.4f}x")
        print(result["caveat"])


if __name__ == "__main__":
    main()
