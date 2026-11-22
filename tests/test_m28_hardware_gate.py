"""M28: synthetic hardware benchmark acceptance tests; no GPU required."""
import copy
import json
import math
import sys

import pytest

import pilot_train
from aetherion.hardware_gate import validate_pilot_benchmark
from aetherion.presets import parameter_breakdown, get_preset


def synthetic_report(**patch):
    steps, batch, accum, context = 3, 1, 4, 256
    tokens = steps * batch * accum * context
    source = {
        "schema_version": 1,
        "kind": "aetherion-synthetic-optimizer-hardware-benchmark",
        "synthetic": True,
        "preset": "125m",
        "device": "cuda:0",
        "hardware": "Synthetic fake GPU (not physical evidence)",
        "torch_version": "test",
        "precision": "bf16",
        "parameters": parameter_breakdown(get_preset("125m"))["total"],
        "context": context, "batch_size": batch, "accumulation": accum,
        "warmup_updates": 1, "measured_updates": steps, "tokens": tokens,
        "seconds": 2.0, "tokens_per_second": tokens / 2.0, "last_loss": 10.0,
        "cuda_peak_allocated_bytes": 5 * 1024**3,
        "cuda_peak_reserved_bytes": 6 * 1024**3,
        "xpu_peak_allocated_bytes": None, "xpu_peak_reserved_bytes": None,
    }
    source.update(patch)
    return source


@pytest.fixture
def report_path(tmp_path):
    path = tmp_path / "bench.json"
    def write(report=None):
        path.write_text(json.dumps(synthetic_report() if report is None else report))
        return path
    return write


def validate(report_path, *, device="cuda:0", peak=8.0):
    return validate_pilot_benchmark(
        report_path, context=256, batch_size=1, accumulation=4,
        precision="bf16", device=device, max_peak_reserved_gib=peak,
    )


def test_exact_match_records_evidence_and_no_fake_memory_claim(report_path):
    path = report_path()
    result = validate(path)
    assert result["preset"] == "125m"
    assert len(result["benchmark_sha256"]) == 64
    assert result["sampled_peak_reserved_bytes"] == 6 * 1024**3
    assert result["synthetic_tokens_per_second"] == 1536.0
    assert "not authenticated" in result["limitations"]
    assert "Synthetic fake GPU" in result["hardware"]


@pytest.mark.parametrize("patch,pattern", [
    ({"preset": "micro"}, "125m"),
    ({"synthetic": False}, "125m"),
    ({"schema_version": 2}, "schema"),
    ({"kind": "other"}, "schema"),
    ({"parameters": 42}, "parameter"),
    ({"device": "cuda:1"}, "device"),
    ({"precision": "fp32"}, "precision"),
    ({"context": 128}, "context"),
    ({"batch_size": 2}, "batch_size"),
    ({"accumulation": 2}, "accumulation"),
    ({"warmup_updates": 0}, "warmup"),
    ({"measured_updates": 2}, "measured"),
    ({"tokens": 3}, "token"),
    ({"seconds": 0.0}, "seconds"),
    ({"seconds": float("inf")}, "seconds"),
    ({"tokens_per_second": 1000}, "tokens/second"),
    ({"last_loss": float("nan")}, "non-finite"),
    ({"hardware": ""}, "hardware"),
    ({"cuda_peak_reserved_bytes": None}, "nonnegative integer"),
    ({"cuda_peak_allocated_bytes": 7 * 1024**3}, "memory"),
])
def test_mismatched_or_untrusted_report_rejected(report_path, patch, pattern):
    with pytest.raises(ValueError, match=pattern):
        validate(report_path(synthetic_report(**patch)))


def test_budget_and_explicit_accelerator_required(report_path):
    path = report_path()
    with pytest.raises(ValueError, match="exceeds declared"):
        validate(path, peak=5)
    with pytest.raises(ValueError, match="explicit accelerator"):
        validate(path, device="cpu")
    with pytest.raises(ValueError, match="explicit accelerator"):
        validate(path, device="cuda")
    with pytest.raises(ValueError, match="finite positive"):
        validate(path, peak=math.nan)


def test_report_limits_and_invalid_json_fail_closed(tmp_path, report_path):
    path = report_path()
    path.write_bytes(b"x" * (64 * 1024 + 1))
    with pytest.raises(ValueError, match="64 KiB"):
        validate(path)
    path.write_text("{broken")
    with pytest.raises(ValueError, match="Invalid benchmark JSON"):
        validate(path)


def test_pilot_execute_requires_report_before_any_training(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(pilot_train, "pilot_command", lambda args:
                        called.append("pilot_preflight") or [sys.executable, "mock_train.py"])
    monkeypatch.setattr(pilot_train.subprocess, "run", lambda *a, **kw:
                        called.append("training_launcher"))
    root_args = [
        "pilot_train.py", "--data-dir", "dummy", "--tokenizer", "dummy",
        "--review", "dummy", "--allow-license", "CC0-1.0", "--steps", "2",
        "--schedule-steps", "4", "--context", "256", "--batch-size", "1",
        "--accumulation", "4", "--device", "cuda:0", "--precision", "bf16",
        "--output", "never-created.pt", "--execute",
    ]
    monkeypatch.setattr(sys, "argv", root_args)
    with pytest.raises(SystemExit) as exc:
        pilot_train.main()
    assert exc.value.code == 2
    assert called == []

    path = tmp_path / "bench.json"
    path.write_text(json.dumps(synthetic_report(device="cuda:1")))
    monkeypatch.setattr(sys, "argv", root_args + [
        "--benchmark-report", str(path), "--max-peak-reserved-gib", "8",
    ])
    with pytest.raises(SystemExit) as exc:
        pilot_train.main()
    assert exc.value.code == 2
    assert called == []

    path.write_text(json.dumps(synthetic_report()))
    pilot_train.main()
    assert called == ["pilot_preflight", "training_launcher"]


def test_pilot_dry_run_remains_available_without_benchmark(monkeypatch, capsys):
    monkeypatch.setattr(pilot_train, "pilot_command", lambda args: ["train.py", "--dry"])
    monkeypatch.setattr(sys, "argv", [
        "pilot_train.py", "--data-dir", "fixture", "--tokenizer", "fixture",
        "--review", "fixture", "--allow-license", "CC0-1.0",
        "--steps", "2", "--schedule-steps", "4", "--output", "fixture.pt",
    ])
    pilot_train.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["executed"] is False
    assert payload["command"] == ["train.py", "--dry"]
