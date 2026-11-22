"""M22 phone benchmark provenance, comparison and privacy gates."""
import copy
import json

import pytest

from compare_mobile_benchmarks import compare_reports, load_reports, validate_report


def sample(fmt="fp32_kv_cache", source="a", model=None, seconds=0.2):
    source_sha = source * 64
    sha = (model or source) * 64
    cached = fmt.endswith("kv_cache")
    def lat(samples, ms):
        return {
            "samples": samples, "min_ms": ms / 2, "mean_ms": ms,
            "median_ms": ms, "p95_ms": ms * 1.5, "max_ms": ms * 2,
        }
    return {
        "schema_version": 3,
        "kind": "aetherion-mobile-local-browser-inference-benchmark",
        "benchmark_protocol": "hi_warmup2_16byte_greedy_v1",
        "scope": "offline-local-single-thread-wasm-byte-token-inference",
        "inference_format": fmt,
        "model_sha256": sha,
        "source_model_sha256": source_sha,
        "model_manifest_schema": (3 if fmt == "int4_kv_cache" else
                                  2 if cached else 1),
        "model_file_bytes": 60000 if fmt.startswith("int4") else 100000,
        "window_byte_tokens": 8,
        "warmup_steps_completed": 2,
        "measured_steps": 16,
        "interrupted": False,
        "elapsed_inference_and_feeds_seconds": seconds,
        "byte_tokens_per_inference_second": 16 / seconds,
        "latency_ms": {
            "overall": lat(16, 10),
            "prefill": lat(2, 15) if cached else None,
            "decode": lat(14, 5) if cached else None,
            "full_window": None if cached else lat(16, 10),
        },
        "device": {
            "user_agent": "Synthetic Phone Browser",
            "hardware_concurrency_reported": 4,
            "installed_standalone": True,
        },
        "prompt_history": "SHOULD NOT LEAK",
    }


def test_validate_and_compare_same_source_fp32_and_int4():
    base = [sample() for _ in range(3)]
    candidate = [sample("int4_kv_cache", model="b", seconds=0.1)
                 for _ in range(3)]
    result = compare_reports(base, candidate)
    assert result["candidate_over_baseline_throughput_ratio"] == 2.0
    assert result["candidate_over_baseline_model_size_ratio"] == 0.6
    assert result["source_model_sha256_prefix"] == "a" * 12
    assert "user_agent" not in result
    assert "SHOULD NOT LEAK" not in json.dumps(result)


@pytest.mark.parametrize("modifier", [
    lambda r: r.update(interrupted=True),
    lambda r: r.update(measured_steps=15),
    lambda r: r.update(source_model_sha256="invalid"),
    lambda r: r.update(benchmark_protocol="other_protocol"),
    lambda r: r["latency_ms"]["decode"].update(samples=13),
    lambda r: r.update(byte_tokens_per_inference_second=10),
    lambda r: r.update(model_manifest_schema=1),
    lambda r: r["device"].pop("user_agent"),
])
def test_reject_incomplete_or_mislabelled_reports(modifier):
    report = sample()
    modifier(report)
    with pytest.raises(ValueError):
        validate_report(report)


def test_comparator_refuses_mismatched_origins_and_devices():
    base = [sample() for _ in range(3)]
    alternative = [sample("int4_kv_cache", source="c", model="b")
                   for _ in range(3)]
    with pytest.raises(ValueError, match="origins differ"):
        compare_reports(base, alternative)

    alternative = [sample("int4_kv_cache", model="b") for _ in range(3)]
    alternative[0]["device"]["user_agent"] = "Another Browser"
    with pytest.raises(ValueError, match="mix browser"):
        compare_reports(base, alternative)

    alternative = [sample("int4_full_window", model="b") for _ in range(3)]
    with pytest.raises(ValueError, match="modes differ"):
        compare_reports(base, alternative)


def test_group_must_not_mix_export_or_context():
    base = [sample() for _ in range(3)]
    candidate = [sample("int4_kv_cache", model="b") for _ in range(3)]
    base[2]["model_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="mix model_sha256"):
        compare_reports(base, candidate)

    base[2]["model_sha256"] = "a" * 64
    base[2]["window_byte_tokens"] = 16
    with pytest.raises(ValueError, match="mix window"):
        compare_reports(base, candidate)


def test_local_loader_rejects_duplicate_and_partial(tmp_path):
    path = tmp_path / "trial.json"
    path.write_text(json.dumps(sample()), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_reports([path, path, path])

    corrupt = copy.deepcopy(sample())
    corrupt["interrupted"] = True
    path.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(ValueError, match="completed"):
        load_reports([path, tmp_path / "missing1.json", tmp_path / "missing2.json"])
