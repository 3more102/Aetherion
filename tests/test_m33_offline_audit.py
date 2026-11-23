"""M33 verifier regression: no PyTorch or live checkpoint loading needed."""
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from verify_selected_test_report import verify_selected_test_report


def digest(data):
    return sha256(data).hexdigest()


@pytest.fixture
def fixture(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    src = {"train": b"train data\n", "validation": b"validation data\n", "test": b"test data!\n"}
    for k, data in src.items():
        (directory / (k + ".txt")).write_bytes(data)
    steps = [1, 2, 3]
    cps = []
    candidates = []
    for i, step in enumerate(steps):
        cp = tmp_path / f"step-{step}.pt"
        cp.write_bytes(f"fake checkpoint for hashing {step}".encode())
        cps.append(cp)
        nll = [1.8, 1.2, 1.5][i]
        candidates.append(dict(step=step, checkpoint_sha256=digest(cp.read_bytes()),
                               evaluated_targets=len(src["validation"]) - 1,
                               mean_nll=nll, perplexity=math.exp(nll)))
    report = {
        "schema_version": 2, "kind": "aetherion-validation-selected-test-evidence",
        "mode": "synthetic_or_unreviewed_byte_demo",
        "selection": {
            "rule": "minimum_validation_mean_next_token_nll_first_tie",
            "candidate_count": 3, "candidates": candidates, "selected_step": 2,
            "selected_checkpoint_sha256": candidates[1]["checkpoint_sha256"],
            "selected_validation_mean_nll": 1.2,
            "selected_validation_perplexity": math.exp(1.2),
            "validation_sha256": digest(src["validation"]),
            "model_config": {"block_size": 8}, "lineage_check": "not_requested",
            "lineage_run_id": None,
        },
        "test": {
            "mean_nll": 1.25, "perplexity": math.exp(1.25),
            "total_nll": 1.25 * (len(src["test"]) - 1),
            "evaluated_targets": len(src["test"]) - 1,
            "context_size": 8, "stride": 4,
            "data_sha256": digest(src["test"]), "digest_source": "operator_pinned",
            "token_unit": "UTF8_byte_tokens",
            "target_coverage": "all_targets_exactly_once",
        },
        "dataset_provenance": {
            "train_sha256": digest(src["train"]),
            "validation_sha256": digest(src["validation"]),
            "prepared_manifest_sha256": None, "tokenizer_sha256": None, "review_sha256": None,
        },
    }
    rp = tmp_path / "report.json"
    rp.write_text(json.dumps(report))
    return directory, cps, rp, report


def run(fixture):
    directory, cps, rp, _ = fixture
    return verify_selected_test_report(report_path=rp, data_dir=directory, checkpoints=cps)


def put(fixture, obj):
    fixture[2].write_text(json.dumps(obj))


def test_success_and_report_is_not_modified(fixture):
    original = fixture[2].read_bytes()
    result = run(fixture)
    assert result["result"] == "hashes_and_report_consistency_passed"
    assert result["selected_step"] == 2
    assert result["candidate_count"] == 3
    assert fixture[2].read_bytes() == original


@pytest.mark.parametrize("change, error", [
    (lambda r: r["selection"].update(selected_step=3), "Selected checkpoint"),
    (lambda r: r["selection"]["candidates"][0].update(mean_nll=0.2, perplexity=math.exp(0.2)), "Selected checkpoint"),
    (lambda r: r["selection"].update(candidate_count=2), "Candidate count"),
    (lambda r: r["selection"]["candidates"][1].update(step=1), "not increasing"),
    (lambda r: r["test"].update(total_nll=0.1), "total_nll"),
    (lambda r: r["test"].update(evaluated_targets=1000), "total_nll"),
    (lambda r: r["test"].update(stride=9), "stride"),
    (lambda r: r["test"].update(mean_nll=float("nan")), "Non-finite JSON"),
    (lambda r: r.update(schema_version=1), "schema-v2"),
])
def test_bad_report_rejected(fixture, change, error):
    r = deepcopy(fixture[3]); change(r); put(fixture, r)
    with pytest.raises(ValueError, match=error):
        run(fixture)


def test_checkpoint_and_data_mutations_refused(fixture):
    directory, cps, rp, obj = fixture
    cps[0].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checkpoint 1 checksum"):
        run(fixture)
    cps[0].write_bytes(b"fake checkpoint for hashing 1")
    (directory / "validation.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="validation data checksum"):
        run(fixture)


def test_duplicate_keys_and_oversized_json_rejected(fixture):
    directory, cps, rp, obj = fixture
    rp.write_text('{"schema_version": 2, "schema_version": 2}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        run(fixture)
    rp.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="too large"):
        run(fixture)


def test_first_equal_nll_wins(fixture):
    r = deepcopy(fixture[3]); c = r["selection"]["candidates"]
    c[0]["mean_nll"] = 1.2; c[0]["perplexity"] = math.exp(1.2)
    r["selection"].update(selected_step=1,
                          selected_checkpoint_sha256=c[0]["checkpoint_sha256"])
    put(fixture, r)
    assert run(fixture)["selected_step"] == 1
    r["selection"]["selected_step"] = 2
    put(fixture, r)
    with pytest.raises(ValueError, match="Selected checkpoint"):
        run(fixture)


def test_cli_create_only(fixture, tmp_path):
    directory, cps, rp, _ = fixture
    target = tmp_path / "audit.json"
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "verify_selected_test_report.py"),
               "--report", str(rp), "--data-dir", str(directory), "--output", str(target)]
    for cp in cps:
        command += ["--checkpoint", str(cp)]
    first = subprocess.run(command, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout) == json.loads(target.read_text())
    again = subprocess.run(command, text=True, capture_output=True)
    assert again.returncode != 0
    assert "File exists" in again.stderr
