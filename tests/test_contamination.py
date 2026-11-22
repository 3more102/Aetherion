"""M25 tests: integrity-gated lexical contamination screening."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

from aetherion.contamination import screen_cross_split, word_shingles
from aetherion.data import prepare


BASE = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma"


def fixture(tmp_path: Path, number_of_similar: int = 2):
    """Use hashed group ordering to deliberately cross train/validation/test."""
    groups = [f"group-{i}" for i in range(12)]
    ordered = sorted(groups, key=lambda g: sha256(f"42:{g}".encode()).hexdigest())
    contaminated = ordered[:1] + ordered[9:10] + ordered[-1:]
    records = []
    for idx, group in enumerate(groups):
        if group in contaminated[:number_of_similar]:
            # Each distinct record has similar-but-not-identical wording.
            text = BASE + (" omega" if group == contaminated[0] else
                           " psi" if group == contaminated[1] else " chi")
        else:
            text = " ".join(f"unique{idx}_word{j}" for j in range(20))
        records.append({"id": f"id-{idx}", "source": group, "group": group,
                        "license": "CC0-1.0", "authorized": True, "text": text})
    raw = tmp_path / "input.jsonl"
    raw.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    dest = tmp_path / "prepared"
    prepare([raw], dest, seed=42)
    return dest


def test_near_duplicates_are_reported_without_text(tmp_path):
    dest = fixture(tmp_path, number_of_similar=2)
    first = screen_cross_split(dest, ["CC0-1.0"])
    second = screen_cross_split(dest, ["CC0-1.0"])
    assert first == second
    assert first["findings_count"] == 1
    hit = first["findings"][0]
    assert {hit["a"]["split"], hit["b"]["split"]} == {"train", "validation"}
    assert hit["jaccard"] >= 0.8
    assert first["requires_manual_review"] is True
    assert BASE not in json.dumps(first)
    assert first["candidate_pairs_compared"] >= 1


def test_distinct_documents_do_not_raise_findings(tmp_path):
    dest = fixture(tmp_path, number_of_similar=1)
    result = screen_cross_split(dest, ["CC0-1.0"])
    assert result["findings_count"] == 0
    assert result["requires_manual_review"] is False


def test_budget_and_argument_validation_fail_closed(tmp_path):
    dest = fixture(tmp_path, number_of_similar=3)
    with pytest.raises(ValueError, match="comparison budget"):
        screen_cross_split(dest, ["CC0-1.0"], max_comparisons=1)
    with pytest.raises(ValueError, match="Document scan budget"):
        screen_cross_split(dest, ["CC0-1.0"], max_documents=2)
    with pytest.raises(ValueError, match="Finding count budget"):
        screen_cross_split(dest, ["CC0-1.0"], max_findings=1)
    for bad_threshold in (float("nan"), float("inf"), -0.2, 1.1):
        with pytest.raises(ValueError, match="threshold"):
            screen_cross_split(dest, ["CC0-1.0"], threshold=bad_threshold)
    with pytest.raises(ValueError, match="window"):
        screen_cross_split(dest, ["CC0-1.0"], window=0)


def test_integrity_and_license_enforced_before_scan(tmp_path):
    dest = fixture(tmp_path)
    with pytest.raises(ValueError, match="License outside allowlist"):
        screen_cross_split(dest, ["not-authorized"])
    path = dest / "train.txt"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        screen_cross_split(dest, ["CC0-1.0"])


def test_cli_exit_codes(tmp_path):
    dest = fixture(tmp_path)
    args = [sys.executable, "screen_corpus.py", str(dest),
            "--allow-license", "CC0-1.0"]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    assert result.returncode == 2, result.stderr
    assert json.loads(result.stdout)["findings_count"] == 1
    report_only = subprocess.run(args + ["--report-only"], capture_output=True,
                                 text=True, check=False)
    assert report_only.returncode == 0, report_only.stderr


def test_shingle_normalization_and_short_text():
    assert word_shingles("ABC  DEF", 5) == word_shingles("abc def", 5)
    assert not word_shingles("   ", 5)
    assert len(word_shingles("one two three four five six", 5)) == 2
