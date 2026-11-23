"""M33: Offline integrity/consistency audit for M32 v2 selected-test evidence.

No model weights are deserialized and no dataset is evaluated. The auditor
checks recorded claims and on-disk hashes, not the truth of reported scores.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

MAX_REPORT_BYTES = 1024 * 1024
MAX_CANDIDATES = 32


def _object_without_duplicates(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Duplicate JSON key: {key}")
        output[key] = value
    return output


def _bad_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def _sha(value) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= set("0123456789abcdef")


def _digest_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def _exact_sha(value, label: str):
    _require(_sha(value), f"Invalid {label} SHA-256")
    return value


def _finite_float(value, label: str, *, minimum: float = 0.0) -> float:
    _require(type(value) in (float, int) and math.isfinite(value) and value >= minimum,
             f"Invalid {label}")
    return float(value)


def _positive_integer(value, label: str) -> int:
    _require(type(value) is int and value > 0, f"Invalid {label}")
    return value


def _metrics(row: dict, label: str, *, total: bool = False):
    _require(type(row) is dict, f"Invalid {label} entry")
    nll = _finite_float(row.get("mean_nll"), f"{label} mean_nll")
    perplexity = _finite_float(row.get("perplexity"), f"{label} perplexity", minimum=1.0)
    _require(nll < 700.0 and math.isclose(math.log(perplexity), nll, rel_tol=1e-8, abs_tol=1e-8),
             f"Inconsistent {label} perplexity / mean_nll")
    targets = _positive_integer(row.get("evaluated_targets"), f"{label} evaluated_targets")
    if total:
        total_nll = _finite_float(row.get("total_nll"), f"{label} total_nll")
        _require(math.isclose(total_nll, nll * targets, rel_tol=1e-8, abs_tol=1e-6),
                 f"Inconsistent {label} total_nll")
    return nll, targets


def verify_selected_test_report(*, report_path: str | Path,
                                data_dir: str | Path,
                                checkpoints: list[str | Path] | tuple[str | Path, ...]) -> dict:
    """Check local report/inputs for integrity and internal selection consistency.

    This never re-scores tokens or authenticates who produced the evidence.
    """
    report_file = Path(report_path)
    _require(report_file.stat().st_size <= MAX_REPORT_BYTES, "Evidence JSON is too large")
    report_bytes = report_file.read_bytes()
    report = json.loads(report_bytes.decode("utf-8"),
                        object_pairs_hook=_object_without_duplicates,
                        parse_constant=_bad_constant)
    _require(type(report) is dict and report.get("schema_version") == 2 and
             report.get("kind") == "aetherion-validation-selected-test-evidence",
             "Expected M32 schema-v2 selected-test evidence")
    selection = report.get("selection")
    _require(type(selection) is dict, "Missing selection record")
    candidates = selection.get("candidates")
    _require(type(candidates) is list and 2 <= len(candidates) <= MAX_CANDIDATES,
             "Require 2–32 recorded validation candidates")
    _require(type(selection.get("candidate_count")) is int and
             selection["candidate_count"] == len(candidates) and
             len(checkpoints) == len(candidates), "Candidate count mismatch")
    _require(selection.get("rule") == "minimum_validation_mean_next_token_nll_first_tie",
             "Unexpected checkpoint selection rule")
    resolved = [Path(path).resolve() for path in checkpoints]
    _require(len(set(resolved)) == len(resolved), "Repeated checkpoint paths")
    previous_step = -1
    candidate_digests = set()
    expected_targets = None
    for index, (entry, path) in enumerate(zip(candidates, resolved), 1):
        _require(type(entry) is dict, "Invalid validation candidate")
        step = _positive_integer(entry.get("step"), "candidate step")
        _require(step > previous_step, "Candidate steps are not increasing")
        previous_step = step
        digest = _exact_sha(entry.get("checkpoint_sha256"), "candidate checkpoint")
        _require(digest not in candidate_digests, "Duplicate checkpoint digests")
        candidate_digests.add(digest)
        _require(_digest_file(path) == digest, f"Checkpoint {index} checksum mismatch")
        _, targets = _metrics(entry, f"candidate {index}")
        if expected_targets is None:
            expected_targets = targets
        _require(targets == expected_targets, "Candidate target coverage mismatch")

    chosen = min(candidates, key=lambda entry: entry["mean_nll"])
    _require(selection.get("selected_step") == chosen["step"] and
             selection.get("selected_checkpoint_sha256") == chosen["checkpoint_sha256"] and
             selection.get("selected_validation_mean_nll") == chosen["mean_nll"] and
             selection.get("selected_validation_perplexity") == chosen["perplexity"],
             "Selected checkpoint does not match first validation-NLL minimum")
    _require(type(selection.get("model_config")) is dict,
             "Missing model configuration")
    _require(selection.get("lineage_check") in ("validated", "not_requested"),
             "Invalid lineage status")
    _require(selection.get("lineage_check") == "validated" or
             selection.get("lineage_run_id") is None,
             "Unvalidated lineage run ID")
    provenance = report.get("dataset_provenance")
    _require(type(provenance) is dict, "Missing dataset provenance")
    directory = Path(data_dir)
    source_digest = {}
    test = report.get("test")
    _require(type(test) is dict, "Missing test result")
    for split in ("train", "validation", "test"):
        expected = (provenance.get(f"{split}_sha256") if split != "test"
                    else test.get("data_sha256"))
        expected = _exact_sha(expected, f"{split} data")
        _require(_digest_file(directory / f"{split}.txt") == expected,
                 f"{split} data checksum mismatch")
        source_digest[split] = expected
    _require(len(set(source_digest.values())) == 3, "Split files have identical contents")
    _require(selection.get("validation_sha256") == source_digest["validation"],
             "Selection validation digest mismatch")
    _, targets = _metrics(test, "test", total=True)
    _require(test.get("target_coverage") == "all_targets_exactly_once",
             "Missing exact-target-coverage claim")
    context = _positive_integer(test.get("context_size"), "test context")
    stride = _positive_integer(test.get("stride"), "test stride")
    _require(stride <= context, "Test stride exceeds model context")
    _require(test.get("token_unit") in ("UTF8_byte_tokens", "BPE_tokens"),
             "Invalid test token unit")
    if report.get("mode") == "synthetic_or_unreviewed_byte_demo":
        _require(test.get("digest_source") == "operator_pinned" and
                 test["token_unit"] == "UTF8_byte_tokens", "Invalid demo provenance")
        _require(targets == (directory / "test.txt").stat().st_size - 1,
                 "Demo test target count differs from test bytes")
    else:
        _require(test.get("digest_source") == "reviewed_prepared_manifest" and
                 test["token_unit"] == "BPE_tokens", "Invalid production provenance")
        manifest_digest = _exact_sha(provenance.get("prepared_manifest_sha256"),
                                     "prepared manifest")
        _require(_digest_file(directory / "manifest.json") == manifest_digest,
                 "Prepared manifest checksum mismatch")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"),
                              object_pairs_hook=_object_without_duplicates,
                              parse_constant=_bad_constant)
        _require(type(manifest) is dict and
                 type(manifest.get("outputs_sha256")) is dict and
                 manifest["outputs_sha256"].get("test.txt") == source_digest["test"],
                 "Prepared test manifest mismatch")
        _exact_sha(provenance.get("tokenizer_sha256"), "tokenizer")
        _exact_sha(provenance.get("review_sha256"), "review")
    return {
        "schema_version": 1,
        "kind": "aetherion-offline-selected-test-integrity-audit",
        "result": "hashes_and_report_consistency_passed",
        "report_sha256": sha256(report_bytes).hexdigest(),
        "candidate_count": len(candidates),
        "selected_step": chosen["step"],
        "selected_checkpoint_sha256": chosen["checkpoint_sha256"],
        "test_sha256": source_digest["test"],
        "test_evaluated_targets_claimed": targets,
        "limitations": (
            "Only file hashes and internal report consistency were checked. "
            "No checkpoint was opened or re-evaluated; NLL, provenance, human review, "
            "lineage and one-time test-set use were not independently verified. "
            "Unsigned reports and artifacts may be forged together."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True, help="Existing M32 schema-v2 report")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", action="append", required=True,
                   help="Retained candidate checkpoint in ascending step order")
    p.add_argument("--output", help="Create-only audit JSON; refuses overwrite")
    args = p.parse_args()
    try:
        result = verify_selected_test_report(report_path=args.report,
                                             data_dir=args.data_dir,
                                             checkpoints=args.checkpoint)
        output = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            with Path(args.output).open("x", encoding="utf-8") as target:
                target.write(output)
    except (ValueError, OSError, KeyError, TypeError, UnicodeError, OverflowError) as exc:
        p.error(str(exc))
    print(output, end="")


if __name__ == "__main__":
    main()
