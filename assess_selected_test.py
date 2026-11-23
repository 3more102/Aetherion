"""M32: Select by held-out validation NLL; evaluate only the winner on test.

A trusted operator must supply a previously fixed, independent test split.
Repeating test evaluation while tuning invalidates claims of untouched testing.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

import torch

from aetherion import Aetherion, ModelConfig
from aetherion.checkpoint import file_sha256
from aetherion.evaluation import evaluate_tokens
from aetherion.tokenizer_runtime import load_verified_tokenizer
from audit_training_history import audit_history
from train import resolve_device


def _sha_is_valid(value: str | None) -> bool:
    return (type(value) is str and len(value) == 64
            and set(value) <= set("0123456789abcdef"))


def evaluate_selected_test(
    *, checkpoints: list[str] | tuple[str, ...], data_dir: str,
    tokenizer: str | None = None, review: str | None = None,
    allow_licenses: tuple[str, ...] | list[str] = (),
    demo: bool = False, stride: int | None = None,
    device: str = "cpu", require_lineage: bool = False,
    expected_test_sha256: str | None = None,
) -> dict:
    """Select the minimum validation-NLL checkpoint, then score test exactly once.

    A test split may still have been seen by the operator. This command
    cannot enforce one-time access, and does not measure assistant usefulness.
    """
    if expected_test_sha256 is not None and not _sha_is_valid(expected_test_sha256):
        raise ValueError("Expected test SHA-256 must be 64 lowercase hex characters")
    if demo and expected_test_sha256 is None:
        raise ValueError("Demo test evaluation requires --expected-test-sha256")
    directory = Path(data_dir)
    test_path = directory / "test.txt"
    if test_path.resolve() in {(directory / "train.txt").resolve(),
                               (directory / "validation.txt").resolve()}:
        raise ValueError("Test file must be a separate path from train/validation")

    # Existing M29–M31 data gates and exact held-out evaluation run first.
    # Production mode includes manifest/license/review/contamination checks.
    audit = audit_history(
        checkpoints=checkpoints, data_dir=data_dir, tokenizer=tokenizer,
        review=review, allow_licenses=allow_licenses, demo=demo,
        stride=stride, device=device, require_lineage=require_lineage,
    )
    history = audit["history"]
    selected_index = min(range(len(history)), key=lambda i: history[i]["mean_nll"])
    chosen = history[selected_index]
    chosen_path = Path(checkpoints[selected_index])
    if file_sha256(chosen_path) != chosen["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint was modified after validation")

    test_bytes = test_path.read_bytes()
    test_digest = sha256(test_bytes).hexdigest()
    dataset = audit["comparable_metadata"]["dataset"]
    if test_digest in {dataset["train_sha256"], dataset["validation_sha256"]}:
        raise ValueError("Test bytes duplicate train or validation split")
    if demo:
        if test_digest != expected_test_sha256:
            raise ValueError("Demo test data does not match operator-pinned SHA-256")
        tokens = torch.tensor(list(test_bytes), dtype=torch.long)
        token_unit = "UTF8_byte_tokens"
        expected_vocab = 256
        expected_digest_source = "operator_pinned"
    else:
        # Corpus integrity was checked before any checkpoint was opened by
        # the production auditor. Recheck the exact manifest and test bytes.
        raw_manifest = (directory / "manifest.json").read_bytes()
        if sha256(raw_manifest).hexdigest() != dataset["prepared_manifest_sha256"]:
            raise ValueError("Prepared manifest changed after validation")
        manifest = json.loads(raw_manifest)
        manifest_digest = manifest["outputs_sha256"].get("test.txt")
        if not _sha_is_valid(manifest_digest) or manifest_digest != test_digest:
            raise ValueError("Prepared test split checksum mismatch")
        if expected_test_sha256 is not None and expected_test_sha256 != test_digest:
            raise ValueError("Operator test digest disagrees with prepared manifest")
        verified, meta = load_verified_tokenizer(tokenizer)
        if (meta["tokenizer_sha256"] != dataset["tokenizer_sha256"] or
                meta["manifest_sha256"] != dataset["prepared_manifest_sha256"]):
            raise ValueError("Tokenizer provenance changed after validation")
        ids = verified.encode(test_bytes.decode("utf-8"),
                              add_special_tokens=False).ids
        tokens = torch.tensor(ids, dtype=torch.long)
        token_unit = "BPE_tokens"
        expected_vocab = verified.get_vocab_size()
        expected_digest_source = "reviewed_prepared_manifest"

    if tokens.numel() < 2:
        raise ValueError("Test split must contain at least two tokens")

    # Load exactly the chosen weights (never re-select using test scores).
    payload = torch.load(chosen_path, map_location="cpu",
                         weights_only=True, mmap=True)
    if (not isinstance(payload, dict) or payload.get("format_version") != 2 or
            payload.get("step") != chosen["step"] or
            payload.get("config") != audit["comparable_metadata"]["model_config"]):
        raise ValueError("Selected checkpoint changed its identity or configuration")
    config = ModelConfig(**payload["config"])
    if config.vocab_size != expected_vocab:
        raise ValueError("Selected checkpoint vocabulary differs from test tokenizer")
    model = Aetherion(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    del payload
    model.to(resolve_device(device))
    measured = evaluate_tokens(model, tokens, stride=stride)
    if measured["evaluated_targets"] != tokens.numel() - 1:
        raise ValueError("Test evaluation did not score every target exactly once")
    if (not math.isfinite(measured["mean_nll"])
            or measured["evaluated_targets"] < 1):
        raise ValueError("Invalid test-set metrics")
    if file_sha256(chosen_path) != chosen["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed during test evaluation")
    if file_sha256(test_path) != test_digest:
        raise ValueError("Test file changed during evaluation")

    return {
        "schema_version": 1,
        "kind": "aetherion-validation-selected-test-evidence",
        "mode": audit["mode"],
        "selection": {
            "rule": "minimum_validation_mean_next_token_nll_first_tie",
            "candidate_count": len(history),
            "selected_step": chosen["step"],
            "selected_checkpoint_sha256": chosen["checkpoint_sha256"],
            "selected_validation_mean_nll": chosen["mean_nll"],
            "selected_validation_perplexity": chosen["perplexity"],
            "validation_sha256": dataset["validation_sha256"],
            "model_config": audit["comparable_metadata"]["model_config"],
            "lineage_check": audit["lineage_check"],
            "lineage_run_id": audit["lineage_run_id"],
        },
        "test": {
            **measured,
            "data_sha256": test_digest,
            "digest_source": expected_digest_source,
            "token_unit": token_unit,
            "target_coverage": "all_targets_exactly_once",
        },
        "dataset_provenance": {
            "train_sha256": dataset["train_sha256"],
            "validation_sha256": dataset["validation_sha256"],
            "prepared_manifest_sha256": dataset["prepared_manifest_sha256"],
            "tokenizer_sha256": dataset["tokenizer_sha256"],
            "review_sha256": dataset["review_sha256"],
        },
        "limitations": (
            "The selected checkpoint was chosen on validation NLL, not test NLL. "
            "The tool cannot prevent earlier test-set exposure, repeated "
            "selection/tuning on these results or forged local checkpoint metadata. "
            "Demo evidence is synthetic/unreviewed. Token-level test loss alone "
            "does not establish assistant quality, safety, licensing or generalization "
            "to other datasets."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="Retained checkpoint, ascending training-step order; repeat 2–32 times")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--review")
    parser.add_argument("--allow-license", action="append", default=[])
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--stride", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-lineage", action="store_true")
    parser.add_argument("--expected-test-sha256",
                        help="Required for demo; optionally pin production manifest test SHA")
    parser.add_argument("--output", help="Create-only JSON evidence; existing files are not replaced")
    args = parser.parse_args()
    try:
        report = evaluate_selected_test(
            checkpoints=args.checkpoint, data_dir=args.data_dir,
            tokenizer=args.tokenizer, review=args.review,
            allow_licenses=args.allow_license, demo=args.demo,
            stride=args.stride, device=args.device,
            require_lineage=args.require_lineage,
            expected_test_sha256=args.expected_test_sha256,
        )
        result = json.dumps(report, indent=2, ensure_ascii=False,
                            allow_nan=False) + "\n"
        if args.output:
            with Path(args.output).open("x", encoding="utf-8") as destination:
                destination.write(result)
    except (ValueError, KeyError, OSError, RuntimeError, TypeError,
            OverflowError) as error:
        parser.error(str(error))
    print(result, end="")


if __name__ == "__main__":
    main()
