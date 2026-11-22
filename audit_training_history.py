"""M30: Re-evaluate distinct checkpoints and report validation progression.

Checks held-out dataset, architecture and training settings at each step.
Matching metadata is not cryptographic proof of a single training lineage.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from assess_checkpoint import assess_checkpoint


MAX_CHECKPOINTS = 32


def audit_history(
    *,
    checkpoints: list[str] | tuple[str, ...],
    data_dir: str,
    tokenizer: str | None = None,
    review: str | None = None,
    allow_licenses: list[str] | tuple[str, ...] = (),
    demo: bool = False,
    stride: int | None = None,
    device: str = "cpu",
) -> dict:
    """Measure checkpoints on identical held-out targets and compare NLL.

    A real pilot must supply independently authorized data and trust the
    local artifacts. This function never trains or modifies checkpoints.
    """
    if not 2 <= len(checkpoints) <= MAX_CHECKPOINTS:
        raise ValueError(f"Require 2 to {MAX_CHECKPOINTS} checkpoints")
    resolved = [str(Path(path).resolve()) for path in checkpoints]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Repeated checkpoint paths are not permitted")

    history = []
    reference = None
    hashes = set()
    for path in checkpoints:
        evidence = assess_checkpoint(
            checkpoint=path, data_dir=data_dir, tokenizer=tokenizer,
            review=review, allow_licenses=allow_licenses, demo=demo,
            stride=stride, device=device,
        )
        cp = evidence["checkpoint"]
        validation = evidence["validation"]
        if cp["sha256"] in hashes:
            raise ValueError("Distinct checkpoint files must have distinct contents")
        hashes.add(cp["sha256"])
        comparable = {
            "mode": evidence["mode"],
            "dataset": evidence["dataset"],
            "model_config": cp["model_config"],
            "seed": cp["seed"],
            "training_hyperparameters": cp["training_hyperparameters"],
            "parameter_count": cp["parameter_count"],
            "validation_context": validation["context_size"],
            "validation_stride": validation["stride"],
            "evaluated_targets": validation["evaluated_targets"],
            "token_unit": validation["token_unit"],
        }
        if reference is None:
            reference = comparable
        elif comparable != reference:
            raise ValueError("Checkpoint history differs in data, architecture, seed, "
                             "hyperparameters, evaluation context or target accounting")

        step = cp["step"]
        if history and step <= history[-1]["step"]:
            raise ValueError("Checkpoint steps must be strictly increasing")
        nll = validation["mean_nll"]
        ppl = validation["perplexity"]
        targets = validation["evaluated_targets"]
        if (not isinstance(nll, (int, float)) or not math.isfinite(nll)
                or nll < 0 or not isinstance(ppl, (int, float))
                or not math.isfinite(ppl) or ppl < 1
                or not math.isclose(ppl, math.exp(nll), rel_tol=1e-6)
                or not isinstance(targets, int) or targets < 1
                or not math.isclose(validation["total_nll"], nll * targets, rel_tol=1e-7, abs_tol=1e-6)):
            raise ValueError("Checkpoint validation metrics are inconsistent")
        previous = history[-1]["mean_nll"] if history else None
        history.append({
            "step": step,
            "checkpoint_sha256": cp["sha256"],
            "evaluated_targets": targets,
            "mean_nll": nll,
            "perplexity": ppl,
            "nll_delta_from_previous": nll - previous if previous is not None else None,
            "nll_delta_from_first": nll - history[0]["mean_nll"] if history else 0.0,
        })

    best = min(history, key=lambda entry: entry["mean_nll"])
    first, last = history[0], history[-1]
    return {
        "schema_version": 1,
        "kind": "aetherion-validated-checkpoint-history",
        "mode": reference["mode"],
        "comparable_metadata": reference,
        "history": history,
        "summary": {
            "checkpoint_count": len(history),
            "first_step": first["step"],
            "last_step": last["step"],
            "best_validation_step": best["step"],
            "best_validation_nll": best["mean_nll"],
            "final_nll_delta": last["mean_nll"] - first["mean_nll"],
            "final_perplexity_ratio": last["perplexity"] / first["perplexity"],
            "final_validation_improved": last["mean_nll"] < first["mean_nll"],
        },
        "limitations": (
            "Each saved checkpoint was independently evaluated on the same validation split. "
            "Matching settings and increasing steps do not prove an uninterrupted training "
            "lineage. Teacher-forced validation metrics alone do not establish real-world "
            "assistant quality, test-set generalization, rights or dataset safety."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True,
                   help="Trusted checkpoint path in increasing training-step order; repeat 2-32 times")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--tokenizer")
    p.add_argument("--review")
    p.add_argument("--allow-license", action="append", default=[])
    p.add_argument("--demo", action="store_true")
    p.add_argument("--stride", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", help="Create-only JSON output; never overwrite evidence")
    p.add_argument("--require-improvement", action="store_true",
                   help="Return exit status 2 if last NLL is not below first NLL")
    args = p.parse_args()
    try:
        result = audit_history(
            checkpoints=args.checkpoint, data_dir=args.data_dir,
            tokenizer=args.tokenizer, review=args.review,
            allow_licenses=args.allow_license, demo=args.demo,
            stride=args.stride, device=args.device,
        )
        payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            with Path(args.output).open("x", encoding="utf-8") as out:
                out.write(payload)
    except (ValueError, KeyError, OSError, RuntimeError, TypeError, OverflowError) as exc:
        p.error(str(exc))
    print(payload, end="")
    if args.require_improvement and not result["summary"]["final_validation_improved"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
