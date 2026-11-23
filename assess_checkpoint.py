"""M29: Auditable per-checkpoint held-out next-token evidence.

Production mode requires the same corpus/attestation/lexical gates as the
125M pilot. Demo mode is explicitly unreviewed and cannot validate general
assistant performance. No training, checkpoint rewriting, or model release.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

import torch

from aetherion import Aetherion, ModelConfig
from aetherion.checkpoint import validate_lineage
from aetherion.contamination import require_no_cross_split_findings
from aetherion.corpus_gate import audit_corpus, require_review
from aetherion.evaluation import evaluate_tokens
from aetherion.presets import get_preset
from train import load_bpe_data, load_data, resolve_device


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assess_checkpoint(
    *,
    checkpoint: str | Path,
    data_dir: str | Path,
    tokenizer: str | None = None,
    review: str | None = None,
    allow_licenses: tuple[str, ...] | list[str] = (),
    demo: bool = False,
    stride: int | None = None,
    device: str = "cpu",
) -> dict:
    """Compute exact validation NLL on the original checkpoint data.

    This function does not read any checkpoint until the requested data
    review, corpus integrity and optional production contamination screen
    have completed.
    """
    directory = Path(data_dir)
    train_file, val_file = directory / "train.txt", directory / "validation.txt"
    review_digest = None
    lexical = None
    if demo:
        if tokenizer or review or allow_licenses:
            raise ValueError("--demo requires byte tokens and no production approval artifacts")
        with redirect_stdout(sys.stderr):
            _, validation, fingerprint = load_data(str(train_file), str(val_file))
        vocab_size = 256
        mode = "synthetic_or_unreviewed_byte_demo"
    else:
        if not tokenizer or not review or not allow_licenses:
            raise ValueError("Production requires reviewed prepared data, tokenizer, review and license allowlist")
        audit = audit_corpus(directory, list(allow_licenses))
        require_review(audit, review)
        lexical = require_no_cross_split_findings(directory, list(allow_licenses))
        review_digest = file_sha256(review)
        _, validation, fingerprint, vocab_size = load_bpe_data(
            str(train_file), str(val_file), tokenizer
        )
        if vocab_size != 32000:
            raise ValueError("Production requires exactly 32000 BPE entries")
        mode = "attested_125m_validation"

    if fingerprint.get("explicit_validation_file") is not True:
        raise ValueError("Evaluation requires a separately supplied validation file")
    chosen_device = resolve_device(device)
    # Verify the original artifact did not change while it was evaluated.
    before_sha = file_sha256(checkpoint)
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(artifact, dict) or artifact.get("format_version") != 2:
        raise ValueError("Require a trusted format-v2 training checkpoint")
    required = ("config", "data", "train_hparams", "seed", "step", "state_dict")
    if any(key not in artifact for key in required):
        raise ValueError("Checkpoint missing required provenance or training fields")
    if (not isinstance(artifact["data"], dict) or artifact["data"] != fingerprint or
            artifact["data"].get("explicit_validation_file") is not True):
        raise ValueError("Checkpoint dataset/tokenizer fingerprints differ")
    if (not isinstance(artifact["train_hparams"], dict) or
            type(artifact["seed"]) is not int or
            type(artifact["step"]) is not int or artifact["step"] < 1 or
            not isinstance(artifact["config"], dict) or
            not isinstance(artifact["state_dict"], dict)):
        raise ValueError("Checkpoint training metadata has invalid types")
    lineage = artifact.get("lineage")
    if lineage is not None:
        lineage = validate_lineage(lineage, step=artifact["step"])
    cfg = ModelConfig(**artifact["config"])
    if cfg.vocab_size != vocab_size:
        raise ValueError("Checkpoint vocabulary differs from validated tokenizer")
    if not demo:
        preset = get_preset("125m")
        if not (
            cfg.d_model == preset.d_model and
            cfg.n_layers == preset.n_layers and
            cfg.n_heads == preset.n_heads and
            cfg.vocab_size == preset.vocab_size and
            cfg.dropout == preset.dropout and
            2 <= cfg.block_size <= preset.block_size
        ):
            raise ValueError("Production checkpoint must match Aetherion-125M architecture")

    model = Aetherion(cfg)
    model.load_state_dict(artifact["state_dict"], strict=True)
    step, seed, hyperparams = artifact["step"], artifact["seed"], artifact["train_hparams"]
    del artifact
    model.to(chosen_device)
    measured = evaluate_tokens(model, validation, stride=stride)
    if measured["evaluated_targets"] != validation.numel() - 1:
        raise ValueError("Validation did not cover every next-token target")
    if file_sha256(checkpoint) != before_sha:
        raise ValueError("Checkpoint changed during evaluation; evidence rejected")
    return {
        "schema_version": 1,
        "kind": "aetherion-single-checkpoint-heldout-validation",
        "mode": mode,
        "checkpoint": {
            "sha256": before_sha,
            "step": step,
            "format_version": 2,
            "seed": seed,
            "model_config": asdict(cfg),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "training_hyperparameters": hyperparams,
            **({"lineage": lineage} if lineage is not None else {}),
        },
        "dataset": {
            "train_sha256": fingerprint["train_sha256"],
            "validation_sha256": fingerprint["validation_sha256"],
            "tokenizer_type": fingerprint.get("tokenizer_type", "bytes"),
            "tokenizer_sha256": fingerprint.get("tokenizer_sha256"),
            "prepared_manifest_sha256": fingerprint.get("prepared_manifest_sha256"),
            "review_sha256": review_digest,
            "lexical_screen_manifest_sha256": lexical["manifest_sha256"] if lexical else None,
            "lexical_screen_candidate_pairs": lexical["candidate_pairs_compared"] if lexical else None,
            "lexical_screen_findings_count": lexical["findings_count"] if lexical else None,
        },
        "validation": {
            **measured,
            "token_unit": "BPE_tokens" if not demo else "UTF8_byte_tokens",
            "target_coverage": "all_targets_exactly_once",
        },
        "limitations": (
            "Only teacher-forced next-token validation NLL/perplexity on one split. "
            "Not test-set evaluation, language understanding, safety, factuality, "
            "assistant usefulness, independent legal/privacy review or evidence of "
            "real 125M training merely because this tool exists."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--review")
    parser.add_argument("--allow-license", action="append", default=[])
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--stride", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", help="Create an evidence JSON file; refuses overwrite")
    args = parser.parse_args()
    try:
        report = assess_checkpoint(
            checkpoint=args.checkpoint, data_dir=args.data_dir,
            tokenizer=args.tokenizer, review=args.review,
            allow_licenses=args.allow_license, demo=args.demo,
            stride=args.stride, device=args.device,
        )
        payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            with Path(args.output).open("x", encoding="utf-8") as stream:
                stream.write(payload)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))
    print(payload, end="")


if __name__ == "__main__":
    main()
