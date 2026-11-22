"""M17: Compare two trusted training checkpoints on one verified held-out split.

Production mode requires prepared, audited, human-attested data and a verified
32k BPE tokenizer. --demo explicitly permits small byte-token CPU fixtures.
Neither mode proves capabilities beyond measured next-token validation loss.
"""
import argparse
from contextlib import redirect_stdout
from hashlib import sha256
import json
from pathlib import Path
import sys

import torch

from aetherion import Aetherion, ModelConfig
from aetherion.corpus_gate import audit_corpus, require_review
from aetherion.evaluation import evaluate_tokens
from aetherion.presets import get_preset
from train import load_bpe_data, load_data, resolve_device


def digest_file(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compare(*, before, after, data_dir, tokenizer=None, review=None,
            allow_licenses=(), demo=False, stride=None, device="cpu"):
    directory = Path(data_dir)
    train_path, val_path = directory / "train.txt", directory / "validation.txt"
    review_sha = None
    if demo:
        if tokenizer or review or allow_licenses:
            raise ValueError("--demo requires byte tokens and no approval artifacts")
        with redirect_stdout(sys.stderr):
            _, val, fingerprint = load_data(str(train_path), str(val_path))
        vocab = 256
        mode = "synthetic_or_unreviewed_byte_demo"
    else:
        if not tokenizer or not review or not allow_licenses:
            raise ValueError("Production requires BPE tokenizer, review and license allowlist")
        corpus_report = audit_corpus(directory, list(allow_licenses))
        require_review(corpus_report, review)
        review_sha = digest_file(review)
        _, val, fingerprint, vocab = load_bpe_data(
            str(train_path), str(val_path), tokenizer
        )
        if vocab != 32000:
            raise ValueError("125M requires exactly 32000 verified BPE entries")
        mode = "attested_125m_pilot"
    selected = resolve_device(device)
    # Reject duplicate paths: evidence requires two distinct checkpoint files.
    if Path(before).resolve() == Path(after).resolve():
        raise ValueError("Before and after checkpoints must be different files")

    def measured(path, expected_metadata):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 2:
            raise ValueError("Require a trusted format-v2 training checkpoint")
        required = ("config", "data", "train_hparams", "seed", "step", "state_dict")
        if any(key not in checkpoint for key in required):
            raise ValueError("Checkpoint missing training metadata")
        if not isinstance(checkpoint["data"], dict) or checkpoint["data"] != fingerprint:
            raise ValueError("Checkpoint dataset/tokenizer fingerprints differ")
        if not isinstance(checkpoint["train_hparams"], dict):
            raise ValueError("Invalid checkpoint training hyperparameters")
        if checkpoint["data"].get("explicit_validation_file") is not True:
            raise ValueError("Explicit validation split required")
        step = checkpoint["step"]
        if type(step) is not int or step < 1:
            raise ValueError("Invalid checkpoint step")
        cfg = ModelConfig(**checkpoint["config"])
        if cfg.vocab_size != vocab:
            raise ValueError("Checkpoint vocabulary differs from tokenizer")
        if not demo:
            preset = get_preset("125m")
            if not (cfg.d_model == preset.d_model and cfg.n_layers == preset.n_layers
                    and cfg.n_heads == preset.n_heads and cfg.vocab_size == preset.vocab_size
                    and cfg.dropout == preset.dropout and 2 <= cfg.block_size <= preset.block_size):
                raise ValueError("Expected Aetherion-125M architecture with valid context")
        metadata = {key: checkpoint[key] for key in
                    ("config", "data", "train_hparams", "seed")}
        if expected_metadata is not None and metadata != expected_metadata:
            raise ValueError("Checkpoints differ in config, data, seed or hyperparameters")
        model = Aetherion(cfg)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        del checkpoint
        model.to(selected)
        metrics = evaluate_tokens(model, val, stride=stride)
        if metrics["evaluated_targets"] != val.numel() - 1:
            raise ValueError("Not all held-out next-token targets were evaluated")
        del model
        return metadata, {"checkpoint_step": step, "checkpoint_sha256": digest_file(path),
                          **metrics}

    meta, first = measured(before, None)
    _, final = measured(after, meta)
    if final["checkpoint_step"] <= first["checkpoint_step"]:
        raise ValueError("After checkpoint step must exceed the before checkpoint step")
    return {
        "schema_version": 1,
        "mode": mode,
        "dataset": {
            "train_sha256": fingerprint["train_sha256"],
            "validation_sha256": fingerprint["validation_sha256"],
            "tokenizer_sha256": fingerprint.get("tokenizer_sha256"),
            "prepared_manifest_sha256": fingerprint.get("prepared_manifest_sha256"),
            "review_sha256": review_sha,
        },
        "model_config": meta["config"],
        "seed": meta["seed"],
        "train_hparams": meta["train_hparams"],
        "before": first,
        "after": final,
        "mean_nll_delta": final["mean_nll"] - first["mean_nll"],
        "perplexity_ratio": final["perplexity"] / first["perplexity"],
        "validation_nll_improved": final["mean_nll"] < first["mean_nll"],
        "limitations": (
            "Exact same-split validation comparison only; not a test-set evaluation, "
            "proof of generalization, verified data rights, or assistant performance."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, help="Trusted earlier v2 checkpoint")
    parser.add_argument("--after", required=True, help="Trusted later v2 checkpoint")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--review")
    parser.add_argument("--allow-license", action="append", default=[])
    parser.add_argument("--demo", action="store_true", help="Allow unreviewed byte-token smoke fixture")
    parser.add_argument("--stride", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", help="Create report; refuse overwrite")
    parser.add_argument("--require-improvement", action="store_true")
    args = parser.parse_args()
    try:
        report = compare(
            before=args.before, after=args.after, data_dir=args.data_dir,
            tokenizer=args.tokenizer, review=args.review,
            allow_licenses=args.allow_license, demo=args.demo,
            stride=args.stride, device=args.device
        )
        payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            with Path(args.output).open("x", encoding="utf-8") as stream:
                stream.write(payload)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(payload, end="")
    if args.require_improvement and not report["validation_nll_improved"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
