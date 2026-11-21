"""Train a byte-level BPE tokenizer from a prepared Aetherion train split.

This module only creates tokenizer artifacts. The Micro trainer still uses raw
bytes; token-ID training and inference will be integrated in a separate step.
A manifest proves input file integrity, not legal authorization or data safety.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

SPECIAL_TOKENS = ("<|unk|>", "<|bos|>", "<|eos|>")


def _hash(data: bytes) -> str:
    return sha256(data).hexdigest()


def verify_training_split(path: Path) -> tuple[str, str]:
    """Require a prepared train.jsonl whose bytes match the split manifest."""
    if path.name != "train.jsonl":
        raise ValueError("Tokenizer fitting requires prepared train.jsonl (never validation/test)")
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Missing prepared dataset manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid dataset manifest.json") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported dataset manifest schema_version")
    expected = manifest.get("outputs_sha256", {}).get("train.jsonl")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Dataset manifest has no train.jsonl SHA-256")
    actual = _hash(path.read_bytes())
    if actual != expected:
        raise ValueError("train.jsonl SHA-256 does not match dataset manifest")
    return actual, _hash(manifest_bytes)


def train_tokenizer(
    train_jsonl: str | Path,
    output_json: str | Path,
    *,
    vocab_size: int = 32000,
    min_frequency: int = 2,
) -> dict:
    """Fit only on the verified training split and save versioned artifacts.

    Reproducibility assumes identical input bytes, parameters and the recorded
    Hugging Face tokenizers version. No validation/test text is read.
    """
    path = Path(train_jsonl)
    output = Path(output_json)
    if vocab_size < 259:
        raise ValueError("vocab_size must be at least 259 (256 bytes + 3 special tokens)")
    if min_frequency < 1:
        raise ValueError("min_frequency must be >= 1")
    if output.suffix.lower() != ".json":
        raise ValueError("Tokenizer artifact filename must end with .json")
    metadata_path = output.with_suffix(".meta.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("Refusing to overwrite tokenizer artifacts")
    train_hash, manifest_hash = verify_training_split(path)

    # Import lazily: the raw-byte Micro path does not require tokenizers.
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    count = 0

    def documents():
        nonlocal count
        with path.open("r", encoding="utf-8") as stream:
            for lineno, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid train.jsonl at line {lineno}") from exc
                if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                    raise ValueError(f"Invalid text in train.jsonl at line {lineno}")
                if not record["text"]:
                    raise ValueError(f"Empty text in train.jsonl at line {lineno}")
                count += 1
                yield record["text"]

    tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS[0]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=list(SPECIAL_TOKENS),
        show_progress=False,
    )
    tokenizer.train_from_iterator(documents(), trainer=trainer)
    if count == 0:
        raise ValueError("Prepared training split contains no documents")
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output), pretty=False)

    metadata = {
        "schema_version": 1,
        "algorithm": "ByteLevel BPE",
        "tokenizers_version": version("tokenizers"),
        "train_sha256": train_hash,
        "manifest_sha256": manifest_hash,
        "tokenizer_sha256": _hash(output.read_bytes()),
        "vocab_size": tokenizer.get_vocab_size(),
        "requested_vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "train_documents": count,
        "special_tokens": list(SPECIAL_TOKENS),
        "training_policy": "verified prepared train.jsonl only",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {"tokenizer_path": str(output), "metadata_path": str(metadata_path), **metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Aetherion byte-level BPE tokenizer")
    parser.add_argument("--train", required=True, help="Prepared train.jsonl")
    parser.add_argument("--output", default="artifacts/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()
    metadata = train_tokenizer(args.train, args.output,
                               vocab_size=args.vocab_size, min_frequency=args.min_frequency)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
