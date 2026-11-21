"""Load auditable BPE training splits without mixing held-out documents."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl_token_ids(path: Path, tokenizer, eos_id: int) -> torch.Tensor:
    token_ids: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"Invalid document text in {path} at line {line_number}")
            if record["text"]:
                # Explicitly insert a boundary after each document.
                token_ids.extend(tokenizer.encode(record["text"], add_special_tokens=False).ids)
                token_ids.append(eos_id)
    if not token_ids:
        raise ValueError(f"No non-empty tokenized documents in {path}")
    return torch.tensor(token_ids, dtype=torch.long)


def load_bpe_splits(
    train_path: str | Path, validation_path: str | Path, tokenizer_path: str | Path
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Require prepared, checksum-matched splits and a tokenizer fitted on train only.

    This verifies file provenance and split wiring, not licenses, content
    contamination, near-duplicates, privacy or training-data suitability.
    """
    from tokenizers import Tokenizer

    train_path = Path(train_path).resolve()
    validation_path = Path(validation_path).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()

    if (train_path.name, validation_path.name) != ("train.jsonl", "validation.jsonl"):
        raise ValueError("BPE mode requires prepared train.jsonl and validation.jsonl")
    if train_path.parent != validation_path.parent:
        raise ValueError("Train and validation JSONL must share a prepared output directory")
    if train_path == validation_path:
        raise ValueError("Training and validation must use separate files")

    manifest_path = train_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_hash = sha256_file(train_path)
    val_hash = sha256_file(validation_path)
    if train_hash == val_hash:
        raise ValueError("Train and validation data have the same checksum")
    for name, actual in (("train", train_hash), ("validation", val_hash)):
        expected = manifest.get("splits", {}).get(name, {}).get("sha256")
        if not expected or actual != expected:
            raise ValueError(f"{name} checksum does not match preparation manifest")

    metadata_path = tokenizer_path.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tokenizer_hash = sha256_file(tokenizer_path)
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported tokenizer metadata schema")
    if metadata.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError("Tokenizer checksum does not match its metadata")
    if metadata.get("train_sha256") != train_hash:
        raise ValueError("Tokenizer was not recorded as fitted on this training split")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocabulary_size = tokenizer.get_vocab_size()
    if metadata.get("vocab_size") != vocabulary_size:
        raise ValueError("Tokenizer vocabulary size does not match metadata")
    eos_id = tokenizer.token_to_id("<|eos|>")
    if eos_id is None:
        raise ValueError("Tokenizer must include <|eos|>")

    train_ids = _load_jsonl_token_ids(train_path, tokenizer, eos_id)
    validation_ids = _load_jsonl_token_ids(validation_path, tokenizer, eos_id)
    if max(train_ids.max().item(), validation_ids.max().item()) >= vocabulary_size:
        raise ValueError("Token ID exceeds tokenizer vocabulary size")

    return train_ids, validation_ids, {
        "tokenizer": {
            "mode": "bpe",
            "sha256": tokenizer_hash,
            "vocab_size": vocabulary_size,
        },
        "dataset": {
            "split_policy": "prepared-document-grouped",
            "train_sha256": train_hash,
            "validation_sha256": val_hash,
            "manifest_sha256": sha256_file(manifest_path),
        },
    }
