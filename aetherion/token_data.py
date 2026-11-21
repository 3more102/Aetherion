"""Token loading and provenance for demo-byte or prepared BPE experiments.

BPE uses only prepared train/validation JSONL splits. Special EOS markers
separate documents, rather than learning to predict JSONL metadata.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_document_tokens(path: str | Path, tokenizer=None) -> torch.Tensor:
    """Read prepared JSONL documents, never the raw JSONL serialization.

    Byte mode encodes UTF-8 and uses newline as an explicit boundary.
    BPE mode inserts a tokenizer EOS ID between documents. Note that training
    may still sample contexts spanning EOS; no packed-document attention mask
    is implemented in this milestone.
    """
    path = Path(path)
    if path.suffix != ".jsonl":
        if tokenizer is not None:
            raise ValueError("BPE requires prepared .jsonl split files")
        return torch.tensor(list(path.read_bytes()), dtype=torch.long)

    eos = tokenizer.token_to_id("<|eos|>") if tokenizer is not None else 10
    if eos is None:
        raise ValueError("Tokenizer must contain <|eos|>")
    tokens: list[int] = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{number}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"Expected a document with text at {path}:{number}")
            document = record["text"]
            if not document:
                continue
            if tokenizer is None:
                tokens.extend(document.encode("utf-8"))
            else:
                tokens.extend(tokenizer.encode(document, add_special_tokens=False).ids)
            tokens.append(eos)
    return torch.tensor(tokens, dtype=torch.long)


def load_training_splits(train_path: str | Path, val_path: str | Path | None, tokenizer=None):
    """Return train IDs, validation IDs, provenance; reject unsafe implicit BPE splits."""
    train_path = Path(train_path)
    if val_path is None:
        if tokenizer is not None or train_path.suffix == ".jsonl":
            raise ValueError("Prepared JSONL / BPE requires an explicit --val-data split")
        data = train_path.read_bytes()
        cutoff = int(len(data) * 0.9)
        train_bytes, val_bytes = data[:cutoff], data[cutoff:]
        print("WARNING: raw 90/10 byte holdout is only a demo; use separated documents")
        return (
            torch.tensor(list(train_bytes), dtype=torch.long),
            torch.tensor(list(val_bytes), dtype=torch.long),
            {
                "split_policy": "demo-byte-90-10",
                "train_sha256": hashlib.sha256(train_bytes).hexdigest(),
                "validation_sha256": hashlib.sha256(val_bytes).hexdigest(),
            },
        )

    val_path = Path(val_path)
    if train_path.resolve() == val_path.resolve():
        raise ValueError("Train and validation files must be different")
    train_sha, val_sha = sha256_file(train_path), sha256_file(val_path)
    if train_sha == val_sha:
        raise ValueError("Train and validation files have identical SHA-256 hashes")
    if tokenizer is not None:
        if train_path.name != "train.jsonl" or val_path.name != "validation.jsonl":
            raise ValueError("BPE requires prepared train.jsonl and validation.jsonl")
    return (
        load_document_tokens(train_path, tokenizer),
        load_document_tokens(val_path, tokenizer),
        {
            "split_policy": "explicit-files",
            "train_sha256": train_sha,
            "validation_sha256": val_sha,
        },
    )
