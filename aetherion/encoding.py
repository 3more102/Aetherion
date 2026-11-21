"""Load auditable byte and BPE token streams for the Micro trainer.

BPE documents are encoded independently and separated by an EOS token.
The tokenizer must have been trained using only train.jsonl.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path: str | Path) -> str:
    """Hash file bytes, not a path or mutable timestamp."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bpe(path: str | Path):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    if tokenizer.token_to_id("<|eos|>") is None:
        raise ValueError("BPE tokenizer must contain <|eos|>")
    return tokenizer


def encode_documents(path: str | Path, tokenizer) -> torch.Tensor:
    """Encode an explicit, pre-split JSONL stream without training on it."""
    eos = tokenizer.token_to_id("<|eos|>")
    if eos is None:
        raise ValueError("BPE tokenizer must contain <|eos|>")
    token_ids: list[int] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {lineno} of {path}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"Missing string 'text' at line {lineno} of {path}")
            if record["text"]:
                token_ids.extend(tokenizer.encode(record["text"], add_special_tokens=False).ids)
                token_ids.append(eos)
    return torch.tensor(token_ids, dtype=torch.long)


def encode_bytes(path: str | Path) -> torch.Tensor:
    return torch.tensor(list(Path(path).read_bytes()), dtype=torch.long)
