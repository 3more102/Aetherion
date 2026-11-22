"""Load BPE tokenizer artifacts and validate their provenance for training/inference.

Compatible with aetherion.tokenizer.train_tokenizer() (metadata *.meta.json).
"""
from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path


def load_verified_tokenizer(path: str | Path):
    """Return (HF tokenizer, metadata); reject incompatible or tampered artifacts."""
    from tokenizers import Tokenizer

    path = Path(path)
    metadata_path = path.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("Unsupported tokenizer metadata schema")
    actual_sha = sha256(path.read_bytes()).hexdigest()
    if metadata.get("tokenizer_sha256") != actual_sha:
        raise ValueError("Tokenizer SHA-256 does not match metadata")
    if metadata.get("tokenizers_version") != version("tokenizers"):
        raise ValueError("Installed tokenizers version differs from artifact metadata")
    tokenizer = Tokenizer.from_file(str(path))
    if tokenizer.get_vocab_size() != metadata.get("vocab_size"):
        raise ValueError("Tokenizer vocabulary differs from artifact metadata")
    special_tokens = metadata.get("special_tokens")
    if (not isinstance(special_tokens, list) or not special_tokens
            or any(tokenizer.token_to_id(t) is None for t in special_tokens)):
        raise ValueError("Tokenizer is missing required special tokens")
    return tokenizer, metadata
