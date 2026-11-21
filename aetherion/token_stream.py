"""Read training tokens without mixing JSON metadata or held-out splits.

The byte-level legacy path is kept for Aetherion-Micro checkpoints. The BPE
path expects a Hugging Face `tokenizers` artifact produced on train.jsonl.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def load_token_stream(path: str | Path, tokenizer=None) -> list[int]:
    """Load bytes or BPE IDs, adding an EOS token after each JSONL document.

    JSONL must only be used with the BPE tokenizer so its metadata never
    accidentally becomes training text. The caller isolates train and val.
    """
    path = Path(path)
    if tokenizer is None:
        if path.suffix.lower() == ".jsonl":
            raise ValueError("JSONL requires --tokenizer; byte mode accepts plain text only")
        ids = list(path.read_bytes())
    elif path.suffix.lower() == ".jsonl":
        eos = tokenizer.token_to_id("<|eos|>")
        if eos is None:
            raise ValueError("Tokenizer must define <|eos|> for document boundaries")
        ids = []
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{number}") from exc
                if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                    raise ValueError(f"Missing text at {path}:{number}")
                if not record["text"].strip():
                    raise ValueError(f"Empty text at {path}:{number}")
                ids.extend(tokenizer.encode(record["text"], add_special_tokens=False).ids)
                ids.append(eos)
    else:
        ids = tokenizer.encode(path.read_text(encoding="utf-8"),
                               add_special_tokens=False).ids
    if not ids:
        raise ValueError(f"No tokens found in {path}")
    return ids
