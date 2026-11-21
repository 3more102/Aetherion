"""Train, hash, load and use a reversible byte-level BPE tokenizer.

Requires tokenizers==0.22.1. The tokenizer is trained exclusively from the
prepared train.jsonl split; validation/test splits are never read for fitting.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
import tokenizers

SPECIAL_TOKENS = ("<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>")
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _read_train_documents(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSONL") from exc
            if not isinstance(doc, dict) or not isinstance(doc.get("text"), str):
                raise ValueError(f"{path}:{number}: expected document with text field")
            if not doc["text"]:
                raise ValueError(f"{path}:{number}: document text cannot be empty")
            yield doc["text"]


class AetherionTokenizer:
    """Small, independently verifiable wrapper around HF tokenizers."""

    def __init__(self, tokenizer: Tokenizer, metadata: dict):
        self._tokenizer = tokenizer
        self.metadata = metadata

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def sha256(self) -> str:
        return self.metadata["tokenizer_sha256"]

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "AetherionTokenizer":
        artifact_dir = Path(artifact_dir)
        metadata_path = artifact_dir / "tokenizer_manifest.json"
        tokenizer_path = artifact_dir / "tokenizer.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported tokenizer artifact schema")
        if metadata.get("library_version") != tokenizers.__version__:
            raise ValueError(
                f"Tokenizer version mismatch: artifact={metadata.get('library_version')}; "
                f"installed={tokenizers.__version__}"
            )
        if _sha256(tokenizer_path) != metadata.get("tokenizer_sha256"):
            raise ValueError("Tokenizer artifact SHA-256 does not match manifest")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        if tokenizer.get_vocab_size() != metadata.get("actual_vocab_size"):
            raise ValueError("Tokenizer vocabulary size does not match manifest")
        for name, expected_id in metadata.get("special_token_ids", {}).items():
            if tokenizer.token_to_id(name) != expected_id:
                raise ValueError(f"Special token {name} ID does not match manifest")
        if set(metadata.get("special_token_ids", {})) != set(SPECIAL_TOKENS):
            raise ValueError("Tokenizer manifest has incomplete special token IDs")
        return cls(tokenizer, metadata)


def train_tokenizer(
    data_dir: str | Path,
    output_dir: str | Path,
    vocab_size: int = 32_000,
    min_frequency: int = 2,
) -> AetherionTokenizer:
    """Train only from a checksum-verified prepared train.jsonl file.

    Vocabulary size is a *target*; a small training corpus cannot necessarily
    produce enough frequent pairs to reach it.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if vocab_size < 260:
        raise ValueError("vocab_size must be at least 260 (256 bytes + 4 special tokens)")
    if min_frequency < 1:
        raise ValueError("min_frequency must be >= 1")
    manifest_path = data_dir / "manifest.json"
    train_path = data_dir / "train.jsonl"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("schema_version") != 1:
        raise ValueError("Unsupported prepared-data manifest schema")
    expected = source_manifest.get("outputs_sha256", {}).get("train.jsonl")
    if expected is None or _sha256(train_path) != expected:
        raise ValueError("Prepared training JSONL checksum mismatch")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing tokenizer artifact directory")

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    # No normalization: a decoded sequence must preserve the input Unicode text.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
        special_tokens=list(SPECIAL_TOKENS),
        show_progress=False,
    )
    tokenizer.train_from_iterator(_read_train_documents(train_path), trainer=trainer)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "tokenizer.json"
    tokenizer.save(str(artifact_path))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "library_version": tokenizers.__version__,
        "tokenizer_sha256": _sha256(artifact_path),
        "source_train_sha256": expected,
        "source_manifest_sha256": _sha256(manifest_path),
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "min_frequency": min_frequency,
        "special_token_ids": {name: tokenizer.token_to_id(name) for name in SPECIAL_TOKENS},
        "normalizer": "none",
        "pre_tokenizer": "ByteLevel(add_prefix_space=False)",
        "decoder": "ByteLevel",
        "training_scope": "verified train.jsonl only; never validation or test",
        "limitations": [
            "No independent license or privacy verification",
            "No automated benchmark-contamination filtering",
            "Vocabulary size may be smaller than the requested target",
        ],
    }
    (output_dir / "tokenizer_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return AetherionTokenizer.load(output_dir)
