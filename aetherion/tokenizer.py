"""Train an experimental byte-level BPE tokenizer from the training split only.

Requires the Hugging Face 'tokenizers' package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SPECIAL_TOKENS = ["<|unk|>", "<|bos|>", "<|eos|>"]


def train_tokenizer(
    train_jsonl: str | Path,
    output_json: str | Path,
    *,
    vocab_size: int = 32000,
    min_frequency: int = 2,
) -> dict:
    """Train BPE with full byte alphabet, then save a reproducible JSON artifact.

    A prepared train.jsonl is required so held-out validation/test text does
    not influence learned merge rules. This is a tokenizer experiment, not a
    claim of a complete pretraining dataset or an integrated LM pipeline.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    path = Path(train_jsonl)
    if path.name != "train.jsonl":
        raise ValueError("Tokenizer must be fitted on prepared train.jsonl only")
    if vocab_size < 259:
        raise ValueError("Vocabulary size must be at least 259 (256 bytes + 3 specials)")
    if min_frequency < 1:
        raise ValueError("min_frequency must be >= 1")

    def documents():
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get("text")
                if not isinstance(text, str):
                    raise ValueError(f"Missing text at line {line_number}")
                if text:
                    yield text

    tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS[0]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )
    tokenizer.train_from_iterator(documents(), trainer=trainer)
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    return {
        "tokenizer_path": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "vocab_size": tokenizer.get_vocab_size(),
        "train_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "special_tokens": SPECIAL_TOKENS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Aetherion byte-level BPE")
    parser.add_argument("--train", required=True, help="Prepared train.jsonl")
    parser.add_argument("--output", default="artifacts/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()
    details = train_tokenizer(args.train, args.output, vocab_size=args.vocab_size,
                              min_frequency=args.min_frequency)
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
