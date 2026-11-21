"""Train an Aetherion byte-level BPE tokenizer from authorized prepared data."""
import argparse
import json
from pathlib import Path

from aetherion.tokenizer import train_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Aetherion BPE tokenizer training")
    parser.add_argument("--data-dir", required=True, type=Path, help="Prepared dataset directory containing manifest.json and train.jsonl")
    parser.add_argument("--output-dir", required=True, type=Path, help="New tokenizer artifact directory")
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    tokenizer = train_tokenizer(args.data_dir, args.output_dir, args.vocab_size, args.min_frequency)
    print(json.dumps({
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_sha256": tokenizer.sha256,
        "source_train_sha256": tokenizer.metadata["source_train_sha256"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
