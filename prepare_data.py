"""Prepare a source-grouped dataset for Aetherion Micro (UTF-8 byte stream)."""
import argparse
import json
from pathlib import Path
from aetherion.data import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare authorized JSONL text for Aetherion")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input JSONL paths")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = prepare(args.inputs, args.output_dir, seed=args.seed)
    print(json.dumps({"splits": result["splits"], "exact_duplicates_removed": result["exact_duplicates_removed"]}, indent=2))
    print("Warning: manual license, privacy, near-duplicate and contamination review is required.")


if __name__ == "__main__":
    main()
