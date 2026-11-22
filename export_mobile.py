"""Export an existing, trusted Aetherion byte-token checkpoint to mobile ONNX."""
import argparse
import json

from aetherion.mobile_export import export_mobile_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Aetherion Micro .pt checkpoint")
    parser.add_argument("--output-dir", default="mobile/artifacts")
    parser.add_argument("--window", type=int, default=64, help="Fixed context (2..checkpoint block_size)")
    args = parser.parse_args()
    print(json.dumps(export_mobile_checkpoint(args.checkpoint, args.output_dir, window=args.window), indent=2))


if __name__ == "__main__":
    main()
