"""Export verified Aetherion-Micro prefill and cached-decode ONNX graphs."""
import argparse
import json

from aetherion.mobile_cached import export_mobile_cached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="mobile/artifacts/cached")
    parser.add_argument("--window", type=int, default=64)
    args = parser.parse_args()
    manifest = export_mobile_cached(args.checkpoint, args.output_dir, window=args.window)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
