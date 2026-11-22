"""Quantize M19 FP32 prefill/decode ONNX exports into matching INT4 graphs."""
import argparse
import json

from aetherion.mobile_cached_quantization import quantize_mobile_cached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="M19 directory containing FP32 graphs and manifest.json")
    parser.add_argument("--output-dir", required=True, help="New directory for INT4 cached graphs")
    parser.add_argument("--block-size", type=int, default=32)
    args = parser.parse_args()
    manifest = quantize_mobile_cached(
        args.source_dir, args.output_dir, block_size=args.block_size
    )
    print(json.dumps({
        "graphs": manifest["graphs"],
        "quantization": manifest["quantization"],
    }, indent=2))


if __name__ == "__main__":
    main()
