"""Quantize both M19 KV-cached ONNX graphs with weight-only INT4."""
import argparse
import json

from aetherion.mobile_cached_quantization import quantize_mobile_cached_int4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Directory with M19 prefill.onnx, decode.onnx, manifest.json")
    parser.add_argument("--output-dir", required=True, help="Separate directory for schema-v3 outputs")
    parser.add_argument("--block-size", type=int, default=32)
    args = parser.parse_args()
    result = quantize_mobile_cached_int4(args.source_dir, args.output_dir, block_size=args.block_size)
    print(json.dumps(result["quantization"], indent=2))


if __name__ == "__main__":
    main()
