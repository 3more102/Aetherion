"""Convert an existing Aetherion mobile ONNX graph into INT4 MatMulNBits."""
import argparse
import json

from aetherion.mobile_quantization import quantize_mobile_int4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Unquantized model.onnx")
    parser.add_argument("--manifest", required=True, help="Matching manifest.json")
    parser.add_argument("--output-dir", required=True, help="Separate directory for INT4 files")
    parser.add_argument("--block-size", type=int, default=32)
    args = parser.parse_args()
    result = quantize_mobile_int4(
        args.model, args.manifest, args.output_dir, block_size=args.block_size
    )
    print(json.dumps(result["quantization"], indent=2))


if __name__ == "__main__":
    main()
