"""Inspect Aetherion architecture without allocating a large model."""
import argparse
import json
from aetherion.presets import PRESETS, preset_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="125m")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report")
    args = parser.parse_args()
    report = preset_report(args.preset)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        counts = report["parameter_breakdown"]
        print(f"Aetherion-{args.preset}: {counts['total']:,} unique parameters (untrained)")
        print(f"SwiGLU hidden size: {counts['swiglu_hidden']:,}")
        print(f"FP32 weight storage only: {report['fp32_weight_only_mib']:.2f} MiB")
        print(report["warning"])


if __name__ == "__main__":
    main()
