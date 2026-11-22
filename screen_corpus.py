"""Screen prepared train/validation/test text for high lexical overlap.

Uses *only* local data. Exits 2 if findings exist (unless --report-only).
"""
import argparse
import json

from aetherion.contamination import screen_cross_split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Prepared dataset directory")
    parser.add_argument("--allow-license", action="append", required=True,
                        help="Explicit license allowlist; repeat for multiple licenses")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--max-documents", type=int, default=20_000)
    parser.add_argument("--max-comparisons", type=int, default=500_000)
    parser.add_argument("--max-findings", type=int, default=1_000)
    parser.add_argument("--report-only", action="store_true",
                        help="Do not fail on findings (never overrides integrity/budget failures)")
    args = parser.parse_args()
    report = screen_cross_split(
        args.directory, args.allow_license,
        threshold=args.threshold, window=args.window,
        max_documents=args.max_documents,
        max_comparisons=args.max_comparisons, max_findings=args.max_findings,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 2 if report["requires_manual_review"] and not args.report_only else 0


if __name__ == "__main__":
    raise SystemExit(main())
