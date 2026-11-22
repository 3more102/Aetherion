"""Audit a prepared corpus before requesting human review."""
import argparse
import json

from aetherion.corpus_gate import audit_corpus


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    p.add_argument("--allow-license", action="append", required=True)
    args = p.parse_args()
    print(json.dumps(audit_corpus(args.directory, args.allow_license), indent=2))


if __name__ == "__main__":
    main()
