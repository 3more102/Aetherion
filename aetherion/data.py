"""Deterministic, document-level corpus preparation for Aetherion.

Input: UTF-8 JSONL; each row must specify id, text, source and license.
License labels are user declarations, NOT independent permission verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def _bucket(group_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}\0{group_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") * 100.0 / (1 << 64)


def prepare_corpus(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    validation_pct: float = 10.0,
    test_pct: float = 10.0,
) -> dict:
    """Validate, normalize, exact-deduplicate, and split documents by group.

    Identical normalized documents cannot cross splits. All documents sharing
    a group_id go into the same split. This does NOT perform near dedup or PII
    scanning: both require a separate audit before real pretraining.
    """
    if not 0 <= validation_pct < 100 or not 0 <= test_pct < 100:
        raise ValueError("Split percentages must be in [0,100)")
    if validation_pct + test_pct >= 100:
        raise ValueError("Validation + test percentages must be below 100")

    source_file = Path(input_path)
    destination = Path(output_dir)
    rows = []
    seen_ids: set[str] = set()
    empty_count = 0
    with source_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}") from exc
            if not isinstance(doc, dict):
                raise ValueError(f"Expected an object at line {line_number}")
            for field in ("id", "text", "source", "license"):
                if not isinstance(doc.get(field), str) or not doc[field].strip():
                    raise ValueError(f"Missing non-empty {field} at line {line_number}")
            document_id = doc["id"].strip()
            if document_id in seen_ids:
                raise ValueError(f"Duplicate document id: {document_id}")
            seen_ids.add(document_id)
            text = _normalize(doc["text"])
            if not text:
                empty_count += 1
                continue
            group_id = doc.get("group_id", document_id)
            if not isinstance(group_id, str) or not group_id.strip():
                raise ValueError(f"Invalid group_id at line {line_number}")
            rows.append({
                "id": document_id,
                "group_id": group_id.strip(),
                "text": text,
                "source": doc["source"].strip(),
                "license": doc["license"].strip(),
                "language": str(doc.get("language", "unknown")),
            })

    rows.sort(key=lambda row: row["id"])
    content_hashes: set[str] = set()
    kept = []
    for row in rows:
        digest = sha256_bytes(row["text"].encode("utf-8"))
        if digest not in content_hashes:
            content_hashes.add(digest)
            kept.append(row)

    splits = {"train": [], "validation": [], "test": []}
    for row in kept:
        bucket = _bucket(row["group_id"], seed)
        split = "test" if bucket < test_pct else (
            "validation" if bucket < test_pct + validation_pct else "train"
        )
        splits[split].append(row)

    destination.mkdir(parents=True, exist_ok=True)
    split_manifest = {}
    for name, documents in splits.items():
        path = destination / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in documents:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        split_manifest[name] = {
            "documents": len(documents),
            "sha256": sha256_file(path),
            "languages": dict(sorted(Counter(row["language"] for row in documents).items())),
        }

    manifest = {
        "schema_version": 1,
        "input_file": source_file.name,
        "input_sha256": sha256_file(source_file),
        "seed": seed,
        "validation_pct": validation_pct,
        "test_pct": test_pct,
        "normalization": "NFC; canonical newlines; strip outer whitespace",
        "deduplication": "exact normalized-text SHA-256 (near dedup not implemented)",
        "pii_screening": "not implemented; manual and automated review required",
        "license_verification": "not performed; source labels must be audited",
        "input_documents": len(seen_ids),
        "empty_excluded": empty_count,
        "exact_duplicates_excluded": len(rows) - len(kept),
        "splits": split_manifest,
        "sources": dict(sorted(Counter(row["source"] for row in kept).items())),
        "licenses": dict(sorted(Counter(row["license"] for row in kept).items())),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare auditable Aetherion JSONL corpus")
    parser.add_argument("--input", required=True, help="JSONL with id,text,source,license")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-pct", type=float, default=10.0)
    parser.add_argument("--test-pct", type=float, default=10.0)
    args = parser.parse_args()
    manifest = prepare_corpus(args.input, args.output_dir, seed=args.seed,
                              validation_pct=args.validation_pct, test_pct=args.test_pct)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
