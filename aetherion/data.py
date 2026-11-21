"""Deterministic, source-grouped preparation of authorized JSONL text datasets.

This module performs *exact* deduplication, not near-duplicate or PII filtering.
All authorization flags are provided by the dataset owner and are not verified.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import unicodedata


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def normalize(text: str) -> str:
    """Preserve content while normalizing line endings and Unicode NFC."""
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def read_documents(paths: list[Path]) -> tuple[list[dict], dict]:
    documents: list[dict] = []
    input_hashes: dict[str, str] = {}
    identifiers: set[tuple[str, str]] = set()
    for path in sorted(paths, key=str):
        raw = path.read_bytes()
        input_hashes[str(path)] = digest(raw)
        for lineno, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object")
            for key in ("id", "source", "license", "text"):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    raise ValueError(f"{path}:{lineno}: required nonempty string '{key}'")
            if item.get("authorized") is not True:
                raise ValueError(f"{path}:{lineno}: requires authorized=true (self-attestation)")
            group = item.get("group", item["source"])
            if not isinstance(group, str) or not group.strip():
                raise ValueError(f"{path}:{lineno}: group must be a nonempty string")
            key = (item["source"], item["id"])
            if key in identifiers:
                raise ValueError(f"{path}:{lineno}: duplicate source/id pair {key}")
            identifiers.add(key)
            text = normalize(item["text"])
            if not text:
                raise ValueError(f"{path}:{lineno}: empty normalized text")
            documents.append({
                "id": item["id"], "source": item["source"],
                "license": item["license"], "group": group,
                "text": text, "sha256": digest(text.encode("utf-8")),
            })
    return documents, input_hashes


def prepare(paths: list[Path], output_dir: Path, seed: int = 42) -> dict:
    """Write deterministic train/validation/test outputs and provenance manifest."""
    if not paths:
        raise ValueError("Provide at least one JSONL input file")
    documents, inputs = read_documents(paths)
    seen: set[str] = set()
    unique: list[dict] = []
    for doc in sorted(documents, key=lambda d: (d["source"], d["id"])):
        if doc["sha256"] not in seen:
            seen.add(doc["sha256"])
            unique.append(doc)
    groups = sorted({doc["group"] for doc in unique},
                    key=lambda g: digest(f"{seed}:{g}".encode("utf-8")))
    if len(groups) < 3:
        raise ValueError("Need at least three distinct source/group values for nonempty splits")
    n = len(groups)
    n_train = max(1, int(n * 0.8))
    n_val = max(1, int(n * 0.1))
    if n_train + n_val >= n:
        n_train = n - n_val - 1
    assignments = {group: ("train" if i < n_train else "validation" if i < n_train + n_val else "test")
                   for i, group in enumerate(groups)}
    by_split = {split: [] for split in ("train", "validation", "test")}
    for doc in unique:
        by_split[assignments[doc["group"]]].append(doc)
    for split, docs in by_split.items():
        if not docs:
            raise ValueError(f"{split} split contains no documents; add more distinct groups")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = [output_dir / f"{split}.{ext}" for split in by_split for ext in ("jsonl", "txt")]
    expected.append(output_dir / "manifest.json")
    if any(p.exists() for p in expected):
        raise FileExistsError("Refusing to overwrite existing prepared dataset files")
    outputs = {}
    stats = {}
    for split, docs in by_split.items():
        docs.sort(key=lambda d: (d["source"], d["id"]))
        jsonl = "".join(json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n" for doc in docs)
        txt = "\n\n".join(doc["text"] for doc in docs) + "\n"
        for ext, content in (("jsonl", jsonl), ("txt", txt)):
            target = output_dir / f"{split}.{ext}"
            raw = content.encode("utf-8")
            target.write_bytes(raw)
            outputs[target.name] = digest(raw)
        stats[split] = {"documents": len(docs), "groups": len({d["group"] for d in docs}),
                        "bytes": len(txt.encode("utf-8")),
                        "licenses": dict(sorted(Counter(d["license"] for d in docs).items()))}
    manifest = {
        "schema_version": 1, "seed": seed,
        "normalization": "Unicode NFC; CRLF/CR to LF; strip document edges",
        "split_method": "SHA-256 seeded ordering of source groups, approximately 80/10/10 by group",
        "inputs_sha256": inputs, "outputs_sha256": outputs,
        "documents_read": len(documents), "exact_duplicates_removed": len(documents)-len(unique),
        "splits": stats,
        "limitations": ["No near-duplicate filtering", "No PII or secret scanning", "License and authorization flags are not independently verified", "No benchmark-contamination filtering"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
