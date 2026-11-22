"""Check prepared corpus integrity and a human review attestation.

An attestation is NOT verification of legal rights or privacy compliance.
"""
from hashlib import sha256
import json
from pathlib import Path
from collections import Counter


def digest(raw):
    return sha256(raw).hexdigest()


def audit_corpus(directory, allowed_licenses):
    directory = Path(directory)
    raw = (directory / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 1 or not allowed_licenses:
        raise ValueError("Require schema-v1 manifest and explicit allowed licenses")
    seen_groups, seen_hashes, seen_ids = set(), set(), set()
    counts = {}
    for split in ("train", "validation", "test"):
        split_groups = set()
        licenses = Counter()
        texts = []
        for ext in ("txt", "jsonl"):
            name = f"{split}.{ext}"
            payload = (directory / name).read_bytes()
            if digest(payload) != manifest["outputs_sha256"].get(name):
                raise ValueError(f"Checksum mismatch: {name}")
            if ext == "txt":
                text_bytes = payload
            else:
                records = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
        if not records:
            raise ValueError(f"Empty split: {split}")
        for record in records:
            license_id = record["license"]
            if license_id not in allowed_licenses:
                raise ValueError(f"License outside allowlist: {license_id}")
            text = record["text"]
            fingerprint = digest(text.encode("utf-8"))
            identity = (record["source"], record["id"])
            if (fingerprint != record["sha256"] or fingerprint in seen_hashes
                    or identity in seen_ids or record["group"] in seen_groups):
                raise ValueError("Duplicate, cross-split group or invalid record fingerprint")
            seen_hashes.add(fingerprint)
            seen_ids.add(identity)
            split_groups.add(record["group"])
            licenses[license_id] += 1
            texts.append(text)
        if ("\n\n".join(texts) + "\n").encode("utf-8") != text_bytes:
            raise ValueError("Prepared text disagrees with JSONL")
        seen_groups.update(split_groups)
        counts[split] = {"documents": len(records), "licenses": dict(licenses)}
    return {"manifest_sha256": digest(raw), "splits": counts,
            "allowed_licenses": sorted(allowed_licenses),
            "limitations": "Integrity and exact overlap checks only; legal rights, PII, near duplicates and contamination require independent review."}


def require_review(report, review_path):
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (review.get("manifest_sha256") != report["manifest_sha256"]
            or sorted(review.get("allowed_licenses", [])) != report["allowed_licenses"]
            or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()
            or any(review.get(key) is not True for key in
                   ("rights_reviewed", "privacy_reviewed", "near_duplicates_reviewed", "contamination_reviewed"))):
        raise ValueError("Missing or mismatched human corpus review attestation")
    return review
