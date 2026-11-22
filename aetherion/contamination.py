"""Deterministic, bounded cross-split lexical contamination screening.

This is an *additional* audit of prepared data, not a legal/privacy review
and not a semantic/benchmark contamination detector. No source text is
written into reports.
"""
from __future__ import annotations

from collections import defaultdict
import json
from math import isfinite
from pathlib import Path

from .corpus_gate import audit_corpus


SPLITS = ("train", "validation", "test")


def word_shingles(text: str, window: int = 5) -> frozenset[tuple[str, ...]]:
    """Casefolded whitespace-token shingles; short documents use one shingle."""
    words = text.casefold().split()
    if not words:
        return frozenset()
    if len(words) < window:
        return frozenset((tuple(words),))
    return frozenset(tuple(words[i:i + window])
                     for i in range(len(words) - window + 1))


def screen_cross_split(
    directory: str | Path,
    allowed_licenses: list[str] | set[str],
    *,
    threshold: float = 0.8,
    window: int = 5,
    max_documents: int = 20_000,
    max_comparisons: int = 500_000,
    max_findings: int = 1_000,
) -> dict:
    """Find high lexical Jaccard overlap across prepared corpus splits.

    Candidate pairs share at least one exact word shingle. Every such
    cross-split candidate is compared, or the scan fails closed on the
    explicit candidate budget. A report of zero findings is *not* proof
    of no paraphrases, PII, or evaluation contamination.
    """
    if (isinstance(threshold, bool) or not isinstance(threshold, (float, int))
            or not isfinite(threshold) or not 0 < threshold <= 1):
        raise ValueError("threshold must be finite and between 0 and 1")
    for name, value in (
        ("window", window),
        ("max_documents", max_documents),
        ("max_comparisons", max_comparisons),
        ("max_findings", max_findings),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if window > 16:
        raise ValueError("window must be at most 16")

    directory = Path(directory)
    # Validate checksums, license allowlist, group isolation, identity and
    # normalized text correspondence before examining any data.
    audit = audit_corpus(directory, allowed_licenses)

    documents: list[tuple[str, str, str, frozenset]] = []
    counts = {}
    for split in SPLITS:
        records = [json.loads(line) for line in
                   (directory / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        counts[split] = len(records)
        for record in records:
            documents.append((split, record["source"], record["id"],
                              word_shingles(record["text"], window)))
            if len(documents) > max_documents:
                raise ValueError("Document scan budget exceeded; use a scalable review tool")

    postings: dict[tuple[str, ...], list[int]] = defaultdict(list)
    findings = []
    comparisons = 0
    for i, (split, source, doc_id, shingles) in enumerate(documents):
        candidate_ids = set()
        for shingle in shingles:
            for prior in postings[shingle]:
                if documents[prior][0] != split:
                    candidate_ids.add(prior)
                    if len(candidate_ids) + comparisons > max_comparisons:
                        raise ValueError("Candidate-pair comparison budget exceeded; review required")
        for prior in sorted(candidate_ids):
            comparisons += 1
            other_split, other_source, other_id, other = documents[prior]
            intersection = len(shingles & other)
            union = len(shingles | other)
            if union and intersection / union >= threshold:
                findings.append({
                    "a": {"split": other_split, "source": other_source, "id": other_id},
                    "b": {"split": split, "source": source, "id": doc_id},
                    "jaccard": round(intersection / union, 6),
                })
                if len(findings) > max_findings:
                    raise ValueError("Finding count budget exceeded; review required")
        for shingle in shingles:
            postings[shingle].append(i)

    return {
        "schema_version": 1,
        "kind": "aetherion-cross-split-lexical-screen",
        "manifest_sha256": audit["manifest_sha256"],
        "allowed_licenses": audit["allowed_licenses"],
        "split_document_counts": counts,
        "word_shingle_window": window,
        "jaccard_threshold": threshold,
        "candidate_pairs_compared": comparisons,
        "findings_count": len(findings),
        "findings": findings,
        "requires_manual_review": bool(findings),
        "limitations": (
            "Lexical shingles only: cannot detect paraphrases, transformed copies, "
            "PII/secrets or benchmark contamination. No original text in report. "
            "An empty finding list is not evidence of clean training data."
        ),
    }
