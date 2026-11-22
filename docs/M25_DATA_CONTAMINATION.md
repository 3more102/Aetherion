# M25 — Cross-split lexical contamination screen

**Scope:** A deterministic local review helper for Aetherion-125M data readiness.
It does **not** train the model, supply a licensed corpus or certify that a
dataset is safe/clean.

## Workflow

After preparing authorized source-grouped JSONL with `prepare_data.py`,
run the existing integrity/license audit, followed by the M25 screen:

```sh
python audit_corpus.py data/processed/v1 --allow-license CC0-1.0
python screen_corpus.py data/processed/v1 --allow-license CC0-1.0
```

Replace `CC0-1.0` with explicitly reviewed license identifiers. The screen
reuses the existing `audit_corpus` checks: split output checksums, the
license allowlist, unique source/id, record/text correspondence and group
isolation. It never uploads dataset text.

The default screen compares **word 5-shingles**, casefolded by Unicode,
and flags train/validation/test document pairs with Jaccard overlap
**at least 0.8**. Findings report only split, source, ID and overlap
score; no source text is written into the JSON report. Each run is
deterministic for the same manifest and parameters.

- Exit code **0**: no lexical findings under configured limits.
- Exit code **2**: potential cross-split contamination; stop and manually review.
- `--report-only`: print findings without exit 2; use only for exploration.
- Integrity, invalid settings or exceeded scan budgets always raise an error.

Threshold and bounds are configurable using `--threshold`,
`--window`, `--max-documents`, `--max-comparisons` and
`--max-findings`. The defaults are 20,000 documents, 500,000
candidate comparisons and 1,000 findings. On a budget overrun the tool
fails closed rather than silently truncating results. This is a
single-machine screening step, not a scalable production data engine.

## Limitations and approval boundary

An empty finding list **does not** prove absence of leakage or
contamination. The lexical method does not detect paraphrases, translations,
semantic duplicates, duplicated fragments shorter than the window,
private information, secrets or benchmark reuse. Similarity is based on
sets of whitespace-separated word shingles, which is language dependent.
It may flag legitimate formulaic text. It does not establish legal rights
or validate license claims.

An independent human reviewer must still assess rights, privacy,
near-duplicates, evaluation contamination and dataset quality.
A completed M25 screen is evidence of one bounded lexical test, not
approval to begin long-running 125M pretraining.

Run regression tests:

```sh
python -m pytest -q tests/test_contamination.py
python -m pytest -q
```
