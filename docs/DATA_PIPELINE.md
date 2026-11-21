# Reproducible data preparation (M1 foundation)

This prepares **user-authorized** JSONL records for the current byte-level Micro training path. It is not a substitute for a licensed corpus, privacy review, near-duplicate detection, or benchmark decontamination.

## Input JSONL schema

Each line must be a JSON object, for example:

```json
{"id":"doc-001","source":"my-corpus-v1","group":"my-corpus-v1","license":"CC0-1.0","authorized":true,"text":"Text that I have the rights to use."}
```

- `id`: document identifier unique within `source`.
- `source`: origin identifier, not necessarily the human-readable title.
- `license`: recorded provenance; **not automatically checked** for correctness.
- `authorized`: explicit self-attestation by the dataset preparer.
- `text`: full text content.
- `group` (optional): split-isolation grouping; defaults to `source`. Keep related sources in one group to reduce leakage.

At least three distinct groups are required for nonempty train/validation/test sets. Identical texts are removed before splitting. Groups are sorted using a seed-dependent SHA-256 key, then allocated approximately 80% / 10% / 10% by **group count**. Actual document ratios may differ, especially for uneven source sizes.

## Commands

```bash
python prepare_data.py /path/to/authorized.jsonl --output-dir data/processed/v1 --seed 42
python train.py --data data/processed/v1/train.txt --val-data data/processed/v1/validation.txt --steps 100 --output checkpoints/experiment-v1.pt
```

Outputs: `train.jsonl`, `validation.jsonl`, `test.jsonl` (document metadata); matching `.txt` files (byte-stream input for Micro), and `manifest.json` (input/output SHA-256 hashes, split counts, provenance summary, limitations). The preparer refuses to overwrite an existing output directory's data files. Do not commit sensitive datasets or derived text to GitHub.

**Important:** `train.py` can ingest the `.txt` files only with the added `--val-data` option. This training example uses raw bytes and remains an educational smoke test. Tokenizer fitting later **must use `train.jsonl` only**. Never fit the tokenizer or tune hyperparameters on the held-out test set. Current exact dedup does not detect paraphrases or near-duplicates. Perform PII/secret scanning and contamination analysis separately before serious training.
