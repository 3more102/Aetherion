# M27 — Mandatory lexical contamination gate for production experiments

**Status:** Production preflight hardening; no model training or real dataset
approval is implied.

## The issue

M25 added a deterministic, bounded cross-split lexical screen, but the
`pilot_train.py` launcher and production mode of
`compare_checkpoints.py` relied on the existing integrity audit and the
human-review attestation without automatically enforcing the new screen.
Therefore, a mistaken or self-attested review could coexist with detected
high-overlap train/validation/test records.

## Behavior

Both production paths now call the same `require_no_cross_split_findings`
acceptance gate **after the checksum/license audit and reviewer attestation,
but before opening weights, constructing the model, or launching training**.

- Cross-split lexical findings prevent execution and include a count, not
  the underlying text, in the error.
- Document, comparison, or finding budget overruns also prevent execution;
  there is no `--report-only` bypass in production launchers.
- A detected corrupt split or disallowed license also prevents launch.
- For accepted production comparisons, reports record the screened
  manifest SHA-256, count of candidate pairs inspected and zero findings.
- The existing explicit synthetic `--demo` checkpoint-comparison mode
  remains separate and is not evidence of approved production data.

The screen uses M25 defaults: 5-word casefolded shingles, Jaccard >= 0.8,
at most 20,000 documents, 500,000 candidate comparisons and 1,000 findings.
This bounded algorithm is **not suitable for arbitrarily large corpora**.
Larger runs require a separately reviewed scalable solution, not bypassing
the guard. Under the current prototype gate a large corpus may be rejected.

## Commands

```sh
python -m pytest -q tests/test_contamination.py tests/test_m27_production_screen.py
python -m pytest -q

# Source and rights review must already be completed; default is dry-run.
python pilot_train.py \
  --data-dir data/processed/v1 \
  --tokenizer artifacts/tokenizer.json \
  --review /private/review.json \
  --allow-license CC0-1.0 \
  --steps 20 --schedule-steps 100 --context 256 \
  --output checkpoints/pilot.pt
```

Do not add `--execute` without actual dataset authorization, a reviewed
hardware budget, and a completed benchmark of the intended configuration.
A print-only command validates gates; it does not train.

## Evidence and limitations

Only SHA-256 integrity, license string allowlisting, source-group separation
and a bounded lexical high-overlap check are automated. Empty findings
**do not** certify dataset rights, privacy, paraphrase freedom, semantic
decontamination, benchmark isolation or language-model quality. Human
independent review remains essential. No approved 125M training corpus,
model weights, held-out quality result or phone performance is claimed.
