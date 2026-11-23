# M33 — Offline integrity check for validation-selected test evidence

M32 now emits **schema_version 2**. In addition to the chosen checkpoint,
it records the validation history of every candidate: ascending step, checkpoint
SHA-256, exact validation target count, validation mean NLL and perplexity.
The reported selection is still based **only on validation NLL**, using the
earliest candidate for an exact tie.

## Verify without loading checkpoints or running inference

Retain the original M32 report, all candidate checkpoints in the same order,
and the original \`train.txt\`, \`validation.txt\` and \`test.txt\` files.

\`\`\`bash
python verify_selected_test_report.py \
  --report reports/selected-test.json \
  --data-dir data/processed/v1 \
  --checkpoint checkpoints/run-step00000001.pt \
  --checkpoint checkpoints/run-step00000002.pt \
  --checkpoint checkpoints/run-step00000003.pt \
  --output reports/selected-test-integrity.json
\`\`\`

The verifier checks:

- Report type/schema, duplicate JSON keys, non-finite values and a 1 MiB
  input limit.
- Exactly 2–32 candidates, distinct SHA-256 digests, strictly increasing
  steps and consistent validation target counts.
- Every **supplied checkpoint file** against its recorded SHA-256 digest,
  streamed without loading tensor weights.
- All three split files against the report's digests; for production, the
  prepared manifest SHA-256 and the pinned test entry.
- The precise selected validation-NLL minimum with first-tie rule.
- Perplexity/NLL arithmetic, test loss × target count, context/stride,
  and the byte-token target count for synthetic demo records.

The create-only output has a SHA-256 of the **report bytes** and a
human-readable limit statement. Reusing an output path fails instead of
overwriting an earlier audit record.

## Scope and limitations

This is an **integrity and internal consistency check**, not an independent
replication of reported NLL. The verifier **does not** load weights, tokenize
real production data, score test tokens, prove model quality or authenticate
licenses, authorship, physical hardware or lineage. An attacker who can edit
both data/checkpoints and a local unsigned report can forge a consistent set.
A successful audit does not establish a one-time untouched holdout.

M32 v1 reports predate the inclusion of per-candidate history; regenerate
the evidence using M32 v2 to enable this offline audit. Regenerating a
test report is **not** appropriate if doing so violates a holdout-use
commitment: preserve the v1 report and disclose its audit limitation.

## Regression verification

\`\`\`bash
python -m pytest -q tests/test_m33_offline_audit.py
python -m pytest -q
\`\`\`

Synthetic fixtures exercise valid reports, selection tampering, arithmetic
inconsistencies, modified data/checkpoints, duplicate JSON keys, report size,
first-tie handling, and exclusive report creation. They do not prove any
Aetherion-125M training or independent physical-device benchmark.
