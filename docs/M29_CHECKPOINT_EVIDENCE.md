# M29 — Single-checkpoint held-out validation evidence

**Status:** Tooling for producing reproducible validation evidence for one
locally trusted Aetherion training checkpoint. It does **not** train a model,
provide pretrained weights, or show a successful real-world 125M run.

## Purpose

The existing evaluate.py reports validation loss and perplexity and
compare_checkpoints.py compares two checkpoints. M29 adds a standalone,
fail-closed, single-checkpoint evidence artifact with recorded model
architecture, checkpoint SHA-256, seed, training hyperparameters and
dataset provenance. Its JSON output distinguishes unreviewed byte-token
diagnostics from attested 125M BPE evaluation.

## Demo use (synthetic/unreviewed)

Train a small independent byte-token diagnostic, using distinct train.txt
and validation.txt files. Then run:

```sh
python assess_checkpoint.py --demo \
  --checkpoint checkpoints/tiny.pt \
  --data-dir data/tiny-demo \
  --stride 8 --device cpu \
  --output reports/tiny-validation.json
```

Create reports/ first. Existing output files are never overwritten.
Demo mode deliberately forbids a tokenizer, review or license allowlist.
This report is labeled synthetic_or_unreviewed_byte_demo, regardless of
how plausible the model's generated text might look.

## Real 125M evaluation (after real training only)

The validated production command requires approved and prepared corpus
splits, independent human review, 32k BPE tokenizer and a locally trusted
format-v2 125M checkpoint from that exact data:

```sh
python assess_checkpoint.py \
  --checkpoint checkpoints/approved-125m-step20.pt \
  --data-dir data/processed/v1 \
  --tokenizer artifacts/tokenizer.json \
  --review /private/review.json \
  --allow-license CC0-1.0 \
  --stride 128 --device cuda:0 \
  --output reports/approved-step20-validation.json
```

These are illustrative paths only. Use actual approved license identifiers.
The production preflight repeats the M27 integrity, group isolation,
allowlist, attestation and bounded lexical contamination gates **before
loading checkpoint weights**. It then verifies the original dataset and
tokenizer hashes, exact Aetherion-125M architecture, full validation-target
coverage, and checkpoint SHA-256 unchanged across the evaluation.

## Report schema

- kind: aetherion-single-checkpoint-heldout-validation
- mode: synthetic_or_unreviewed_byte_demo or attested_125m_validation
- checkpoint: SHA-256, format, step, seed, architecture, parameter count,
  training hyperparameters
- dataset: train/validation hashes, tokenizer hash/type, prepared corpus and
  review digests, and lexical-screen provenance when production-approved
- validation: number of next-token targets, sum and mean negative log
  likelihood, perplexity, context window and stride, plus explicit
  UTF-8-byte vs BPE-token units and all-target coverage indicator

The scorer is teacher-forced next-token validation; every target is counted
exactly once for the requested stride. It does not measure free-running chat
performance. The M17 comparator remains the way to measure evidence of
relative NLL improvement between two genuinely comparable checkpoints.

## Automated regression gates

```sh
python -m pytest -q tests/test_m29_checkpoint_evidence.py
python -m pytest -q
```

CI uses actual small CPU-trained byte-token fixture weights. It checks
reproducibility, evaluation parity with evaluate.py, hashes, exact target
coverage, tampered data, malformed checkpoint metadata, output collision
and rejection of production data before weights are opened.

## Important limitations

A self-attested corpus review is not legal, privacy, or dataset-quality
certification; the lexical screen cannot detect semantic/paraphrase leakage
or all benchmark contamination. NLL/perplexity from one held-out split are
not proof of factual accuracy, safety, independent generalization,
reasoning, human preference or competitive assistant quality. No
non-synthetic 125M checkpoint, actual training loss, or mobile validation
is included by this change. Checkpoints must originate from a trusted
local source; weights-only loading is not a general malicious-file sandbox.
