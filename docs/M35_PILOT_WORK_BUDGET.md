# M35 — Explicit token and checkpoint-write limits for 125M pilots

**Status:** The execution launcher checks a declared upper work budget before
hardware-report parsing, corpus scanning, model initialization or training.
No production corpus, physical accelerator or trained 125M weights are provided.

## Why this matters

The 125M launcher already requires a reviewed corpus, a matching synthetic
hardware benchmark and a point-in-time live GPU check. An inadvertent high
`--steps`, context, batch size, accumulation or save interval could still
schedule large numbers of optimizer updates or checkpoint writes.

M35 requires **both** `--max-total-tokens` and
`--max-checkpoint-writes` whenever `pilot_train.py --execute` is used.
These values are ceilings entered by the operator, not built-in budgets
or representations of affordable cost.

## Exact formulas

The trainer samples fixed-length next-token microbatches. For a requested
total optimizer-step target, the conservative upper count is:

```text
sampled token positions = steps × batch_size × accumulation × context
```

This counts **training positions sampled**, including repeated corpus spans.
It is not the number of *unique* licensed corpus tokens and is not a
measured data throughput, physical GPU speed, FLOP, energy or dollar amount.

The number of planned checkpoint writes for an unresumed run is:

```text
if save_every = 0: writes = 1 (final checkpoint)
otherwise: writes = ceil(steps / save_every)
```

The trainer defaults to `save_every=50` if omitted. With
`--retain-step-checkpoints`, each write produces a separate
create-only file; otherwise, later writes overwrite the same output path.
The operator must cap *writes* in either mode.

On resume, `steps` is the **total target**, not the remaining steps.
The formula deliberately assumes the whole target run, so it overcounts
the actual work and saves on resume. This is a conservative admission gate.

## Example

After supplying a reviewed dataset and independently measured matching
M28 benchmark on an actual accelerator:

```sh
python pilot_train.py --data-dir data/processed/v1 \
  --tokenizer artifacts/tokenizer.json --review /private/review.json \
  --allow-license CC0-1.0 --steps 20 --schedule-steps 100 \
  --context 256 --batch-size 1 --accumulation 4 --device cuda:0 \
  --precision bf16 --output checkpoints/pilot.pt \
  --benchmark-report reports/125m-cuda0.json \
  --max-peak-reserved-gib 12 --max-total-tokens 20480 \
  --max-checkpoint-writes 1 --execute
```

For this illustrative run, `20 × 1 × 4 × 256 = 20,480`
sampled token positions and one final checkpoint write are planned.
The 12 GiB ceiling is illustrative, not a recommendation. Choose caps
based on the real review and authorization process. A launch may still
fail due to memory, disk, device availability or other system factors.

## Validation

```sh
python -m pytest -q tests/test_m35_pilot_budget.py
python -m pytest -q
```

The isolated budget module has 15 CPU-only regression checks. Two
additional integration tests require absent or insufficient limits to
reject before benchmark/corpus access. Existing M28/M34 launcher tests
are updated to satisfy the explicit caps. Full CI must pass before merge.

**Limitations:** This is not a runtime time limit, allocation of free
GPU memory, financial cap, preflight for available disk space, telemetry,
or independent evidence of data rights or model quality. It bounds the
requested work, but unexpected failures can still produce partial files.
