# M28 — Benchmark-backed execution preflight for Aetherion-125M

**Status:** Automated validation of a **user-supplied synthetic accelerator
benchmark**, not evidence that a physical GPU has been measured here.
Aetherion-125M remains untrained.

## Why this gate exists

The 125M pilot launcher previously required reviewed corpus data and a
verified tokenizer but allowed `--execute` without a matching recorded
hardware benchmark. The current 125M architecture has **123,688,704 unique
parameters**. Its weight-only estimates substantially understate training
memory because gradients, optimizer state, activations, scratch buffers and
runtime allocations add overhead.

M28 requires a recent, deliberate measured report as *user-supplied*
evidence before `pilot_train.py --execute`. The validator checks the
reported 125M configuration and numbers, but **cannot authenticate** that
the JSON was produced on the claimed physical device.

## 1 — Measure exact configuration

On your intended CUDA or XPU hardware (not on this CI environment):

```sh
python benchmark_training.py \
  --preset 125m --device cuda:0 --precision bf16 \
  --context 256 --batch-size 1 --accumulation 4 \
  --warmup 1 --steps 3 \
  --output reports/125m-cuda0-bench.json
```

Create the `reports` directory first, keep this report private when it
contains identifiers you do not want to publish, and inspect the values.
The benchmark performs real synthetic optimizer updates for this architecture;
it does **not** load a licensed corpus or measure model learning. A GPU
out-of-memory failure is a valid stop signal; do not fake measurements.

The report is now schema version 1 with the fixed kind
`aetherion-synthetic-optimizer-hardware-benchmark`. The `--output` option
creates a new JSON file and refuses overwrite. An existing report can still
be printed to stdout without `--output`.

## 2 — Dry-run corpus and tokenizer checks

```sh
python pilot_train.py \
  --data-dir data/processed/v1 --tokenizer artifacts/tokenizer.json \
  --review /private/review.json --allow-license CC0-1.0 \
  --steps 20 --schedule-steps 100 --context 256 \
  --batch-size 1 --accumulation 4 --device cuda:0 --precision bf16 \
  --output checkpoints/pilot.pt
```

This prints a command but does **not** launch training. Existing M25/M27
corpus integrity, rights-attestation and lexical-leakage checks still apply.
Replace every example path and allowlisted license with actual reviewed
inputs.

## 3 — Explicit training execution (only after approval)

Repeat the same pilot command with:

```sh
  --benchmark-report reports/125m-cuda0-bench.json \
  --max-peak-reserved-gib 12 \
  --max-total-tokens 20480 --max-checkpoint-writes 1 \
  --execute
```

The 12 GiB number is **illustrative, not a hardware recommendation**;
choose an independently justified memory ceiling for the actual machine.
For `--execute`, the gate requires:

- An explicit accelerator index such as `cuda:0` or `xpu:0` (no `auto` or `cpu`).
- Exact preset/parameter count, context, batch size, accumulation, device
  and precision matching the proposed training run.
- At least one warmup update and three measured optimizer updates.
- Coherent measured token count and token/second rate, finite loss and
  positive duration.
- Positive recorded accelerator peak allocated/reserved bytes, with peak
  reserved memory not exceeding the supplied GiB ceiling.
- Valid, bounded (64 KiB max) benchmark JSON.

Failure prevents `subprocess.run` from launching the trainer. Successful
preflight prints a SHA-256 of the exact JSON report and its synthetic
measurements before launching. Neither the report nor the hash proves
the report is authentic or the physical device has free RAM.

## Verification

```sh
python -m pytest -q tests/test_m28_hardware_gate.py
python -m pytest -q
```

The tests use **fabricated clearly labeled benchmark fixtures** so no GPU is
needed in CI. No live 125M training, actual CUDA/XPU memory result,
real-world device throughput or pretrained weights is established by a
passing test.

## Limits and next evidence

A small synthetic benchmark is not representative of a longer training run,
different data-loader memory pressure, thermal throttling, fragmentation,
future model changes, or multiple GPUs. Even a valid report is not a peak
system-RAM guarantee. The preflight does not validate the actual identity of
the GPU at execution time. Hardware operators must cross-check reported
device information, the current GPU, available resources and report age.

Data rights, privacy, held-out contamination, seed reproducibility,
checkpoint evaluation and useful assistant behavior must be independently
validated. The user must explicitly approve real training and its cost.
