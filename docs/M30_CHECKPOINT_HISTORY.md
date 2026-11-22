# M30 — Retained training checkpoints and comparable validation history

**Scope:** Reproducibility infrastructure. No real Aetherion-125M run,
pretrained assistant checkpoint, approved training corpus or physical mobile
performance is supplied by M30.

## Problem addressed

The existing `train.py --save-every` rewrote one `--output` file. That is
useful for a latest-state resume checkpoint, but loses earlier snapshots
and prevents a genuine step-by-step held-out NLL comparison unless
operators manually copy files between steps.

## Opt-in retained checkpoints

With a **separate explicit validation split**, run a tiny local demonstration:

```sh
python train.py \
  --data data/tiny-demo/train.txt \
  --val-data data/tiny-demo/validation.txt \
  --steps 4 --save-every 1 --batch-size 1 --block-size 8 \
  --d-model 16 --layers 1 --heads 2 --device cpu \
  --retain-step-checkpoints --output checkpoints/tiny.pt
```

The opt-in flag saves to:

```text
checkpoints/tiny-step00000001.pt
checkpoints/tiny-step00000002.pt
checkpoints/tiny-step00000003.pt
checkpoints/tiny-step00000004.pt
```

This mode **does not write** `checkpoints/tiny.pt`; the original
`--output` is a filename prefix for named snapshots. Without
`--retain-step-checkpoints`, the previous replace-latest behavior is
unchanged for legacy training workflows.

Snapshotted checkpoint files use a fsynced temporary artifact plus an
atomic hard-link create-if-absent operation to prevent existing snapshot
files being overwritten, including by concurrent writers. The local
filesystem must support hard links. A failed attempt to reuse a step
raises `FileExistsError`; it does not silently replace evidence.
Each saved snapshot retains the existing format-v2 optimizer and RNG
state and can be used with `--resume path/to/stepXXXXXXXX.pt`. Restart
with a *larger total* `--steps` value and leave the previous files intact.

Checkpointing requires disk space proportional to model size and optimizer
state, particularly for the proposed 125M pilot; confirm free space and
storage permissions before enabling frequent snapshots.

## Re-evaluate an ordered checkpoint history

```sh
python audit_training_history.py \
  --demo --data-dir data/tiny-demo --device cpu --stride 8 \
  --checkpoint checkpoints/tiny-step00000001.pt \
  --checkpoint checkpoints/tiny-step00000002.pt \
  --checkpoint checkpoints/tiny-step00000003.pt \
  --checkpoint checkpoints/tiny-step00000004.pt \
  --output reports/tiny-history.json
```

Create `reports/` beforehand. The CLI only creates a report when the
destination does not already exist. It never writes checkpoint weights.
The command re-runs the full M29 single-checkpoint assessment on each
supplied file and rejects incompatible training settings, corpus/tokenizer
fingerprints, seed, model architecture, validation context, stride, or
target count. Steps must be **strictly increasing**; exact duplicate
paths or checkpoint contents are rejected. The input limit is 2–32
snapshots to bound accidental overlong evaluations.

History JSON lists each step, checkpoint SHA-256, held-out next-token NLL
and perplexity, per-step NLL delta, total NLL delta, final perplexity
ratio, and the checkpoint having the best validation NLL. The optional
`--require-improvement` flag returns code 2 when the **last** checkpoint
has not improved over the first; a regression is still a valid recorded
result.

## Production 125M history

Omit `--demo` and pass `--tokenizer`, `--review` and one or more
`--allow-license` options with an authorized and prepared corpus.
M29/M27 recheck the complete production approval gates *before opening
each trusted checkpoint*. All checkpoints must match the 125M model and
must have been measured against the same reviewed validation data.

If a training pilot is launched via `pilot_train.py --execute`, the
existing M28 accelerator hardware evidence gate applies. This M30
history command is a **read-only evaluator** and does not bypass pilot
training approval.

## Evidence interpretation

A series of matching dataset hashes, architecture, seed and hyperparameters
is **not cryptographic proof** that two files share an uninterrupted
training lineage: weights could have been substituted or checkpoint
metadata edited. File SHA-256 identifies bytes, but does not authenticate
the claims in a locally supplied checkpoint. Retaining snapshots and
re-evaluating them provides stronger comparability, not attestation of
training ancestry or intellectual capability.

Teacher-forced held-out validation NLL does not measure test-set
generalization, reasoning, factual reliability, instruction following or
assistant safety. The `--demo` path uses unreviewed byte-token fixtures
and cannot be used to claim useful pretrained language ability.

## Test coverage

```sh
python -m pytest -q tests/test_m30_checkpoint_history.py
python -m pytest -q
```

CPU tests run a tiny actual optimizer sequence, verify separate step
artifacts, resume preservation, no-overwrite collisions, metadata
mismatch rejection, exact validation-target accounting, exclusive
report saving, and missing production review gates. No real 125M weights
or production GPU/phone results are included.
