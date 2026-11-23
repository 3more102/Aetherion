# M31 — SHA-linked retained checkpoint lineage

**Scope:** Optional provenance checks for the existing M30 retained checkpoint
workflow. This is *not* training evidence for Aetherion-125M and does not
authenticate the operator, weights, or optimizer trajectory.

## Why

M30 compares checkpoint metadata and held-out next-token NLL, but two files
with matching hyperparameters and ascending steps could have been produced by
unrelated runs. M31 allows the existing history auditor to reject a broken
recorded save chain before reporting it as a linked history.

## Generate linked snapshots

Use the existing M30 flag (no new command is needed):

```sh
python train.py --data data/tiny-demo/train.txt \
  --val-data data/tiny-demo/validation.txt \
  --steps 4 --save-every 1 --batch-size 1 --block-size 8 \
  --d-model 16 --layers 1 --heads 2 --device cpu \
  --retain-step-checkpoints --output checkpoints/tiny.pt
```

Each retained snapshot includes an optional `lineage` dictionary within the
existing format-v2 checkpoint:

- `version=1` and a fresh random 128-bit hexadecimal `run_id` for a new chain.
- `parent_sha256`: SHA-256 of the previous **saved checkpoint's complete file**,
  or `null` for a newly started chain.
- `parent_step`: previous saved checkpoint's training step, or `null` at root.

Uninterrupted saves and retained-checkpoint resumes carry the same run ID and
link back to the actual parent file. SHA-256 is computed in bounded memory;
when resuming a retained chain, the parent is hashed both before and after
weight/optimizer/RNG restoration. A changed file aborts the continuation.

The original `--output` path is still only a naming prefix in retention
mode. Snapshots remain exclusive-create and are never silently overwritten.

## Verify root-to-tip linkage and held-out metrics

```sh
python audit_training_history.py --demo --data-dir data/tiny-demo \
  --checkpoint checkpoints/tiny-step00000001.pt \
  --checkpoint checkpoints/tiny-step00000002.pt \
  --checkpoint checkpoints/tiny-step00000003.pt \
  --checkpoint checkpoints/tiny-step00000004.pt \
  --require-lineage --output reports/tiny-linked-history.json
```

The optional `--require-lineage` enforces:

1. At least two and at most 32 unique checkpoints, in ascending step order.
2. The **first supplied snapshot is the root**, with no parent.
3. Every subsequent snapshot has the exact previous supplied snapshot's SHA-256
   and step and the identical run ID.
4. Existing M29/M30 checks still apply: every checkpoint must independently
   match the held-out dataset, tokenizer, model configuration, hyperparameters
   and exact validation-token coverage.

The report labels `lineage_check: validated` and records the run ID. If the
flag is omitted, old behavior remains and the result explicitly says
`lineage_check: not_requested`.

If a checkpoint is missing in the middle, supply the entire chain rather than
skipping it. The history auditor requires a root and cannot validate a partial
suffix or an alternative untracked branch.

## Legacy and resume behavior

Format-v2 checkpoints with no `lineage` field remain valid for generation,
evaluation, legacy resume and normal replace-latest training. Resuming from one
of these while **enabling retained snapshots** starts a *new* root at the
first checkpoint saved after resumption and prints a warning. Do not interpret
this root as evidence linking earlier legacy files.

A linked chain can be resumed and appended with higher `--steps`; neither the
source checkpoint nor earlier snapshots are rewritten.

## Limits and threats

- **SHA links are not signatures.** Someone who can rewrite the checkpoint
  files can create a consistent fake chain, new run IDs and fake optimizer
  states. Independent trusted timestamps, externally anchored digests, signed
  provenance and controlled training logs are outside M31.
- The chain records the previous **saved** step, not each intervening optimizer
  update; it cannot prove uninterrupted computation between saves.
- Re-evaluated held-out NLL on identical targets measures only a narrow
  teacher-forced property, not accuracy, instruction following, safety or
  general intelligence. Selection on validation data must not be represented
  as untouched test-set performance.
- This feature does **not** provide a pretrained model, approved corpus,
  verified hardware, physical-device performance or production readiness.
- Arbitrary checkpoint files are untrusted inputs. Use only locally trusted
  artifacts; `weights_only=True` reduces, but does not eliminate, risk.
