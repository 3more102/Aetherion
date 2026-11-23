# M36 — Checkpoint destination and free-space preflight

**Status:** Implemented for the guarded 125M `pilot_train.py --execute` launcher.
It is a point-in-time storage safety check; **no 125M training has been executed**.

## Problem

A multi-hour training run can otherwise fail when the checkpoint volume is
full, or overwrite an earlier non-retained checkpoint. M35 limits planned
checkpoint *write operations*, not available storage bytes.

## Execution requirements

Every `--execute` invocation now needs a positive
`--min-free-checkpoint-gib` budget chosen by the operator. Immediately before
starting the trainer, M36 checks free bytes on the nearest existing parent
filesystem for `--output` and refuses to execute when below the floor.

For ordinary (non-retained) checkpoint outputs, existing files are rejected by
default. Operators may explicitly opt in to replacement with
`--allow-overwrite-output`. This is destructive and should be used only when
the checkpoint has been backed up and replacement is intended.

For `--retain-step-checkpoints`, M36 scans preexisting names matching the
trainer's scheduled `<stem>-stepXXXXXXXX<suffix>` outputs, including the
mandatory final snapshot, and blocks collisions. It never permits the
overwrite flag in retained mode. It checks only the planned full-target
schedule: on a resumed run it may conservatively reject a previous snapshot
that the trainer would not actually write again.

The preflight refuses symlink or directory targets, malformed budgets,
unreadable free-space statistics, insufficient space and the wrong output
parent type. The direct `train.py` command retains its previous behavior;
these safeguards are specific to the reviewed 125M pilot launcher.

## Example

After obtaining **your own authorized and reviewed corpus** and a genuinely
measured matching M28 hardware benchmark on your machine:

```sh
python pilot_train.py --data-dir data/processed/v1 \
  --tokenizer artifacts/tokenizer.json --review /private/review.json \
  --allow-license CC0-1.0 --steps 20 --schedule-steps 100 \
  --context 256 --batch-size 1 --accumulation 4 \
  --device cuda:0 --precision bf16 --output checkpoints/pilot.pt \
  --benchmark-report reports/125m-cuda0.json \
  --max-peak-reserved-gib 12 --max-total-tokens 20480 \
  --max-checkpoint-writes 1 --min-free-checkpoint-gib 2 \
  --execute
```

The 2 GiB floor and 12 GiB accelerator peak ceiling are **examples only**.
Neither is a scientifically derived requirement for this model or a guarantee
that the checkpoint fits.

## Verification

```sh
python -m pytest -q tests/test_m36_checkpoint_destination.py
python -m pytest -q
```

The standalone disk checker has 18 CPU-only regression cases, with two
additional launcher fail-closed checks. CI should verify the full Python test
suite and existing mobile ONNX export/inference compatibility.

## Limitations

- Disk is checked at one instant; no capacity is reserved or locked.
- A different mount or filesystem can appear after the check.
- Other processes can change paths or consume space between check and save.
- The floor is operator-supplied, **not** an estimate of actual checkpoint size,
  number of bytes eventually written, total disk cost or total required storage.
- A successful preflight does **not** prove permission to use data, quality of
  generated text, actual model training, or a real accelerator benchmark.
