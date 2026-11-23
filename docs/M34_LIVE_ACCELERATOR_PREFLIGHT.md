# M34 — Live accelerator identity and memory admission check

**State:** pre-execution safeguard only. No real GPU measurement, 125M
training, trained model weights, or mobile inference result is produced by CI.

## Background

The M28 gate validates a *user-supplied* synthetic optimizer benchmark for
the intended 125M configuration. Before M34, a benchmark claiming `cuda:0`
could be accepted even if that device was no longer present, had changed,
or lacked free memory when the training process was started.

## Behavior

With `pilot_train.py --execute`, the existing hardware report, dataset
integrity, rights review, contamination screen and tokenizer gates run
first. **Immediately before spawning the trainer**, M34 then:

1. Checks that the explicitly indexed CUDA/XPU device is available.
2. Reads its current device name and requires it to equal the hardware name
   in the M28-validated benchmark summary.
3. Reads the accelerator's current free/total memory and device capacity.
4. Checks that recorded synthetic peak reserved memory does not exceed
   device capacity, and currently free memory is **at least 125%** of that
   recorded peak reserved memory.
5. Rejects missing device-inspection APIs, inconsistent memory accounting
   and changed device identity instead of launching the trainer.

The passing snapshot appears as `live_accelerator_preflight` beside the
original M28 `hardware_preflight` report on stdout. The existing
dry-run path (without `--execute`) never touches a GPU and does not
require a hardware benchmark.

For example, run an M28 benchmark on the **same indexed accelerator**,
then use the normal approved launch command:

```sh
python benchmark_training.py --preset 125m --device cuda:0 \
  --precision bf16 --context 256 --batch-size 1 --accumulation 4 \
  --warmup 1 --steps 3 --output reports/125m-cuda0.json

python pilot_train.py --data-dir data/processed/v1 \
  --tokenizer artifacts/tokenizer.json --review /private/review.json \
  --allow-license CC0-1.0 --steps 20 --schedule-steps 100 \
  --context 256 --batch-size 1 --accumulation 4 \
  --device cuda:0 --precision bf16 --output checkpoints/pilot.pt \
  --benchmark-report reports/125m-cuda0.json \
  --max-peak-reserved-gib 12 \
  --max-total-tokens 20480 --max-checkpoint-writes 1 \\
  --min-free-checkpoint-gib 2 --execute
```

Paths and the 12 GiB ceiling are examples only. They are **not** a
recommendation to launch costly training without reviewed data and approval.

## Test

```sh
python -m pytest -q tests/test_m34_live_device_gate.py \
    tests/test_m28_hardware_gate.py
python -m pytest -q
```

The new module has 12 standalone CPU-mock regression tests. Existing
M28 tests cover the launcher ordering and require M34 verification before
the training subprocess. Full CI must pass before merging.

## Limitations

- Live free memory is a **point-in-time observation**, not reserved capacity.
- The 25% headroom multiplier is a heuristic, not a proven safety margin.
  A long or differently shaped training run may still OOM or thrash.
- The M28 benchmark remains **self-reported and unauthenticated**; matching
  a device name does not prove the benchmark ran on that device.
- Memory APIs can be unavailable on some PyTorch/driver combinations;
  the gate deliberately refuses execution in that case.
- GPU scheduling, other processes, thermal stability, network failures,
  held-out evaluation and rights compliance need separate controls.
