# Resumable Aetherion-Micro training

This stage adds a **versioned, atomic training checkpoint** with:

- Model and AdamW optimizer weights/state.
- Step number, model config, seed, data SHA-256 values, and training hyperparameters.
- Torch CPU RNG state, plus CUDA/MPS RNG state where applicable.
- Strict rejection of changed datasets, model settings, optimizer-affecting settings, or device type.

Run with two separately prepared data splits (see `docs/DATA_PIPELINE.md`):

```bash
python train.py \
  --data data/processed/v1/train.txt \
  --val-data data/processed/v1/validation.txt \
  --steps 200 --save-every 50 \
  --output checkpoints/micro-v1.pt
```

Continue the same run to **500 total steps** using the **same hyperparameters and exact same data files**:

```bash
python train.py \
  --data data/processed/v1/train.txt \
  --val-data data/processed/v1/validation.txt \
  --steps 500 --save-every 50 \
  --resume checkpoints/micro-v1.pt \
  --output checkpoints/micro-v1.pt
```

`--steps` always specifies the **total target step**, not additional steps.
`--save-every 0` writes only at the final step. A final-step checkpoint
can also be resumed, because validation batches do not consume the training
minibatch RNG stream. The file is written to a temporary sibling path then
atomically replaced, avoiding partially written checkpoint files.

The standard `generate.py` loader continues to work with these files using
the `state_dict` and `config` entries.

## Verification

```bash
python -m pytest -q tests/test_resume.py
```

The test compares a continuous six-step CPU run with a three-step run
followed by a resume to six steps: model tensors, AdamW state, and CPU RNG
state must match exactly. It also tests rejection of altered data and
inference-only/legacy checkpoints.

## Limits

- Exact bitwise continuation is tested on **CPU** with a fixed PyTorch build,
  model, training configuration, and dataset. GPU kernels, nondeterministic
  operations, hardware changes, or changing framework versions may prevent
  bitwise reproducibility.
- A prior model-only checkpoint cannot resume training; it can still be
  used for inference.
- Checkpoints are for trusted files; do not load untrusted checkpoints.
- This is a Micro-prototype reliability improvement, **not** a pretrained
  assistant or evidence of production readiness.
