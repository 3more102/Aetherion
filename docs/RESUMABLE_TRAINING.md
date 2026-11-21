# Resumable Micro training (M2)

This is a training-system capability, **not** a pretrained language-model release.

## Run

Start with distinct prepared training and validation data:

```bash
python train.py --data data/processed/v1/train.txt --val-data data/processed/v1/validation.txt \
  --steps 100 --batch-size 8 --block-size 128 --save-every 25 --output checkpoints/micro.pt
```

Continue to **200 total optimization steps** (not 200 more):

```bash
python train.py --data data/processed/v1/train.txt --val-data data/processed/v1/validation.txt \
  --steps 200 --batch-size 8 --block-size 128 --save-every 25 \
  --resume checkpoints/micro.pt --output checkpoints/micro.pt
```

The checkpoint includes model weights, AdamW optimizer state, completed step,
CPU/device RNG state, config, training settings, and SHA-256 fingerprints for
both dataset splits. `--save-every 0` (default) writes only at the end; positive
values also save at that interval. Checkpoints are replaced atomically.

For a reproducible continuation, use the **same model flags, batch size,
learning rate, evaluation interval, seed, byte streams, and device type**.
The trainer rejects mismatches. CLI `--steps` must exceed the saved step.
The earlier model-only checkpoints remain valid for `generate.py`, but cannot
resume training because optimizer/RNG state is missing.

Validation sampling uses a step-specific RNG independent of training samples,
so short runs performing an extra final-step evaluation do not change the
training sequence. Tests confirm bitwise equality between interrupted and
uninterrupted CPU runs on the same PyTorch build. Cross-hardware or different
PyTorch/CUDA versions are **not** guaranteed bitwise reproducible.

**Limitations:** No automatic learning-rate schedule, gradient accumulation,
mixed precision, multi-GPU sharding, tokenizer integration, or dataset-rights
verification is implemented here. `torch.load(weights_only=True)` is used for
checkpoint loading; only load model artifacts from trusted sources.
