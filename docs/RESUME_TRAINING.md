# Resuming Aetherion-Micro training

The Micro byte-token trainer now saves atomically and supports restart from an
optimizer-and-RNG checkpoint. Use authorized, source-isolated train and validation
files for anything beyond the synthetic smoke test.

```bash
python train.py --steps 50 --save-every 10 --output checkpoints/micro.pt
python train.py --steps 100 --save-every 10 --resume checkpoints/micro.pt --output checkpoints/micro.pt
```

`--steps` is the **total** optimizer step target, not the number of additional steps.
Retain identical `--data`, `--val-data`, model dimensions, `--lr`,
`--batch-size`, and `--seed` when resuming. Use the same optional arguments on
both commands (the example relies on their defaults).

The checkpoint contains model weights, AdamW optimizer moments, processed step,
CPU/CUDA torch RNG, dedicated batch-sampler RNG, configuration, data SHA-256
fingerprints and training hyperparameters. Writes use a same-directory temporary
file and atomic rename. Checkpoints remain compatible with `generate.py`.

Validation uses a step-specific random seed, so validation performed at an
earlier interruption boundary does not perturb subsequent training batches.

## Verification

```bash
python -m pytest -q tests/test_resume.py
```

The regression compares full CPU training with an interrupted/resumed run,
checks optimizer state and detects changed input datasets. It does **not**
establish bitwise reproduction across different devices, PyTorch versions,
CUDA topology or hardware. A checkpoint created by the original Micro trainer
lacks optimizer/RNG state and cannot be resumed (but can still be used for
inference). Byte-token Micro is not a production model or a GPT-class assistant.
