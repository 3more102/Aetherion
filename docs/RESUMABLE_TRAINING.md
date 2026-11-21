# Resumable Aetherion-Micro training

Aetherion-Micro can save and resume **version-2** training checkpoints. These contain the model weights, optimizer state, current optimizer step, source-data SHA-256 hashes, and CPU/sampler RNG states (plus CUDA or MPS RNG state on the active accelerator). Checkpoints are replaced atomically.

## Train, interrupt, and resume

Use authorized, separately prepared training and validation files. For example:

```bash
python train.py --data data/processed/v1/train.txt --val-data data/processed/v1/validation.txt \
  --steps 100 --batch-size 8 --block-size 128 --d-model 128 --layers 4 --heads 4 \
  --seed 42 --device cpu --save-every 25 --output checkpoints/experiment.pt

python train.py --data data/processed/v1/train.txt --val-data data/processed/v1/validation.txt \
  --steps 200 --batch-size 8 --block-size 128 --d-model 128 --layers 4 --heads 4 \
  --seed 42 --device cpu --save-every 25 \
  --resume checkpoints/experiment.pt --output checkpoints/experiment.pt
```

`--steps` is the **total target optimizer step**, not the number of additional steps. The resumed run starts at the saved step plus one. `--save-every 25` saves progress every 25 optimizer steps; the final step is always saved. `--save-every 0` writes only the final checkpoint. Each save atomically replaces the output; use distinct paths when you need to retain past versions.

The architecture, seed, device, batch size, learning rate, and exact data hashes/split mode must match the checkpoint. A data/hyperparameter mismatch raises an error rather than silently continuing on a new experiment. Use the same explicit validation file to avoid the unsafe demo-only raw byte split.

## Reproducibility and compatibility

- Validation and training use separate CPU sampling RNG streams so evaluating at a partial run's final step does not change subsequent training batches.
- The regression test compares an uninterrupted six-step CPU run with three steps followed by three resumed steps, including exact model parameters and AdamW state. This is **not** a claim of identical results across different CUDA hardware, kernel algorithms or library versions.
- `generate.py` continues to load model/config keys from both old and new checkpoints with `weights_only=True`.
- Previous inference-only checkpoints do not contain optimizer/RNG state and **cannot** be resumed.
- This baseline uses fixed-rate AdamW. LR warmup/cosine schedule, AMP/grad accumulation and larger-model benchmarks are future work.
- Version-2 checkpoints are PyTorch files; load only trusted checkpoints and protect corpus rights/privacy separately.
