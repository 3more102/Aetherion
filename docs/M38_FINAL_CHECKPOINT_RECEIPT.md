# M38 — Verify the final checkpoint before reporting a guarded 125M pilot complete

**Status:** Engineering safeguard; no approved corpus, 125M training results,
pretrained assistant weights, or physical GPU measurements are included.

## New completion gate

The guarded `pilot_train.py --execute` launcher now treats a successful
`train.py` process exit as **necessary, but insufficient**. With the M37
exclusive output lock still held, it checks the expected final checkpoint:

- Correct filename (`output.pt` or `output-stepXXXXXXXX.pt`).
- Non-symlink, readable format-v2 resumable checkpoint at the exact target step.
- 125M decoder configuration, tokenizer mode/vocabulary, seed, and the
  trainer's expected optimizer hyperparameters.
- Exact expected state-dict keys, tensor dimensions, FP32 parameter values'
  **dtype**, and tied embedding/head storage.
- Nonempty optimizer state and saved training RNG states.
- Data, manifest and tokenizer SHA-256 fingerprints against current files.
- Checkpoint SHA-256 stability before and after inspecting the artifact.

If any condition fails, the launcher reports failure rather than declaring
successful checkpoint verification. It does **not** delete an incomplete
checkpoint; preserve it for diagnosis.

The check re-hashes the final checkpoint twice. Large artifacts and corpora
make this a potentially slow, storage-intensive verification pass. The
checkpoint is opened with PyTorch `weights_only=True, mmap=True` to avoid an
eager copy of all mapped tensors, but checkpoints should still be from trusted
local runs.

## Optional permanent receipt

Append `--completion-report reports/pilot-125m-completion.json` to the
approved M35/M36 launch command. The JSON file is atomically created without
overwriting any previous receipt; an existing report destination is rejected
before training. Without this option, verification still runs and prints the
receipt to stdout when successful.

The receipt includes the final step, checkpoint path/size/SHA-256, exact model
config, parameter count from expected tensor shapes and the train,
validation, manifest, and tokenizer digests. Keep receipts and checkpoints
together; the receipt cannot reconstruct missing model weights.

The CLI's other mandatory approval, corpus, synthetic benchmark, GPU memory,
token budget, disk floor and output-lock checks remain in effect.

## Verification

```bash
python -m pytest -q tests/test_m38_pilot_completion.py
python -m pytest -q
```

CPU-only fixtures test the checker, checkpoint mutations, held-out data
changes, retained snapshot path selection, no-overwrite receipt semantics and
simulated launcher success/failure behavior. All Python and mobile ONNX CI
checks must pass before merge.

## Limitations

The receipt checks **structural and hash consistency**, not genuine SGD
history, correct math, trustworthy self-reported metadata, legal data rights,
held-out generalization, safety, mobile speed, or conversational intelligence.
Without a separately anchored signature, both a checkpoint and its receipt
may be jointly modified by someone controlling the files. **Never use a
successful M38 result as a claim of a trained or useful 125M assistant.**
