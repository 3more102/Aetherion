# Aetherion Micro: BPE training and generation

This experimental path integrates the existing ByteLevel BPE tokenizer with
Aetherion-Micro. **It is a pipeline verification tool, not a trained assistant.**

## Reproducible workflow

Prepare a corpus you have verified you are allowed to use, following
[the data-pipeline guide](EXECUTION_PLAN.md). The corpus preparer keeps related
`group_id` values together when making splits.

```bash
python -m aetherion.data --input data/raw/corpus.jsonl --output-dir data/processed --seed 42
python -m aetherion.tokenizer --train data/processed/train.jsonl --output artifacts/tokenizer.json --vocab-size 32000
python train.py --data data/processed/train.jsonl --val-data data/processed/validation.jsonl \
  --tokenizer artifacts/tokenizer.json --steps 5 --batch-size 2 \
  --block-size 32 --d-model 32 --layers 2 --heads 4 \
  --output checkpoints/bpe-micro.pt
python generate.py --checkpoint checkpoints/bpe-micro.pt \
  --tokenizer artifacts/tokenizer.json --prompt "Hello مرحبا" --tokens 16
python -m pytest -q
```

**What is guaranteed by the code:**

- BPE fitting uses `train.jsonl` and not the held-out documents.
- Training accepts separate `--data` and `--val-data` inputs; BPE requires a
  separate validation file and rejects identical input bytes/paths.
- A JSONL token stream encodes **only** its `text` values, adding the tokenizer's
  `<|eos|>` after each document. This is a concatenated token stream, not
  document-masked attention.
- Model `vocab_size` is read from the BPE tokenizer.
- Checkpoints store tokenizer SHA-256, train/val file SHA-256, and token counts.
  Inference rejects a missing or differently hashed tokenizer.
- Legacy plain-text 256-byte training/checkpoints remain supported. Raw JSONL
  is rejected in byte mode to avoid learning dataset metadata accidentally.

**Limits and next gates:** These scripts materialize token streams in memory,
do not resume optimizer state, and do not measure serious training capability.
Exact deduplication and group splits are not enough: verify licenses, conduct
PII/secret and near-duplicate screening, and check benchmark contamination
before training on real corpora. A 125M pretraining run requires a measured
hardware/budget pilot. Use the training split only to fit tokenizer merges.

Do not commit private corpora, credentials, model weights or tokenizer artifacts
unless distribution rights are independently established.
