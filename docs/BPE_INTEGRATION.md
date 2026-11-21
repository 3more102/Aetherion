# BPE integration (experimental, Micro only)

The `feat/bpe-train-generate-integration` branch is stacked on the source-grouped
data/BPE work in PR #11. The new pipeline converts prepared documents to token IDs
while preserving the original UTF-8 byte training mode.

## Run with authorized local data

```bash
python -m aetherion.data --input data/raw/corpus.jsonl --output-dir data/processed --seed 42
python -m aetherion.tokenizer --train data/processed/train.jsonl --output artifacts/tokenizer.json --vocab-size 32000
python train.py --data data/processed/train.jsonl --val-data data/processed/validation.jsonl --tokenizer artifacts/tokenizer.json --steps 100 --output checkpoints/micro-bpe.pt
python generate.py --checkpoint checkpoints/micro-bpe.pt --tokenizer artifacts/tokenizer.json --prompt "Hello" --tokens 50
```

These commands are a wiring smoke test, **not** a large-scale training recipe.
The model uses randomly initialized parameters until training completes.

## Validation invariants

- Input names are exactly `train.jsonl` and `validation.jsonl` in the **same**
  prepared data directory; validation must be passed explicitly in BPE mode.
- Both split checksums must match `manifest.json`; identical train and validation
  checksums are refused. The manifest must come from corpus preparation.
- Tokenizer training consumes only `train.jsonl`. The tokenizer command emits
  `tokenizer.meta.json` recording the SHA-256 of the training split, tokenizer
  artifact and vocabulary size. Modifying either file invalidates the check.
- Document boundaries are marked with `<|eos|>` between each encoded text.
- Checkpoints contain tokenizer mode, vocabulary, tokenizer SHA-256, split
  checksums and manifest checksum. Generation refuses BPE inference without
  the matching tokenizer. Old checkpoints without tokenizer metadata remain
  readable as byte checkpoints.
- The tokenizer and data checks use hashes to detect mismatches; **they do not**
  authenticate the author of the files or grant rights to use their content.

## Limitations / next work

The BPE loader currently materializes all tokens in RAM. It is suitable for a
Micro test and **must** be replaced with memory-mapped, packed token shards and
a resumable sharded dataloader before larger-scale training. The trainer does
not yet save optimizer state, scheduler state, sampler position or RNG states.
Evaluation still samples validation batches rather than calculating a full
held-out aggregate. There is no training/evaluation pipeline for the planned
Aetherion-125M model yet.

The split preparer only performs normalized exact duplicate removal and
source/group separation where group IDs correctly capture related documents.
Near-duplicate, personal-information, secret, license, source-authorization
and benchmark-contamination reviews remain independent prerequisites.
Synthetic demo metrics cannot establish assistant quality.

## Verification

```bash
python -m pytest -q
python -m compileall -q aetherion train.py generate.py
```

The integration regression test trains a tiny CPU BPE model for one step,
generates from its checkpoint and rejects modified tokenizer/split artifacts.
CI results must be inspected at the exact head commit; a local test pass is
not evidence of an actually trained production model.
