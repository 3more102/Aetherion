# Aetherion v1: reversible byte-level BPE

**Status:** engineering feature; 32,000 vocabulary entries are a target, not guaranteed on small input corpora. This trains a tokenizer, not a language model.

## Data rights and prerequisites

Only use documents that you are authorized to train on. The data preparer performs exact deduplication and source-grouped splits, but it does **not** independently verify licensing, remove near duplicates, detect PII/secrets or decontaminate benchmarks. See [data preparation](DATA_PIPELINE.md).

Install from the project root (Python 3.10+):

```bash
pip install -r requirements.txt
```

## End-to-end commands

1. Write your own UTF-8 `data/raw/authorized.jsonl` with at least three distinct `source`/`group` values and document IDs, `license`, `authorized: true` and `text`.

2. Prepare isolated splits, train BPE, train Micro and sample text:

```bash
python prepare_data.py data/raw/authorized.jsonl --output-dir data/processed/v1 --seed 42
python train_tokenizer.py --data-dir data/processed/v1 --output-dir artifacts/tokenizer-v1 --vocab-size 32000 --min-frequency 2
python train.py --data data/processed/v1/train.txt --val-data data/processed/v1/validation.txt --tokenizer-dir artifacts/tokenizer-v1 --steps 100 --batch-size 2 --block-size 64 --output checkpoints/bpe-micro.pt
python generate.py --checkpoint checkpoints/bpe-micro.pt --tokenizer-dir artifacts/tokenizer-v1 --prompt "Hello مرحبا" --tokens 20
```

**Do not treat generated output from 100 steps of tiny Micro training as meaningful assistant behavior.** It is only a pipeline integration test.

## Artifact checks

Training is restricted to checksum-verified `train.jsonl` from the prepared dataset manifest; validation and test documents are **not** used to fit BPE. The tokenizer does not apply Unicode normalization so `decode(encode(text))` should preserve input text, including Arabic, emoji, newlines and spaces. Prepared data itself is normalized separately by `prepare_data.py`.

Artifacts in `artifacts/tokenizer-v1/`:

- `tokenizer.json`: serialized trained BPE tokenizer.
- `tokenizer_manifest.json`: SHA-256 of tokenizer, train data and prepared manifest; tokenizers dependency version; requested/actual vocabulary sizes; special token IDs; pipeline settings and known limitations.

The four special tokens are `<|pad|>`, `<|bos|>`, `<|eos|>`, `<|unk|>` with IDs 0–3. Training does **not** automatically insert BOS/EOS around each document yet; a later token packing milestone must define sequence boundaries before extended pretraining.

`train.py --tokenizer-dir` verifies both the source prepared manifest and the tokenizer's source hashes. The checkpoint stores tokenizer SHA-256 and vocabulary size; `generate.py` refuses missing or mismatched tokenizer artifacts. Existing 256-byte Micro checkpoints remain supported without `--tokenizer-dir`.

### Determinism and reproducibility

- Train data iterator preserves document order; full 256-byte BPE alphabet is explicitly sorted.
- The dependency is pinned (`tokenizers==0.22.1`); the artifact records its version and checksum.
- The *exact* tokenizer artifact must be retained for future inference. Re-training with different corpora, ordering, library versions, or configuration creates a different model vocabulary.
- Two consecutive tiny-data training runs are compared in automated tests. This does not guarantee identical output on every future platform without verification.

## Future improvements

- Document-level BOS/EOS packing and scalable, memory-mapped token shards.
- Train-split-only near duplicate, PII, secret and benchmark contamination checks.
- Configurable language balancing, subword fertility and multilingual tokenization evaluation.
- Resumable large-scale pretraining with an independent held-out evaluation set.

## Technical references

- [Hugging Face: BPE tokenizer from scratch](https://huggingface.co/docs/course/chapter6/8)
- [Hugging Face: training tokenizers from iterators](https://huggingface.co/docs/tokenizers/main/training_from_memory)
