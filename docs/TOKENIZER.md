# Aetherion BPE tokenizer (M1)

Aetherion supports byte-level BPE tokenizer training and **verified BPE token IDs** in the Micro trainer and generator. Legacy raw-byte Micro checkpoints remain supported. No pretrained assistant capability is implied.

## Prepare authorized inputs

Use the dataset preparation CLI first, after separately verifying the rights,
privacy and quality of the source text:

```bash
python prepare_data.py /path/to/authorized.jsonl --output-dir data/processed/v1
```

This writes `train.jsonl`, `validation.jsonl`, `test.jsonl` and
`manifest.json`. The tokenizer must fit **only** on `train.jsonl`.
The tokenizer command refuses to proceed if the training file checksum differs
from `manifest.json`, if a different split filename is supplied, or if
the output would overwrite existing artifacts.

## Train and inspect

```bash
pip install -r requirements.txt
python -m aetherion.tokenizer \
  --train data/processed/v1/train.jsonl \
  --output artifacts/tokenizer.json \
  --vocab-size 32000 --min-frequency 2
```

Outputs:

- `artifacts/tokenizer.json`: Hugging Face Tokenizers serialized model,
  including 256-byte alphabet and special tokens.
- `artifacts/tokenizer.meta.json`: tokenizer version, train and manifest
  SHA-256, tokenizer SHA-256, actual and requested vocabulary sizes, document
  count, special tokens and training configuration.

Reserved tokens: `<|unk|>`, `<|bos|>`, `<|eos|>`. No chat template is
defined yet. On a small corpus, the learned vocabulary can be smaller than the
requested 32,000 tokens.

## Inspect a round-trip

```python
from tokenizers import Tokenizer
t = Tokenizer.from_file("artifacts/tokenizer.json")
sample = "Hello, مرحبا 🌍"
ids = t.encode(sample).ids
assert t.decode(ids) == sample
print(ids)
```

For deterministic artifacts, hold training input bytes, configuration, Python
environment and the pinned `tokenizers` package version constant. The model's
future vocabulary size must exactly match the trained tokenizer's *actual*
vocabulary size, and future checkpoints must record and verify its SHA-256.

The manifest verifies data integrity **only**; it does not independently verify
copyright permission, privacy screening, near-duplicate removal, benchmark
contamination, or suitability for pretraining.

## Use BPE for model training and generation

The trainer now supports `--tokenizer` pointing to the existing `tokenizer.json` artifact. Its `.meta.json` is verified at load time, and the trainer requires matching prepared text files and split manifest. The checkpoint stores the exact tokenizer SHA-256 and actual vocabulary size. The generator refuses a missing or mismatched tokenizer.

```bash
python train.py --data data/processed/v1/train.txt \
  --val-data data/processed/v1/validation.txt \
  --tokenizer artifacts/tokenizer.json \
  --steps 100 --batch-size 2 --block-size 32 \
  --save-every 20 --output checkpoints/bpe-micro.pt

python generate.py --checkpoint checkpoints/bpe-micro.pt \
  --tokenizer artifacts/tokenizer.json --prompt "Hello مرحبا" --tokens 20
```

To resume, pass the exact same dataset and tokenizer with `--resume checkpoints/bpe-micro.pt` and increase the total `--steps` value. Existing byte-only checkpoints still work without a tokenizer. Changing the tokenizer, source files, or manifest invalidates checkpoint resume.

**Scope:** This is still a tiny engineering pipeline. The prepared `.txt` streams use newline boundaries, not explicit per-document BOS/EOS packing. Real pretraining needs token packing, held-out evaluation, licensed large-scale data and measured hardware feasibility.
