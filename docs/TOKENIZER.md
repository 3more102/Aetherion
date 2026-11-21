# Aetherion BPE tokenizer (M1)

Aetherion supports a **separate experimental byte-level BPE tokenizer trainer**.
The current Micro model is still a 256-byte-token demo. BPE token IDs are **not**
yet consumed by `train.py` or `generate.py`; using BPE artifacts with a
byte-only checkpoint would be invalid.

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
