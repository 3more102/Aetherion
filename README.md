# Aetherion

**Aetherion** is an open-source experimental language-model project. Our goal is to understand, implement, train, evaluate, and eventually serve an original decoder-only Transformer.

> **Status: Phase 0 / engineering prototype.** This repository starts with a small model using randomly initialized weights and supports byte-level or experimental BPE tokenization. It is **not** a competitive general-purpose assistant, and no frontier-level capabilities are claimed.

## Project tracks

| Track | Goal | State |
|---|---|---|
| Aetherion-Micro | Reproducible train/generate/test loop on CPU or GPU | Phase 0 |
| Aetherion-125M | Trained tokenizer, clean data, ~125M parameter baseline | Planned |
| Aetherion-Instruct | Instruction tuning, evaluation, model card | Planned |
| Aetherion-Scale | Distributed training and larger model feasibility | Research |

## Quick start

Python 3.10+ and PyTorch:

```bash
git clone https://github.com/3more102/Aetherion.git
cd Aetherion
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
python train.py --steps 5 --batch-size 2 --d-model 32 --layers 2 --heads 4 --block-size 32
python generate.py --prompt "Hello" --tokens 30
```

The sample corpus is synthetic demonstration text, not sufficient to build a chat assistant. Training a strong model from scratch requires vastly more data and compute.

## M1: experimental corpus and BPE tokenizer

Provide a UTF-8 JSONL file with one document per line, with required string fields `id`, `text`, `source`, and `license`. Optional fields: `language` and `group_id` (for keeping related documents in the same split). For example, a **test-only** record:

```json
{"id":"sample-001","text":"Hello, مرحبا!","source":"local test fixture","license":"test-only","language":"mixed","group_id":"fixture-1"}
```

Run preparation and tokenizer training:

```bash
python -m aetherion.data --input data/raw/corpus.jsonl --output-dir data/processed --seed 42
python -m aetherion.tokenizer --train data/processed/train.jsonl --output artifacts/tokenizer.json --vocab-size 32000
```

The tokenizer command now writes both `tokenizer.json` and `tokenizer.meta.json` (training-split and tokenizer SHA-256 hashes). Use the prepared **document-separated validation split** to train the Micro model with token IDs:

```bash
python train.py --data data/processed/train.jsonl --val-data data/processed/validation.jsonl \
  --tokenizer artifacts/tokenizer.json --steps 100 --output checkpoints/micro-bpe.pt
python generate.py --checkpoint checkpoints/micro-bpe.pt \
  --tokenizer artifacts/tokenizer.json --prompt "Hello" --tokens 30
```

BPE training fails if the split checksums differ from `data/processed/manifest.json`, or the tokenizer sidecar was fitted on a different training split. Generation verifies the checkpoint's tokenizer SHA-256 and vocabulary size. The old byte-only `train.py`/`generate.py` commands remain supported; byte mode's default last-10%-of-bytes validation is **demo-only**. BPE validation is document-disjoint according to the prepared split groups, but language quality remains unmeasured.

See [BPE integration notes](docs/BPE_INTEGRATION.md) for invariants and limitations.

**Important limitations:** only exact normalized-text deduplication is performed. Near-duplicate removal, PII/secret scanning, actual license verification, and evaluation-set contamination screening must be completed before real model training. Data labels alone do not prove permission. No private or unlicensed corpora should be committed.

## Architecture

Input bytes → token embeddings → N × (RMSNorm → causal attention + RoPE → residual → RMSNorm → SwiGLU → residual) → RMSNorm → tied LM head → next-token loss.

The research target for Aetherion-125M is **14 layers, width 768, 12 attention heads, 32k vocabulary, 2048 context and SwiGLU**, estimated around **124M parameters** with tied embeddings. This is a target, not implemented or trained in the Micro prototype.

## Documentation

- [Detailed execution plan](docs/EXECUTION_PLAN.md)
- [Engineering roadmap](docs/ROADMAP.md)
- [Architecture and design choices](docs/ARCHITECTURE.md)
- [Dataset and evaluation policy](docs/DATA_AND_EVALUATION.md)

## Principles

1. Reproducible tests, pinned experiments, transparent limitations.
2. No claim of pretrained or evaluated capability without measured evidence.
3. Use data with appropriate licenses and provenance; respect privacy, copyright, and opt-outs.
4. Separate pretraining, instruction tuning, inference and evaluation.
5. Publish safety evaluations and limitations before any serious deployment.

## References

- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Hugging Face: train a causal language model from scratch](https://huggingface.co/docs/course/chapter7/6)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/main/quicktour)

## License

MIT (source code). Datasets and later model weights require independent licensing decisions.
