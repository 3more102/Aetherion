# Aetherion

**Aetherion** is an open-source experimental language-model project. Our goal is to understand, implement, train, evaluate, and eventually serve an original decoder-only Transformer.

> **Status: engineering prototype.** Training, checkpointing, instruction tuning, cached inference, evaluation and experimental DDP/XPU paths are implemented and tested, but no large pretrained assistant weights or frontier-level capabilities are claimed. Synthetic tests are not evidence of useful general-purpose intelligence.

## Project tracks

| Track | Goal | State |
|---|---|---|
| Aetherion-Micro | Reproducible training/generation and learning-signal tests | Implemented (synthetic validation) |
| Aetherion-125M | Reviewed corpus, 32k BPE, hardware pilot and measured held-out metrics | Gated pilot tooling; untrained |
| Aetherion-Instruct | Response-only SFT and local KV-cached chat | Implemented tooling; no verified assistant checkpoint |
| Aetherion-Scale | CPU/CUDA DDP prototype, activation checkpointing, size presets | Implemented infrastructure; large models untrained |

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

## Reproducible dataset preparation (new)

Provide **your own authorized**, source-identified JSONL documents (schema in [Data pipeline](docs/DATA_PIPELINE.md)). The preparer creates separate train/validation/test files, exact-deduplication reports, and SHA-256 manifests. Dataset licenses, privacy and near-duplicates still require independent review.

```bash
python prepare_data.py /path/to/authorized.jsonl --output-dir data/processed/v1
python train.py --data data/processed/v1/train.txt --val-data data/processed/v1/validation.txt --steps 100 --output checkpoints/experiment-v1.pt
```

## Experimental BPE tokenizer (M1)

After preparing a licensed/authorized corpus, fit a byte-level BPE tokenizer on **only** the verified training split:

```bash
python -m aetherion.tokenizer --train data/processed/v1/train.jsonl \
  --output artifacts/tokenizer.json --vocab-size 32000
```

The tokenizer and its metadata record SHA-256 checksums, a pinned library version and special tokens. See [Tokenizer guide](docs/TOKENIZER.md). **It is now supported by the Micro trainer and generator** through a verified tokenizer artifact. Legacy byte-level checkpoints remain compatible.

```bash
python train.py --data data/processed/v1/train.txt --val-data data/processed/v1/validation.txt \
  --tokenizer artifacts/tokenizer.json --steps 100 --output checkpoints/bpe-micro.pt
python generate.py --checkpoint checkpoints/bpe-micro.pt --tokenizer artifacts/tokenizer.json --prompt "Hello مرحبا" --tokens 20
```

Checkpoints store the tokenizer fingerprint and support exact optimizer/RNG resume via `--resume`.

## Architecture

Input bytes → token embeddings → N × (RMSNorm → causal attention + RoPE → residual → RMSNorm → SwiGLU → residual) → RMSNorm → tied LM head → next-token loss.

The research target for Aetherion-125M is **14 layers, width 768, 12 attention heads, 32k vocabulary, 2048 context and SwiGLU**, estimated around **124M parameters** with tied embeddings. This is a target, not implemented or trained in the Micro prototype.

## Mobile inference prototype (M17–M18)

Aetherion-Micro now has a **mobile web/WASM prototype** using verified
byte-token ONNX exports. M18 adds optional weight-only INT4 MatMulNBits
conversion and an on-device benchmark report. The prototype does not
ship assistant-quality model weights or a native mobile app, and phone
compatibility is not universal.

See [M17 mobile setup](docs/M17_MOBILE.md) and [M18 INT4/benchmark guide](docs/M18_MOBILE_INT4.md).

## Documentation

- [Eight-week execution plan (priorities, exit gates, risks)](docs/EXECUTION_PLAN.md)
- [Engineering roadmap](docs/ROADMAP.md)
- [Architecture and design choices](docs/ARCHITECTURE.md)
- [Data preparation and split manifest](docs/DATA_PIPELINE.md)
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

## Resumable Micro training

After the initial training run, restart from an optimizer/RNG checkpoint with `--resume` and a larger total `--steps` value. See [the resume guide](docs/RESUME_TRAINING.md). No pretrained assistant capability is claimed.

## Verified validation evaluation

Run `python evaluate.py --checkpoint checkpoints/demo.pt` for the demo, or supply the original prepared data and tokenizer paths for a BPE checkpoint. See [validation evaluation](docs/VALIDATION_EVALUATION.md) for exact target accounting, perplexity and comparison limitations.

## M17: Verified held-out training evidence

After saving two checkpoints from the same training run, use [M17 checkpoint comparison](docs/M17_CHECKPOINT_COMPARISON.md) to verify corpus and tokenizer fingerprints, equal training settings and exact validation-token coverage before comparing perplexity. Real 125M comparisons require a reviewed corpus and 32k BPE artifact; `--demo` is explicitly limited to unreviewed byte-token fixtures. Passing CI or improving a synthetic sequence is **not** a claim of real pretrained language ability.
