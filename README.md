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

## Mobile inference prototype (M17–M23)

Aetherion-Micro has an experimental local ONNX Runtime Web/WASM
inference path. M17 supports full-window FP32, M18 weight-only INT4,
M19 prefill and FP32 KV-cached decode, and M20 weight-only INT4
prefill/decode with FP32 K/V tensors.

**M21 adds an installable offline-first static PWA package**, generated
192/512-pixel icons, a downloadable deployment ZIP containing the pinned
WebAssembly runtime, and an opt-in JSON benchmark with separate prefill,
cached-decode and full-window timings. No prompts, checkpoints or model
weights are bundled or sent to an inference server. The PWA requires an
HTTPS host, browser install support and user-imported ONNX model files.
It is **not** a native APK/IPA or a validated conversational assistant.

Guides: [M17 setup](docs/M17_MOBILE.md),
[M18 INT4](docs/M18_MOBILE_INT4.md),
[M19 KV caching](docs/M19_MOBILE_KV_CACHE.md),
[M20 combined INT4 + cache](docs/M20_MOBILE_INT4_CACHE.md), and
[M21 PWA packaging/device benchmark](docs/M21_MOBILE_PWA_BENCH.md), and
[M22 source-verified device comparison](docs/M22_MOBILE_DEVICE_VALIDATION.md).

**M22 adds a local benchmark report comparison gate:** schema-v3 results
retain the original FP32 graph SHA across INT4 conversions, and a Python
CLI rejects mismatched models, contexts, browser hints and partial tests.
It requires at least three reports per variant. No physical-device results
or pretrained model weights are shipped with this tooling.

**M23 optionally stores a verified model export in origin-scoped IndexedDB**:
choose Save local copy after importing ONNX files, Restore saved model after
reopening the PWA, and Delete saved copy to remove its stored record. Storage
is unencrypted, opt-in, limited to 256 MiB, and can be rejected or evicted by
the browser. Restored models must pass the usual SHA and graph checks. No
trained assistant model is supplied. See the
[M23 local model guide](docs/M23_LOCAL_MODEL_VAULT.md).

## M23–M24 mobile reliability

M23 adds **opt-in local storage** of SHA-256-verified ONNX exports, with
revalidation on restore. M24 adds a **local device API readiness check**
(no telemetry) and early 256 MiB ONNX / 128 KiB manifest import limits.
API readiness is not proof of phone compatibility, sufficient RAM,
successful inference, or useful language ability.

See [M23 stored model vault](docs/M23_LOCAL_MODEL_VAULT.md) and
[M24 device readiness and size limits](docs/M24_DEVICE_READINESS.md).

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

## M25: Core-model data contamination screening

The 125M research model is **still untrained**. Before authorizing any real
training pilot, use the [M25 cross-split lexical screening guide](docs/M25_DATA_CONTAMINATION.md)
to check the integrity of prepared data and flag potentially overlapping
train/validation/test texts without uploading raw documents:

```bash
python screen_corpus.py data/processed/v1 --allow-license CC0-1.0
```

Findings produce exit code 2 by default. Even a zero-finding report is not
proof of a legally authorized, private, benchmark-clean training corpus; a
separate human review and provenance attestation are still mandatory.

## M17: Verified held-out training evidence

After saving two checkpoints from the same training run, use [M17 checkpoint comparison](docs/M17_CHECKPOINT_COMPARISON.md) to verify corpus and tokenizer fingerprints, equal training settings and exact validation-token coverage before comparing perplexity. Real 125M comparisons require a reviewed corpus and 32k BPE artifact; `--demo` is explicitly limited to unreviewed byte-token fixtures. Passing CI or improving a synthetic sequence is **not** a claim of real pretrained language ability.
