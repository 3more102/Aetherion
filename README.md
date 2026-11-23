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

## M26: conservative mobile runtime shape preflight

The mobile app now rejects context windows above 4096 byte tokens,
cached models above 48 layers / 64 heads / head dimensions of 256,
and manifests projecting more than 64 MiB for **one logical FP32
K/V-cache copy**. These checks run before reading ONNX graph bytes or
creating inference sessions, apply equally to restored local models,
and retain legacy M17/M18 exports within limits. Model-loading status
reports estimated cache size.

The estimate excludes graph weights, temporary tensors, duplicate cache
copies, WASM allocations, browser overhead and other memory; **passing
these checks does not guarantee a model will run on a phone**. See
[M26 mobile execution guardrails](docs/M26_MOBILE_RUNTIME_BUDGET.md).

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


## M33: Offline selected-test integrity audit

M32 now emits schema-v2 selected-test evidence recording **every validation
candidate** (checkpoint SHA-256, step, validation NLL/perplexity, target count)
as well as the validation-selected winner. The new
\`verify_selected_test_report.py\` checks input file hashes, the first-tie
minimum-validation-NLL decision and internal test-metric arithmetic without
loading weights or re-scoring the test split. It can write an exclusive,
no-overwrite JSON integrity report. This **does not independently verify
reported NLL, training, data rights or assistant ability**; unsigned
artifacts may all be forged together. See
[M33 offline integrity audit](docs/M33_OFFLINE_TEST_EVIDENCE_AUDIT.md).

## M32: Validation-selected, independently pinned test evaluation

The new `assess_selected_test.py` re-evaluates retained checkpoints on the
same held-out validation split and selects the **minimum validation NLL**
checkpoint. It then evaluates only that selected checkpoint on a separate
`test.txt` using exact target accounting and checksum checks. Demo runs
require an operator-pinned test SHA-256; real 125M evidence uses the reviewed
prepared test manifest and M27–M31 gates. An opt-in `--require-lineage` also
checks the entire M31 save chain. Results are create-only JSON, not proof of
conversational ability or a trained 125M model. Repeated use of the test set
for tuning defeats the intended holdout separation. See
[M32 selected test evaluation](docs/M32_SELECTED_TEST_EVALUATION.md).

## M31: Opt-in SHA-linked checkpoint history

Retained snapshots now include a random run identifier and the SHA-256 digest
and step of the previous saved checkpoint. Resuming a linked snapshot extends
that record. With `audit_training_history.py --require-lineage`, the auditor
requires the full root-to-tip chain and rejects missing, altered or unrelated
snapshots before accepting a linked held-out history. Legacy checkpoints and
the default audit mode remain supported. Hash links do **not** authenticate the
trainer or prove uninterrupted optimizer updates; they can be forged by anyone
who controls the files. See the [M31 lineage guide](docs/M31_CHECKPOINT_LINEAGE.md).

## M30: Retained checkpoints and validation progression

Training now supports the opt-in `--retain-step-checkpoints` flag to
save uniquely numbered, no-overwrite snapshots rather than replacing
one latest checkpoint. The new `audit_training_history.py` can
re-evaluate 2–32 trusted checkpoints on identical held-out validation
tokens, requiring matched dataset/tokenizer hashes, model config,
seed, training hyperparameters and evaluation settings. Its report
records per-step NLL/perplexity, changes from prior snapshots and
best observed validation step.

Snapshot creation requires a filesystem with hard-link support.
Matching settings are **not proof of a shared training lineage**.
Synthetic byte fixtures remain diagnostic only; no real 125M
training result is implied. See
[M30 checkpoint-history guide](docs/M30_CHECKPOINT_HISTORY.md).

## M29: Per-checkpoint validation evidence

After a trusted training checkpoint exists, `assess_checkpoint.py` can
produce a collision-safe, provenance-checked JSON report for one held-out
validation split. It records the SHA-256 checkpoint digest, exact
next-token NLL/perplexity and target count, model configuration, seed,
training settings and data/tokenizer hashes. Real 125M mode enforces
independent review, a 32k BPE tokenizer, and the M27 lexical screen
before opening weights; `--demo` explicitly labels unreviewed byte
fixtures. This is not training, a test-set quality benchmark, or proof
of conversational ability. See
[M29 checkpoint evaluation evidence](docs/M29_CHECKPOINT_EVIDENCE.md).

## M28: benchmark-backed 125M execution safety gate

The `pilot_train.py` dry-run remains available with approved dataset,
review and verified tokenizer. To actually use `--execute`, M28 requires
a **user-supplied matching 125M hardware benchmark report** (at least
three measured synthetic optimizer updates), an explicit indexed CUDA/XPU
device, and a declared accelerator peak-reserved-memory budget. A missing,
mismatched or over-budget report rejects the run before training starts.

The `benchmark_training.py --output ...` command writes report JSON
without overwriting existing evidence. These are measured-report format
checks, **not independently authenticated GPU measurements** or proof
of available memory, language quality, or pretrained weights. See
[M28 accelerator benchmark preflight](docs/M28_HARDWARE_BENCHMARK_GATE.md).

## M27: Production training contamination gate

The 125M pilot launcher and non-demo held-out checkpoint comparator now
**reject detected cross-split lexical contamination automatically** before
starting training or loading checkpoint weights. They fail closed on scan
budget overruns, corrupted prepared files and disallowed license IDs.
Reviewer attestations remain necessary but cannot bypass these detected
failures. The check is bounded and lexical only; no claim of full benchmark
decontamination, data licensing, privacy compliance or trained weights.
See [M27 production screening gate](docs/M27_PRODUCTION_CONTAMINATION_GATE.md).

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
