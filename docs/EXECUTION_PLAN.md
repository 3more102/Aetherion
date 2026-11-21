# Aetherion execution plan — 2026-11-21

## Mission

Build and evaluate an **original, open-source decoder-only language model** using a reproducible software and training stack. The repository is a research implementation, not a pretrained GPT-6/Gemini competitor. Scale only after quality, licensing, and compute gates pass.

## Current verified repository baseline

- PyTorch model: causal SDPA, RoPE, RMSNorm, SwiGLU, tied embeddings.
- Byte-token educational training and sampling scripts.
- Smoke tests and GitHub Actions workflow.
- Aetherion-125M is a **proposed** architecture, not trained weights.
- Existing M0 checklist describes criteria; execution on GitHub CI must be checked independently.

## Workstreams and acceptance gates

| Stage | Deliverable | Pass condition |
|---|---|---|
| M0 — Micro | Reproducible 5-step training, load, generation | CI green; same configuration reloads |
| M1a — Corpus | Licensed JSONL + deterministic group splits + manifest | No exact overlaps; provenance/sha256; manual license review |
| M1b — Tokenizer | ByteLevel BPE 32k experiment | Multilingual round-trip; tokenizer trained on train split only |
| M1c — Data QA | Near-duplicate checks, PII screening, contamination scan | Documented results and exclusions, no held-out leakage |
| M1d — Integration | Tokenized shards + model vocab support in train/infer | End-to-end BPE train/evaluate/generate |
| M2a — 125M preset | Parameter-count and one-batch overfit tests | No NaNs; causal isolation; architecture math verified |
| M2b — Compute pilot | GPU memory/throughput profile, estimated cost | Cloud environment and approved finite budget |
| M2c — Pretraining | Resumable train + eval | Validation NLL/perplexity, checkpoint recovery, training log |
| M3 — Instruct | Curated SFT data and chat template | Measurable held-out instruction following |
| M4 — Scale | FSDP/distributed + optimized serving | Reproducible gain and documented economics |
| M5 — Release | API, model card, safety/eval report, provenance | Public claims match actual evaluated artifacts |

## Concrete first four sprints

**Sprint 1 (repo baseline):** audit existing modules, run tests, confirm CI, record known gaps. Initial implementation is already in main; check the run results separately.

**Sprint 2 (data/tokenizer):** add corpus JSONL validation, stable group-level hash splits, manifest and exact dedup, fit a BPE tokenizer on training documents only, multilingual round-trip tests. This first implementation does **not** validate input licensing, detect near duplicates, or scan PII — those remain release blockers for real corpora.

**Sprint 3 (data/model integration):** stream documents from approved dataset inventories, implement near dedup and PII review, train 32k tokenizer, pack token IDs with document EOS into train/validation/test shards, update train/generate for tokenizer version, track artifact hashes, and add loss/perplexity evaluation.

**Sprint 4 (125M readiness):** implement explicit 125M config, parameter-count assertion, one-batch overfit, device profiling (FP32/BF16 as supported), gradient accumulation, warmup/cosine LR, interrupted/resumed run tests and GPU cost estimate. Do not begin substantive 125M pretraining before approval of data, budget and reproducibility gates.

## Target technical stack

- Model: custom PyTorch Transformer (avoid confusing library use with pretrained weights).
- Tokenizer experiment: Hugging Face tokenizers ByteLevel BPE.
- Data: JSONL documents with id, text, source, license, optional language/group_id; JSON SHA-256 manifests.
- Optimization roadmap: AdamW, autocast when supported, gradient accumulation, gradient clipping, checkpoint resume.
- Tracking: seed, git SHA, training tokens, model config, tokenizer hash, dataset manifests, losses, GPU, elapsed time, checkpoints.
- Evaluation: held-out NLL/perplexity, multilingual samples, leakage check, instruction/safety tests before release.
- Serving later: FastAPI or compatible inference server after a validated checkpoint exists.

## Release gates

1. **Legal:** inspect each dataset's actual authorization and redistribution terms; labels in a manifest are not proof.
2. **Data:** remove private/secret/PII records, near duplicates and benchmark contamination. Exact hash dedup is insufficient on its own.
3. **Compute:** measure throughput, memory, cost, and approved cloud-GPU capacity.
4. **Claims:** call the model untrained until a checkpoint is trained; report all benchmarks with methodology and limits.
5. **Security:** never commit data dumps, credentials, tokens or large checkpoints; review prompt/tool execution before serving.

## Next concrete GitHub tasks

- Integrate BPE tokenizer with training and generation.
- Add token packing, split-level vocab checks and measured token counts.
- Add near deduplication/PII audit reporting.
- Profile and test the 125M architecture before commissioning cloud training.

## References

- PyTorch SDPA: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- Hugging Face Tokenizers: https://huggingface.co/docs/tokenizers/
- PyTorch reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
