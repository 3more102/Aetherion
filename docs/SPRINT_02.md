# Aetherion — Sprint 02: measurable baseline and data readiness

**Status:** execution plan, not a claim of trained capability. Work continues from the existing byte-level Aetherion-Micro implementation.

## Goal and exit criteria

Prove that training, checkpoint loading and held-out evaluation are repeatable before expanding model size. Later, introduce a licensed, versioned tokenizer/dataset pipeline and validate the proposed 125M architecture on available compute.

| Priority | Deliverable | Acceptance test | Risk / dependency |
|---|---|---|---|
| P0 | Full-dataset held-out NLL/perplexity | Each validation target scored once; fixed checkpoint/data yields the same metrics | The bundled demo has repeated text; metrics are not indicative of generalization |
| P0 | Dataset identity in checkpoint | SHA-256 stored; evaluation rejects a changed data file | No dataset provenance yet |
| P0 | CI evaluation smoke | Train a tiny model; reload; evaluate automatically | GitHub-hosted CPU run time |
| P1 | Tokenizer interface + 32k BPE experiment | Multilingual round-trip, unknown characters, deterministic artifact and hash | Corpus licensing and language mix |
| P1 | Data manifest and split pipeline | Document-level disjoint IDs, dedup, provenance, reproducible splits | Approved datasets required |
| P1 | Resumable train state + metrics | Resume reproduces uninterrupted next steps to tolerance; validate and log tokens/sec | RNG capture, optimizer/scheduler state |
| P2 | ~125M model preset and pilot | Exact parameter count + shape/gradient/causality tests; report VRAM and tokens/sec | GPU access, budget approval |
| P2 | Model card and evaluation harness | Publish loss, perplexity, model/data details and limitations | Uncontaminated test set |

## Execution order

1. **Validation infrastructure:** finish NLL/perplexity CLI, dataset checksum guard, test and CI check.
2. **Tokenizer and datasets:** select legitimately usable text, document provenance and licenses, split at document/source level **before** fitting tokenizer; dedup and inspect for PII.
3. **Training engineering:** introduce configs, gradient accumulation, LR warmup/cosine, mixed precision, resume, fixed-seed tests and machine-readable logging.
4. **Pilot benchmark:** run a small-scale overfit test, then a short ~125M benchmark. Record actual peak memory, tokens/sec and tokens budget before longer training.
5. **Evaluation and decisions:** run on disjoint held-out data, inspect degeneration/repetition, publish an evidence-based go/no-go decision for instruction tuning and scaling.

## Required experiment record

Code SHA, model hyperparameters, seed, PyTorch/Python version, tokenizer SHA, data manifest SHA, number of optimization steps, number of tokens, optimizer and schedule, hardware, throughput, peak memory, checkpoint SHA, held-out loss/perplexity, context window and evaluation split identity.

## Release gates

- **Ready for training pilot:** unit/CI checks green; tokenizer and dataset provenance documented; no known train/validation/test overlap; full checkpoint resume tested.
- **Ready for instruction tuning:** pretrained checkpoint demonstrates generalization on suitable independent evaluation, not just memorization of demo text.
- **Ready to scale:** available GPUs, data quantity and quality, measured training efficiency, and approved cost envelope.
- **Ready to publish weights/service:** trained weights, model card, licensing review, basic safety and misuse evaluations. Source-code MIT licensing does not transfer to third-party datasets.

## Reference implementations

- PyTorch SDPA: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- Tokenizers: https://huggingface.co/docs/tokenizers/main/quicktour
- LM Evaluation Harness: https://github.com/EleutherAI/lm-evaluation-harness
