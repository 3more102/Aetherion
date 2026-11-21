# Aetherion — Sprint 01 execution plan

## Objective
Turn Aetherion-Micro into a reproducible, tested research baseline before scaling. Do not claim frontier-level capability.

## Priorities and deliverables
| Priority | Work package | Definition of done |
|---|---|---|
| P0 | Verify existing baseline | Clean environment installs; pytest passes; 5 training steps complete; checkpoint reloads; generation runs |
| P0 | Continuous integration | GitHub Actions executes tests on push and pull request |
| P0 | Reproducibility | Log seed, commit SHA, PyTorch version, config, dataset identifier, token count, train/validation losses |
| P1 | Tokenizer | Byte-BPE training/encode/decode, deterministic serialization, multilingual round-trip tests |
| P1 | Data ingestion | Licensed input only; provenance manifest, checksums, document-level split, dedup and PII checks |
| P1 | Evaluation | Held-out loss and perplexity, fixed prompts, contamination checks |
| P2 | 125M pilot | Correctness tests, parameter count, GPU peak memory, throughput and cost estimate before full pretraining |

## Week-by-week plan
- **Week 1:** baseline tests, CI, configuration and run metadata.
- **Week 2:** tokenizer and data processing; freeze validation/test split.
- **Week 3:** 125M preset and correctness tests; benchmark throughput.
- **Week 4:** mixed precision, scheduler, gradient accumulation, resumable checkpoints.
- **Week 5:** short pretraining pilot if compute and licensed data are ready.
- **Week 6:** held-out evaluation, comparisons, model card draft.
- **Week 7:** instruction-tuning experiments only if a suitable pretrained checkpoint exists.
- **Week 8:** inference API and demo; release only with documented limitations.

## Initial issue backlog
1. `ci: add CPU smoke-test workflow`
2. `test: verify causal attention and checkpoint round-trip`
3. `feat: introduce tokenizer abstraction and BPE baseline`
4. `feat: data manifest and document-level split`
5. `feat: eval command for held-out perplexity`
6. `bench: profile Aetherion-125M memory and tokens/sec`

## Experiment record
For every run store: Git SHA, seed, config, tokenizer hash, dataset manifest hash, optimizer/scheduler, tokens processed, train/validation loss, elapsed seconds, hardware, peak memory, and checkpoint hash.

## Go/no-go gates
- **Micro → 125M:** tests pass, no train/test overlap, stable checkpoint resume, hardware estimate recorded.
- **125M → instruction tuning:** meaningful pretraining run and held-out evaluation, no leakage evidence.
- **125M → 1B:** measured scaling benefit and approved GPU/data budget.
- **Release:** model card, data provenance, licensing review, safety and quality evaluation.

## Dependencies
- PyTorch: https://pytorch.org/docs/stable/
- Tokenizers: https://huggingface.co/docs/tokenizers/
- GitHub Actions: https://docs.github.com/en/actions
