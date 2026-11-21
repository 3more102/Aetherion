# Data governance and evaluation

## Dataset requirements

Every corpus needs a manifest entry with: source name, source URL or local origin, collection date, license/authorization, language, transformations, document count, token count, checksum and exclusions. **Do not commit private or unauthorized data.**

Processing steps:
1. Parse text and record source identity.
2. Normalize where explicitly justified; retain reproducible scripts.
3. Filter secrets, sensitive personal data and low-quality content.
4. Deduplicate documents before splitting.
5. Split by document/source group: training, validation, held-out test.
6. Fit tokenizer on training portion only.
7. Check overlaps with evaluation datasets and publish contamination limits.
8. Store hashes and manifests; avoid shipping large training corpora in git.

## Evaluation levels

- **Unit:** tensor shapes, finite losses/gradients, no future-token attention, save/reload equivalence.
- **Training:** loss curve, tokens/sec, GPU memory, checkpoint recovery and reproducibility.
- **Language modeling:** held-out cross-entropy/perplexity; report tokenizer, corpus, evaluation size and split.
- **Assistant:** instruction-following, factuality, calibration, multilingual tests, prompt-injection resistance, harmful-use/refusal evaluation.
- **Release:** model card, data summary, license, hardware, compute, limitations, performance and known risks.

## Claims discipline

A demo sampler does not establish reasoning capability. A small corpus may be memorized. Don't compare raw perplexities across distinct tokenizers without additional explanation. No "human level" or "frontier" claims without independent tests.
