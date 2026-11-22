# M3–M7: gated training pilot

M3: accumulation averages equal-size microbatches and counts optimizer steps.
Fix --schedule-steps before training; resume using the same horizon and warmup.
Default settings retain old FP32 constant-LR resume compatibility.

M4: FP32 master parameters and BF16 autocast on CPU/CUDA. No FP16, XPU,
distributed training or activation checkpointing yet. BF16 hardware support
must be measured; do not infer performance from CPU smoke tests.

M5: measure synthetic training updates on the actual target hardware:

    python benchmark_training.py --preset micro --device cpu --context 32
    python benchmark_training.py --preset 125m --device cuda --precision bf16 --context 256 --steps 3

The latter allocates the full model and optimizer. Check available VRAM first.
Peak allocated/reserved memory includes warmup and initialization, not external
processes. Synthetic throughput is not corpus throughput.

M6: prepare only an independently authorized corpus, then run:

    python audit_corpus.py data/processed/v1 --allow-license CC0-1.0

Use license IDs appropriate to the actual corpus and approvals. Exact
deduplication, cross-split group checks and checksums cannot establish rights
or absence of PII, near duplicates or benchmark contamination. An actual
reviewer must conduct those reviews and write a JSON attestation with:
manifest_sha256, allowed_licenses (array), reviewer (nonempty string),
rights_reviewed, privacy_reviewed, near_duplicates_reviewed,
contamination_reviewed (all true only after review).

M7: use the existing tokenizer trainer on the reviewed train split; require
an actual 32,000-entry BPE vocabulary. A guarded dry run is default:

    python pilot_train.py --data-dir data/processed/v1 --tokenizer artifacts/tokenizer.json --review review.json --allow-license CC0-1.0 --steps 10 --schedule-steps 100 --warmup-steps 5 --device cuda --output checkpoints/pilot.pt

Only after reviewing memory/throughput, corpus rights and the printed command,
add --execute. Keep the same schedule horizon and options for resume.
The 256-token pilot context is shorter than the 2048-token architecture
target; full-context memory and quality are not established.

Exit gates: finite loss, successful save/resume, actual device benchmark,
human-reviewed corpus and independent validation report after training.
No real dataset or 125M weights are supplied by this change.
