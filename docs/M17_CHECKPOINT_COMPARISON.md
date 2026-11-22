# M17: Verified checkpoint-to-checkpoint held-out comparison

The earlier validation command (`evaluate.py`) evaluates one checkpoint.
M17 compares **two separately saved checkpoints from the same training
experiment** on identical validation token IDs, context and stride. This
produces evidence of whether validation next-token NLL actually improved
between training steps. It does **not** train the model.

## Real 125M pilot prerequisites

1. Prepare a genuinely authorized, source-grouped corpus with
   `prepare_data.py`, and independently review rights, privacy, near-duplicates
   and benchmark contamination.
2. Run `audit_corpus.py` and create a human attestation matching the manifest.
   This software validates attestations structurally, **not** the truth of the
   human review or licensing claim.
3. Fit and verify the 32,000-entry BPE tokenizer exclusively on the training
   split. Run `pilot_train.py` with an explicitly approved model hardware budget.
4. Preserve at least two sequential training checkpoints from the **same**
   experiment. For example, save a step-10 checkpoint separately, then
   resume to step 50 with unchanged data, seed, schedule horizon and
   hyperparameters. Do not overwrite the step-10 checkpoint.

```bash
python compare_checkpoints.py \
  --before checkpoints/pilot-step10.pt \
  --after checkpoints/pilot-step50.pt \
  --data-dir data/processed/v1 \
  --tokenizer artifacts/tokenizer.json \
  --review review.json \
  --allow-license CC0-1.0 \
  --stride 128 \
  --device cuda \
  --output reports/pilot-comparison.json \
  --require-improvement
```

Replace the sample license with the licenses actually authorized for
your corpus. The command will refuse missing approval artifacts, mismatched
source fingerprints or training settings, changed architecture, wrong step
order, or a non-125M production configuration. A measured regression is
reported as a valid result; `--require-improvement` additionally exits 2
when validation NLL has not improved. JSON output is created only if the
target path does not already exist. It never writes/overwrites weights.

## Explicit synthetic diagnostic mode

For small byte-token fixtures **only**:

```bash
python compare_checkpoints.py --demo \
  --before /tmp/step1.pt --after /tmp/step3.pt \
  --data-dir /tmp/byte-demo --device cpu
```

The files `/tmp/byte-demo/train.txt` and `validation.txt` must be distinct
explicit files matching the checkpoint. This mode requires no tokenizer or
human approval and is labeled as synthetic/unreviewed in the report. It
cannot be used to make production dataset or 125M claims.

## Report and interpretation

The report records both SHA-256 checkpoint hashes, training steps, the
original train/validation and tokenizer fingerprints, seed, architecture,
hyperparameters, stride, token target counts, mean next-token NLL and
perplexity. Perplexity ratio and mean-NLL delta quantify the change.
The comparator evaluates **all validation targets** for each checkpoint;
it does not touch the test split, run new training or infer causality.
It uses trusted local PyTorch weights-only loading (not a general safe
untrusted-checkpoint sandbox); high-memory 125M checkpoint evaluation
still requires real RAM/VRAM. Comparisons do not establish unseen-domain
generalization, factual accuracy, instruction following, or ChatGPT-like
capabilities.

CI covers synthetic CPU checkpoint creation/resume, exact target coverage,
JSON evidence, write collision, tampered split, mismatched seed, invalid
step order, and production preflight refusal. **No real corpus, actual
125M training weights, 125M GPU memory run, or production comparison
results are included.**
