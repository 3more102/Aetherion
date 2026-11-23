# M32 — Validation-selected checkpoint, separately pinned test evaluation

**Scope:** One checkpoint is selected using *validation* next-token NLL from the
M30/M31 history auditor. Only after that decision does the tool compute
next-token loss on a distinct `test.txt` split. The tool does not train a
125M assistant or establish general intelligence.

## Why this milestone

Running the test split on every checkpoint and choosing the lowest test
perplexity leaks test performance into model selection. M32 keeps the
selection metric explicit, reports which checkpoint was chosen, and
evaluates the test split **only for that checkpoint**.

The software cannot prevent an operator from rerunning it or previously
examining the test set. Publish any such reuse in the experiment record,
and do not describe repeatedly examined test data as untouched.

## Deterministic synthetic demo

Prepare separate small `train.txt`, `validation.txt`, and `test.txt`
files under a local `data/tiny-demo` directory. Train and retain at least
two snapshots following [M31](M31_CHECKPOINT_LINEAGE.md). Pin the exact test
bytes *before* starting the selection:

```sh
python -c "from hashlib import sha256; from pathlib import Path; print(sha256(Path('data/tiny-demo/test.txt').read_bytes()).hexdigest())"
```

Then replace `<pinned-test-sha256>` with the displayed digest:

```sh
python assess_selected_test.py --demo --data-dir data/tiny-demo \
  --checkpoint checkpoints/tiny-step00000001.pt \
  --checkpoint checkpoints/tiny-step00000002.pt \
  --checkpoint checkpoints/tiny-step00000003.pt \
  --require-lineage --stride 8 --device cpu \
  --expected-test-sha256 <pinned-test-sha256> \
  --output reports/tiny-selected-test.json
```

Create the reports directory beforehand. `--output` uses exclusive
create mode and will reject a filename that already exists. No test
documents or prompt text are written into the JSON evidence.

## Production 125M evidence preconditions

For any real 125M evaluation:

- Omit `--demo` and supply `--tokenizer`, `--review`, and one or more
  `--allow-license` arguments.
- Use the same authorized prepared corpus and pinned 32k-BPE tokenizer
  required by the existing M27–M30 production gates. The tool reruns
  manifest integrity, human-review attestation and lexical cross-split
  screening *before* opening checkpoint weights.
- `test.txt` must match the previously prepared `manifest.json` digest;
  optionally supply `--expected-test-sha256` as an additional pin.
- The test split must be distinct from train/validation and contain at
  least two tokens after tokenization.
- The selected checkpoint is checked against the validation-recorded
  SHA-256 both before and after test evaluation; `test.txt` is checked
  before and after evaluation as well.
- If desired, `--require-lineage` enforces the M31 root-to-tip save chain.

**Do not use test NLL to reselect a different checkpoint or tune further
without labeling this test set as used for development.** Obtain a genuinely
fresh holdout for any subsequent final test.

## Evidence contents and interpretation

The resulting JSON records:

- The validation-only selection rule and number of candidate checkpoints.
- Selected step, checkpoint SHA-256, validation mean NLL/perplexity and
  dataset/tokenizer fingerprints.
- Independent exact next-token target coverage on test, total and mean
  NLL, test perplexity, context/stride, token unit and test digest source.
- Whether M31 lineage checking was requested and passed.

Scores are **not** instruction-following, truthfulness, reasoning, safety,
legal-rights or privacy evaluations. A demo run is unreviewed synthetic
evidence only. An unsigned checkpoint or provenance file can still be
forged by a party with write access.

## Regression verification

```sh
python -m pytest -q tests/test_m32_selected_test.py
python -m pytest -q
```

These tests exercise actual tiny CPU training and independent evaluation,
explicit test pinning, test split duplication, corrupted checkpoint
detection, CLI no-overwrite behavior and production fail-closed gating.
They are not a measured Aetherion-125M deployment.

## M33 compatibility

M32 reports now use `schema_version: 2` and add `selection.candidates` with
all validation candidate steps, SHA-256 hashes, target counts, mean NLL and
perplexity. This makes the validation-only selection rule independently
checkable for internal consistency using the
[M33 offline verifier](M33_OFFLINE_TEST_EVIDENCE_AUDIT.md). The verifier
checks file hashes and report arithmetic, not the actual model scores.
M32 schema-v1 reports remain historical evidence but cannot be audited by
M33 without re-evaluating, which may violate a one-time test-use policy.
