# M22 — provenance-gated phone benchmark comparisons

**Status:** Implemented comparison tooling and automated regression gates.
These are not measurements of physical phones. A device benchmark becomes
evidence only after running M21/M22 on a real device and retaining the JSON.

## Why M22

An INT4 ONNX graph has a different SHA-256 fingerprint from its FP32 source.
Comparing them using the exported-file fingerprints alone cannot prove they
came from the same model. M22 adds `source_model_sha256` to **benchmark
schema v3** so FP32/INT4 reports can be matched to their exact source graph.

Source provenance is taken from the validated **M17/M19 FP32 manifest SHA**
or the **M18/M20 source graph metadata**, never guessed from filenames.
Model file bytes, inference format, manifest schema, and device hints remain
in the report. The local app does not upload files, benchmark results or
prompt text. The source hash is **unsigned manifest metadata**, not a
cryptographic attestation of the conversion process; independent trust in
that manifest and its creation still matters.

M21 schema-v2 reports **cannot** prove this pairing and are intentionally
not accepted by the M22 comparator. Collect new M22 reports from the updated
PWA; saved old reports are not automatically upgraded.

## Compare repeated real phone reports

Deploy the updated static PWA over HTTPS. Import the exact corresponding
FP32 ONNX model and manifest, run **Benchmark (16 steps)**, and save the JSON.
Repeat at least three times with the same phone, browser, app install mode
and context window. Repeat with the INT4 graph(s) derived from the *same*
FP32 source. Use distinct JSON filenames.

Run locally from the repository root:

```sh
python compare_mobile_benchmarks.py \
  --baseline fp32-run1.json fp32-run2.json fp32-run3.json \
  --candidate int4-run1.json int4-run2.json int4-run3.json
```

Add `--json` for a machine-readable summary, suitable for version-controlled
reports after manual review. The summary omits raw user-agent strings, dates,
prompts and file paths; the original JSON files remain on the local machine.

Validation rejects: partial/cancelled benchmarks, nonstandard protocols,
fewer than three independent trials per group, absent SHA-256 provenance,
different source models or context sizes, mismatched browser/user-agent hints,
mixed cached/full-window inference, inconsistent export fingerprints, and
invalid phase counts/timings.

The result reports per-group **median measured byte-tokens/s**, the median
of per-run overall p95 latencies, ONNX file bytes, and ratios. These are
*descriptive* results; the comparator does not perform a statistical
significance test.

## Testing without a physical phone

```sh
python -m pytest -q tests/test_mobile_benchmark_compare.py
node mobile/benchmark_smoke.mjs
```

The Python tests use synthetic JSON fixtures. The Node smoke test checks
the browser-produced schema-v3 provenance fields. CI runs these tests in
addition to the existing ONNX/WASM execution and PWA packaging gates.

## Measurement boundaries

- Browser user-agent equality **does not establish device identity**.
  Verify identical handset, OS/browser versions, temperature, background
  tasks and power/battery conditions manually.
- The benchmark uses 2 warmup byte steps followed by 16 measured steps
  on a fixed two-byte input. It is not a representative application
  performance benchmark or tokenizer-agnostic language metric.
- The file byte ratio is not peak RAM. Browser JS heap readings exclude
  WebAssembly and other process memory. There is no validated energy or
  thermal measurement.
- Better inference speed **does not imply** conversational quality,
  pretrained capability, or correct generated language.
- M22 does not validate any specific Android/iOS phone, publish model
  weights, or provide native APK/IPA binaries.

See [M21 PWA and raw benchmarks](M21_MOBILE_PWA_BENCH.md) for model loading
and installation instructions.
