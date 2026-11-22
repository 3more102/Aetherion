# M20 — INT4 quantization combined with mobile KV caching

## Delivered (experimental, not a released assistant)

M20 combines the M18 weight-only INT4 ONNX MatMulNBits optimization
with the M19 separate prefill/decode KV-cache inference interface.

- FP32 attention **key/value cache stays float32**. Weight-only INT4 does
  not imply 4-bit activations, embeddings, or cache tensors.
- Schema 3 manifest verifies BOTH INT4 graphs with SHA-256 hashes and
  records the source FP32 graph hashes, per-graph node counts, exact
  file-size ratios and deterministic synthetic FP32-vs-INT4 logit differences.
- M17 FP32 full-window, M18 INT4 full-window and M19 FP32 cached
  exports remain supported by the same mobile browser interface.
- Context overflow rebuilds the cache and restarts relative RoPE
  positions. This is needed to remain consistent with the original
  Aetherion-Micro sliding-window inference.
- No actual Android or iOS performance benchmark has been collected.
  This change does not include a mobile app binary, hosted model,
  or independently trained general-purpose assistant.

## Export

On your development computer:

    python -m pip install -r requirements.txt
    python -m pip install onnx==1.23.2 onnxruntime==1.32.0 onnx_ir

Export from a trusted, compatible byte-level Aetherion-Micro checkpoint:

    python export_mobile_cached.py \
      --checkpoint checkpoints/my-micro.pt \
      --window 64 \
      --output-dir mobile/artifacts/m19_fp32

Convert both ONNX graphs:

    python quantize_mobile_cached.py \
      --source-dir mobile/artifacts/m19_fp32 \
      --output-dir mobile/artifacts/m20_int4 \
      --block-size 32

M20 outputs:

- prefill.int4.onnx
- decode.int4.onnx
- manifest.json (schema_version 3)

The original M19 schema-2 artifacts are preserved in their separate
directory. The exporter validates their provenance hashes before
writing INT4 files and rejects a same-directory output.

The M20 quantizer reports actual file sizes for both graphs and
numerical differences across multiple deterministic cached-generation
steps, including context-boundary rebuilds. Do not assume a particular
reduction factor or token throughput: each model and device differs.

## Run locally in phone browser

1. Within mobile/, run npm install && npm run stage-runtime.
2. Publish mobile/ with vendor/ to a **trusted HTTPS** origin.
3. Open the page in a modern Android or iOS browser with WebAssembly.
4. Select prefill.int4.onnx, decode.int4.onnx and manifest.json
   from the same M20 export.
5. Tap Verify & load model, then Generate or Benchmark (16 steps).

Use exactly the same benchmark settings, checkpoint, phone and browser
when comparing the M19 FP32 cached model against the M20 INT4 cached
model. Model load time, prompt prefill, the fixed-size K/V tensor
transfers and thermal state can affect results. INT4 could be slower.

The benchmark displays time per generated UTF-8 **byte token**.
Do not compare these byte token counts directly to BPE-based LLM
tokens-per-second measurements.

The browser does not send inference prompts to a server. Initial
distribution of the static web application and the ONNX Runtime Web
files may require the network. The service worker is best-effort, so
offline access is not universally guaranteed on all phones.

## Validation

    python -m pytest -q \
      tests/test_mobile_export.py \
      tests/test_mobile_quantization.py \
      tests/test_mobile_cached.py \
      tests/test_mobile_cached_quantization.py

    python -m mobile.ci_synthetic_models mobile/artifacts/ci
    node mobile/runtime_smoke.mjs mobile/artifacts/ci
    node mobile/app_state_smoke.mjs

Tests inspect the ONNX runtime shapes and cache update path, execute
the quantized models on ONNX Runtime CPU and WebAssembly, compare to
FP32 for deterministic synthetic inputs, and reject tampered manifests.
This is **numerical/infrastructure validation only**, not demonstrated
natural-language quality, mobile speedup, energy consumption or
compatibility with every phone.

## Roadmap after M20

- Measure supported real Android and iOS devices with the same checkpoint,
  temperature, browser and workload: prefill latency, cached-decode
  tokens/s, peak memory and battery drain.
- Add WebGPU/CoreML/NNAPI execution only after device-specific tests.
- Optimize KV tensor transfers and persistent buffer allocation.
- Evaluate trained model quality and safety on independently held-out
  authorized data. No quality claims from random-weight test fixtures.
- Package an installable native app only after validating device security,
  model management, runtime compatibility and offline licensing.

Runtime references:
- https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- https://onnxruntime.ai/docs/tutorials/web/
