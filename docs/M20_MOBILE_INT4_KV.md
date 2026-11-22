# M20 — INT4 weight-only plus KV-cached WebAssembly inference

M20 combines M18's ONNX MatMulNBits weight-only quantization with M19's
FP32 KV cache. Both *prefill* and *decode* ONNX graphs are quantized.
The K/V cache **remains FP32**, so this is **not** end-to-end 4-bit
inference. No native Android or iOS app or trained assistant weights ship.

## Reproducible export

Required developer environment: Python 3.11, PyTorch, ONNX 1.23.2,
ONNX Runtime 1.32.0, onnx_ir.

    python -m pip install -r requirements.txt
    python -m pip install onnx==1.23.2 onnxruntime==1.32.0 onnx_ir

Export a trusted Aetherion-Micro byte-token checkpoint into M19's
two-graph FP32 format:

    python export_mobile_cached.py --checkpoint checkpoints/my-micro.pt \
      --window 64 --output-dir mobile/artifacts/fp32_cached

Convert that to a separate output directory:

    python quantize_mobile_cached.py \
      --source-dir mobile/artifacts/fp32_cached \
      --output-dir mobile/artifacts/int4_cached \
      --block-size 32

The converter writes:
- prefill.onnx — INT4 blockwise weights with FP32 activations/cache
- decode.onnx — INT4 blockwise weights with FP32 activations/cache
- manifest.json — M19 schema v2 plus M20 quantization diagnostics,
  per-file hashes and original/INT4 file sizes.

Source FP32 files are not overwritten. The converter rejects an
already-quantized source, modified model hashes, unexpected graph
contracts, invalid block sizes, or conversions that produce no
MatMulNBits operators.

## Mobile web app

Run from mobile/:

    npm install
    npm run stage-runtime

Serve the mobile directory over HTTPS. On a modern phone browser,
select **prefill.onnx**, **decode.onnx**, and **manifest.json** from the
same M20 export. The mobile interface validates both model hashes before
initializing local WebAssembly sessions.

For an M19 export, the same app still uses FP32 cached inference.
For M17/M18 exports, the same app still uses the original single-graph
FP32/INT4 full-window inference.

Tap **Benchmark (16 steps)** for a local timing report containing
actual per-step latency (median and p95), byte tokens/sec, number of
prefill/decode calls, browser user-agent, browser-reported RAM when
supported, and a calculated FP32 KV-cache tensor payload size. Note that
the estimated cache bytes are a model-shape calculation, **not a
measured device-memory footprint**.

## Engineering validation

CI covers:
- Quantization of both graphs to real MatMulNBits operators.
- FP32 graph SHA integrity and unchanged input/output tensor contracts.
- Paired FP32 vs INT4 execution across deterministic synthetic prefills
  and repeated decodes; each version carries its own updated K/V states.
- ONNX Runtime CPU and ONNX Runtime Web/WebAssembly inference.
- Browser manifest validation and backward compatibility with M17–M19.
- Existing repository test suite.

The generated manifest reports synthetic logit drift and greedy-token
agreement without suppressing mismatches. Do not interpret these as a
language capability or task accuracy score.

## Known limitations / next validation

- The project does not currently contain a released, independently
  evaluated high-quality mobile conversational checkpoint.
- INT4 quantization can worsen output quality or latency. Test using a
  trained checkpoint and held-out evaluations before release.
- KV caching still rebuilds at the context-window boundary to preserve
  relative RoPE semantics. It transfers updated FP32 cache tensors
  between JS and WASM, and therefore does not guarantee a speedup.
- No actual Android/iPhone results, battery measurements or RAM
  measurements have been collected by this CI. Physical devices are
  required to establish compatibility.
- SHA-256 checks verify file consistency, not publisher authenticity.
- Models using BPE tokenization and native APK/IPA packaging remain
  separate engineering milestones.

References:
- [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
