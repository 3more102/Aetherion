# M19 — Mobile KV cache: ONNX prefill and single-byte decode

Aetherion-Micro now supports an experimental two-graph CPU/WebAssembly
inference contract that avoids recalculating the entire fixed window for
each next byte. Its output must match the original Aetherion FP32 model.

## Requirements / exporter

Python 3.11, PyTorch, ONNX 1.23.2, ONNX Runtime 1.32.0:

    pip install -r requirements.txt
    pip install onnx==1.23.2 onnxruntime==1.32.0 onnx_ir

Export **only a trusted, trained or synthetic Aetherion byte-level Micro
checkpoint**, with the same parameter limit used by M17:

    python export_mobile_cached.py \
      --checkpoint checkpoints/my-micro.pt \
      --window 64 \
      --output-dir mobile/artifacts/m19

Results:
- prefill.onnx — full right-padded [1, window] input, selected logits, plus
  all layer K/V tensors.
- decode.onnx — [1,1] new token, past_length, per-layer fixed-shape K/V,
  and updated logits/cache.
- manifest.json — schema_version 2 with per-file SHA-256, config, context
  window, model checkpoint hash, RoPE cache policy and exact layer geometry.

**A model.onnx made by M17 or model.int4.onnx made by M18 is a schema 1
single-graph artifact, not an M19 cached export.** These continue to work
using full-window inference. INT4 conversion from M18 does not yet support
the M19 dual-graph format; the M19 cache graph is FP32.

## Browser app

Stage ONNX Runtime Web and host mobile/ over HTTPS:

    cd mobile
    npm install
    npm run stage-runtime

On Android or iOS, open the HTTPS mobile app, select *prefill.onnx* as
the main model, *decode.onnx* as the second model and *manifest.json*
as the manifest. SHA-256 checks ensure both supplied graphs match the
manifest. All prompts and cache data stay on-device; the host is only
needed to serve the static application shell and runtime assets.

To benchmark, tap **Benchmark (16 steps)**. The JSON report distinguishes
the number of prefill calls from cached decode calls. Tests must compare
same device, same model, same context and same browser; tokens are UTF-8
bytes, not LLM BPE tokens.

## Exact inference state machine

1. Reset state for every new prompt or independent benchmark.
2. Prefill with the most recent min(window, prompt_length) bytes,
   right-padded with zeros. Set last_position = length - 1.
3. Store K/V buffers shaped [1, heads, window, head_dim] for each layer.
   Future padded entries are ignored; a mask excludes them.
4. The next output token is computed from prefill's next_logits. Append it.
5. If cache_length < window, call decode with the appended token,
   past_length = cache_length, and all previous layer caches. Update caches,
   cache_length and next_logits. Continue until the context becomes full.
6. If cache_length == window, rebuild the cache with the last window
   tokens and reset relative RoPE positions to zero. Do not append to
   an already-full cache: Aetherion's original generator uses the same
   sliding-window rebase semantics.

Decode creates attention for only one new query, while maintaining and
returning fixed-capacity K/V tensors. This does **not** eliminate the cost
of transferring the cache tensors through the WebAssembly JavaScript API;
speedup must be demonstrated empirically on physical devices.

## Verification gates

- Pure PyTorch prefill/decode compared with the reference model for
  prefix lengths 1, 2, 3, 5, 7, 8.
- Real CPU ONNX Runtime prefill/decode parity across several decode steps,
  dynamic RoPE indexing, and sliding-window rebuild.
- Exact input/output contract and both SHA-256 hashes.
- Node ONNX Runtime Web/WASM prefill/decode smoke test compared against
  the existing M17 FP32 full-window output.
- Existing M17/M18 tests and repository CI unaffected.

These tests validate numerical correctness on deterministic **synthetic**
model weights. They do not prove language understanding, phone support,
battery or memory use. Real Android/iOS device benchmarks and independently
validated trained checkpoints remain required.

Sources:
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
- [ONNX Runtime Mobile](https://onnxruntime.ai/docs/get-started/with-mobile.html)
