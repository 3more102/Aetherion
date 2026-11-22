// M19: test browser generation control flow without a physical phone.
// Real ONNX/WASM graph execution is separately checked in runtime_smoke.mjs.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Script, createContext } from "node:vm";

const fakeNodes = new Map();
const context = createContext({
  console, TextEncoder, TextDecoder, BigInt64Array, Float32Array, Uint8Array,
  navigator: {}, location: { protocol: "http:" },
  document: { getElementById(id) {
    if (!fakeNodes.has(id)) {
      fakeNodes.set(id, {
        textContent: "", disabled: false,
        addEventListener() {},
        classList: { toggle() {} },
      });
    }
    return fakeNodes.get(id);
  }},
  ort: {
    Tensor: class Tensor {
      constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
    },
  },
});
const script = readFileSync(new URL("./app.js", import.meta.url), "utf8");
new Script(script, { filename: "app.js" }).runInContext(context);

const sha = "a".repeat(64);
const validateManifest = context.validateManifest;
const m19 = {
  schema_version: 2, architecture: "aetherion-micro",
  tokenizer_type: "utf8_bytes", vocab_size: 256,
  window: 4, n_layers: 1, n_heads: 4, head_dim: 8,
  cache_policy: "rebuild_relative_rope_at_window_boundary",
  graphs: {
    prefill: { filename: "prefill.onnx", sha256: sha },
    decode: { filename: "decode.onnx", sha256: sha },
  },
};
assert.equal(validateManifest(m19).schema_version, 2);
// M20 shares M19's two-graph session contract but requires exact INT4
// metadata and both original source hashes.
const m20 = {
  ...m19,
  graphs: {
    prefill: { ...m19.graphs.prefill, source_sha256: "b".repeat(64) },
    decode: { ...m19.graphs.decode, source_sha256: "c".repeat(64) },
  },
  quantization: {
    type: "weight_only_symmetric_int4", format: "MatMulNBits",
    cache_dtype: "float32",
    prefill_quantized_matmul_nodes: 1,
    decode_quantized_matmul_nodes: 1,
  },
};
assert.equal(validateManifest(m20).quantization.format, "MatMulNBits");
assert.throws(() => validateManifest({
  ...m20, quantization: { ...m20.quantization, cache_dtype: "int4" },
}));
assert.throws(() => validateManifest({
  ...m20, graphs: { ...m20.graphs, decode: { ...m20.graphs.decode, source_sha256: "invalid" } },
}));
assert.throws(() => validateManifest({ ...m19, cache_policy: "unlimited" }));
assert.throws(() => validateManifest({
  ...m19, graphs: { ...m19.graphs, decode: { filename: "decode.onnx", sha256: "bad" } },
}));

const v1 = {
  schema_version: 1, architecture: "aetherion-micro", tokenizer_type: "utf8_bytes",
  vocab_size: 256, window: 4, onnx_sha256: sha,
};
assert.equal(validateManifest(v1).schema_version, 1);

const state = new Script("state").runInContext(context);
let firstCalls = 0, decodedCalls = 0;
const logits = (token) => {
  const values = new Float32Array(256);
  values[token] = 3.5;
  return { data: values };
};
state.session = { async run(feeds) {
  firstCalls++;
  assert.equal(feeds.tokens.dims[1], 4);
  return { next_logits: logits(70), key_0: { data: 42 }, value_0: { data: 43 } };
}};
state.decodeSession = { async run(feeds) {
  decodedCalls++;
  assert.equal(Number(feeds.past_length.data[0]), decodedCalls);
  assert.ok(feeds.past_key_0 && feeds.past_value_0);
  return { next_logits: logits(71), key_0: { data: 44 }, value_0: { data: 45 } };
}};
state.manifest = m19;
context.resetCache();
const sequence = [65];
for (let i = 0; i < 5; i++) {
  const next = await context.runByteStep(sequence);
  sequence.push(next);
}
assert.equal(firstCalls, 2, "Must prefill again when context becomes full");
assert.equal(decodedCalls, 3, "Must not decode into an already full cache");
assert.equal(state.cacheLength, 4);
assert.equal(state.cacheStats.prefill, 2);
assert.equal(state.cacheStats.decode, 3);

context.resetCache();
assert.equal(state.cache, null);
assert.equal(state.cacheLength, 0);
state.manifest = v1;
const v1step = await context.runByteStep([65]);
assert.equal(v1step, 70);
assert.equal(state.cache, null);
assert.equal(firstCalls, 3);
console.log("M19 browser cache policy and M17/M18 fallback smoke OK");
