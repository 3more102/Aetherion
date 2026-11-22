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
assert.throws(() => validateManifest({ ...m19, cache_policy: "unlimited" }));
assert.throws(() => validateManifest({
  ...m19, graphs: { ...m19.graphs, decode: { filename: "decode.onnx", sha256: "bad" } },
}));

const v1 = {
  schema_version: 1, architecture: "aetherion-micro", tokenizer_type: "utf8_bytes",
  vocab_size: 256, window: 4, onnx_sha256: sha,
};
assert.equal(validateManifest(v1).schema_version, 1);

const m20 = {
  ...m19, schema_version: 3,
  graphs: {
    prefill: { filename: "prefill.int4.onnx", sha256: sha },
    decode: { filename: "decode.int4.onnx", sha256: sha },
  },
  source_graphs: m19.graphs,
  quantization: {
    type: "weight_only_symmetric_int4",
    format: "MatMulNBits",
    block_size: 32,
    kv_cache_dtype: "float32",
    converted_matmul_nodes: { prefill: 9, decode: 9 },
  },
};
assert.equal(validateManifest(m20).schema_version, 3);
assert.throws(() => validateManifest({ ...m20, quantization: undefined }));
assert.throws(() => validateManifest({ ...m20, quantization: { ...m20.quantization, block_size: 17 } }));
assert.throws(() => validateManifest({ ...m20, graphs: m19.graphs }));
assert.throws(() => validateManifest({ ...m20, source_graphs: undefined }));
assert.throws(() => validateManifest({ ...m19, quantization: m20.quantization }));


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
  assert.equal(Number(feeds.past_length.data[0]), state.cacheLength);
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
state.manifest = m20;
context.resetCache();
const m20seq = [65];
for (let i = 0; i < 5; i++) m20seq.push(await context.runByteStep(m20seq));
assert.equal(state.cacheStats.prefill, 2);
assert.equal(state.cacheStats.decode, 3);
console.log("M20 schema-3 cached state and legacy M17/M18/M19 fallback smoke OK");
