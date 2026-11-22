// Device benchmark report contract: phase isolation, privacy, RAM estimate and
// local JSON export readiness. All metrics here are deterministic mock values.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Script, createContext } from "node:vm";

const nodes = new Map();
const context = createContext({
  console, Date, TextEncoder, TextDecoder, BigInt64Array, Uint8Array, Float32Array,
  navigator: { userAgent: "Synthetic test browser", deviceMemory: 8, hardwareConcurrency: 4 },
  location: { protocol: "http:", hostname: "localhost" },
  performance: { now: () => 123.4, memory: { usedJSHeapSize: 120000 } },
  document: { getElementById(id) {
    if (!nodes.has(id)) {
      nodes.set(id, {
        textContent: "", disabled: true, hidden: true,
        addEventListener() {}, classList: { toggle() {} },
      });
    }
    return nodes.get(id);
  }},
});
new Script(readFileSync(new URL("./model-vault.js", import.meta.url), "utf8"),
  { filename: "model-vault.js" }).runInContext(context);
new Script(readFileSync(new URL("./app.js", import.meta.url), "utf8"),
  { filename: "app.js" }).runInContext(context);

const state = new Script("state").runInContext(context);
const sha = "a".repeat(64);
state.manifest = {
  schema_version: 3, window: 8, n_layers: 2, n_heads: 4, head_dim: 8,
  graphs: { prefill: { sha256: sha }, decode: { sha256: sha } },
  source_graphs: { prefill: { sha256: "b".repeat(64) } },
};
state.modelBytes = 200811;
state.modelLoadAndVerifyMs = 1500;
state.cacheStats = { prefill: 3, decode: 17 };
const phases = { prefill: [20, 30], decode: [2, 4, 6], full_window: [] };
const report = context.displayBenchmark(phases, 2, 62, 64000, 120000, false);
assert.equal(report.schema_version, 3);
assert.equal(report.benchmark_protocol, "hi_warmup2_16byte_greedy_v1");
assert.equal(report.source_model_sha256, "b".repeat(64));
assert.equal(report.inference_format, "int4_kv_cache");
assert.equal(report.measured_steps, 5);
assert.equal(report.warmup_steps_completed, 2);
assert.equal(report.latency_ms.prefill.median_ms, 25);
assert.equal(report.latency_ms.decode.median_ms, 4);
assert.equal(report.latency_ms.decode.p95_ms, 6);
assert.equal(report.latency_ms.full_window, null);
assert.equal(report.model_file_bytes, 200811);
assert.equal(report.model_load_and_sha_verify_ms, 1500);
assert.equal(report.memory.estimated_one_copy_fp32_kv_cache_bytes,
  2 * 2 * 4 * 8 * 8 * 4);
assert.equal(report.memory.browser_js_heap_after_bytes_if_available, 120000);
assert.equal(report.byte_tokens_per_inference_second, 80.645);
assert.equal(report.device.user_agent, "Synthetic test browser");
assert.equal(nodes.get("download-report").disabled, false);
assert.equal(state.lastBenchmark, report);
const serialized = nodes.get("benchmark-report").textContent;
assert.equal(JSON.parse(serialized).schema_version, 3);
assert.ok(!serialized.includes("my private prompt"));
assert.ok(!serialized.includes("ONNX weights"));

state.manifest = { schema_version: 1, window: 8, onnx_sha256: sha };
const legacy = context.displayBenchmark({
  prefill: [], decode: [], full_window: [10, 10, 20],
}, 1, 40, null, null, false);
assert.equal(legacy.inference_format, "fp32_full_window");
assert.equal(legacy.source_model_sha256, sha);
assert.equal(legacy.memory.estimated_one_copy_fp32_kv_cache_bytes, 0);
assert.equal(legacy.latency_ms.full_window.p95_ms, 20);
assert.equal(legacy.latency_ms.decode, null);

console.log("M21 deterministic device benchmark phases, memory labeling and local JSON readiness passed");
