"use strict";

const $ = (id) => document.getElementById(id);
const state = { session: null, decodeSession: null, manifest: null, cache: null,
  cacheLength: 0, cacheStats: { prefill: 0, decode: 0 }, running: false, cancel: false };
const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8", { fatal: false });

function status(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}

function setRunning(running) {
  state.running = running;
  $("generate").disabled = running || !state.session;
  $("benchmark").disabled = running || !state.session;
  $("load").disabled = running;
  $("stop").disabled = !running;
}

function hex(bytes) {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function sha256(bytes) {
  if (!crypto.subtle) {
    throw new Error("HTTPS or localhost is required for SHA-256 verification.");
  }
  return hex(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)));
}

function isSha256(value) {
  return typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
}

function cacheTensorNames(layers, prefix = "") {
  const names = [];
  for (let i = 0; i < layers; i++) {
    names.push(prefix + "key_" + i, prefix + "value_" + i);
  }
  return names;
}

function validateManifest(manifest) {
  if (!manifest ||
      manifest.architecture !== "aetherion-micro" ||
      manifest.tokenizer_type !== "utf8_bytes" ||
      manifest.vocab_size !== 256 ||
      !Number.isInteger(manifest.window) || manifest.window < 2) {
    throw new Error("Invalid Aetherion byte-level manifest.");
  }
  if (manifest.schema_version === 2) {
    const validGraph = (g, name) => g && g.filename === name && isSha256(g.sha256);
    if (!manifest.graphs ||
        !validGraph(manifest.graphs.prefill, "prefill.onnx") ||
        !validGraph(manifest.graphs.decode, "decode.onnx") ||
        !Number.isInteger(manifest.n_layers) || manifest.n_layers < 1 ||
        !Number.isInteger(manifest.n_heads) || manifest.n_heads < 1 ||
        !Number.isInteger(manifest.head_dim) || manifest.head_dim < 2 ||
        manifest.cache_policy !== "rebuild_relative_rope_at_window_boundary") {
      throw new Error("Unsupported M19/M20 KV-cache graph contract.");
    }
    if (manifest.quantization !== undefined &&
        (!manifest.quantization ||
         manifest.quantization.type !== "weight_only_symmetric_int4" ||
         manifest.quantization.format !== "MatMulNBits" ||
         manifest.quantization.cache_dtype !== "float32" ||
         !Number.isInteger(manifest.quantization.prefill_quantized_matmul_nodes) ||
         manifest.quantization.prefill_quantized_matmul_nodes < 1 ||
         !Number.isInteger(manifest.quantization.decode_quantized_matmul_nodes) ||
         manifest.quantization.decode_quantized_matmul_nodes < 1 ||
         !isSha256(manifest.graphs.prefill.source_sha256) ||
         !isSha256(manifest.graphs.decode.source_sha256))) {
      throw new Error("Unsupported M20 INT4 KV-cache manifest.");
    }
    }
  } else if (manifest.schema_version === 1) {
    if (!isSha256(manifest.onnx_sha256)) {
      throw new Error("Missing M17/M18 model SHA-256.");
    }
    if (manifest.quantization !== undefined &&
        (!manifest.quantization ||
         manifest.quantization.type !== "weight_only_symmetric_int4" ||
         manifest.quantization.format !== "MatMulNBits" ||
         !Number.isInteger(manifest.quantization.quantized_matmul_nodes) ||
         manifest.quantization.quantized_matmul_nodes < 1)) {
      throw new Error("Unrecognized M18 quantization metadata.");
    }
  } else {
    throw new Error("Unsupported mobile manifest version.");
  }
  return manifest;
}

function resetCache() {
  state.cache = null;
  state.cacheLength = 0;
  state.cacheStats = { prefill: 0, decode: 0 };
}

function requireContract(session, expectedInputs, expectedOutputs) {
  if (JSON.stringify(session.inputNames) !== JSON.stringify(expectedInputs) ||
      JSON.stringify(session.outputNames) !== JSON.stringify(expectedOutputs)) {
    throw new Error("ONNX graph input/output names do not match the mobile contract.");
  }
}

async function loadModel() {
  if (state.running) return;
  setRunning(true);
  const previous = [state.session, state.decodeSession];
  state.session = null;
  state.decodeSession = null;
  state.manifest = null;
  resetCache();
  const onnxFile = $("onnx").files[0];
  const decodeFile = $("decode").files[0];
  const manifestFile = $("manifest").files[0];
  let prefillSession = null, decodeSession = null;
  try {
    for (const session of previous) {
      if (session) await session.release();
    }
    if (!onnxFile || !manifestFile) {
      throw new Error("Select a prefill/model ONNX file and manifest.json.");
    }
    if (typeof ort === "undefined") {
      throw new Error("ONNX Runtime is missing. Run npm install and npm run stage-runtime.");
    }
    status("Verifying model files...");
    const manifest = validateManifest(JSON.parse(await manifestFile.text()));
    if (manifest.schema_version === 2 && !decodeFile) {
      throw new Error("M19 requires decode.onnx in addition to prefill.onnx.");
    }
    const firstBytes = await onnxFile.arrayBuffer();
    const firstSha = manifest.schema_version === 2 ?
      manifest.graphs.prefill.sha256 : manifest.onnx_sha256;
    if ((await sha256(firstBytes)) !== firstSha) {
      throw new Error("Prefill/model SHA-256 mismatch.");
    }
    let decodeBytes = null;
    if (manifest.schema_version === 2) {
      decodeBytes = await decodeFile.arrayBuffer();
      if ((await sha256(decodeBytes)) !== manifest.graphs.decode.sha256) {
        throw new Error("Cached decode SHA-256 mismatch.");
      }
    }
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = "./vendor/";
    prefillSession = await ort.InferenceSession.create(new Uint8Array(firstBytes), {
      executionProviders: ["wasm"],
    });
    if (manifest.schema_version === 2) {
      const cacheNames = cacheTensorNames(manifest.n_layers);
      requireContract(prefillSession, ["tokens", "last_position"], ["next_logits", ...cacheNames]);
      decodeSession = await ort.InferenceSession.create(new Uint8Array(decodeBytes), {
        executionProviders: ["wasm"],
      });
      requireContract(decodeSession,
        ["token_ids", "past_length", ...cacheTensorNames(manifest.n_layers, "past_")],
        ["next_logits", ...cacheNames]);
    } else {
      requireContract(prefillSession, ["tokens", "last_position"], ["next_logits"]);
    }
    state.session = prefillSession;
    state.decodeSession = decodeSession;
    state.manifest = manifest;
    const mode = manifest.schema_version === 2 ?
      (manifest.quantization ? "INT4 + FP32 KV cache" : "FP32 + KV cache") :
      (manifest.quantization ? "INT4 full-window" : "FP32 full-window");
    status("Model loaded locally. Window " + manifest.window + " bytes; " + mode +
           ". Conversation quality depends on your actual checkpoint.");
  } catch (error) {
    if (prefillSession) await prefillSession.release().catch(() => {});
    if (decodeSession) await decodeSession.release().catch(() => {});
    status("Load failed: " + error.message, true);
  } finally {
    setRunning(false);
  }
}

function argmax(values) {
  if (values.length !== 256) {
    throw new Error("Expected 256 byte logits; got " + values.length);
  }
  let best = 0;
  for (let i = 1; i < values.length; i++) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}

async function runByteStep(tokens) {
  const window = state.manifest.window;
  if (state.manifest.schema_version === 2 &&
      state.cache && state.cacheLength < window) {
    const feeds = {
      token_ids: new ort.Tensor("int64", BigInt64Array.of(BigInt(tokens.at(-1))), [1, 1]),
      past_length: new ort.Tensor("int64", BigInt64Array.of(BigInt(state.cacheLength)), [1]),
    };
    for (let i = 0; i < state.manifest.n_layers; i++) {
      feeds["past_key_" + i] = state.cache["key_" + i];
      feeds["past_value_" + i] = state.cache["value_" + i];
    }
    const outputs = await state.decodeSession.run(feeds);
    const newCache = {};
    for (const key of cacheTensorNames(state.manifest.n_layers)) {
      newCache[key] = outputs[key];
    }
    state.cache = newCache;
    state.cacheLength++;
    state.cacheStats.decode++;
    return argmax(outputs.next_logits.data);
  }
  const context = tokens.slice(-window);
  const input = new BigInt64Array(window);
  context.forEach((token, j) => { input[j] = BigInt(token); });
  const feeds = {
    tokens: new ort.Tensor("int64", input, [1, window]),
    last_position: new ort.Tensor("int64",
      BigInt64Array.of(BigInt(context.length - 1)), [1]),
  };
  const outputs = await state.session.run(feeds);
  if (state.manifest.schema_version === 2) {
    state.cache = {};
    for (const key of cacheTensorNames(state.manifest.n_layers)) {
      state.cache[key] = outputs[key];
    }
    state.cacheLength = context.length;
    state.cacheStats.prefill++;
  }
  return argmax(outputs.next_logits.data);
}

function displayBenchmark(samples, warmups, tokens, seconds) {
  const sorted = [...samples].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const p95 = sorted[Math.ceil(sorted.length * 0.95) - 1];
  const report = {
    schema_version: 1,
    kind: "aetherion-mobile-synthetic-on-device-benchmark",
    inference_format: state.manifest.schema_version === 2 ?
      (state.manifest.quantization ? "int4_kv_cache" : "fp32_kv_cache") :
      (state.manifest.quantization ? "int4_full_window" : "fp32_full_window"),
    prefill_calls: state.cacheStats.prefill,
    cached_decode_calls: state.cacheStats.decode,
    window_byte_tokens: state.manifest.window,
    model_sha256: state.manifest.schema_version === 2 ?
      state.manifest.graphs.prefill.sha256 : state.manifest.onnx_sha256,
    device_ram_gb_reported_by_browser: navigator.deviceMemory ?? null,
    estimated_kv_cache_bytes: state.manifest.schema_version === 2 ?
      2 * state.manifest.n_layers * state.manifest.n_heads *
        state.manifest.window * state.manifest.head_dim * 4 : 0,
    browser_heap_used_bytes_if_available: performance.memory?.usedJSHeapSize ?? null,
    warmup_steps: warmups,
    measured_steps: tokens,
    elapsed_seconds: Number(seconds.toFixed(4)),
    tokens_per_second: Number((tokens / seconds).toFixed(3)),
    median_step_ms: Number(median.toFixed(2)),
    p95_step_ms: Number(p95.toFixed(2)),
    user_agent: navigator.userAgent,
    disclaimer: "Local synthetic benchmark, not a validated device matrix or assistant quality measurement.",
  };
  $("benchmark-report").textContent = JSON.stringify(report, null, 2);
}

async function benchmark() {
  if (!state.session || state.running) return;
  state.cancel = false;
  resetCache();
  setRunning(true);
  $("benchmark-report").textContent = "";
  status("Benchmark: warming up on the current device...");
  // Start with one byte, so M19/M20 cache decode is actually exercised.
  // A long prompt would fill a small test window and force prefill every step.
  const tokens = [65];
  const warmups = 2;
  const steps = 16;
  const timings = [];
  let completed = 0;
  try {
    for (let i = 0; i < warmups + steps && !state.cancel; i++) {
      const begin = performance.now();
      const next = await runByteStep(tokens);
      const elapsed = performance.now() - begin;
      tokens.push(next);
      if (i >= warmups) {
        timings.push(elapsed);
        completed++;
      }
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    if (completed) {
      const seconds = timings.reduce((a, b) => a + b, 0) / 1000;
      displayBenchmark(timings, Math.min(warmups, tokens.length), completed, seconds);
    }
    status(state.cancel ? "Benchmark interrupted." :
           "Benchmark completed locally; measurements are for this browser and model only.");
  } catch (error) {
    status("Benchmark failed: " + error.message, true);
  } finally {
    setRunning(false);
  }
}

async function generate() {
  if (!state.session || state.running) return;
  const prompt = $("prompt").value;
  const total = Number($("count").value);
  if (!Number.isInteger(total) || total < 1 || total > 64) {
    status("Token count must be between 1 and 64.", true);
    return;
  }
  const tokens = Array.from(encoder.encode(prompt));
  if (tokens.length === 0) {
    status("Enter a nonempty prompt.", true);
    return;
  }
  state.cancel = false;
  resetCache();
  setRunning(true);
  status("Running locally in this browser...");
  $("output").textContent = prompt;
  const start = performance.now();
  let generated = 0;
  try {
    for (let i = 0; i < total && !state.cancel; i++) {
      const next = await runByteStep(tokens);
      tokens.push(next);
      generated++;
      $("output").textContent = decoder.decode(Uint8Array.from(tokens));
      // Allow rendering and cancellation between model steps.
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    const seconds = (performance.now() - start) / 1000;
    status((state.cancel ? "Stopped. " : "Complete. ") +
           generated + " tokens in " + seconds.toFixed(2) + " seconds on device (" +
           (generated / Math.max(seconds, 1e-9)).toFixed(2) + " byte tokens/s, including UI scheduling).");
  } catch (error) {
    status("Inference failed: " + error.message, true);
  } finally {
    setRunning(false);
  }
}

$("load").addEventListener("click", loadModel);
$("generate").addEventListener("click", generate);
$("benchmark").addEventListener("click", benchmark);
$("stop").addEventListener("click", () => { state.cancel = true; });
if ("serviceWorker" in navigator && location.protocol === "https:") {
  navigator.serviceWorker.register("./sw.js").catch(() => {});
}
status("Select an exported checkpoint model and its manifest. No network inference API is used.");
