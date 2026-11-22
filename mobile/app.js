"use strict";

const $ = (id) => document.getElementById(id);
const state = { session: null, decodeSession: null, manifest: null, cache: null,
  cacheLength: 0, cacheStats: { prefill: 0, decode: 0 }, running: false, cancel: false,
  modelBytes: 0, modelLoadAndVerifyMs: null, lastBenchmark: null,
  saveCandidate: null, hasStoredModel: false, storageBusy: false };
const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8", { fatal: false });

function status(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}

function updateVaultButtons() {
  const busy = state.running || state.storageBusy || !modelVault.available();
  $("save-model").disabled = busy || !state.saveCandidate;
  $("restore-model").disabled = busy || !state.hasStoredModel;
  $("delete-model").disabled = busy || !state.hasStoredModel;
}

function setRunning(running) {
  state.running = running;
  $("generate").disabled = running || !state.session;
  $("benchmark").disabled = running || !state.session;
  $("load").disabled = running;
  $("stop").disabled = !running;
  updateVaultButtons();
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
  if (manifest.schema_version === 2 || manifest.schema_version === 3) {
    const v3 = manifest.schema_version === 3;
    const prefillName = v3 ? "prefill.int4.onnx" : "prefill.onnx";
    const decodeName = v3 ? "decode.int4.onnx" : "decode.onnx";
    const validGraph = (graph, expected) =>
      graph && graph.filename === expected && isSha256(graph.sha256);
    if (!manifest.graphs ||
        !validGraph(manifest.graphs.prefill, prefillName) ||
        !validGraph(manifest.graphs.decode, decodeName) ||
        !Number.isInteger(manifest.n_layers) || manifest.n_layers < 1 ||
        !Number.isInteger(manifest.n_heads) || manifest.n_heads < 1 ||
        !Number.isInteger(manifest.head_dim) || manifest.head_dim < 2 ||
        manifest.cache_policy !== "rebuild_relative_rope_at_window_boundary") {
      throw new Error("Unsupported M19/M20 KV-cache graph contract.");
    }
    if (v3) {
      const q = manifest.quantization;
      if (!q || q.type !== "weight_only_symmetric_int4" ||
          q.format !== "MatMulNBits" || q.kv_cache_dtype !== "float32" ||
          !Number.isInteger(q.block_size) || q.block_size < 16 ||
          (q.block_size & (q.block_size - 1)) !== 0 ||
          !q.converted_matmul_nodes ||
          !Number.isInteger(q.converted_matmul_nodes.prefill) ||
          !Number.isInteger(q.converted_matmul_nodes.decode) ||
          q.converted_matmul_nodes.prefill < 1 ||
          q.converted_matmul_nodes.decode < 1 ||
          !manifest.source_graphs ||
          !isSha256(manifest.source_graphs.prefill?.sha256) ||
          !isSha256(manifest.source_graphs.decode?.sha256)) {
        throw new Error("Invalid M20 INT4 metadata.");
      }
    } else if (manifest.quantization !== undefined) {
      throw new Error("M19 schema 2 cannot contain INT4 metadata.");
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
         manifest.quantization.quantized_matmul_nodes < 1 ||
         !isSha256(manifest.source_onnx_sha256))) {
      throw new Error("Unrecognized M18 quantization metadata.");
    }
  } else {
    throw new Error("Unsupported mobile manifest version.");
  }
  return manifest;
}

function isCachedManifest(manifest) {
  return manifest.schema_version === 2 || manifest.schema_version === 3;
}

// Fail before reading ONNX bytes or allocating sessions on large imports.
function enforceImportBudget(manifest, prefillSize, decodeSize = 0) {
  if (!Number.isSafeInteger(prefillSize) || prefillSize <= 0 ||
      !Number.isSafeInteger(decodeSize) || decodeSize < 0) {
    throw new Error("Model files must have valid nonempty byte sizes.");
  }
  if (isCachedManifest(manifest) && decodeSize === 0) {
    throw new Error("Cached models require a nonempty decode ONNX graph.");
  }
  if (prefillSize + decodeSize > 256 * 1024 * 1024) {
    throw new Error("Combined ONNX model exceeds the 256 MiB import safety limit.");
  }
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

async function loadModel(source = null) {
  if (state.running) return;
  setRunning(true);
  const previous = [state.session, state.decodeSession];
  state.session = null;
  state.decodeSession = null;
  state.manifest = null;
  state.lastBenchmark = null;
  state.modelBytes = 0;
  state.modelLoadAndVerifyMs = null;
  state.saveCandidate = null;
  updateVaultButtons();
  $("download-report").disabled = true;
  resetCache();
  const fromStorage = source !== null;
  const onnxFile = fromStorage ? null : $("onnx").files[0];
  const decodeFile = fromStorage ? null : $("decode").files[0];
  const manifestFile = fromStorage ? null : $("manifest").files[0];
  let prefillSession = null, decodeSession = null;
  try {
    for (const session of previous) {
      if (session) await session.release();
    }
    if (!fromStorage && (!onnxFile || !manifestFile)) {
      throw new Error("Select a prefill/model ONNX file and manifest.json.");
    }
    if (typeof ort === "undefined") {
      throw new Error("ONNX Runtime is missing. Run npm install and npm run stage-runtime.");
    }
    status("Verifying model files...");
    const loadingStart = performance.now();
    if (!fromStorage && manifestFile.size > 128 * 1024) {
      throw new Error("Model manifest exceeds the 128 KiB import limit.");
    }
    if (fromStorage && (typeof source.manifestText !== "string" ||
        source.manifestText.length > 128 * 1024)) {
      throw new Error("Stored model manifest exceeds the 128 KiB limit.");
    }
    const manifest = validateManifest(JSON.parse(
      fromStorage ? source.manifestText : await manifestFile.text()));
    if (isCachedManifest(manifest) && !fromStorage && !decodeFile) {
      throw new Error("M19 requires decode.onnx in addition to prefill.onnx.");
    }
    enforceImportBudget(manifest,
      fromStorage ? source.prefillBytes?.byteLength : onnxFile.size,
      isCachedManifest(manifest) ?
        (fromStorage ? source.decodeBytes?.byteLength : decodeFile.size) : 0);
    const firstBytes = fromStorage ? source.prefillBytes : await onnxFile.arrayBuffer();
    const firstSha = isCachedManifest(manifest) ?
      manifest.graphs.prefill.sha256 : manifest.onnx_sha256;
    if ((await sha256(firstBytes)) !== firstSha) {
      throw new Error("Prefill/model SHA-256 mismatch.");
    }
    let decodeBytes = null;
    if (isCachedManifest(manifest)) {
      decodeBytes = fromStorage ? source.decodeBytes : await decodeFile.arrayBuffer();
      if (!decodeBytes || (await sha256(decodeBytes)) !== manifest.graphs.decode.sha256) {
        throw new Error("Cached decode SHA-256 mismatch.");
      }
    }
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = "./vendor/";
    prefillSession = await ort.InferenceSession.create(new Uint8Array(firstBytes), {
      executionProviders: ["wasm"],
    });
    if (isCachedManifest(manifest)) {
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
    state.modelBytes = firstBytes.byteLength + (decodeBytes ? decodeBytes.byteLength : 0);
    state.modelLoadAndVerifyMs = performance.now() - loadingStart;
    state.saveCandidate = fromStorage ? null : { onnxFile, decodeFile, manifestFile };
    updateVaultButtons();
    const mode = manifest.schema_version === 3 ? "INT4 weights + FP32 KV cache" :
      manifest.schema_version === 2 ? "FP32 + KV cache" :
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

async function refreshSavedModelStatus() {
  if (!modelVault.available()) {
    $("vault-note").textContent = "Local browser model storage is unavailable. You can still import files each session.";
    updateVaultButtons();
    return;
  }
  try {
    state.hasStoredModel = await modelVault.exists();
    $("vault-note").textContent = state.hasStoredModel ?
      "A saved model exists on this browser origin. Restore or delete it explicitly." :
      "No model saved on this device. Import and verify files, then choose Save local copy.";
  } catch (error) {
    state.hasStoredModel = false;
    $("vault-note").textContent = "Local storage unavailable: " + error.message;
  }
  updateVaultButtons();
}

async function saveLoadedModel() {
  if (state.running || state.storageBusy || !state.saveCandidate) return;
  state.storageBusy = true;
  updateVaultButtons();
  try {
    const { onnxFile, decodeFile, manifestFile } = state.saveCandidate;
    const size = onnxFile.size + (isCachedManifest(state.manifest) ? decodeFile.size : 0);
    if (size > 256 * 1024 * 1024) {
      throw new Error("Model exceeds the 256 MiB offline storage safety limit.");
    }
    const manifestText = await manifestFile.text();
    const manifest = validateManifest(JSON.parse(manifestText));
    if (JSON.stringify(manifest) !== JSON.stringify(state.manifest)) {
      throw new Error("Imported manifest changed since validation.");
    }
    const prefillBytes = await onnxFile.arrayBuffer();
    const decodeBytes = isCachedManifest(manifest) ?
      await decodeFile.arrayBuffer() : null;
    const expectedPrefill = isCachedManifest(manifest) ?
      manifest.graphs.prefill.sha256 : manifest.onnx_sha256;
    if ((await sha256(prefillBytes)) !== expectedPrefill ||
        (decodeBytes && (await sha256(decodeBytes)) !== manifest.graphs.decode.sha256)) {
      throw new Error("Model integrity changed since import.");
    }
    await modelVault.save({ manifestText, prefillBytes, decodeBytes });
    state.hasStoredModel = true;
    $("vault-note").textContent = "Model saved locally to this website's browser storage. It is not encrypted.";
    status("Model saved on this browser. The storage copy can be deleted independently of the loaded model.");
  } catch (error) {
    status("Saving local model failed: " + error.message, true);
  } finally {
    state.storageBusy = false;
    updateVaultButtons();
  }
}

async function restoreSavedModel() {
  if (state.running || state.storageBusy || !state.hasStoredModel) return;
  state.storageBusy = true;
  updateVaultButtons();
  try {
    const stored = await modelVault.read();
    if (!stored) throw new Error("No local model is stored.");
    state.storageBusy = false;
    updateVaultButtons();
    await loadModel(stored);  // Always recheck SHA-256 and ONNX graph contracts.
  } catch (error) {
    status("Restoring local model failed: " + error.message, true);
  } finally {
    state.storageBusy = false;
    updateVaultButtons();
  }
}

async function deleteSavedModel() {
  if (state.running || state.storageBusy || !state.hasStoredModel) return;
  state.storageBusy = true;
  updateVaultButtons();
  try {
    await modelVault.remove();
    state.hasStoredModel = false;
    $("vault-note").textContent = "Stored model deleted. The active inference session, if any, is unaffected.";
    status("Browser-stored model deleted; browser or device backups may retain copies.");
  } catch (error) {
    status("Deleting stored model failed: " + error.message, true);
  } finally {
    state.storageBusy = false;
    updateVaultButtons();
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
  if (isCachedManifest(state.manifest) &&
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
  if (isCachedManifest(state.manifest)) {
    state.cache = {};
    for (const key of cacheTensorNames(state.manifest.n_layers)) {
      state.cache[key] = outputs[key];
    }
    state.cacheLength = context.length;
    state.cacheStats.prefill++;
  }
  return argmax(outputs.next_logits.data);
}

function summarizeLatencies(samples) {
  if (!samples.length) return null;
  const sorted = [...samples].sort((a, b) => a - b);
  const sum = sorted.reduce((a, b) => a + b, 0);
  const round = (value) => Number(value.toFixed(3));
  return {
    samples: samples.length,
    min_ms: round(sorted[0]),
    mean_ms: round(sum / samples.length),
    median_ms: round(sorted.length % 2 ?
      sorted[Math.floor(sorted.length / 2)] :
      (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2),
    p95_ms: round(sorted[Math.ceil(sorted.length * 0.95) - 1]),
    max_ms: round(sorted[sorted.length - 1]),
  };
}

function browserHeapBytes() {
  // performance.memory is nonstandard and absent on most iOS browsers.
  const bytes = performance?.memory?.usedJSHeapSize;
  return Number.isSafeInteger(bytes) && bytes >= 0 ? bytes : null;
}

function modelFormat(manifest) {
  return manifest.schema_version === 3 ? "int4_kv_cache" :
    manifest.schema_version === 2 ? "fp32_kv_cache" :
    (manifest.quantization ? "int4_full_window" : "fp32_full_window");
}

function sourceModelSha256(manifest) {
  // Link INT4 and FP32 exports only when their manifests prove common origin.
  // Do not infer provenance from the name or the locally imported file.
  if (manifest.schema_version === 3) return manifest.source_graphs.prefill.sha256;
  if (manifest.schema_version === 1 && manifest.quantization) {
    return manifest.source_onnx_sha256;
  }
  return isCachedManifest(manifest) ?
    manifest.graphs.prefill.sha256 : manifest.onnx_sha256;
}

function displayBenchmark(phases, warmups, elapsedMs, memoryBefore, memoryAfter, interrupted) {
  const all = [...phases.prefill, ...phases.decode, ...phases.full_window];
  const measured = all.length;
  const cacheBytes = isCachedManifest(state.manifest) ?
    2 * state.manifest.n_layers * state.manifest.n_heads *
      state.manifest.window * state.manifest.head_dim * 4 : 0;
  const report = {
    schema_version: 3,
    kind: "aetherion-mobile-local-browser-inference-benchmark",
    benchmark_protocol: "hi_warmup2_16byte_greedy_v1",
    generated_at_utc: new Date().toISOString(),
    scope: "offline-local-single-thread-wasm-byte-token-inference",
    inference_format: modelFormat(state.manifest),
    model_sha256: isCachedManifest(state.manifest) ?
      state.manifest.graphs.prefill.sha256 : state.manifest.onnx_sha256,
    source_model_sha256: sourceModelSha256(state.manifest),
    model_manifest_schema: state.manifest.schema_version,
    model_file_bytes: state.modelBytes,
    model_load_and_sha_verify_ms: state.modelLoadAndVerifyMs === null ?
      null : Number(state.modelLoadAndVerifyMs.toFixed(3)),
    window_byte_tokens: state.manifest.window,
    warmup_steps_completed: warmups,
    measured_steps: measured,
    interrupted: Boolean(interrupted),
    prefill_calls_including_warmup: state.cacheStats.prefill,
    cached_decode_calls_including_warmup: state.cacheStats.decode,
    elapsed_inference_and_feeds_seconds: Number((elapsedMs / 1000).toFixed(4)),
    byte_tokens_per_inference_second: measured ?
      Number((1000 * measured / elapsedMs).toFixed(3)) : null,
    latency_ms: {
      overall: summarizeLatencies(all),
      prefill: summarizeLatencies(phases.prefill),
      decode: summarizeLatencies(phases.decode),
      full_window: summarizeLatencies(phases.full_window),
    },
    memory: {
      estimated_one_copy_fp32_kv_cache_bytes: cacheBytes,
      browser_js_heap_before_bytes_if_available: memoryBefore,
      browser_js_heap_after_bytes_if_available: memoryAfter,
      device_ram_gb_reported_by_browser: navigator.deviceMemory ?? null,
      limitation: "KV estimate excludes temporary/multiple caches, weights, WASM heap and browser overhead. JS heap is not measured peak process RAM.",
    },
    device: {
      user_agent: navigator.userAgent ?? null,
      hardware_concurrency_reported: navigator.hardwareConcurrency ?? null,
      installed_standalone: (typeof window !== "undefined" &&
        window.matchMedia?.("(display-mode: standalone)")?.matches) ||
        navigator.standalone === true,
    },
    disclaimer: "Untrained-model technical benchmark only. No network upload. Byte tokens are not LLM BPE tokens; no language-quality, battery-life or cross-device speed claims.",
  };
  state.lastBenchmark = report;
  $("benchmark-report").textContent = JSON.stringify(report, null, 2);
  $("download-report").disabled = false;
  return report;
}

function downloadBenchmark() {
  if (!state.lastBenchmark) return;
  // No automatic telemetry. JSON is saved only from an explicit user click.
  const blob = new Blob([JSON.stringify(state.lastBenchmark, null, 2) + "\n"],
                        { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "aetherion-m22-device-benchmark.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function benchmark() {
  if (!state.session || state.running) return;
  state.cancel = false;
  resetCache();
  state.lastBenchmark = null;
  $("download-report").disabled = true;
  setRunning(true);
  $("benchmark-report").textContent = "";
  status("Benchmark: warming up on the current device...");
  // Short prompt tests real cached decoding with small context fixtures.
  const tokens = Array.from(encoder.encode("Hi"));
  const warmups = 2;
  const steps = 16;
  const phases = { prefill: [], decode: [], full_window: [] };
  const memoryBefore = browserHeapBytes();
  let completed = 0;
  let warmed = 0;
  try {
    for (let i = 0; i < warmups + steps && !state.cancel; i++) {
      const type = !isCachedManifest(state.manifest) ? "full_window" :
        state.cache && state.cacheLength < state.manifest.window ? "decode" : "prefill";
      const begin = performance.now();
      const next = await runByteStep(tokens);
      const elapsed = performance.now() - begin;
      tokens.push(next);
      if (i >= warmups) {
        phases[type].push(elapsed);
        completed++;
      } else {
        warmed++;
      }
      // Excluded from timing. Allows UI paint and stop between inference calls.
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    if (completed) {
      const elapsedMs = Object.values(phases).flat().reduce((a, b) => a + b, 0);
      displayBenchmark(phases, warmed,
        elapsedMs, memoryBefore, browserHeapBytes(), state.cancel);
    }
    status(state.cancel ? "Benchmark interrupted (partial results saved if measured)." :
           "Benchmark complete. Export JSON to compare the same model on another phone.");
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

$("check-device").addEventListener("click", async () => {
  $("check-device").disabled = true;
  try {
    const report = await deviceCheck.inspect();
    $("device-report").textContent = JSON.stringify(report, null, 2);
  } catch (error) {
    $("device-report").textContent = "Local device check failed: " + error.message;
  } finally {
    $("check-device").disabled = false;
  }
});
$("load").addEventListener("click", () => loadModel());
$("save-model").addEventListener("click", saveLoadedModel);
$("restore-model").addEventListener("click", restoreSavedModel);
$("delete-model").addEventListener("click", deleteSavedModel);
$("generate").addEventListener("click", generate);
$("benchmark").addEventListener("click", benchmark);
$("download-report").addEventListener("click", downloadBenchmark);
$("stop").addEventListener("click", () => { state.cancel = true; });
let deferredInstallPrompt = null;
if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
  window.addEventListener("beforeinstallprompt", (event) => {
    event.preventDefault();
    deferredInstallPrompt = event;
    $("install").hidden = false;
  });
  window.addEventListener("appinstalled", () => {
    deferredInstallPrompt = null;
    $("install").hidden = true;
    $("install-note").textContent = "Aetherion has been added to this device.";
  });
}
$("install").addEventListener("click", async () => {
  if (!deferredInstallPrompt) return;
  const event = deferredInstallPrompt;
  deferredInstallPrompt = null;
  $("install").hidden = true;
  await event.prompt();
  await event.userChoice;
});
if ("serviceWorker" in navigator && (location.protocol === "https:" || location.hostname === "localhost")) {
  navigator.serviceWorker.register("./sw.js").catch(() => {});
}
status("Select an exported checkpoint model and its manifest. No network inference API is used.");
refreshSavedModelStatus();
