"use strict";

const $ = (id) => document.getElementById(id);
const state = { session: null, manifest: null, running: false, cancel: false };
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

function validateManifest(manifest) {
  if (!manifest || manifest.schema_version !== 1 ||
      manifest.architecture !== "aetherion-micro" ||
      manifest.tokenizer_type !== "utf8_bytes" ||
      manifest.vocab_size !== 256 ||
      !Number.isInteger(manifest.window) || manifest.window < 2 ||
      !/^[a-f0-9]{64}$/.test(manifest.onnx_sha256 || "")) {
    throw new Error("Invalid or incompatible Aetherion M17 manifest.");
  }
  if (manifest.quantization !== undefined &&
      (!manifest.quantization ||
       manifest.quantization.type !== "weight_only_symmetric_int4" ||
       manifest.quantization.format !== "MatMulNBits" ||
       !Number.isInteger(manifest.quantization.quantized_matmul_nodes) ||
       manifest.quantization.quantized_matmul_nodes < 1)) {
    throw new Error("Unrecognized quantization metadata.");
  }
  return manifest;
}

async function loadModel() {
  if (state.running) return;
  setRunning(true);
  const oldSession = state.session;
  state.session = null;
  state.manifest = null;
  const onnxFile = $("onnx").files[0];
  const manifestFile = $("manifest").files[0];
  try {
    if (oldSession) await oldSession.release();
    if (!onnxFile || !manifestFile) {
      throw new Error("Select both model.onnx and manifest.json.");
    }
    if (typeof ort === "undefined") {
      throw new Error("ONNX Runtime is missing. Run npm install and npm run stage-runtime.");
    }
    status("Reading and verifying model...");
    const manifest = validateManifest(JSON.parse(await manifestFile.text()));
    const bytes = await onnxFile.arrayBuffer();
    if ((await sha256(bytes)) !== manifest.onnx_sha256) {
      throw new Error("Model SHA-256 mismatch. Refusing to load the wrong or modified model.");
    }
    ort.env.wasm.numThreads = 1; // No SharedArrayBuffer or cross-origin isolation needed.
    ort.env.wasm.wasmPaths = "./vendor/";
    const session = await ort.InferenceSession.create(new Uint8Array(bytes), {
      executionProviders: ["wasm"],
    });
    if (!session.inputNames.includes("tokens") ||
        !session.inputNames.includes("last_position") ||
        !session.outputNames.includes("next_logits")) {
      await session.release();
      throw new Error("ONNX input/output names do not match the mobile contract.");
    }
    state.session = session;
    state.manifest = manifest;
    status("Model loaded locally. Window: " + manifest.window +
           " byte tokens. Format: " +
           (manifest.quantization ? "INT4 MatMulNBits" : "FP32") +
           ". Output quality depends on the supplied checkpoint.");
  } catch (error) {
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
  const context = tokens.slice(-window);
  const input = new BigInt64Array(window);
  context.forEach((token, j) => { input[j] = BigInt(token); });
  const feeds = {
    tokens: new ort.Tensor("int64", input, [1, window]),
    last_position: new ort.Tensor("int64",
      BigInt64Array.from([BigInt(context.length - 1)]), [1]),
  };
  const results = await state.session.run(feeds);
  return argmax(results.next_logits.data);
}

function displayBenchmark(samples, warmups, tokens, seconds) {
  const sorted = [...samples].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const p95 = sorted[Math.ceil(sorted.length * 0.95) - 1];
  const report = {
    schema_version: 1,
    kind: "aetherion-mobile-synthetic-on-device-benchmark",
    inference_format: state.manifest.quantization ? "int4" : "fp32",
    window_byte_tokens: state.manifest.window,
    model_sha256: state.manifest.onnx_sha256,
    device_ram_gb_reported_by_browser: navigator.deviceMemory ?? null,
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
  setRunning(true);
  $("benchmark-report").textContent = "";
  status("Benchmark: warming up on the current device...");
  const tokens = Array.from(encoder.encode("Aetherion M18 device test"));
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
