// M24: Local device API preflight. No network calls, fingerprints or telemetry.
"use strict";
const deviceCheck = (() => {
  const WASM_HEADER = Uint8Array.from([0, 97, 115, 109, 1, 0, 0, 0]);

  async function inspect(env = globalThis) {
    const nav = env.navigator || {};
    const wasm = env.WebAssembly;
    let wasmApi = false;
    try {
      wasmApi = Boolean(wasm && typeof wasm.validate === "function" &&
        typeof wasm.instantiate === "function" && wasm.validate(WASM_HEADER));
    } catch {
      wasmApi = false;
    }
    const required = {
      secure_context: env.isSecureContext === true,
      webassembly: wasmApi,
      bigint64_array: typeof env.BigInt64Array === "function",
      sha256_subtle_crypto: typeof env.crypto?.subtle?.digest === "function",
      onnx_runtime_wasm_script: typeof env.ort?.InferenceSession?.create === "function",
    };
    const missing = Object.entries(required)
      .filter(([, supported]) => !supported).map(([name]) => name);
    let quota = null;
    let usage = null;
    try {
      if (typeof nav.storage?.estimate === "function") {
        const estimate = await nav.storage.estimate();
        if (Number.isSafeInteger(estimate?.quota) && estimate.quota >= 0) {
          quota = estimate.quota;
        }
        if (Number.isSafeInteger(estimate?.usage) && estimate.usage >= 0) {
          usage = estimate.usage;
        }
      }
    } catch {
      // A storage estimate is optional, not a requirement for inference.
    }
    const memory = Number.isFinite(nav.deviceMemory) && nav.deviceMemory > 0 ?
      nav.deviceMemory : null;
    return {
      schema_version: 1,
      kind: "aetherion-local-mobile-api-preflight",
      status: missing.length ? "missing_required_apis" : "environment_ready_to_attempt",
      required,
      missing_required_apis: missing,
      optional: {
        indexeddb_for_opt_in_model_save: typeof env.indexedDB?.open === "function",
        service_worker_for_offline_shell: "serviceWorker" in nav,
        browser_reported_memory_gb_if_available: memory,
        browser_storage_quota_bytes_if_available: quota,
        browser_storage_usage_bytes_if_available: usage,
      },
      limitations: "API availability is not a device benchmark. Memory sufficiency, " +
        "ONNX operator support, persistent storage, real offline operation and " +
        "usable pretrained model quality require separate validation. No data is uploaded.",
    };
  }
  return { inspect };
})();
