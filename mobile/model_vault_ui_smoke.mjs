// M23 browser/UI flow smoke: load, explicit save, relaunch/restore, tamper rejection.
// Deterministic mock IDB/ORT, not a physical phone or real ONNX execution.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { createContext, Script } from "node:vm";

function makeStorage() {
  let current = null, created = false, writes = 0;
  return {
    get writes() { return writes; },
    corrupt() {
      assert.ok(current);
      current = { ...current, prefillBytes: new Uint8Array([99]).buffer };
    },
    open(name, version) {
      assert.equal(name, "aetherion-local-model-v1");
      assert.equal(version, 1);
      const req = {};
      queueMicrotask(() => {
        const db = {
          objectStoreNames: { contains: () => created },
          createObjectStore() { created = true; },
          close() {},
          transaction(storeName, mode) {
            assert.equal(storeName, "models");
            const tx = { error: null };
            tx.objectStore = () => {
              const doRequest = (action, value, key) => {
                assert.equal(key, "selected");
                const op = {};
                queueMicrotask(() => {
                  try {
                    if (action === "put") {
                      assert.equal(mode, "readwrite");
                      current = structuredClone(value);
                      writes++;
                    } else if (action === "delete") current = null;
                    else if (action === "count") op.result = current ? 1 : 0;
                    else if (action === "get") op.result =
                      current ? structuredClone(current) : undefined;
                    op.onsuccess?.();
                    tx.oncomplete?.();
                  } catch (err) {
                    tx.error = err;
                    tx.onabort?.();
                  }
                });
                return op;
              };
              return {
                put: (value, key) => doRequest("put", value, key),
                delete: (key) => doRequest("delete", null, key),
                count: (key) => doRequest("count", null, key),
                get: (key) => doRequest("get", null, key),
              };
            };
            return tx;
          },
        };
        req.result = db;
        if (!created) req.onupgradeneeded?.();
        req.onsuccess?.();
      });
      return req;
    },
  };
}

function makeFile(data, name) {
  const bytes = Uint8Array.from(data);
  return {
    name, size: bytes.byteLength,
    async arrayBuffer() { return bytes.buffer.slice(0); },
    async text() { return new TextDecoder().decode(bytes); },
  };
}

function hash(data) {
  return createHash("sha256").update(data).digest("hex");
}

function makeApp(storage) {
  const elements = new Map();
  const dom = {
    getElementById(id) {
      if (!elements.has(id)) {
        const handlers = new Map();
        elements.set(id, {
          disabled: false, textContent: "", files: [], value: "", hidden: false,
          handlers,
          classList: { toggle() {} },
          addEventListener(name, callback) { handlers.set(name, callback); },
        });
      }
      return elements.get(id);
    },
  };
  let sessionCreations = 0;
  const ort = {
    env: { wasm: {} },
    InferenceSession: {
      async create(bytes) {
        sessionCreations++;
        const marker = bytes[0];
        const cached = marker === 10 || marker === 11;
        return {
          inputNames: marker === 11 ?
            ["token_ids", "past_length", "past_key_0", "past_value_0"] :
            ["tokens", "last_position"],
          outputNames: cached ?
            ["next_logits", "key_0", "value_0"] : ["next_logits"],
          async release() {},
        };
      },
    },
  };
  const context = createContext({
    console, ArrayBuffer, Uint8Array, Float32Array, BigInt64Array,
    TextEncoder, TextDecoder, Promise, Date, Error, indexedDB: storage,
    navigator: {}, location: { protocol: "http:", hostname: "localhost" },
    document: dom, ort,
    crypto: {
      subtle: {
        async digest(algorithm, buffer) {
          assert.equal(algorithm, "SHA-256");
          const digest = createHash("sha256").update(Buffer.from(buffer)).digest();
          return digest.buffer.slice(digest.byteOffset, digest.byteOffset + digest.byteLength);
        },
      },
    },
    performance: { now: () => 1.5 },
    setTimeout,
  });
  for (const file of ["model-vault.js", "app.js"]) {
    new Script(readFileSync(new URL("./" + file, import.meta.url), "utf8"), {
      filename: file,
    }).runInContext(context);
  }
  return {
    node: dom.getElementById.bind(dom),
    get state() { return new Script("state").runInContext(context); },
    get sessions() { return sessionCreations; },
    async click(id) {
      const handler = dom.getElementById(id).handlers.get("click");
      assert.ok(handler, "Missing event handler " + id);
      await handler();
    },
    async flush() {
      await new Promise((resolve) => setImmediate(resolve));
    },
  };
}

const storage = makeStorage();
const first = makeApp(storage);
await first.flush();
assert.equal(storage.writes, 0, "No implicit model persistence");

// M26: reject unsafe advertised window before any ONNX session creation.
const oversize = makeApp(storage);
await oversize.flush();
const dummyModel = Uint8Array.from([1, 2, 3]);
oversize.node("onnx").files = [makeFile(dummyModel, "model.onnx")];
oversize.node("manifest").files = [makeFile(
  new TextEncoder().encode(JSON.stringify({
    schema_version: 1, architecture: "aetherion-micro",
    tokenizer_type: "utf8_bytes", vocab_size: 256,
    window: 4097, onnx_sha256: hash(dummyModel),
  })), "manifest.json",
)];
await oversize.click("load");
assert.equal(oversize.sessions, 0, "Oversized context must fail before ONNX initialization");
assert.match(oversize.node("status").textContent, /4096/);
assert.equal(storage.writes, 0, "Failed import must not persist model bytes");
assert.equal(first.node("restore-model").disabled, true);
const source = Uint8Array.from([1, 2, 3, 4, 5]);
const manifest = {
  schema_version: 1, architecture: "aetherion-micro",
  tokenizer_type: "utf8_bytes", vocab_size: 256, window: 4,
  onnx_sha256: hash(source),
};
first.node("onnx").files = [makeFile(source, "model.onnx")];
first.node("manifest").files = [makeFile(
  new TextEncoder().encode(JSON.stringify(manifest)), "manifest.json",
)];
await first.click("load");
assert.equal(first.sessions, 1);
assert.equal(first.node("generate").disabled, false);
assert.equal(first.node("save-model").disabled, false);
assert.equal(storage.writes, 0);
await first.click("save-model");
assert.equal(storage.writes, 1, "Only a dedicated save click may write weights");
assert.equal(first.node("restore-model").disabled, false);

// A new browser app session retains the origin-scoped saved model record.
const reloaded = makeApp(storage);
await reloaded.flush();
assert.equal(reloaded.node("restore-model").disabled, false);
await reloaded.click("restore-model");
assert.equal(reloaded.sessions, 1);
assert.equal(reloaded.node("generate").disabled, false);
assert.equal(reloaded.node("save-model").disabled, true,
  "Restoring must not silently rewrite the saved model");
assert.equal(storage.writes, 1);
await reloaded.click("delete-model");
assert.equal(reloaded.node("restore-model").disabled, true);
assert.equal(reloaded.node("generate").disabled, false,
  "Deleting the disk copy must not silently terminate active inference");

// Tampering with the saved bytes must prevent creating the ONNX session.
await first.click("save-model");
storage.corrupt();
const tampered = makeApp(storage);
await tampered.flush();
await tampered.click("restore-model");
assert.equal(tampered.sessions, 0);
assert.equal(tampered.node("generate").disabled, true);
assert.match(tampered.node("status").textContent, /SHA-256 mismatch/i);

// Cached prefill/decode pair: save + restore both graphs and contracts.
const cached = makeApp(storage);
await cached.flush();
const prefill = Uint8Array.from([10, 20, 30]);
const decode = Uint8Array.from([11, 22, 33]);
const cachedManifest = {
  schema_version: 2, architecture: "aetherion-micro",
  tokenizer_type: "utf8_bytes", vocab_size: 256, window: 4,
  n_layers: 1, n_heads: 4, head_dim: 8,
  cache_policy: "rebuild_relative_rope_at_window_boundary",
  graphs: {
    prefill: { filename: "prefill.onnx", sha256: hash(prefill) },
    decode: { filename: "decode.onnx", sha256: hash(decode) },
  },
};
cached.node("onnx").files = [makeFile(prefill, "prefill.onnx")];
cached.node("decode").files = [makeFile(decode, "decode.onnx")];
cached.node("manifest").files = [makeFile(
  new TextEncoder().encode(JSON.stringify(cachedManifest)), "manifest.json",
)];
await cached.click("load");
assert.equal(cached.sessions, 2);
await cached.click("save-model");
const cachedRestart = makeApp(storage);
await cachedRestart.flush();
await cachedRestart.click("restore-model");
assert.equal(cachedRestart.sessions, 2, "Dual ONNX prefill/decode must be restored");
assert.equal(cachedRestart.state.manifest.schema_version, 2);
assert.equal(cachedRestart.node("generate").disabled, false);

console.log("M23 UI load/save/relaunch/restore/delete and tampered-ONNX rejection smoke passed");
