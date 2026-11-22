// Node.js smoke gate for the same ONNX Runtime Web/WASM package shipped to phones.
// This is NOT a substitute for physical Android/iPhone testing.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-web";

const root = resolve(process.argv[2] || "mobile/artifacts/ci");
const runtime = join(dirname(fileURLToPath(import.meta.url)), "vendor");
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = runtime + "/";

async function testModel(filename) {
  const bytes = new Uint8Array(readFileSync(filename));
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
  });
  try {
    assert.deepEqual(session.inputNames, ["tokens", "last_position"]);
    assert.deepEqual(session.outputNames, ["next_logits"]);
    const tokens = new BigInt64Array(8);
    [65, 66, 67].forEach((t, i) => { tokens[i] = BigInt(t); });
    const feeds = {
      tokens: new ort.Tensor("int64", tokens, [1, 8]),
      last_position: new ort.Tensor("int64", BigInt64Array.from([2n]), [1]),
    };
    const result = await session.run(feeds);
    const logits = result.next_logits.data;
    assert.equal(logits.length, 256);
    assert.ok(Array.from(logits).every(Number.isFinite));
    console.log(filename.split("/").at(-1), "ORT Web WASM output shape OK");
  } finally {
    await session.release();
  }
}

await testModel(join(root, "fp32", "model.onnx"));
await testModel(join(root, "int4", "model.int4.onnx"));


// M19: run true prefill -> cached single-token decode in the shipped WASM
// engine; compare the decoded token logits with a full-window FP32 graph.
const cachedRoot = join(root, "cached");
const prefill = await ort.InferenceSession.create(
  new Uint8Array(readFileSync(join(cachedRoot, "prefill.onnx"))),
  { executionProviders: ["wasm"] }
);
const decode = await ort.InferenceSession.create(
  new Uint8Array(readFileSync(join(cachedRoot, "decode.onnx"))),
  { executionProviders: ["wasm"] }
);
const legacy = await ort.InferenceSession.create(
  new Uint8Array(readFileSync(join(root, "fp32", "model.onnx"))),
  { executionProviders: ["wasm"] }
);
try {
  const inputs = new BigInt64Array(8);
  inputs[0] = 65n; inputs[1] = 66n; inputs[2] = 67n;
  const first = await prefill.run({
    tokens: new ort.Tensor("int64", inputs, [1, 8]),
    last_position: new ort.Tensor("int64", BigInt64Array.of(2n), [1]),
  });
  assert.equal(first.next_logits.data.length, 256);
  const feeds = {
    token_ids: new ort.Tensor("int64", BigInt64Array.of(68n), [1, 1]),
    past_length: new ort.Tensor("int64", BigInt64Array.of(3n), [1]),
  };
  for (let i = 0; i < 2; i++) {
    feeds["past_key_" + i] = first["key_" + i];
    feeds["past_value_" + i] = first["value_" + i];
  }
  const second = await decode.run(feeds);
  assert.equal(second.next_logits.data.length, 256);
  const full = new BigInt64Array(8);
  full.set([65n, 66n, 67n, 68n]);
  const expected = await legacy.run({
    tokens: new ort.Tensor("int64", full, [1, 8]),
    last_position: new ort.Tensor("int64", BigInt64Array.of(3n), [1]),
  });
  const maxDiff = Array.from(second.next_logits.data).reduce((worst, x, i) =>
    Math.max(worst, Math.abs(x - expected.next_logits.data[i])), 0);
  assert.ok(maxDiff < 0.0002, "Cached WASM logits differ from full window: " + maxDiff);
  console.log("M19 cached prefill + decode ORT Web WASM parity OK (max difference", maxDiff, ")");
} finally {
  await prefill.release();
  await decode.release();
  await legacy.release();
}


// M20: execute BOTH quantized cached graphs under the same WASM runtime
// shipped to mobile browsers. Verify expected outputs, cache shapes, finite
// numbers, and paired numerical drift from the M19 FP32 cached graphs.
const m20root = join(root, "cached_int4");
const quantPrefill = await ort.InferenceSession.create(
  new Uint8Array(readFileSync(join(m20root, "prefill.int4.onnx"))),
  { executionProviders: ["wasm"] }
);
const quantDecode = await ort.InferenceSession.create(
  new Uint8Array(readFileSync(join(m20root, "decode.int4.onnx"))),
  { executionProviders: ["wasm"] }
);
const refPrefill = await ort.InferenceSession.create(
  new Uint8Array(readFileSync(join(cachedRoot, "prefill.onnx"))),
  { executionProviders: ["wasm"] }
);
const refDecode = await ort.InferenceSession.create(
  new Uint8Array(readFileSync(join(cachedRoot, "decode.onnx"))),
  { executionProviders: ["wasm"] }
);
try {
  assert.deepEqual(quantPrefill.inputNames, ["tokens", "last_position"]);
  assert.deepEqual(quantDecode.inputNames,
    ["token_ids", "past_length", "past_key_0", "past_value_0", "past_key_1", "past_value_1"]);
  assert.deepEqual(quantPrefill.outputNames,
    ["next_logits", "key_0", "value_0", "key_1", "value_1"]);
  assert.deepEqual(quantDecode.outputNames, quantPrefill.outputNames);
  const history = [65n, 66n];
  let cacheA = null, cacheB = null, length = 0;
  let compared = 0, rebuilds = 0, decodeCalls = 0, worstError = 0;
  // Use identical input bytes for both modes. No autoregressive sampling
  // from quantized logits: this isolates inference numerical error.
  for (const token of [67n, 68n, 69n, 70n, 71n, 72n, 73n, 74n]) {
    history.push(token);
    let outA, outB;
    if (cacheA && cacheB && length < 8) {
      const makeFeeds = (cache) => {
        const feeds = {
          token_ids: new ort.Tensor("int64", BigInt64Array.of(token), [1, 1]),
          past_length: new ort.Tensor("int64", BigInt64Array.of(BigInt(length)), [1]),
        };
        for (let layer = 0; layer < 2; layer++) {
          feeds["past_key_" + layer] = cache["key_" + layer];
          feeds["past_value_" + layer] = cache["value_" + layer];
        }
        return feeds;
      };
      outA = await refDecode.run(makeFeeds(cacheA));
      outB = await quantDecode.run(makeFeeds(cacheB));
      decodeCalls++;
      length++;
    } else {
      const last = history.slice(-8);
      const padded = new BigInt64Array(8);
      padded.set(last);
      const feeds = {
        tokens: new ort.Tensor("int64", padded, [1, 8]),
        last_position: new ort.Tensor("int64", BigInt64Array.of(BigInt(last.length - 1)), [1]),
      };
      outA = await refPrefill.run(feeds);
      outB = await quantPrefill.run(feeds);
      if (cacheA) rebuilds++;
      length = last.length;
    }
    cacheA = outA; cacheB = outB;
    for (const cache of [cacheA, cacheB]) {
      for (let layer = 0; layer < 2; layer++) {
        for (const kind of ["key", "value"]) {
          const output = cache[kind + "_" + layer];
          assert.deepEqual(output.dims, [1, 4, 8, 8]);
          assert.ok(Array.from(output.data).every(Number.isFinite));
        }
      }
    }
    const a = Array.from(outA.next_logits.data);
    const b = Array.from(outB.next_logits.data);
    assert.equal(a.length, 256);
    assert.equal(b.length, 256);
    for (let i = 0; i < a.length; i++) {
      assert.ok(Number.isFinite(a[i]) && Number.isFinite(b[i]));
      worstError = Math.max(worstError, Math.abs(a[i] - b[i]));
    }
    compared++;
  }
  assert.ok(decodeCalls > 0 && rebuilds > 0, "Must test decoding and cache-boundary rebuild");
  assert.ok(worstError < 1, "Unexpected large synthetic INT4 logit drift: " + worstError);
  console.log("M20 INT4 KV-cache ONNX Runtime Web WASM passed:",
              { compared, decodeCalls, rebuilds, worstError });
} finally {
  await quantPrefill.release();
  await quantDecode.release();
  await refPrefill.release();
  await refDecode.release();
}
