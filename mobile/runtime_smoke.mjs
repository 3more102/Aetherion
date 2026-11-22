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
