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
