// Stage the pinned ONNX Runtime Web assets for private, self-hosted offline use.
import { copyFileSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { join } from "node:path";
const source = join("node_modules", "onnxruntime-web", "dist");
const target = "vendor";
if (!existsSync(source)) {
  throw new Error("Run npm install in mobile/ before staging runtime files.");
}
mkdirSync(target, { recursive: true });
let copied = 0;
for (const filename of readdirSync(source)) {
  if (filename.startsWith("ort") && /\.(js|mjs|wasm)$/.test(filename)) {
    copyFileSync(join(source, filename), join(target, filename));
    copied++;
  }
}
if (!existsSync(join(target, "ort.wasm.min.js")) || !copied) {
  throw new Error("Expected ort.wasm.min.js missing; inspect the pinned package distribution.");
}
console.log("Staged " + copied + " ONNX Runtime assets to mobile/vendor");
