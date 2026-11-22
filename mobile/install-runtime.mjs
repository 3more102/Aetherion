// Stage the pinned ONNX Runtime Web assets for private, self-hosted offline use.
import { copyFileSync, existsSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
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
// The PWA install must include whichever pinned WASM variant ORT selects.
execFileSync(process.execPath, ["generate-icons.mjs"], { stdio: "inherit" });
const runtimeFiles = readdirSync(target)
  .filter((name) => name.startsWith("ort") && /\\.(?:js|mjs|wasm)$/.test(name))
  .map((name) => "./vendor/" + name);
const requiredAssets = [
  "./", "./index.html", "./app.js", "./manifest.webmanifest", "./icon.svg",
  "./icons/icon-192.png", "./icons/icon-512.png",
  ...runtimeFiles,
];
if (!runtimeFiles.includes("./vendor/ort.wasm.min.js") ||
    !runtimeFiles.some((p) => p.endsWith(".wasm"))) {
  throw new Error("Offline PWA needs the pinned ORT entrypoint and WebAssembly binary.");
}
writeFileSync("precache-assets.json",
  JSON.stringify({ schema_version: 1, assets: requiredAssets }, null, 2) + "\\n");
console.log("Staged " + copied + " ONNX Runtime assets and generated install/offline assets");
