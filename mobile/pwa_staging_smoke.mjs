// Executable PWA staging and scoped-offline service worker test.
// Does not require a phone, browser installation, or any pretrained weights.
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { inflateSync } from "node:zlib";
import { createContext, Script } from "node:vm";

const inventory = JSON.parse(readFileSync("precache-assets.json", "utf8"));
const manifest = JSON.parse(readFileSync("manifest.webmanifest", "utf8"));
assert.equal(inventory.schema_version, 1);
assert.ok(inventory.assets.length >= 8);
assert.equal(new Set(inventory.assets).size, inventory.assets.length);
assert.equal(manifest.start_url, "./");
assert.equal(manifest.scope, "./");
assert.equal(manifest.display, "standalone");
assert.ok(inventory.assets.includes("./vendor/ort.wasm.min.js"));
assert.ok(inventory.assets.some((f) => f.endsWith(".wasm")));
assert.ok(!inventory.assets.some((f) => f.endsWith(".onnx")));
for (const asset of inventory.assets) {
  const path = asset === "./" ? "index.html" : asset.replace(/^\.\//, "");
  assert.ok(existsSync(path), "Offline asset missing: " + asset);
}
for (const [size, filename] of [
  [192, "icons/icon-192.png"], [512, "icons/icon-512.png"],
]) {
  const png = readFileSync(filename);
  assert.equal(png.subarray(0, 8).toString("hex"), "89504e470d0a1a0a");
  assert.equal(png.readUInt32BE(16), size);
  assert.equal(png.readUInt32BE(20), size);
  assert.equal(png[25], 6, "RGBA PNG expected");
  const idatSize = png.readUInt32BE(33);
  assert.equal(png.toString("ascii", 37, 41), "IDAT");
  const pixels = inflateSync(png.subarray(41, 41 + idatSize));
  assert.equal(pixels.length, size * (1 + size * 4));
  assert.equal(pixels[0], 0, "PNG filter 0 expected");
}

const listeners = {};
const databases = new Map([
  ["unrelated-cache", new Map([["do-not-remove", "unrelated"]])],
  ["aetherion-mobile-m20-v1", new Map()],
]);
const cacheMock = (name) => {
  if (!databases.has(name)) databases.set(name, new Map());
  const entries = databases.get(name);
  return {
    async addAll(urls) {
      for (const url of urls) {
        const path = url === "./" ? "index.html" : url.replace(/^\.\//, "");
        if (!existsSync(path)) throw new Error("Asset not available: " + url);
        entries.set(url, { ok: true, clone() { return this; } });
      }
    },
    async match(key) { return entries.get(key) ?? null; },
    async put(key, response) { entries.set(key, response); },
  };
};
let activated = false, skipped = false, networkRequests = 0;
const mockSelf = {
  location: { origin: "https://example.test" },
  registration: { scope: "https://example.test/mobile/" },
  clients: { async claim() { activated = true; } },
  async skipWaiting() { skipped = true; },
  addEventListener(kind, handler) { listeners[kind] = handler; },
};
const context = createContext({
  self: mockSelf, URL,
  caches: {
    async open(name) { return cacheMock(name); },
    async keys() { return [...databases.keys()]; },
    async delete(name) { return databases.delete(name); },
  },
  async fetch(url) {
    networkRequests++;
    if (url === "./precache-assets.json") {
      return { ok: true, async json() { return inventory; } };
    }
    return { ok: true, clone() { return this; } };
  },
});
new Script(readFileSync("sw.js", "utf8"), { filename: "sw.js" }).runInContext(context);
let install;
listeners.install({ waitUntil(promise) { install = promise; } });
await install;
assert.ok(skipped);
const stored = databases.get("aetherion-mobile-m21-v1");
assert.ok(stored);
for (const path of inventory.assets) {
  assert.ok(stored.has(path), "Runtime not precached for offline use: " + path);
}
let activation;
listeners.activate({ waitUntil(promise) { activation = promise; } });
await activation;
assert.ok(activated);
assert.ok(databases.has("unrelated-cache"), "Unrelated caches must not be removed");
assert.ok(!databases.has("aetherion-mobile-m20-v1"), "Old Aetherion cache must be removed");
let modelIntercepted = false;
listeners.fetch({
  request: { method: "GET", url: "https://example.test/mobile/model.onnx", mode: "same-origin" },
  respondWith() { modelIntercepted = true; },
});
assert.equal(modelIntercepted, false, "Models must never be cached by SW");
assert.ok(networkRequests >= 1);
console.log("M21 PWA PNG, offline runtime, scoped cache cleanup and no-model-cache smoke passed");
