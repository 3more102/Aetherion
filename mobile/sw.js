// M21 offline-first installable PWA shell. Models remain local file imports:
// no checkpoints, prompts, or arbitrary same-origin API responses are cached.
"use strict";
const CACHE = "aetherion-mobile-m24-v1";
const PREFIX = "aetherion-mobile-";
const SHELL = [
  "./", "./index.html", "./app.js", "./model-vault.js", "./device-check.js", "./manifest.webmanifest",
  "./icon.svg", "./precache-assets.json",
  "./icons/icon-192.png", "./icons/icon-512.png",
];

async function installShell() {
  const cache = await caches.open(CACHE);
  await cache.addAll(SHELL);
  const response = await fetch("./precache-assets.json", { cache: "no-store" });
  if (!response.ok) throw new Error("PWA precache manifest could not be loaded");
  const data = await response.json();
  if (!data || data.schema_version !== 1 ||
      !Array.isArray(data.assets) || data.assets.length < SHELL.length ||
      data.assets.length > 200 ||
      !data.assets.every((name) =>
        typeof name === "string" && name.startsWith("./") &&
        !name.includes("..") && !name.includes("?") &&
        !name.endsWith(".onnx"))) {
    throw new Error("Invalid local PWA precache asset list");
  }
  // Installation is considered complete only when the packaged runtime is
  // cached. A device that cannot store the assets can still use the online app.
  await cache.addAll(data.assets);
  await self.skipWaiting();
}

self.addEventListener("install", (event) => {
  event.waitUntil(installShell());
});
self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter((key) =>
      key.startsWith(PREFIX) && key !== CACHE).map((key) => caches.delete(key)));
    await self.clients.claim();
  })());
});
self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin ||
      !url.pathname.startsWith(new URL(self.registration.scope).pathname) ||
      url.pathname.endsWith(".onnx")) return;

  if (request.mode === "navigate") {
    event.respondWith((async () => {
      try {
        const response = await fetch(request);
        if (response.ok) {
          event.waitUntil(caches.open(CACHE).then((cache) =>
            cache.put("./index.html", response.clone())));
        }
        return response;
      } catch {
        const cache = await caches.open(CACHE);
        const fallback = await cache.match("./index.html");
        if (fallback) return fallback;
        throw new Error("Offline app shell not available");
      }
    })());
    return;
  }
  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    const manifest = await cache.match("./precache-assets.json");
    // Only the explicitly bundled application shell/runtime is cacheable.
    // Never persist arbitrary origin-scoped JSON/API/model responses.
    if (!manifest || url.search) return fetch(request);
    const data = await manifest.json();
    const scopePath = new URL(self.registration.scope).pathname;
    const relativeAsset = "./" + url.pathname.slice(scopePath.length);
    if (!Array.isArray(data.assets) || !data.assets.includes(relativeAsset)) {
      return fetch(request);
    }
    const cached = await cache.match(relativeAsset);
    if (cached) return cached;
    const response = await fetch(request);
    if (response.ok) {
      event.waitUntil(cache.put(relativeAsset, response.clone()));
    }
    return response;
  })());
});
