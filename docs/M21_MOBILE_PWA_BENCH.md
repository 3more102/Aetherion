# M21 — Installable mobile PWA and reproducible device benchmark

**Status:** Source code and GitHub Actions packaging. This is a static
installable WebAssembly application, **not** a signed Android APK, an
iOS IPA, or a release of a trained conversational model.

The app operates with the M17–M20 byte-level ONNX exports. Aetherion
currently has no independently validated useful general-purpose
assistant checkpoint. The mobile frontend will generate low-quality
output when supplied with an untrained test checkpoint.

## Prepare a deployable PWA

On a development machine, install Node.js 22 and Python 3.11+.
From the repository root:

    cd mobile
    npm install
    npm run stage-runtime
    npm run check
    npm run check-pwa
    npm run pack-pwa

The build creates mobile/dist/aetherion-mobile-pwa.zip. Alternatively,
a successful **Mobile ONNX export parity** GitHub Actions run uploads an
artifact named aetherion-mobile-m21-pwa, kept for 14 days. This
artifact is a ZIP of a static website with ONNX Runtime Web staged
locally, not a store-distributed native application.

The ZIP includes: index.html, app.js, sw.js, the web manifest,
192- and 512-pixel PNG install icons, icon.svg, the pinned ONNX
Runtime Web distribution under vendor/, the offline precache
manifest, and PACKAGE_INFO.json listing asset lengths and SHA-256
hashes. It deliberately excludes model ONNX files, model checkpoints,
browser logs, prompt histories, credentials and dependencies such as
node_modules/.

Unzip directly into a static **HTTPS** website directory. The host
must serve .wasm assets. A localhost-based development server can
also be used; visiting the HTML by file:// is not supported. Do
not configure the host to redirect JavaScript/WASM asset URLs to
HTML fallbacks.

## Install on mobile

- Android: open the HTTPS URL in a compatible browser. Use its
  "Install app" / "Add to Home screen" menu, or the in-page Install
  button if the browser dispatches beforeinstallprompt.
- iPhone: open in Safari and use Share → Add to Home Screen. The
  JavaScript beforeinstallprompt workflow is not available on iOS.
- Once installed, open the app and import your matching ONNX model
  file(s) and manifest. Browsers do not automatically preserve local
  file-picker selections across sessions.

The M21 service worker precaches the static shell and all pinned
JavaScript/WebAssembly runtime assets before completing installation.
It removes only old Aetherion caches, never other same-origin apps'
caches. No ONNX model is fetched or precached by the service worker.
If the browser denies the required cache storage, the service worker
installation may fail while the online web page still works. Even when
the precache succeeds, offline persistence depends on the browser's
storage/eviction policy. Therefore, **offline availability on every
phone is not guaranteed**.

## Benchmark actual devices

Use the **same checkpoint**, context window, browser and phone for
paired FP32-versus-INT4 comparisons. Reset model and browser environment
between trials; record device OS/browser versions and thermal/battery
conditions yourself. The app uses a one-thread ONNX Runtime WebAssembly
execution provider.

1. Import the three files of a matching M19 FP32 or M20 INT4 cached
   export and select Verify & load model. The app hashes every ONNX
   graph before running it.
2. Tap **Benchmark (16 steps)**. The benchmark uses a two-byte prompt,
   two warmup steps, and up to sixteen measured generated UTF-8 bytes.
3. Tap **Save benchmark JSON** to save the resulting file manually.
   Nothing is uploaded by the app.
4. Repeat at least 3–5 times on each device and format. Compare
   per-phase timings with similar prefill/decode call counts.

Benchmark JSON v2 records:
- UTC timestamp; manifest schema; quantization format; prefill SHA-256;
  exact ONNX file bytes and time spent reading, hashing and initializing.
- Separate prefill, cached decode and legacy full-window sample counts,
  mean/median/p95/min/max latency, plus aggregate inference time.
- Byte token throughput **excluding UI rendering and scheduling**.
- Pre/decode call counts (including warmups), a *calculated* single
  FP32 KV-cache tensor payload size, browser-reported memory information
  if available, and user-agent/device hints.
- Whether the benchmark was interrupted; explicit caveats.

The browser performance.memory property is nonstandard and reports only
exposed JavaScript heap size. It is **not** peak device RAM, WASM heap size
or actual model RSS. The KV estimate excludes duplicate caches and other
temporary tensors. Battery/energy measurement requires external
instruments or system profiling. On phones lacking the property, the
JSON reports null instead of fabricated figures.

Aetherion emits **UTF-8 byte tokens**, not BPE tokens; do not compare
these tokens/s directly with commercial LLM throughput.

## Verification / deployment checks

    python -m pytest -q tests/test_mobile_package.py
    cd mobile
    npm run check-pwa
    node benchmark_smoke.mjs

- Python checks enforce deterministic ZIP packaging, SHA inventory,
  mandatory runtime and icon files, and exclusion of private artifacts.
- Node checks inflate generated PNGs, exercise the service-worker
  offline install/cache isolation contract, and verify model files are
  not cached.
- Benchmark checks validate memory caveats, per-phase statistics and
  JSON export readiness using deterministic mock data.
- Existing Python ONNX Runtime, M17–M20 WASM regression tests and
  repository smoke training continue to run in CI.

**Not validated by M21:** Actual install on any particular Android or
iPhone, battery usage, thermal behavior, native APK/IPA packaging,
BPE inference, or quality of generated language. Supply reproducible
on-device JSON reports to ground the next optimization milestone.

## Official references

- [MDN — Making PWAs installable](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps/Guides/Making_PWAs_installable)
- [MDN — Installing and uninstalling web apps](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps/Guides/Installing)
- [ONNX Runtime — Deploying ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/deploy.html)
- [MDN — Service Worker API](https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API)
