# M23 — Opt-in local ONNX model storage (IndexedDB)

**Status:** Experimental browser feature. M23 reduces file-picker friction;
it does **not** provide pretrained assistant weights, an APK, an IPA, or
guaranteed deployment on every Android/iOS device.

## What changes

The M17–M22 static PWA required reselecting ONNX model files and their
manifest on each browser session. M23 adds a small, explicit model vault:

1. Import and **Verify & load model** using the M17–M20 ONNX graph(s) and
   compatible manifest. Graph SHA-256 and ONNX input/output contracts are
   checked by the existing inference loader.
2. Select **Save local copy** to store that export and its manifest in
   this browser's origin-scoped IndexedDB. Nothing is saved automatically.
3. When reopening the application later, select **Restore saved model**.
   A saved model is re-read and reverified against SHA-256 and graph
   contracts before inference is enabled.
4. Select **Delete saved copy** to delete the IndexedDB record. If a model
   is already running in memory, this does **not** release its active
   inference session. Delete other caches/backups through your device
   settings if required.

All model formats currently accepted by the M17–M20 browser loader remain
supported: schema-v1 FP32 or INT4 full-window; schema-v2 FP32 cached;
schema-v3 INT4 cached.

## Data boundaries and risks

- Model saving is **opt-in**, disabled until a successful ONNX load, and
  restricted to a combined 256 MiB of imported ONNX bytes, plus a bounded
  manifest. The application keeps just **one** saved export at a time.
- Model bytes and manifest stay in **IndexedDB on the current website
  origin**. The service worker still does not cache model files.
  Prompt history, generated text, credentials and benchmark logs are **not**
  saved by M23.
- **Not encrypted**: other JavaScript executing on the same trusted origin,
  or anyone with access to local browser data, may access this copy.
  Choose a trusted HTTPS host. This is not a protected secret vault.
- Storage is not guaranteed on every device. Browsers may reject a write
  because of storage quotas, privacy mode, storage settings or
  platform restrictions. Browser eviction, cache/data clearing and device
  changes may remove the saved copy. Deleting the IndexedDB entry does not
  promise secure wiping or deletion of backups.
- The save operation rechecks file hashes. Restore again validates the
  declared manifest, the actual SHA-256 of the files, and the ONNX graph
  I/O contract. **Manifest metadata is not a signed attestation**.
- M23 does not collect real-device latency or battery data, and does
  not demonstrate useful language output. It uses the same one-thread
  ONNX Runtime Web WASM engine as M22.

## Verification

From the repository root:

```sh
node mobile/model_vault_smoke.mjs
node mobile/app_state_smoke.mjs
node mobile/benchmark_smoke.mjs
python -m pytest -q tests/test_mobile_package.py
```

The new Node test uses a deterministic mock IndexedDB and checks
save/read/replace/delete, atomic rollback on simulated QuotaExceededError,
invalid records, explicit opt-in, and missing-storage behavior.
The Mobile ONNX export parity workflow enforces the smoke checks and
existing package-integrity gates.

**Still requires manual validation**: Android Chrome, iOS Safari and
installed-standalone PWA data survival, storage quota behavior, reload
across browser restarts and storage eviction. Revisit the same HTTPS origin
and use Restore saved model; do not assume saved file-picker selections.

See [M21 package/install guide](M21_MOBILE_PWA_BENCH.md) and
[M22 phone benchmark provenance](M22_MOBILE_DEVICE_VALIDATION.md).
