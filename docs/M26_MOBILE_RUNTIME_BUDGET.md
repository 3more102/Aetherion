# M26 — Conservative mobile manifest shape and KV-cache budget

**Status:** Implemented as local browser validation with automated regression
tests. **No real handset has been qualified.**

## Why

The earlier M24 256 MiB combined ONNX-file-size guard restricted imported
*files*, but did not bound model execution parameters. A small or malformed
manifest could advertise a huge context or excessive K/V dimensions and cause
avoidable allocations or resource exhaustion after import.

## New preflight

`validateManifest` now calls `enforceRuntimeShapeBudget` **before ONNX
files are read or an inference session is created**. Both user-selected files
and restored IndexedDB files pass through the same guard.

| Advertised property | Maximum accepted |
| --- | ---: |
| Context window (byte tokens) | 4096 |
| Layers (cached schema v2/v3) | 48 |
| Heads (cached schema v2/v3) | 64 |
| Head dimension (cached schema v2/v3) | 256 |
| One logical FP32 K/V-cache estimate (cached v2/v3) | 64 MiB |

The cache estimate is calculated as:

```text
2 (key + value) × layers × heads × head_dim × window × 4 bytes
```

This uses the declared FP32 K/V activations of M19/M20. The browser also
reports the resulting estimate after a successful cached-model load.
Full-window v1 exports have no K/V cache, but still obey the window bound.

**Not a peak-RAM bound:** the estimate excludes ONNX graph weights, model
activations, intermediate tensors, multiple live cache copies, WASM arena
allocations, execution-provider overhead and the browser/OS itself.
The app does not measure available physical RAM or enforce a true RAM cap.
Likewise, a small declared shape does not prove the actual ONNX tensor shapes
or operators are compatible. Existing SHA-256 and I/O name checks still apply.

## Validation

```sh
node mobile/app_state_smoke.mjs
node mobile/model_vault_ui_smoke.mjs
cd mobile && npm run check && npm run check-pwa
```

Tests reject over-window v1/v2 manifests, cached dimension excesses,
one-copy KV estimates above 64 MiB and an oversized import before any
inference-session creation or IndexedDB write. They also verify valid,
small M17/M18/M19/M20 cases and remove stale M24 PWA shell cache on upgrade.

Separate GitHub CI additionally exercises ONNX Runtime CPU and
WebAssembly synthetic parity. This is **not physical mobile testing**.

## Remaining release blockers

- Test genuine Android and iPhone hardware (cold starts, memory pressure,
  background/foreground restoration, browser eviction and offline launch).
- Capture at least three complete, provenance-matched device measurements per
  model format with the existing M22 benchmark-comparison gate.
- Review a legally authorized and privacy-screened training dataset.
- Train and evaluate a useful checkpoint; existing synthetic weights do not
  establish language quality or practical conversation.
- Keep the PWA scope and origin trusted; IndexedDB model storage is not
  encrypted and may be evicted.
