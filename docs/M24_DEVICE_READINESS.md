# M24 — Mobile device readiness and bounded model import

**Status:** Browser API preflight and early import-size guards.
This is not physical Android/iOS validation or universal phone support.

## Use

Open the existing static HTTPS PWA and select **Check this device**.
The local check verifies API availability: secure context, WebAssembly,
BigInt64Array, SubtleCrypto SHA-256 and the shipped ONNX Runtime Web script.
It reports missing required APIs separately from optional IndexedDB,
service-worker and browser-provided memory/quota estimates.

An `environment_ready_to_attempt` result **does not** prove that the
phone has enough free RAM, can execute a particular ONNX graph or retains
model storage. The check makes no network requests, does not store data,
and does not include a user-agent fingerprint.

After selecting files, **Verify & load model** rejects a manifest larger
than 128 KiB and combined ONNX graphs larger than 256 MiB **before**
reading ONNX byte buffers or allocating sessions. Cached exports must
contain a nonempty decode graph. These are input limits, not RAM ceilings:
WASM decoding and temporary graph buffers can consume much more memory.

## Validation gates

```sh
node mobile/device_check_smoke.mjs
node mobile/app_state_smoke.mjs
cd mobile && npm run check
```

The Node tests cover supported/unsupported APIs, optional storage and
exceptions, invalid estimates, and input-size guard behavior.
CI also verifies the packaged PWA and offline precache references.

Real Android/iOS tests, storage retention across restarts, memory-pressure
behavior, true inference throughput, and available pretrained conversational
weights remain outstanding.
