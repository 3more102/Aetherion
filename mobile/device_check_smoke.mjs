// M24 deterministic browser-feature checks; these do not emulate a phone.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createContext, Script } from "node:vm";

const sandbox = createContext({ Uint8Array });
new Script(readFileSync(new URL("./device-check.js", import.meta.url), "utf8"),
  { filename: "device-check.js" }).runInContext(sandbox);
const inspect = new Script("deviceCheck.inspect").runInContext(sandbox);
let estimates = 0;
const ready = {
  isSecureContext: true, WebAssembly, BigInt64Array,
  crypto: { subtle: { digest() {} } },
  ort: { InferenceSession: { create() {} } },
  indexedDB: { open() {} },
  navigator: {
    serviceWorker: {}, deviceMemory: 4,
    storage: { async estimate() { estimates++; return { quota: 8192, usage: 1024 }; } },
  },
};
const ok = await inspect(ready);
assert.equal(ok.status, "environment_ready_to_attempt");
assert.equal(ok.schema_version, 1);
assert.deepEqual(Array.from(ok.missing_required_apis), []);
assert.equal(ok.optional.browser_storage_quota_bytes_if_available, 8192);
assert.equal(ok.optional.browser_reported_memory_gb_if_available, 4);
assert.equal(estimates, 1);
assert.ok(!JSON.stringify(ok).includes("userAgent"));

const missing = await inspect({ ...ready, isSecureContext: false,
  WebAssembly: undefined, BigInt64Array: undefined, crypto: {},
  ort: undefined, indexedDB: undefined, navigator: {} });
assert.equal(missing.status, "missing_required_apis");
assert.equal(missing.missing_required_apis.length, 5);
assert.equal(missing.optional.indexeddb_for_opt_in_model_save, false);
assert.equal(missing.optional.browser_storage_quota_bytes_if_available, null);

const noStorage = await inspect({ ...ready, indexedDB: undefined,
  navigator: { storage: { async estimate() { throw Error("quota blocked"); } } } });
assert.equal(noStorage.status, "environment_ready_to_attempt");
assert.equal(noStorage.optional.indexeddb_for_opt_in_model_save, false);
assert.equal(noStorage.optional.browser_storage_quota_bytes_if_available, null);
assert.equal(noStorage.optional.browser_reported_memory_gb_if_available, null);

const badEstimates = await inspect({ ...ready,
  navigator: { storage: { async estimate() { return { quota: -1, usage: Infinity }; } } } });
assert.equal(badEstimates.optional.browser_storage_quota_bytes_if_available, null);
assert.equal(badEstimates.optional.browser_storage_usage_bytes_if_available, null);
console.log("M24 local device readiness checks passed");
