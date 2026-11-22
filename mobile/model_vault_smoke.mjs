// M23 IndexedDB vault regression: mocked transaction behavior, no real phone.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Script, createContext } from "node:vm";

function fakeIndexedDB() {
  let created = false;
  let record = null;
  let writes = 0;
  let errorOnNextWrite = false;
  const db = {
    objectStoreNames: { contains: (name) => created && name === "models" },
    createObjectStore(name) {
      assert.equal(name, "models");
      created = true;
    },
    close() {},
    transaction(name, mode) {
      assert.equal(name, "models");
      const tx = {
        error: null,
        objectStore(requested) {
          assert.equal(requested, "models");
          function request(kind, value, key) {
            assert.equal(key, "selected");
            const op = {};
            queueMicrotask(() => {
              try {
                if (kind === "put") {
                  assert.equal(mode, "readwrite");
                  if (errorOnNextWrite) {
                    errorOnNextWrite = false;
                    throw new Error("QuotaExceededError");
                  }
                  record = structuredClone(value);
                  writes++;
                } else if (kind === "delete") {
                  assert.equal(mode, "readwrite");
                  record = null;
                } else if (kind === "count") {
                  op.result = Number(record !== null);
                } else if (kind === "get") {
                  op.result = record === null ? undefined : structuredClone(record);
                }
                op.onsuccess?.();
                tx.oncomplete?.();
              } catch (err) {
                tx.error = err;
                tx.onabort?.();
              }
            });
            return op;
          }
          return {
            put: (value, key) => request("put", value, key),
            get: (key) => request("get", null, key),
            count: (key) => request("count", null, key),
            delete: (key) => request("delete", null, key),
          };
        },
      };
      return tx;
    },
  };
  return {
    open(name, version) {
      assert.equal(name, "aetherion-local-model-v1");
      assert.equal(version, 1);
      const request = {};
      queueMicrotask(() => {
        request.result = db;
        if (!created) request.onupgradeneeded?.();
        request.onsuccess?.();
      });
      return request;
    },
    writes: () => writes,
    failWrite: () => { errorOnNextWrite = true; },
    corruptStored: (item) => { record = item; },
  };
}

const idb = fakeIndexedDB();
const browser = createContext({ indexedDB: idb, ArrayBuffer, Date, Promise, Error });
new Script(readFileSync(new URL("./model-vault.js", import.meta.url), "utf8"),
  { filename: "model-vault.js" }).runInContext(browser);
const vault = new Script("modelVault").runInContext(browser);
assert.equal(vault.available(), true);
assert.equal(await vault.exists(), false);
assert.equal(idb.writes(), 0, "Startup storage check must not write model data");

const bytes = new Uint8Array([1, 2, 3, 4]).buffer;
const manifestText = JSON.stringify({ schema_version: 1, note: "synthetic fixture" });
const source = { manifestText, prefillBytes: bytes, decodeBytes: null };
await vault.save(source);
assert.equal(await vault.exists(), true);
assert.equal(idb.writes(), 1);
const got = await vault.read();
assert.equal(got.version, 1);
assert.equal(got.manifestText, manifestText);
assert.deepEqual(Array.from(new Uint8Array(got.prefillBytes)), [1, 2, 3, 4]);
assert.equal(got.decodeBytes, null);
assert.ok(got.savedAtUtc);

await vault.save({ ...source, prefillBytes: new Uint8Array([5, 6]).buffer,
  decodeBytes: new Uint8Array([7, 8]).buffer });
assert.deepEqual(Array.from(new Uint8Array((await vault.read()).decodeBytes)), [7, 8]);

idb.failWrite();
await assert.rejects(vault.save(source), /QuotaExceededError/);
assert.deepEqual(Array.from(new Uint8Array((await vault.read()).prefillBytes)),
  [5, 6], "Failed atomic write must not destroy prior saved model");
await assert.rejects(vault.save({ ...source, prefillBytes: new ArrayBuffer(0) }), /invalid/);
await assert.rejects(vault.save({ ...source, manifestText: "x".repeat(256 * 1024 + 1) }), /invalid/);
await assert.rejects(vault.save({ ...source, prefillBytes: Object.defineProperty(new ArrayBuffer(1), "byteLength", { value: 256 * 1024 * 1024 + 1 }) }),
  /limit/);
idb.corruptStored({ version: 8, manifestText: "broken" });
await assert.rejects(vault.read(), /invalid/);
await vault.remove();
assert.equal(await vault.read(), null);
assert.equal(await vault.exists(), false);

// No implicit storage is available when the browser has disabled IndexedDB.
const unavailable = createContext({ ArrayBuffer, Date, Promise, Error });
new Script(readFileSync(new URL("./model-vault.js", import.meta.url), "utf8"))
  .runInContext(unavailable);
const disabledVault = new Script("modelVault").runInContext(unavailable);
assert.equal(disabledVault.available(), false);
await assert.rejects(disabledVault.exists(), /IndexedDB/);
console.log("M23 opt-in IndexedDB save/restore/delete, quotas, corruption and atomic rollback smoke passed");
