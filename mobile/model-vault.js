"use strict";

// M23: Opt-in, origin-scoped IndexedDB persistence. No prompt history,
// credentials or telemetry are written. Model bytes are never service-worker
// cached and are not encrypted; users must trust the HTTPS origin.
const modelVault = (() => {
  const NAME = "aetherion-local-model-v1";
  const STORE = "models";
  const KEY = "selected";
  const LIMIT_BYTES = 256 * 1024 * 1024;
  const MAX_MANIFEST_CHARS = 256 * 1024;

  function available() {
    return typeof indexedDB !== "undefined" &&
      typeof indexedDB.open === "function";
  }

  function guardRecord(record) {
    if (!record || record.version !== 1 ||
        typeof record.manifestText !== "string" ||
        record.manifestText.length < 2 ||
        record.manifestText.length > MAX_MANIFEST_CHARS ||
        !(record.prefillBytes instanceof ArrayBuffer) ||
        record.prefillBytes.byteLength < 1 ||
        !(record.decodeBytes === null || record.decodeBytes instanceof ArrayBuffer)) {
      throw new Error("Stored model is invalid; delete it and import ONNX files again.");
    }
    if (record.prefillBytes.byteLength +
        (record.decodeBytes ? record.decodeBytes.byteLength : 0) > LIMIT_BYTES) {
      throw new Error("Stored model exceeds the 256 MiB local limit.");
    }
    return record;
  }

  function action(operation, payload = null) {
    if (!available()) {
      return Promise.reject(new Error("This browser does not expose IndexedDB."));
    }
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(NAME, 1);
      let finished = false;
      const fail = (error) => {
        if (finished) return;
        finished = true;
        reject(error || new Error("Local model storage failed."));
      };
      request.onerror = () => fail(request.error);
      request.onblocked = () =>
        fail(new Error("Model storage upgrade is blocked by another tab."));
      request.onupgradeneeded = () => {
        if (!request.result.objectStoreNames.contains(STORE)) {
          request.result.createObjectStore(STORE);
        }
      };
      request.onsuccess = () => {
        const db = request.result;
        if (finished) { db.close(); return; }
        try {
          const tx = db.transaction(STORE, operation === "save" || operation === "remove" ?
            "readwrite" : "readonly");
          const store = tx.objectStore(STORE);
          let result;
          let op;
          if (operation === "save") op = store.put(payload, KEY);
          else if (operation === "remove") op = store.delete(KEY);
          else if (operation === "exists") op = store.count(KEY);
          else if (operation === "read") op = store.get(KEY);
          else throw new Error("Unsupported vault operation");
          op.onsuccess = () => { result = op.result; };
          tx.oncomplete = () => {
            db.close();
            if (!finished) {
              finished = true;
              resolve(operation === "exists" ? Boolean(result) :
                operation === "read" ? (result || null) : undefined);
            }
          };
          const abort = () => { db.close(); fail(tx.error || new Error("Storage transaction aborted.")); };
          tx.onerror = abort;
          tx.onabort = abort;
        } catch (error) {
          db.close();
          fail(error);
        }
      };
    });
  }

  async function save(entry) {
    const payload = guardRecord({
      version: 1,
      manifestText: entry.manifestText,
      prefillBytes: entry.prefillBytes,
      decodeBytes: entry.decodeBytes ?? null,
      savedAtUtc: new Date().toISOString(),
    });
    await action("save", payload);
  }

  async function read() {
    const result = await action("read");
    return result === null ? null : guardRecord(result);
  }

  return Object.freeze({
    available,
    save,
    read,
    exists: () => action("exists"),
    remove: () => action("remove"),
  });
})();
