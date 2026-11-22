# M17 — Aetherion Mobile (experimental)

## What is implemented

An ONNX export path and an Android/iOS **browser** inference preview
using ONNX Runtime Web (single-threaded CPU/WebAssembly). It does not
require a model API or PyTorch on the phone. A fixed-window graph, SHA-256
manifest, UTF-8 byte tokenizer and greedy decoder form the first mobile
inference contract. There is no native APK or IPA yet.

**Not a pretrained assistant.** The repository has no released, independently
validated chat checkpoint. The demo training corpus and untrained checkpoints
cannot support claims of useful conversation, safety, or reasoning.

## Build / export on the development computer

1. Install Python 3.10+ requirements and ONNX export/test tools:

       python -m pip install -r requirements.txt
       python -m pip install onnx onnxruntime

2. Use a **trusted checkpoint** created by the Aetherion trainer
   (a tiny local engineering smoke run is shown, not useful pretraining):

       python train.py --steps 2 --batch-size 2 --d-model 32 --layers 2 --heads 4 --block-size 32 --output checkpoints/mobile-smoke.pt

3. Export a 32-byte context for the smoke checkpoint:

       python export_mobile.py --checkpoint checkpoints/mobile-smoke.pt --window 32 --output-dir mobile/artifacts

   For other trained Micro checkpoints, choose --window up to their
   block_size. Export produces model.onnx and manifest.json.
   It fails for BPE checkpoints, non-byte vocabularies, oversized parameter
   counts, and invalid context windows. No conversion to GGUF is implied.

4. Validate inference parity (with optional ONNX/ONNX Runtime installed):

       python -m pytest -q tests/test_mobile_export.py

## Run on a mobile browser

1. On the host/development machine, stage the pinned runtime locally:

       cd mobile
       npm install
       npm run stage-runtime
       npm run check

2. Publish the complete mobile/ static directory, including vendor/, on a
   trusted **HTTPS** origin. An HTTPS origin is required for SubtleCrypto
   checksum validation and service-worker caching. The local development
   computer can use http://localhost for testing but a phone on your LAN
   generally cannot use insecure HTTP for this setup.
3. Open the HTTPS page in current Safari on iOS or Chrome on Android.
   Select the locally stored model.onnx and manifest.json exported in step 3.
   Tap **Verify & load model**, enter a prompt, then **Generate**.
4. The browser performs inference locally. Model selection must be repeated
   after a page reload. Prompts and model contents are never sent to a
   remote inference API by this application. Hosting and initial runtime
   download still require a network connection.

The service worker caches the app shell and self-hosted runtime files after
they are fetched. Offline reuse depends on complete caching and the browser
retaining that cache. The model files remain user-selected and are not
persistently cached. Avoid describing the PWA as universally offline.

## Exact interface (schema_version 1)

* tokens: int64 [1, window], UTF-8 byte values 0..255, right-padded with 0
* last_position: int64 [1], zero-based index of last real prompt token
* next_logits: float32 [1, 256], next-byte logits at last_position
* generation: greedy (argmax) with no EOS or chat formatting
* computation: full causal window recomputed at every byte, **no KV cache**
* RoPE and normalization: original weights and forward semantics preserved
* context: fixed to --window, sliding left for prompts longer than the window

A mobile byte token is not a character and may produce an incomplete UTF-8
sequence until later tokens are generated. The UI uses replacement decoding
for incomplete or invalid sequences.

## Constraints and follow-up gates

- Modern 64-bit browser and sufficient free memory required. No guarantee of
  support for feature phones, outdated OS/browser versions, or extremely
  memory-constrained devices.
- Maximum M17 model size is 5,000,000 parameters (checked, not measured).
  No performance/energy/memory benchmark or phone-device pass is claimed.
- No model weights, authorized data corpus, or executable APK/IPA is shipped.
- No BPE integration on phone: add verified tokenizer parity before enabling.
- Next: quantized ORT graph with measured accuracy, proper KV caching,
  Android/iOS device CI, version-pinned model card and independent evaluations.

Runtime background:
https://onnxruntime.ai/docs/tutorials/web/
https://onnxruntime.ai/docs/get-started/with-mobile.html
