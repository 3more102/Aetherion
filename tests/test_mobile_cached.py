"""Real PyTorch/ORT checks for M19 cached prefill, decode and context rebuild."""
from dataclasses import asdict
from hashlib import sha256
import json

import pytest
import torch

from aetherion.model import Aetherion, ModelConfig
from aetherion.mobile_cached import (
    MobileCachedPrefill,
    MobileCachedDecode,
    export_mobile_cached,
    cache_names,
)


@pytest.fixture
def small():
    torch.manual_seed(1907)
    cfg = ModelConfig(block_size=12, d_model=32, n_heads=4, n_layers=2)
    return Aetherion(cfg).eval()


def _padded(ids, window=8):
    result = torch.zeros((1, window), dtype=torch.long)
    result[:, :ids.shape[-1]] = ids
    return result


@pytest.mark.parametrize("length", (1, 2, 5, 8))
def test_cached_prefill_matches_full_torch(small, length):
    ids = torch.randint(0, 256, (1, length))
    with torch.inference_mode():
        reference = small(ids)[0][:, -1, :]
        cached = MobileCachedPrefill(small, 8)(_padded(ids), torch.tensor([length - 1]))
        reference_cached, layer_cache = small.forward_cached(ids)
    torch.testing.assert_close(cached[0], reference, atol=1e-5, rtol=1e-4)
    for i, (key, value) in enumerate(layer_cache):
        torch.testing.assert_close(cached[1 + 2 * i][:, :, :length], key, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(cached[2 + 2 * i][:, :, :length], value, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("length", (1, 3, 7))
def test_cached_decode_matches_torch_full(small, length):
    ids = torch.randint(0, 256, (1, length))
    next_id = torch.randint(0, 256, (1, 1))
    with torch.inference_mode():
        cached = MobileCachedPrefill(small, 8)(_padded(ids), torch.tensor([length - 1]))
        decoded = MobileCachedDecode(small, 8)(next_id, torch.tensor([length]), *cached[1:])
        reference = small(torch.cat((ids, next_id), dim=1))[0][:, -1, :]
    torch.testing.assert_close(decoded[0], reference, atol=1e-5, rtol=1e-4)


@pytest.fixture
def exported(tmp_path, small):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    checkpoint = tmp_path / "tiny.pt"
    torch.save({"config": asdict(small.cfg), "state_dict": small.state_dict()}, checkpoint)
    dest = tmp_path / "cached"
    manifest = export_mobile_cached(checkpoint, dest, window=8)
    return dest, manifest, small


def test_cached_onnx_shapes_sha_and_parity(exported):
    import numpy as np
    import onnx
    import onnxruntime as ort

    root, manifest, model = exported
    assert manifest["schema_version"] == 2
    assert manifest["cache_policy"] == "rebuild_relative_rope_at_window_boundary"
    assert json.loads((root / "manifest.json").read_text()) == manifest
    for kind in ("prefill", "decode"):
        filename = root / manifest["graphs"][kind]["filename"]
        onnx.checker.check_model(str(filename))
        assert sha256(filename.read_bytes()).hexdigest() == manifest["graphs"][kind]["sha256"]
    prefill = ort.InferenceSession(str(root / "prefill.onnx"), providers=["CPUExecutionProvider"])
    decode = ort.InferenceSession(str(root / "decode.onnx"), providers=["CPUExecutionProvider"])
    assert [i.name for i in prefill.get_inputs()] == ["tokens", "last_position"]
    assert [i.name for i in prefill.get_outputs()] == ["next_logits", *cache_names(model.cfg.n_layers)]
    assert [i.name for i in decode.get_inputs()] == [
        "token_ids", "past_length", *cache_names(model.cfg.n_layers, prefix="past_")
    ]
    assert [i.name for i in decode.get_outputs()] == ["next_logits", *cache_names(model.cfg.n_layers)]

    rng = np.random.default_rng(1907)
    for length in (1, 3, 6):
        ids = rng.integers(0, 256, size=length, dtype=np.int64)
        padded = np.zeros((1, 8), dtype=np.int64)
        padded[0, :length] = ids
        out = prefill.run(None, {
            "tokens": padded, "last_position": np.array([length - 1], dtype=np.int64)
        })
        with torch.inference_mode():
            original = model(torch.tensor(ids).view(1, -1))[0][:, -1, :].numpy()
        np.testing.assert_allclose(out[0], original, atol=1e-4, rtol=1e-4)
        caches = out[1:]
        for step in range(8 - length):
            next_token = int(rng.integers(0, 256))
            inputs = {
                "token_ids": np.array([[next_token]], dtype=np.int64),
                "past_length": np.array([length + step], dtype=np.int64),
            }
            inputs.update(dict(zip(cache_names(model.cfg.n_layers, prefix="past_"), caches)))
            out = decode.run(None, inputs)
            caches = out[1:]
            ids = np.append(ids, next_token)
            with torch.inference_mode():
                original = model(torch.tensor(ids).view(1, -1))[0][:, -1, :].numpy()
            np.testing.assert_allclose(out[0], original, atol=2e-4, rtol=2e-4)
        # Simulate sliding-window reset: renumber relative RoPE from zero.
        suffix = np.concatenate((ids[1:], np.array([99], dtype=np.int64)))
        padded = suffix.reshape(1, 8)
        reset = prefill.run(None, {
            "tokens": padded, "last_position": np.array([7], dtype=np.int64)
        })
        with torch.inference_mode():
            original = model(torch.tensor(suffix).view(1, -1))[0][:, -1, :].numpy()
        np.testing.assert_allclose(reset[0], original, atol=2e-4, rtol=2e-4)


def test_export_cached_rejects_out_of_range_window(tmp_path, small):
    checkpoint = tmp_path / "tiny.pt"
    torch.save({"config": asdict(small.cfg), "state_dict": small.state_dict()}, checkpoint)
    with pytest.raises(ValueError, match="window"):
        export_mobile_cached(checkpoint, tmp_path / "out", window=13)
