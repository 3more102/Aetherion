"""Regression gates for the mobile graph, including actual ONNX Runtime parity."""
from dataclasses import asdict
from hashlib import sha256
import json

import pytest
import torch

from aetherion.model import Aetherion, ModelConfig
from aetherion.mobile_export import (
    MobileNextLogits, export_mobile_checkpoint, load_mobile_checkpoint,
)


def tiny_model():
    torch.manual_seed(19)
    cfg = ModelConfig(block_size=16, d_model=32, n_heads=4, n_layers=2)
    return Aetherion(cfg).eval()


@pytest.mark.parametrize("length", [1, 4, 8])
def test_mobile_padded_logits_match_unpadded_pytorch(length):
    model = tiny_model()
    ids = torch.randint(0, 256, (1, length), dtype=torch.long)
    padded = torch.zeros((1, 8), dtype=torch.long)
    padded[:, :length] = ids
    with torch.inference_mode():
        expected = model(ids)[0][:, -1, :]
        actual = MobileNextLogits(model, window=8)(
            padded, torch.tensor([length - 1], dtype=torch.long)
        )
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_export_rejects_bpe_checkpoint(tmp_path):
    model = tiny_model()
    checkpoint = tmp_path / "bpe.pt"
    torch.save({
        "config": asdict(model.cfg),
        "state_dict": model.state_dict(),
        "data": {"tokenizer_type": "bpe"},
    }, checkpoint)
    with pytest.raises(ValueError, match="BPE"):
        load_mobile_checkpoint(checkpoint)


def test_export_rejects_incorrect_byte_vocab(tmp_path):
    cfg = ModelConfig(vocab_size=300, block_size=8, d_model=32, n_heads=4, n_layers=1)
    model = Aetherion(cfg)
    checkpoint = tmp_path / "bad.pt"
    torch.save({"config": asdict(cfg), "state_dict": model.state_dict()}, checkpoint)
    with pytest.raises(ValueError, match="256-ID"):
        load_mobile_checkpoint(checkpoint)


def test_onnx_runtime_parity_and_manifest(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    model = tiny_model()
    checkpoint = tmp_path / "tiny.pt"
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict()}, checkpoint)
    manifest = export_mobile_checkpoint(checkpoint, tmp_path / "export", window=8)
    onnx_file = tmp_path / "export" / "model.onnx"
    onnx.checker.check_model(str(onnx_file))
    assert manifest["onnx_sha256"] == sha256(onnx_file.read_bytes()).hexdigest()
    assert json.loads((tmp_path / "export" / "manifest.json").read_text()) == manifest

    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    assert [item.name for item in session.get_inputs()] == ["tokens", "last_position"]
    assert [item.name for item in session.get_outputs()] == ["next_logits"]

    for length in (1, 4, 8):
        ids = torch.randint(0, 256, (1, length), dtype=torch.long)
        padded = torch.zeros((1, 8), dtype=torch.long)
        padded[:, :length] = ids
        with torch.inference_mode():
            expected = model(ids)[0][:, -1, :].numpy()
        predicted = session.run(
            ["next_logits"],
            {"tokens": padded.numpy(), "last_position": np.array([length - 1], dtype=np.int64)},
        )[0]
        np.testing.assert_allclose(predicted, expected, rtol=1e-4, atol=1e-4)
