"""Test real MatMulNBits conversion, graph loading, numerical sanity and guardrails."""
from dataclasses import asdict
from hashlib import sha256
import json

import pytest
import torch

from aetherion.mobile_export import export_mobile_checkpoint
from aetherion.mobile_quantization import quantize_mobile_int4
from aetherion.model import Aetherion, ModelConfig


@pytest.fixture
def base(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch.manual_seed(18)
    cfg = ModelConfig(block_size=16, d_model=32, n_heads=4, n_layers=2)
    model = Aetherion(cfg).eval()
    checkpoint = tmp_path / "tiny.pt"
    torch.save({"config": asdict(cfg), "state_dict": model.state_dict()}, checkpoint)
    output = tmp_path / "fp32"
    export_mobile_checkpoint(checkpoint, output, window=8)
    return output


def test_int4_quantization_and_measured_parity(base, tmp_path):
    import onnx
    source = base / "model.onnx"
    manifest = base / "manifest.json"
    output = tmp_path / "int4"
    result = quantize_mobile_int4(source, manifest, output, block_size=32)
    quantized = output / "model.int4.onnx"
    quantized_manifest = output / "manifest.int4.json"
    assert quantized.is_file() and quantized_manifest.is_file()
    assert result == json.loads(quantized_manifest.read_text())
    assert result["onnx_sha256"] == sha256(quantized.read_bytes()).hexdigest()
    graph = onnx.load(str(quantized), load_external_data=False)
    assert any(node.op_type == "MatMulNBits" for node in graph.graph.node)
    assert result["quantization"]["quantized_matmul_nodes"] >= 1
    assert result["quantization"]["synthetic_parity"]["sample_count"] == 3
    assert result["quantization"]["synthetic_parity"]["max_abs_logit_error"] < 1.0
    assert result["quantization"]["int4_onnx_bytes"] == quantized.stat().st_size
    assert result["quantization"]["fp32_onnx_bytes"] == source.stat().st_size


def test_int4_rejects_tampered_model(base, tmp_path):
    source = base / "model.onnx"
    fake = tmp_path / "fake.onnx"
    fake.write_bytes(source.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        quantize_mobile_int4(fake, base / "manifest.json", tmp_path / "int4")


def test_int4_rejects_invalid_block_size(base, tmp_path):
    with pytest.raises(ValueError, match="power of 2"):
        quantize_mobile_int4(base / "model.onnx", base / "manifest.json",
                            tmp_path / "int4", block_size=24)
