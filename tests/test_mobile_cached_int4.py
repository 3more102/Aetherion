"""M20: real, source-verified INT4 prefill/decode ONNX Runtime testing."""
from dataclasses import asdict
from hashlib import sha256
import json

import pytest
import torch

from aetherion.mobile_cached import export_mobile_cached
from aetherion.mobile_cached_quantization import quantize_mobile_cached
from aetherion.model import Aetherion, ModelConfig


@pytest.fixture(scope="module")
def m19(tmp_path_factory):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    root = tmp_path_factory.mktemp("m20")
    torch.manual_seed(20)
    cfg = ModelConfig(block_size=12, d_model=32, n_heads=4, n_layers=2)
    model = Aetherion(cfg).eval()
    checkpoint = root / "tiny.pt"
    torch.save({"config": asdict(cfg), "state_dict": model.state_dict()}, checkpoint)
    export_mobile_cached(checkpoint, root / "fp32", window=8)
    return root / "fp32"


def test_m20_convert_load_and_multi_step_parity(m19, tmp_path):
    import numpy as np
    import onnx
    import onnxruntime as ort

    manifest = quantize_mobile_cached(m19, tmp_path / "int4", block_size=32)
    loaded = json.loads((tmp_path / "int4" / "manifest.json").read_text())
    assert loaded == manifest
    assert manifest["schema_version"] == 2
    assert manifest["quantization"]["cache_dtype"] == "float32"
    assert manifest["quantization"]["type"] == "weight_only_symmetric_int4"
    for kind in ("prefill", "decode"):
        graph_path = tmp_path / "int4" / (kind + ".onnx")
        onnx.checker.check_model(str(graph_path))
        graph = onnx.load(str(graph_path), load_external_data=False)
        assert any(n.op_type == "MatMulNBits" for n in graph.graph.node)
        assert manifest["graphs"][kind]["sha256"] == sha256(graph_path.read_bytes()).hexdigest()
        assert manifest["graphs"][kind]["source_sha256"] == sha256((m19 / (kind + ".onnx")).read_bytes()).hexdigest()
        assert manifest["graphs"][kind]["int4_bytes"] == graph_path.stat().st_size
        assert manifest["quantization"]["synthetic_parity"][kind]["sample_count"] >= 3
        assert np.isfinite(manifest["quantization"]["synthetic_parity"][kind]["max_abs_logit_error"])
        assert manifest["quantization"]["synthetic_parity"][kind]["max_abs_logit_error"] < 1.0
        session = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
        assert session.get_providers()[0] == "CPUExecutionProvider"


def test_m20_reject_tampered_prefill(m19, tmp_path):
    source = tmp_path / "tampered"
    source.mkdir()
    (source / "manifest.json").write_bytes((m19 / "manifest.json").read_bytes())
    for name in ("prefill.onnx", "decode.onnx"):
        (source / name).write_bytes((m19 / name).read_bytes())
    with (source / "prefill.onnx").open("ab") as out:
        out.write(b"modified")
    with pytest.raises(ValueError, match="SHA-256"):
        quantize_mobile_cached(source, tmp_path / "int4")


def test_m20_reject_same_directory(m19):
    with pytest.raises(ValueError, match="overwrite"):
        quantize_mobile_cached(m19, m19)


def test_m20_reject_bad_block_size(m19, tmp_path):
    with pytest.raises(ValueError, match="power of 2"):
        quantize_mobile_cached(m19, tmp_path / "other", block_size=24)


def test_m20_reject_already_quantized(m19, tmp_path):
    source = tmp_path / "quantized_source"
    source.mkdir()
    manifest = json.loads((m19 / "manifest.json").read_text())
    manifest["quantization"] = {"type": "already_quantized"}
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unquantized"):
        quantize_mobile_cached(source, tmp_path / "other")
