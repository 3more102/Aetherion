"""M20: exercise real INT4 cached prefill/decode and schema/hash guards."""
from dataclasses import asdict
from hashlib import sha256
import json

import pytest
import torch

from aetherion.mobile_cached import export_mobile_cached
from aetherion.mobile_cached_quantization import quantize_mobile_cached_int4
from aetherion.model import Aetherion, ModelConfig


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    root = tmp_path_factory.mktemp("int4_cached")
    torch.manual_seed(2020)
    cfg = ModelConfig(block_size=12, d_model=32, n_heads=4, n_layers=2)
    model = Aetherion(cfg).eval()
    checkpoint = root / "synthetic.pt"
    torch.save({"config": asdict(cfg), "state_dict": model.state_dict()}, checkpoint)
    source = root / "original"
    exported = export_mobile_cached(checkpoint, source, window=8)
    target = root / "int4"
    converted = quantize_mobile_cached_int4(source, target)
    return source, target, exported, converted


def test_m20_both_graphs_load_and_parity(fixture):
    import onnx
    import onnxruntime as ort

    source, target, base, m20 = fixture
    assert base["schema_version"] == 2
    assert m20["schema_version"] == 3
    assert m20["source_graphs"] == base["graphs"]
    assert m20["cache_policy"] == "rebuild_relative_rope_at_window_boundary"
    assert m20["quantization"]["kv_cache_dtype"] == "float32"
    assert m20["quantization"]["synthetic_parity"]["logit_comparisons"] == 15
    assert m20["quantization"]["synthetic_parity"]["max_abs_logit_error"] < 1.0
    assert m20["quantization"]["synthetic_parity"]["context_rebuilds"] > 0
    assert json.loads((target / "manifest.json").read_text()) == m20
    for kind in ("prefill", "decode"):
        path = target / m20["graphs"][kind]["filename"]
        assert m20["graphs"][kind]["sha256"] == sha256(path.read_bytes()).hexdigest()
        graph = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(graph)
        assert any(node.op_type == "MatMulNBits" for node in graph.graph.node)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        assert "next_logits" in sess.output_names if hasattr(sess, "output_names") else (
            "next_logits" in [out.name for out in sess.get_outputs()]
        )


def test_m20_rejects_tampered_source(fixture, tmp_path):
    source, _, _, _ = fixture
    fake = tmp_path / "bad"
    fake.mkdir()
    for name in ("prefill.onnx", "decode.onnx", "manifest.json"):
        (fake / name).write_bytes((source / name).read_bytes())
    (fake / "decode.onnx").write_bytes((fake / "decode.onnx").read_bytes() + b"x")
    with pytest.raises(ValueError, match="decode FP32 ONNX SHA-256"):
        quantize_mobile_cached_int4(fake, tmp_path / "out")


def test_m20_rejects_existing_quantized_or_wrong_dir(fixture, tmp_path):
    source, target, _, _ = fixture
    with pytest.raises(ValueError, match="separate directory"):
        quantize_mobile_cached_int4(source, source)
    with pytest.raises(ValueError, match="schema-2"):
        quantize_mobile_cached_int4(target, tmp_path / "other")


@pytest.mark.parametrize("block", [0, 8, 24, 33, True])
def test_m20_rejects_invalid_blocks(fixture, tmp_path, block):
    source, _, _, _ = fixture
    with pytest.raises(ValueError, match="power of 2"):
        quantize_mobile_cached_int4(source, tmp_path / "bad", block_size=block)
