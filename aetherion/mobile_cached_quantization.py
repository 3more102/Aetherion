"""M20: INT4 weight-only conversion of M19's two KV-cached ONNX graphs.

This is not a new training checkpoint or model-quality evaluation. Attention
K/V activations remain FP32; only eligible constant MatMul weights become
blockwise INT4 MatMulNBits operators.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path

from .mobile_cached import cache_names


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _graph_io(session):
    return ([item.name for item in session.get_inputs()],
            [item.name for item in session.get_outputs()])


def _assert_logits(array, label):
    import numpy as np
    if array.shape != (1, 256) or not np.isfinite(array).all():
        raise ValueError(f"Invalid {label} next_logits shape or non-finite data")


def _quantize_graph(source: Path, dest: Path, block_size: int) -> int:
    import onnx
    from onnxruntime.quantization import matmul_nbits_quantizer, quant_utils

    onnx.checker.check_model(str(source))
    config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
        bits=4,
        block_size=block_size,
        is_symmetric=True,
        accuracy_level=4,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
        quant_axes=(("MatMul", 0),),
    )
    model = quant_utils.load_model_with_shape_infer(source)
    quantizer = matmul_nbits_quantizer.MatMulNBitsQuantizer(model, algo_config=config)
    quantizer.process()
    quantizer.model.save_model_to_file(str(dest), False)
    converted = onnx.load(str(dest), load_external_data=False)
    onnx.checker.check_model(converted)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in converted.graph.initializer):
        raise ValueError("External ONNX weight files are not supported by this mobile loader")
    count = sum(node.op_type == "MatMulNBits" for node in converted.graph.node)
    if count < 1:
        raise ValueError(f"INT4 conversion of {source.name} produced zero MatMulNBits operators")
    return count


class _Parity:
    """Record synthetic logit comparisons, without judging semantic quality."""

    def __init__(self):
        self.samples = 0
        self.agreements = 0
        self.max_abs = 0.0
        self.total_mean = 0.0

    def compare(self, reference, quantized, label):
        import numpy as np
        _assert_logits(reference, label + " FP32")
        _assert_logits(quantized, label + " INT4")
        delta = np.abs(reference - quantized)
        self.samples += 1
        self.agreements += int(int(reference.argmax()) == int(quantized.argmax()))
        self.max_abs = max(self.max_abs, float(delta.max()))
        self.total_mean += float(delta.mean())

    def report(self) -> dict:
        return {
            "sample_count": self.samples,
            "max_abs_logit_error": round(self.max_abs, 8),
            "mean_abs_logit_error": round(self.total_mean / max(1, self.samples), 8),
            "argmax_agreement_count": self.agreements,
            "argmax_agreement_denominator": self.samples,
        }


def quantize_mobile_cached(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    block_size: int = 32,
) -> dict:
    """Create INT4 prefill + decode graphs and manifest from an M19 export.

    Validates source hashes and I/O contracts, generates both quantized
    graphs, measures paired prefill/decode fidelity through the full context
    boundary, then writes schema-2 integrity metadata. Output must differ
    from source directory so the original FP32 files remain intact.
    """
    import numpy as np
    import onnxruntime as ort

    source = Path(source_dir)
    destination = Path(output_dir)
    if type(block_size) is not int or block_size < 16 or block_size & (block_size - 1):
        raise ValueError("INT4 block_size must be a power of 2 >= 16")
    if source.resolve() == destination.resolve():
        raise ValueError("INT4 output directory cannot overwrite the FP32 export")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 2 or
        manifest.get("architecture") != "aetherion-micro" or
        manifest.get("tokenizer_type") != "utf8_bytes" or
        manifest.get("vocab_size") != 256 or
        manifest.get("quantization") is not None or
        manifest.get("cache_policy") != "rebuild_relative_rope_at_window_boundary"):
        raise ValueError("Expected a compatible, unquantized M19 schema-2 export")
    window = manifest.get("window")
    layers = manifest.get("n_layers")
    heads = manifest.get("n_heads")
    head_dim = manifest.get("head_dim")
    if (type(window) is not int or window < 2 or
        type(layers) is not int or layers < 1 or
        type(heads) is not int or heads < 1 or
        type(head_dim) is not int or head_dim < 2 or head_dim % 2):
        raise ValueError("Invalid M19 cache shape or window")
    names = cache_names(layers)
    cache_input_names = cache_names(layers, prefix="past_")
    expected = {
        "prefill": (["tokens", "last_position"], ["next_logits", *names]),
        "decode": (["token_ids", "past_length", *cache_input_names], ["next_logits", *names]),
    }
    originals = {}
    baseline_sessions = {}
    for kind in ("prefill", "decode"):
        metadata = manifest.get("graphs", {}).get(kind, {})
        # Refuse arbitrary paths and mismatched artifacts. Only M19 exact
        # filenames are legal for the two-graph browser contract.
        if metadata.get("filename") != kind + ".onnx":
            raise ValueError(f"Unexpected {kind} source filename")
        path = source / (kind + ".onnx")
        if _hash(path) != metadata.get("sha256"):
            raise ValueError(f"FP32 {kind} SHA-256 mismatch")
        baseline_sessions[kind] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        if _graph_io(baseline_sessions[kind]) != expected[kind]:
            raise ValueError(f"Unexpected M19 {kind} graph interface")
        originals[kind] = path

    destination.mkdir(parents=True, exist_ok=True)
    models = {}
    reports = {}
    converted_nodes = {}
    for kind in ("prefill", "decode"):
        output = destination / (kind + ".onnx")
        if output.resolve() == originals[kind].resolve():
            raise ValueError("Cannot overwrite FP32 export")
        converted_nodes[kind] = _quantize_graph(originals[kind], output, block_size)
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        if _graph_io(session) != expected[kind]:
            raise ValueError(f"INT4 {kind} graph interface differs from FP32")
        models[kind] = session
        reports[kind] = {
            "filename": output.name,
            "sha256": _hash(output),
            "source_sha256": _hash(originals[kind]),
            "fp32_bytes": originals[kind].stat().st_size,
            "int4_bytes": output.stat().st_size,
            "quantized_matmul_nodes": converted_nodes[kind],
        }

    rng = np.random.default_rng(2020)
    stats = {"prefill": _Parity(), "decode": _Parity()}
    # Paired state streams: FP32 and INT4 consume their own updated K/V,
    # so accumulated drift is measured rather than ignored.
    for length in sorted(set((1, max(1, window // 2), max(1, window - 1), window))):
        inputs = np.zeros((1, window), dtype=np.int64)
        inputs[0, :length] = rng.integers(0, 256, size=length, dtype=np.int64)
        prefill_feed = {
            "tokens": inputs, "last_position": np.asarray([length - 1], dtype=np.int64)
        }
        full = baseline_sessions["prefill"].run(None, prefill_feed)
        compressed = models["prefill"].run(None, prefill_feed)
        stats["prefill"].compare(full[0], compressed[0], "prefill")
        original_cache, quantized_cache = full[1:], compressed[1:]
        for position in range(length, window):
            token = int(rng.integers(0, 256))
            common = {
                "token_ids": np.asarray([[token]], dtype=np.int64),
                "past_length": np.asarray([position], dtype=np.int64),
            }
            ref_feed = {**common, **dict(zip(cache_input_names, original_cache))}
            int4_feed = {**common, **dict(zip(cache_input_names, quantized_cache))}
            ref_out = baseline_sessions["decode"].run(None, ref_feed)
            int4_out = models["decode"].run(None, int4_feed)
            stats["decode"].compare(ref_out[0], int4_out[0], "decode")
            original_cache, quantized_cache = ref_out[1:], int4_out[1:]
            for j, (a, b) in enumerate(zip(original_cache, quantized_cache)):
                required = (1, heads, window, head_dim)
                if a.shape != required or b.shape != required:
                    raise ValueError(f"Unexpected cache output {j} shape")

    manifest_quantized = dict(manifest)
    manifest_quantized.update({
        "graphs": reports,
        "quantization": {
            "type": "weight_only_symmetric_int4",
            "format": "MatMulNBits",
            "block_size": block_size,
            "cache_dtype": "float32",
            "prefill_quantized_matmul_nodes": converted_nodes["prefill"],
            "decode_quantized_matmul_nodes": converted_nodes["decode"],
            "synthetic_parity": {kind: stats[kind].report() for kind in stats},
            "validation_scope": "deterministic synthetic token windows; not language quality",
        },
        "status": "int4_kv_cached_inference_not_language_quality_validated",
    })
    (destination / "manifest.json").write_text(
        json.dumps(manifest_quantized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_quantized
