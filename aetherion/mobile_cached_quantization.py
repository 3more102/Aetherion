"""M20: checked INT4 conversion for both M19 cached ONNX graphs.

The key/value cache stays FP32; only supported constant-weight MatMul nodes
are converted to ORT MatMulNBits. This does not guarantee faster inference.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _contract(session, inputs: list[str], outputs: list[str]) -> None:
    if ([node.name for node in session.get_inputs()] != inputs or
            [node.name for node in session.get_outputs()] != outputs):
        raise ValueError("Unexpected cached ONNX input/output names")


def quantize_mobile_cached_int4(
    source_dir: str | Path, output_dir: str | Path, *, block_size: int = 32
) -> dict:
    """Convert M19 prefill/decode and record deterministic numerical comparisons.

    Requires an unmodified schema-2 M19 manifest and its two source graphs.
    Rejects same-directory output, missing hashes and unsupported graph layouts.
    """
    if (not isinstance(block_size, int) or isinstance(block_size, bool)
            or block_size < 16 or block_size & (block_size - 1)):
        raise ValueError("INT4 block_size must be a power of 2 >= 16")
    source = Path(source_dir)
    target = Path(output_dir)
    if source.resolve() == target.resolve():
        raise ValueError("Output must use a separate directory from FP32 source")
    manifest_file = source / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 2 or
            manifest.get("architecture") != "aetherion-micro" or
            manifest.get("tokenizer_type") != "utf8_bytes" or
            manifest.get("vocab_size") != 256 or
            manifest.get("quantization") is not None or
            manifest.get("cache_policy") != "rebuild_relative_rope_at_window_boundary"):
        raise ValueError("Expected an unquantized M19 schema-2 byte-level manifest")
    window = manifest.get("window")
    layers = manifest.get("n_layers")
    heads = manifest.get("n_heads")
    head_dim = manifest.get("head_dim")
    if (not all(isinstance(x, int) and not isinstance(x, bool) and x > 0
                for x in (window, layers, heads, head_dim)) or window < 2):
        raise ValueError("Invalid cached graph geometry")
    sources = {}
    for kind in ("prefill", "decode"):
        metadata = manifest.get("graphs", {}).get(kind, {})
        filename = f"{kind}.onnx"
        if metadata.get("filename") != filename:
            raise ValueError(f"Unexpected {kind} graph filename")
        path = source / filename
        if _sha(path) != metadata.get("sha256"):
            raise ValueError(f"{kind} FP32 ONNX SHA-256 mismatch")
        sources[kind] = path

    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import matmul_nbits_quantizer, quant_utils
    from .mobile_cached import cache_names

    names = cache_names(layers)
    inputs = {
        "prefill": ["tokens", "last_position"],
        "decode": ["token_ids", "past_length", *cache_names(layers, prefix="past_")],
    }
    outputs = ["next_logits", *names]
    originals = {}
    for kind, path in sources.items():
        onnx.checker.check_model(str(path))
        originals[kind] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        _contract(originals[kind], inputs[kind], outputs)

    target.mkdir(parents=True, exist_ok=True)
    sizes = {}
    quantized = {}
    converted = {}
    graphs = {}
    for kind, path in sources.items():
        dest = target / f"{kind}.int4.onnx"
        config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
            bits=4,
            block_size=block_size,
            is_symmetric=True,
            accuracy_level=4,
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=("MatMul",),
            quant_axes=(("MatMul", 0),),
        )
        quantizer = matmul_nbits_quantizer.MatMulNBitsQuantizer(
            quant_utils.load_model_with_shape_infer(path), algo_config=config
        )
        quantizer.process()
        quantizer.model.save_model_to_file(str(dest), False)
        graph = onnx.load_model(str(dest), load_external_data=False)
        onnx.checker.check_model(graph)
        count = sum(node.op_type == "MatMulNBits" for node in graph.graph.node)
        if count < 1:
            raise ValueError(f"{kind} INT4 export converted zero MatMul nodes")
        if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
            raise ValueError("External weights not supported by mobile file picker")
        quantized[kind] = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
        _contract(quantized[kind], inputs[kind], outputs)
        sizes[kind] = (path.stat().st_size, dest.stat().st_size)
        converted[kind] = count
        graphs[kind] = {"filename": dest.name, "sha256": _sha(dest)}

    # Keep identical byte inputs through each graph (no sampling from
    # quantized logits). Compare numerical differences without promising
    # synthetic argmax agreement or meaningful language quality.
    rng = np.random.default_rng(2020)
    max_error = 0.0
    errors = []
    matching_argmax = 0
    comparisons = 0
    sequence_count = 0
    for length in sorted(set((1, min(3, window), window))):
        ids = rng.integers(0, 256, size=length, dtype=np.int64)
        padded = np.zeros((1, window), dtype=np.int64)
        padded[0, :length] = ids
        feed = {"tokens": padded, "last_position": np.array([length - 1], dtype=np.int64)}
        ref = originals["prefill"].run(None, feed)
        approx = quantized["prefill"].run(None, feed)
        count = length

        def compare(a, b):
            nonlocal max_error, matching_argmax, comparisons
            if a.shape != (1, 256) or b.shape != (1, 256):
                raise ValueError("Unexpected logits output shape")
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("Non-finite FP32 or INT4 logits")
            delta = np.abs(a - b)
            max_error = max(max_error, float(delta.max()))
            errors.append(float(delta.mean()))
            matching_argmax += int(a.argmax() == b.argmax())
            comparisons += 1

        compare(ref[0], approx[0])
        # Cross window boundary, then re-prefill with a zero-based RoPE
        # window: parity test also checks cached decode after rebuilding.
        for step in range(4):
            next_byte = int(rng.integers(0, 256))
            ids = np.append(ids, next_byte)
            if count >= window:
                context = ids[-window:]
                padded = context.reshape(1, window)
                feed = {"tokens": padded,
                        "last_position": np.array([window - 1], dtype=np.int64)}
                ref = originals["prefill"].run(None, feed)
                approx = quantized["prefill"].run(None, feed)
                count = window
                sequence_count += 1
            else:
                def to_inputs(cache):
                    result = {
                        "token_ids": np.array([[next_byte]], dtype=np.int64),
                        "past_length": np.array([count], dtype=np.int64),
                    }
                    result.update(zip(cache_names(layers, prefix="past_"), cache[1:]))
                    return result
                ref = originals["decode"].run(None, to_inputs(ref))
                approx = quantized["decode"].run(None, to_inputs(approx))
                count += 1
            compare(ref[0], approx[0])

    original_bytes = sum(t[0] for t in sizes.values())
    quantized_bytes = sum(t[1] for t in sizes.values())
    report = {
        "type": "weight_only_symmetric_int4",
        "format": "MatMulNBits",
        "block_size": block_size,
        "kv_cache_dtype": "float32",
        "converted_matmul_nodes": converted,
        "files": {k: {"fp32_bytes": sizes[k][0], "int4_bytes": sizes[k][1]}
                  for k in ("prefill", "decode")},
        "combined_fp32_bytes": original_bytes,
        "combined_int4_bytes": quantized_bytes,
        "combined_size_ratio": round(quantized_bytes / original_bytes, 6),
        "synthetic_parity": {
            "logit_comparisons": comparisons,
            "max_abs_logit_error": round(max_error, 8),
            "mean_abs_logit_error": round(sum(errors) / len(errors), 8),
            "argmax_agreement_count": matching_argmax,
            "context_rebuilds": sequence_count,
            "scope": "deterministic untrained input paths, not held-out language quality",
        },
    }
    result = dict(manifest)
    result.update({
        "schema_version": 3,
        "graphs": graphs,
        "source_graphs": manifest["graphs"],
        "quantization": report,
        "status": "int4_kv_cached_inference_export_only_not_language_validated",
    })
    (target / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
