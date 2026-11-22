"""Offline INT4 weight-only conversion for the M17 ONNX mobile inference graph.

Uses ONNX Runtime's MatMulNBits QOperator. No inference/training cloud calls.
Quantized files are independent of the original FP32 model and checkpoint.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path



def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def quantize_mobile_int4(
    model_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    block_size: int = 32,
) -> dict:
    """Create a self-contained INT4 ONNX model and integrity-checked manifest.

    Checks synthetic inference parity against FP32 and records its error.
    Measured error here is *not* a language-quality metric.
    """
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import matmul_4bits_quantizer, quant_utils

    source = Path(model_path)
    manifest_file = Path(manifest_path)
    destination = Path(output_dir)
    if block_size < 16 or block_size & (block_size - 1):
        raise ValueError("INT4 block_size must be a power of 2 >= 16")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or
            manifest.get("architecture") != "aetherion-micro" or
            manifest.get("tokenizer_type") != "utf8_bytes" or
            manifest.get("vocab_size") != 256 or
            manifest.get("quantization") is not None):
        raise ValueError("Expected a compatible, unquantized M17 UTF-8-byte manifest")
    if _digest(source) != manifest.get("onnx_sha256"):
        raise ValueError("FP32 ONNX SHA-256 mismatch")
    window = manifest.get("window")
    if not isinstance(window, int) or isinstance(window, bool) or window < 2:
        raise ValueError("Invalid M17 context window")
    onnx.checker.check_model(str(source))
    fp32 = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    if ([item.name for item in fp32.get_inputs()] != ["tokens", "last_position"] or
            [item.name for item in fp32.get_outputs()] != ["next_logits"]):
        raise ValueError("Unexpected M17 graph interface")

    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "model.int4.onnx"
    output_manifest = destination / "manifest.int4.json"
    if output.resolve() == source.resolve() or output_manifest.resolve() == manifest_file.resolve():
        raise ValueError("INT4 output would overwrite a source artifact")

    config = matmul_4bits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=block_size,
        is_symmetric=True,
        accuracy_level=4,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
        quant_axes=(("MatMul", 0),),
    )
    model = quant_utils.load_model_with_shape_infer(source)
    quantizer = matmul_4bits_quantizer.MatMul4BitsQuantizer(
        model, algo_config=config
    )
    quantizer.process()
    quantizer.model.save_model_to_file(str(output), False)
    graph = onnx.load_model(str(output), load_external_data=False)
    onnx.checker.check_model(graph)
    converted = sum(node.op_type == "MatMulNBits" for node in graph.graph.node)
    if converted < 1:
        raise ValueError("INT4 conversion produced zero MatMulNBits operators")
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
        raise ValueError("External initializer data is not supported by the mobile file picker")
    int4 = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])

    worst_abs = 0.0
    mean_abs_samples = []
    same_top_token = 0
    rng = np.random.default_rng(2017)
    for length in sorted(set((1, max(1, window // 2), window))):
        ids = np.zeros((1, window), dtype=np.int64)
        ids[0, :length] = rng.integers(0, 256, size=length)
        inputs = {"tokens": ids, "last_position": np.array([length - 1], dtype=np.int64)}
        reference = fp32.run(["next_logits"], inputs)[0]
        predicted = int4.run(["next_logits"], inputs)[0]
        if reference.shape != (1, 256) or predicted.shape != reference.shape:
            raise ValueError("Quantized logits shape differs from FP32 output")
        if not np.isfinite(reference).all() or not np.isfinite(predicted).all():
            raise ValueError("Non-finite quantized or reference logits")
        error = np.abs(predicted - reference)
        worst_abs = max(worst_abs, float(error.max()))
        mean_abs_samples.append(float(error.mean()))
        same_top_token += int(reference.argmax() == predicted.argmax())

    stats = {
        "sample_count": len(mean_abs_samples),
        "max_abs_logit_error": round(worst_abs, 8),
        "mean_abs_logit_error": round(sum(mean_abs_samples) / len(mean_abs_samples), 8),
        "argmax_agreement_count": same_top_token,
        "argmax_agreement_denominator": len(mean_abs_samples),
        "validation_scope": "three deterministic synthetic input windows; not language-quality evaluation",
    }
    new_manifest = dict(manifest)
    new_manifest.update({
        "onnx_sha256": _digest(output),
        "source_onnx_sha256": manifest["onnx_sha256"],
        "quantization": {
            "type": "weight_only_symmetric_int4",
            "format": "MatMulNBits",
            "block_size": block_size,
            "quantized_matmul_nodes": converted,
            "fp32_onnx_bytes": source.stat().st_size,
            "int4_onnx_bytes": output.stat().st_size,
            "file_size_ratio_int4_over_fp32": round(output.stat().st_size / source.stat().st_size, 5),
            "synthetic_parity": stats,
        },
        "status": "quantized_inference_export_only_not_quality_validated",
    })
    output_manifest.write_text(
        json.dumps(new_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return new_manifest
