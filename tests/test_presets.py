"""Architecture preset and weight-count checks without allocating full weights."""
import pytest

from aetherion.presets import get_preset, PRESET_NAMES
from inspect_model import inspect_preset


def test_preset_names_and_validation():
    assert PRESET_NAMES == ("micro", "aetherion-125m")
    with pytest.raises(ValueError, match="Unknown preset"):
        get_preset("unverified-huge-model")
    assert get_preset("micro").vocab_size == 256
    assert get_preset("aetherion-125m").vocab_size == 32_000


def test_125m_parameter_count_and_tied_weights():
    report = inspect_preset("aetherion-125m")
    assert report["parameters"] == 123_688_704
    assert report["embedding_parameters"] == 24_576_000
    assert report["tied_input_output_embeddings"] is True
    assert report["configuration"]["block_size"] == 2048
    assert report["configuration"]["d_model"] == 768
    assert report["configuration"]["n_layers"] == 14
    assert report["configuration"]["n_heads"] == 12
    assert report["theoretical_weight_storage_mib"] == round(123_688_704 * 2 / (1024 ** 2), 2)


def test_inspection_does_not_change_config_or_require_cuda():
    first = get_preset("micro")
    first.d_model = 64
    assert get_preset("micro").d_model == 128
    half = inspect_preset("micro", "fp16")
    full = inspect_preset("micro", "fp32")
    assert full["parameters"] == half["parameters"]
    assert full["theoretical_weight_storage_mib"] >= 2 * half["theoretical_weight_storage_mib"] - 0.02
    with pytest.raises(ValueError, match="Unsupported weight dtype"):
        inspect_preset("micro", "int4")
