import subprocess
import sys

import pytest

from mlx_raclate.registry import (
    ARCHITECTURES,
    PIPELINES,
    infer_pipeline_from_architecture,
    resolve_model_class,
)


pytestmark = pytest.mark.release


def test_registry_import_does_not_load_mlx():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mlx_raclate.registry; "
            "assert not any(n == 'mlx' or n.startswith('mlx.') for n in sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_registry_contains_the_runtime_model_types():
    assert set(ARCHITECTURES) == {
        "bidirectional_pplx_qwen3",
        "colbert_zero",
        "gemma3_text",
        "granite_embedding",
        "lateon",
        "lfm2",
        "lfm2_bidirectional",
        "modernbert",
        "neobert",
        "openai_privacy_filter",
        "qwen3",
        "t5gemma_encoder",
    }


def test_supported_pipeline_resolves_lazy_class_names():
    reference = resolve_model_class("modernbert", "zero-shot-classification")

    assert reference.module == "mlx_raclate.models.modernbert"
    assert reference.model_class == "ModelForMaskedLM"
    assert reference.args_class == "ModelArgs"


@pytest.mark.parametrize("model_type", ["colbert_zero", "granite_embedding", "lateon"])
def test_placeholder_heads_are_not_advertised(model_type):
    with pytest.raises(ValueError, match="not implemented"):
        resolve_model_class(model_type, "masked-lm")


def test_architecture_specific_error_lists_real_capabilities():
    with pytest.raises(ValueError, match="embeddings.*sentence-transformers"):
        resolve_model_class("qwen3", "masked-lm")


def test_unknown_model_and_pipeline_fail_cleanly():
    with pytest.raises(ValueError, match="Model type 'unknown' is not supported"):
        resolve_model_class("unknown", "embeddings")
    with pytest.raises(ValueError, match="Pipeline 'unknown' is not supported"):
        resolve_model_class("modernbert", "unknown")


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [
        ("ModernBertForSequenceClassification", "text-classification"),
        ("Lfm2BidirectionalForMaskedLM", "masked-lm"),
        ("Lfm2BidirectionalForTokenClassification", "token-classification"),
        ("Lfm2BidirectionalModel", "embeddings"),
        ("ModernBertModel", None),
        (None, None),
    ],
)
def test_pipeline_inference(architecture, expected):
    assert infer_pipeline_from_architecture(architecture) == expected


def test_pipeline_names_are_stable():
    assert PIPELINES == (
        "embeddings",
        "masked-lm",
        "text-classification",
        "token-classification",
        "sentence-transformers",
        "zero-shot-classification",
        "sentence-similarity",
    )
