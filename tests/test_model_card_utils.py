"""Tests for installable model-card inference templates."""

from mlx_raclate.tuner.model_card_utils import (
    _load_inference_example_module,
    get_inference_code,
)


def test_token_classification_template_is_loaded_from_runtime_package():
    module = _load_inference_example_module("token-classification")

    assert module.__name__ == (
        "mlx_raclate.inference_examples.token_classification"
    )


def test_token_classification_model_card_code_is_generated_without_tests_package():
    code = get_inference_code(
        pipeline="token-classification",
        model_path="org/model",
    )

    assert 'model_path = "org/model"' in code
    assert "postprocess_token_classification_output" in code
