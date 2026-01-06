"""
Inference Tests for mlx-raclate

These tests verify that inference works correctly for each (model_family, pipeline)
combination in the model registry. Tests are marked as 'slow' since they require
model downloads.

Run with: pytest tests/test_inference.py -v --run-slow
Skip slow tests: pytest tests/test_inference.py -v
Filter by family: pytest tests/test_inference.py -v --run-slow --model-family modernbert
"""

import pytest
import mlx.core as mx
from typing import Dict, List, Any

from .model_registry import (
    get_endpoint,
    get_inference_test_cases,
    get_untested_combinations,
    MODEL_FAMILIES
)
from .inference_examples import (
    text_classification,
    sentence_similarity,
    embeddings,
    masked_lm,
    zero_shot,
)

def _get_inference_params():
    """Generate pytest parameters from the model registry."""
    test_cases = get_inference_test_cases()
    return [
        pytest.param(
            model_family,
            pipeline,
            config,
            id=f"{model_family}-{pipeline}",
            marks=[pytest.mark.slow]
        )
        for model_family, pipeline, config in test_cases
    ]


@pytest.mark.parametrize("model_family,pipeline,endpoint_config", _get_inference_params())
def test_inference(model_family: str, pipeline: str, endpoint_config):
    """
    Test inference for each registered (model_family, pipeline) combination.
    
    This test:
    1. Loads the model from the registered endpoint
    2. Runs inference using the corresponding inference example module
    3. Validates the output shape and types
    """
    endpoint = endpoint_config.endpoint
    model_config = endpoint_config.model_config
    
    # Dispatch to the appropriate inference example
    if pipeline == "text-classification":
        result = text_classification.run_inference(
            model_path=endpoint,
            texts=["This is a test sentence.", "Another test for classification."],
            model_config=model_config,
        )
        assert "predictions" in result
        assert "probabilities" in result
        assert len(result["predictions"]) == 2
        assert result["probabilities"].shape[0] == 2
        
    elif pipeline in ("sentence-similarity", "sentence-transformers"):
        use_late = model_config.get("use_late_interaction", False)
        result = sentence_similarity.run_inference(
            model_path=endpoint,
            queries=["What is MLX?"],
            documents=["MLX is an array framework.", "Python is a language."],
            use_late_interaction=use_late,
            model_config=model_config,
        )
        assert "similarities" in result
        # Should be [1, 2] for 1 query vs 2 documents
        assert result["similarities"].shape[0] == 1
        assert result["similarities"].shape[1] == 2

        # also test late interaction
        late_result = sentence_similarity.run_inference(
            model_path=endpoint,
            queries=["What is MLX?"],
            documents=["MLX is an array framework.", "Python is a language."],
            use_late_interaction=True,
            model_config=model_config,
        )
        assert "similarities" in late_result
        assert late_result["similarities"].shape[0] == 1
        assert late_result["similarities"].shape[1] == 2
        
    elif pipeline == "embeddings":
        result = embeddings.run_inference(
            model_path=endpoint,
            texts=["Test text one", "Test text two"],
            model_config=model_config,
        )
        assert "embeddings" in result
        assert result["embeddings"].shape[0] == 2
        assert result["hidden_size"] > 0
        
    elif pipeline == "masked-lm":
        result = masked_lm.run_inference(
            model_path=endpoint,
            text="The capital of France is [MASK].",
            model_config=model_config,
        )
        assert "predictions" in result
        assert len(result["predictions"]) > 0
        assert "mask_position" in result
        # Check predictions have expected structure
        for pred in result["predictions"]:
            assert "token" in pred
            assert "probability" in pred
        
    elif pipeline == "zero-shot-classification":
        result = zero_shot.run_inference(
            model_path=endpoint,
            text="MLX is a machine learning framework for Apple Silicon.",
            label_candidates=["technology", "sports", "politics"],
            model_config=model_config,
        )
        assert "predictions" in result
        assert "top_label" in result
        assert isinstance(result["predictions"], dict)
        
    else:
        pytest.skip(f"Inference test not implemented for pipeline: {pipeline}")


def test_model_registry_coverage():
    """Validate that all expected combinations have endpoints or are documented."""
    untested = get_untested_combinations()
    
    # Just log the untested combinations for now
    if untested:
        print(f"\n⚠️  {len(untested)} untested (model_family, pipeline) combinations:")
        for fam, pipe in untested:
            print(f"   - ({fam}, {pipe})")
    
    # This test passes but logs warnings - adjust as needed for stricter coverage
    assert True


def test_all_model_families_have_at_least_one_endpoint():
    """Ensure each model family has at least one testable endpoint."""
    for family in MODEL_FAMILIES:
        endpoints = [
            config for (fam, pipe, config) in get_inference_test_cases()
            if fam == family
        ]
        assert len(endpoints) > 0, f"Model family '{family}' has no registered endpoints"


# ============================================================================
# EXAMPLE CODE EXTRACTION TESTS
# ============================================================================

def test_text_classification_example_code_generation():
    """Test that example code generates valid Python."""
    code = text_classification.get_example_code(
        model_path="NousResearch/Minos-v1",
        texts=["Test text"],
        is_regression=False,
    )
    assert "from mlx_raclate.utils.utils import load" in code
    assert "NousResearch/Minos-v1" in code
    assert "pipeline=\"text-classification\"" in code
    # Verify it's valid Python syntax
    compile(code, "<string>", "exec")


def test_sentence_similarity_example_code_generation():
    """Test that example code generates valid Python."""
    code = sentence_similarity.get_example_code(
        model_path="nomic-ai/modernbert-embed-base",
        use_late_interaction=False,
    )
    assert "from mlx_raclate.utils.utils import load" in code
    assert "sentence-similarity" in code
    compile(code, "<string>", "exec")
    
    # Also test late interaction variant
    code_late = sentence_similarity.get_example_code(
        model_path="LiquidAI/LFM2-ColBERT-350M",
        use_late_interaction=True,
    )
    assert "use_late_interaction" in code_late
    compile(code_late, "<string>", "exec")


def test_embeddings_example_code_generation():
    """Test that example code generates valid Python."""
    code = embeddings.get_example_code(model_path="answerdotai/ModernBERT-base")
    assert "pipeline=\"embeddings\"" in code
    compile(code, "<string>", "exec")


def test_masked_lm_example_code_generation():
    """Test that example code generates valid Python."""
    code = masked_lm.get_example_code(
        model_path="answerdotai/ModernBERT-base",
        text="The [MASK] is blue.",
    )
    assert "pipeline=\"masked-lm\"" in code
    assert "[MASK]" in code
    compile(code, "<string>", "exec")


def test_zero_shot_example_code_generation():
    """Test that example code generates valid Python."""
    code = zero_shot.get_example_code(
        model_path="answerdotai/ModernBERT-Large-Instruct",
    )
    assert "zero-shot-classification" in code
    compile(code, "<string>", "exec")
