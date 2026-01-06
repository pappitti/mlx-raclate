from typing import List, Optional
import importlib

# Pipeline to module mapping
_PIPELINE_TO_MODULE = {
    "text-classification": "text_classification",
    "sentence-similarity": "sentence_similarity",
    "sentence-transformers": "sentence_similarity",  # Same module, similar code
    "embeddings": "embeddings",
    "masked-lm": "masked_lm",
    "zero-shot-classification": "zero_shot",
}


def get_inference_code(
    pipeline: str,
    model_path: str = "{{MODEL_PATH}}",
    **kwargs,
) -> str:
    """
    Get inference example code for a model card.
    
    This function returns clean, runnable Python code that can be directly
    used in HuggingFace model cards. The code comes from the same source
    as the test suite, ensuring consistency.
    
    Args:
        pipeline: The pipeline type (e.g., "text-classification", "sentence-similarity")
        model_path: The model path to use in the example. Use "{{MODEL_PATH}}" as a
            placeholder if the actual path isn't known yet.
        **kwargs: Additional arguments passed to the specific pipeline's get_example_code()
            function. Common options:
            - text: str - for masked-lm, zero-shot
            - texts: List[str] - for text-classification, embeddings
            - text_pairs: List[str] - for text-classification
            - documents: List[str] - for text-classification
            - queries: List[str] - for text-classification
            - is_regression: bool - for text-classification
            - use_late_interaction: bool - for sentence-similarity (ColBERT-style)
            - label_candidates: List or Dict - for zero-shot
    
    Returns:
        Formatted Python code string ready for inclusion in a model card.

    """
    if pipeline not in _PIPELINE_TO_MODULE:
        raise ValueError(
            f"Unknown pipeline: {pipeline}. "
            f"Supported pipelines: {list(_PIPELINE_TO_MODULE.keys())}"
        )
    
    module_name = _PIPELINE_TO_MODULE[pipeline]
    
    # Try importing from tests.inference_examples first (development)
    # Fall back to relative import if tests not available
    try:
        module = importlib.import_module(f"tests.inference_examples.{module_name}")
    except ImportError:
        # If running from within the library, try relative path
        try:
            import tests.inference_examples
            module = getattr(tests.inference_examples, module_name)
        except (ImportError, AttributeError):
            raise ImportError(
                f"Could not import inference example module for {pipeline}. "
                "Make sure the tests package is installed or accessible."
            )
    
    # Call the module's get_example_code function
    return module.get_example_code(model_path=model_path, **kwargs)


def get_available_pipelines() -> List[str]:
    """Get list of pipelines that have model card code templates."""
    return list(_PIPELINE_TO_MODULE.keys())


def generate_model_card_section(
    pipeline: str,
    model_path: str,
    title: str = "Usage with mlx-raclate",
    **kwargs,
) -> str:
    """
    Generate a complete model card section with title and code block.
    
    Args:
        pipeline: The pipeline type
        model_path: The model path
        title: Section title
        **kwargs: Additional arguments for get_inference_code()
        
    Returns:
        Markdown-formatted section for a model card
    """
    code = get_inference_code(pipeline=pipeline, model_path=model_path, **kwargs)
    
    return f"""## {title}

This model can be used with [mlx-raclate](https://github.com/pappitti/mlx-raclate) for native inference on Apple Silicon.

```python
{code}
```
"""


def get_code_for_trained_model(
    pipeline: str,
    model_path: str,
    base_model: str,
    training_task: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Generate model card content for a newly trained model.
    
    This is intended to be called after training, to generate the
    inference example code for the model card before uploading to HuggingFace.
    
    Args:
        pipeline: Pipeline the model was trained for
        model_path: Path where the model will be uploaded (e.g., "my-org/my-model")
        base_model: The base model used for training
        training_task: Optional description of the training task
        **kwargs: Additional arguments for the code template
        
    Returns:
        Complete markdown section for the model card
    """
    section = generate_model_card_section(
        pipeline=pipeline,
        model_path=model_path,
        **kwargs,
    )
    
    # Add metadata about training
    metadata = f"""
### Model Details

- **Base Model**: [{base_model}](https://huggingface.co/{base_model})
- **Pipeline**: `{pipeline}`
- **Framework**: [mlx-raclate](https://github.com/pappitti/mlx-raclate) (MLX)
"""
    if training_task:
        metadata += f"- **Training Task**: {training_task}\n"
    
    return section + metadata


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI for generating model card code snippets."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate inference code for model cards"
    )
    parser.add_argument(
        "pipeline",
        choices=get_available_pipelines(),
        help="Pipeline type"
    )
    parser.add_argument(
        "--model-path",
        default="{{MODEL_PATH}}",
        help="Model path/ID for the example"
    )
    parser.add_argument(
        "--late-interaction",
        action="store_true",
        help="Use late interaction for sentence-similarity"
    )
    parser.add_argument(
        "--full-section",
        action="store_true",
        help="Generate full markdown section instead of just code"
    )
    
    args = parser.parse_args()
    
    kwargs = {}
    if args.late_interaction:
        kwargs["use_late_interaction"] = True
    
    if args.full_section:
        output = generate_model_card_section(
            pipeline=args.pipeline,
            model_path=args.model_path,
            **kwargs,
        )
    else:
        output = get_inference_code(
            pipeline=args.pipeline,
            model_path=args.model_path,
            **kwargs,
        )
    
    print(output)


if __name__ == "__main__":
    main()
