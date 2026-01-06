import mlx.core as mx
from typing import List, Dict, Any, Optional

def run_inference(
    model_path: str,
    texts: List[str],
    text_pairs: Optional[List[str]] = None,
    is_regression: bool = False,
    top_k: int = 5,
    model_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run text classification inference.
    
    Args:
        model_path: HuggingFace model ID or local path
        texts: List of texts to classify
        text_pairs: Optional second text for pair classification
        is_regression: Whether this is a regression task
        top_k: Number of top predictions to return
        model_config: Additional model configuration
        
    Returns:
        Dictionary containing:
        - predictions: List of prediction dicts per input
        - probabilities: Raw probability tensor
        - id2label: Label mapping (if available)
    """
    from mlx_raclate.utils.utils import load
    
    # Merge user config with defaults
    config = {"is_regression": is_regression}
    if model_config:
        config.update(model_config)
    
    # Load model and tokenizer
    model, tokenizer = load(
        model_path,
        model_config=config,
        pipeline="text-classification"
    )
    
    max_length = getattr(model.config, "max_position_embeddings", 512)
    id2label = getattr(model.config, "id2label", None)
    
    # Tokenize
    tokens = tokenizer._tokenizer(
        texts,
        text_pairs,
        return_tensors="mlx",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    # Run inference
    outputs = model(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        return_dict=True
    )
    
    probabilities = outputs["probabilities"]
    
    # Process predictions
    predictions = []
    for i in range(probabilities.shape[0]):
        probs = probabilities[i]
        
        if is_regression:
            predictions.append({
                "score": float(probs[0]),
            })
        else:
            # Get top-k predictions
            sorted_indices = mx.argsort(probs)[::-1]
            top_indices = sorted_indices[:min(len(probs), top_k)]
            top_probs = probs[top_indices]
            
            pred = {
                "top_predictions": [
                    {
                        "label": id2label[str(idx)] if id2label else str(idx),
                        "score": float(prob),
                    }
                    for idx, prob in zip(top_indices.tolist(), top_probs.tolist())
                ]
            }
            predictions.append(pred)
    
    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "id2label": id2label,
    }


_EXAMPLE_CODE_TEMPLATE = '''import mlx.core as mx
from mlx_raclate.utils.utils import load

# Load model and tokenizer
model, tokenizer = load(
    "{model_path}",
    model_config={{"is_regression": {is_regression}}},
    pipeline="text-classification"
)

# Prepare input texts
texts = {texts}

# Tokenize
max_length = getattr(model.config, "max_position_embeddings", 512)
tokens = tokenizer._tokenizer(
    texts,
    return_tensors="mlx",
    padding=True,
    truncation=True,
    max_length=max_length
)

# Run inference
outputs = model(
    input_ids=tokens["input_ids"],
    attention_mask=tokens["attention_mask"],
    return_dict=True
)

# Get predictions
probabilities = outputs["probabilities"]
id2label = model.config.id2label

for i, text in enumerate(texts):
    probs = probabilities[i]
    sorted_indices = mx.argsort(probs)[::-1]
    top_indices = sorted_indices[:5]
    
    print(f"Text: {{text}}")
    print("Predictions:")
    for idx, prob in zip(top_indices.tolist(), probs[top_indices].tolist()):
        label = id2label[str(idx)]
        print(f"  {{label}}: {{prob:.3f}}")
    print()
'''


def get_example_code(
    model_path: str = "{{MODEL_PATH}}",
    texts: Optional[List[str]] = None,
    is_regression: bool = False,
) -> str:

    if texts is None:
        texts = [
            "This movie was absolutely fantastic!",
            "I didn't enjoy this product at all.",
        ]
    
    return _EXAMPLE_CODE_TEMPLATE.format(
        model_path=model_path,
        texts=repr(texts),
        is_regression=str(is_regression),
    ).strip()


if __name__ == "__main__":
    # Print example code for model card
    print("Example code for model card:")
    print("-" * 40)
    print(get_example_code("NousResearch/Minos-v1"))
