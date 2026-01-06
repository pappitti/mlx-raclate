import mlx.core as mx
from typing import List, Dict, Any, Optional

def run_inference(
    model_path: str,
    text: str,
    top_k: int = 5,
    model_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run masked language model inference.
    
    Args:
        model_path: HuggingFace model ID or local path
        text: Text with [MASK] token(s)
        top_k: Number of top predictions to return
        model_config: Additional model configuration
        
    Returns:
        Dictionary containing:
        - predictions: List of predicted tokens with probabilities
        - mask_position: Position of the mask token
        - logits: Raw logits
    """
    from mlx_raclate.utils.utils import load
    
    config = model_config or {}
    
    # Load model and tokenizer
    model, tokenizer = load(
        model_path,
        model_config=config,
        pipeline="masked-lm"
    )
    
    # Tokenize
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])  # Add batch dimension
    
    # Find mask position
    mask_token_id = tokenizer.mask_token_id
    mask_position = tokens.index(mask_token_id)
    
    # Run inference
    outputs = model(input_ids=input_ids, return_dict=True)
    
    # Get predictions for the masked token
    logits = outputs["logits"]
    masked_token_predictions = logits[0, mask_position]
    
    # Get top-k predictions
    probs = mx.softmax(masked_token_predictions)
    sorted_indices = mx.argsort(probs)[::-1]
    top_indices = sorted_indices[:top_k].astype(mx.int32)
    top_probs = probs[top_indices]
    
    predictions = [
        {
            "token": tokenizer.decode([idx]),
            "probability": float(prob),
        }
        for idx, prob in zip(top_indices.tolist(), top_probs.tolist())
    ]
    
    return {
        "predictions": predictions,
        "mask_position": mask_position,
        "logits": logits,
    }


_EXAMPLE_CODE_TEMPLATE = '''import mlx.core as mx
from mlx_raclate.utils.utils import load

# Load model and tokenizer
model, tokenizer = load(
    "{model_path}",
    pipeline="masked-lm"
)

# Prepare input text with [MASK] token
text = "{text}"

# Tokenize
tokens = tokenizer.encode(text)
input_ids = mx.array([tokens])  # Add batch dimension

# Find mask position
mask_token_id = tokenizer.mask_token_id
mask_position = tokens.index(mask_token_id)

# Run inference
outputs = model(input_ids=input_ids, return_dict=True)

# Get predictions for the masked token
logits = outputs["logits"]
masked_token_predictions = logits[0, mask_position]

# Get top 5 predictions
probs = mx.softmax(masked_token_predictions)
sorted_indices = mx.argsort(probs)[::-1]
top_indices = sorted_indices[:5].astype(mx.int32)
top_probs = probs[top_indices]

print(f"Input: {{text}}")
print("\\nTop 5 predictions for [MASK]:")
for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
    token = tokenizer.decode([idx])
    print(f"  {{token}}: {{prob:.3f}}")
'''


def get_example_code(
    model_path: str = "{{MODEL_PATH}}",
    text: str = "The capital of France is [MASK].",
) -> str:

    return _EXAMPLE_CODE_TEMPLATE.format(
        model_path=model_path,
        text=text,
    ).strip()

if __name__ == "__main__":
    print("Example code for model card:")
    print("-" * 40)
    print(get_example_code("answerdotai/ModernBERT-base"))
