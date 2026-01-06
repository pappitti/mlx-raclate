"""
Zero-Shot Classification Inference Example

Based on the AnswerAI paper: https://arxiv.org/html/2502.03793v2

Example usage:
    from tests.inference_examples.zero_shot import run_inference
    
    results = run_inference(
        model_path="answerdotai/ModernBERT-Large-Instruct",
        text="MLX is an array framework for machine learning.",
        label_candidates=["technology", "sports", "politics"]
    )
"""

import mlx.core as mx
from typing import List, Dict, Any, Optional, Union

def run_inference(
    model_path: str,
    text: str,
    label_candidates: Union[List[str], Dict[str, str]],
    top_k: int = 10,
    model_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run zero-shot classification inference.
    
    Args:
        model_path: HuggingFace model ID or local path
        text: Text to classify
        label_candidates: List of labels or dict of {label: description}
        top_k: Number of top predictions to consider
        model_config: Additional model configuration
        
    Returns:
        Dictionary containing:
        - predictions: Dict mapping labels to probabilities
        - top_label: The most likely label
        - raw_tokens: Top-k raw token predictions
    """
    from mlx_raclate.utils.utils import load
    
    config = model_config or {}
    
    # Load model and tokenizer
    model, tokenizer = load(
        model_path,
        model_config=config,
        pipeline="zero-shot-classification"
    )
    
    max_length = getattr(model.config, "max_position_embeddings", 512)
    
    # Format label candidates
    if isinstance(label_candidates, dict):
        categories = "\n".join([
            f"{i}: {k.lower().strip()} ({v})" 
            for i, (k, v) in enumerate(label_candidates.items())
        ])
        label_keys = list(label_candidates.keys())
    else:
        categories = "\n".join([
            f"{i}: {label.lower().strip()}" 
            for i, label in enumerate(label_candidates)
        ])
        label_keys = label_candidates
    
    # Create prompt with mask
    prompt = f"""You will be given a text and categories to classify the text.

TEXT: {text}

Read the text carefully and select the right category from the list. Only provide the index of the category:
{categories}

ANSWER: [unused0][MASK]
"""
    
    # Tokenize
    tokens = tokenizer.encode(
        prompt,
        return_tensors="mlx",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    # Find mask position
    mask_token_id = tokenizer.mask_token_id
    mask_position = mx.argmax(tokens == mask_token_id)
    
    # Run inference
    outputs = model(input_ids=tokens, return_dict=True)
    
    # Get predictions for the masked token
    logits = outputs["logits"]
    masked_token_predictions = logits[0, mask_position]
    
    # Get probabilities
    probs = mx.softmax(masked_token_predictions)
    
    # Create label mapping
    label_mapping = {str(i): label.lower().strip() for i, label in enumerate(label_keys)}
    label_probs = {k.lower().strip(): 0.0 for k in label_keys}
    
    # Get top-k predictions
    sorted_indices = mx.argsort(probs)[::-1]
    top_indices = sorted_indices[:top_k].astype(mx.int32)
    top_probs = probs[top_indices]
    
    raw_tokens = []
    for idx, prob_val in zip(top_indices.tolist(), top_probs.tolist()):
        token = tokenizer.decode([idx]).lower().strip()
        prob = prob_val
        raw_tokens.append({"token": token, "probability": prob})
        
        # Accumulate probabilities (model may return label name or index)
        if token in label_probs:
            label_probs[token] += prob
        elif token in label_mapping:
            label = label_mapping[token]
            if label in label_probs:
                label_probs[label] += prob
    
    # Sort by probability
    sorted_probs = dict(sorted(label_probs.items(), key=lambda x: x[1], reverse=True))
    top_label = list(sorted_probs.keys())[0] if sorted_probs else None
    
    return {
        "predictions": sorted_probs,
        "top_label": top_label,
        "raw_tokens": raw_tokens,
    }


_EXAMPLE_CODE_TEMPLATE = '''import mlx.core as mx
from mlx_raclate.utils.utils import load

# Load model and tokenizer
model, tokenizer = load(
    "{model_path}",
    pipeline="zero-shot-classification"
)

# Prepare text and label candidates
text_to_classify = "{text}"

label_candidates = {label_candidates}

# Format categories for prompt
if isinstance(label_candidates, dict):
    categories = "\\n".join([
        f"{{i}}: {{k.lower().strip()}} ({{v}})" 
        for i, (k, v) in enumerate(label_candidates.items())
    ])
    label_keys = list(label_candidates.keys())
else:
    categories = "\\n".join([
        f"{{i}}: {{label.lower().strip()}}" 
        for i, label in enumerate(label_candidates)
    ])
    label_keys = label_candidates

# Create prompt with mask token
prompt = f"""You will be given a text and categories to classify the text.

TEXT: {{text_to_classify}}

Read the text carefully and select the right category from the list. Only provide the index of the category:
{{categories}}

ANSWER: [unused0][MASK]
"""

# Tokenize
max_length = getattr(model.config, "max_position_embeddings", 512)
tokens = tokenizer.encode(
    prompt,
    return_tensors="mlx",
    padding=True,
    truncation=True,
    max_length=max_length
)

# Find mask position and run inference
mask_token_id = tokenizer.mask_token_id
mask_position = mx.argmax(tokens == mask_token_id)

outputs = model(input_ids=tokens, return_dict=True)
logits = outputs["logits"]
masked_token_predictions = logits[0, mask_position]

# Get probabilities and map to labels
probs = mx.softmax(masked_token_predictions)
label_mapping = {{str(i): label.lower().strip() for i, label in enumerate(label_keys)}}
label_probs = {{k.lower().strip(): 0.0 for k in label_keys}}

sorted_indices = mx.argsort(probs)[::-1][:10]
for idx in sorted_indices.tolist():
    token = tokenizer.decode([idx]).lower().strip()
    prob = float(probs[idx])
    if token in label_probs:
        label_probs[token] += prob
    elif token in label_mapping:
        label_probs[label_mapping[token]] += prob

# Print results
print(f"Text: {{text_to_classify}}\\n")
print("Label probabilities:")
for label, prob in sorted(label_probs.items(), key=lambda x: x[1], reverse=True):
    print(f"  {{label}}: {{prob:.3f}}")
'''


def get_example_code(
    model_path: str = "{{MODEL_PATH}}",
    text: str = "MLX is an array framework for machine learning on Apple Silicon.",
    label_candidates: Optional[Union[List[str], Dict[str, str]]] = None,
) -> str:
    
    if label_candidates is None:
        label_candidates = {
            "artificial intelligence": "The study of computer science that focuses on the creation of intelligent machines.",
            "physics": "The study of matter, energy, and the fundamental forces of nature.",
            "biology": "The study of living organisms.",
        }
    
    return _EXAMPLE_CODE_TEMPLATE.format(
        model_path=model_path,
        text=text,
        label_candidates=repr(label_candidates),
    ).strip()
    

if __name__ == "__main__":
    print("Example code for model card:")
    print("-" * 40)
    print(get_example_code("answerdotai/ModernBERT-Large-Instruct"))
