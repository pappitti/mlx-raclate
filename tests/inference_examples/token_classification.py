import mlx.core as mx
from typing import List, Dict, Any, Optional

def run_inference(
    model_path: str,
    texts: List[str],
    model_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run token classification (NER) inference.
    
    Args:
        model_path: HuggingFace model ID or local path
        texts: List of texts for token classification
        model_config: Additional model configuration
        
    Returns:
        Dictionary containing:
        - predictions: List of token-level predictions per input
        - id2label: Label mapping (if available)
    """
    from mlx_raclate.utils.utils import load
    
    config = model_config or {}
    
    # Load model and tokenizer
    model, tokenizer = load(
        model_path,
        model_config=config,
        pipeline="token-classification"
    )
    
    max_length = getattr(model.config, "max_position_embeddings", 512)
    id2label = getattr(model.config, "id2label", None)
    
    # Tokenize
    tokens = tokenizer._tokenizer(
        texts,
        return_tensors="mlx",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    
    # Store offset mapping and remove from model inputs
    offset_mapping = tokens.pop("offset_mapping", None)
    
    # Run inference
    outputs = model(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        return_dict=True
    )
    
    logits = outputs["logits"]
    
    # Get predictions
    predictions = []
    for i in range(logits.shape[0]):
        token_logits = logits[i]
        token_predictions = mx.argmax(token_logits, axis=-1)
        
        pred_list = []
        for j, pred_idx in enumerate(token_predictions.tolist()):
            token = tokenizer.decode([tokens["input_ids"][i][j].item()])
            label = id2label[str(pred_idx)] if id2label else str(pred_idx)
            pred_list.append({
                "token": token,
                "label": label,
                "label_id": pred_idx,
            })
        
        predictions.append(pred_list)
    
    return {
        "predictions": predictions,
        "id2label": id2label,
        "logits": logits,
    }


_EXAMPLE_CODE_TEMPLATE = '''import mlx.core as mx
from mlx_raclate.utils.utils import load

# Load model and tokenizer
model, tokenizer = load(
    "{model_path}",
    pipeline="token-classification"
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
logits = outputs["logits"]
predictions = mx.argmax(logits, axis=-1)
id2label = model.config.id2label

# Process and print results
for i, text in enumerate(texts):
    print(f"Text: {{text}}")
    print("Token predictions:")
    for j, pred_idx in enumerate(predictions[i].tolist()):
        token = tokenizer.decode([tokens["input_ids"][i][j].item()])
        label = id2label[str(pred_idx)] if id2label else str(pred_idx)
        if label != "O":  # Skip non-entity tokens
            print(f"  {{token}}: {{label}}")
    print()
'''


def get_example_code(
    model_path: str = "{{MODEL_PATH}}",
    texts: Optional[List[str]] = None,
) -> str:

    if texts is None:
        texts = [
            "John works at Apple in California.",
            "Microsoft was founded by Bill Gates.",
        ]
    
    return _EXAMPLE_CODE_TEMPLATE.format(
        model_path=model_path,
        texts=repr(texts),
    ).strip()


if __name__ == "__main__":
    print("Example code for model card:")
    print("-" * 40)
    print(get_example_code("my-org/my-ner-model"))
