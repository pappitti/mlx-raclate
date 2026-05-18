from typing import List, Dict, Any, Optional

from mlx_raclate.utils.token_classification import (
    postprocess_token_classification_output,
    viterbi_transition_biases_from_calibration,
)


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
        - grouped_spans: BIOES-decoded grouped spans per input
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
    transition_biases = viterbi_transition_biases_from_calibration(
        getattr(model, "viterbi_calibration", None)
    )
    
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
    probabilities = outputs["probabilities"]
    processed = postprocess_token_classification_output(
        logits=logits,
        probabilities=probabilities,
        id2label=id2label,
        texts=texts,
        offsets=None if offset_mapping is None else offset_mapping.tolist(),
        transition_biases=transition_biases,
    )

    predictions = processed["predictions"]
    for i, pred_list in enumerate(predictions):
        for j, prediction in enumerate(pred_list):
            prediction["token"] = tokenizer.decode([tokens["input_ids"][i][j].item()])
    
    return {
        "predictions": predictions,
        "grouped_spans": processed["grouped_spans"],
        "id2label": id2label,
        "logits": logits,
    }


_EXAMPLE_CODE_TEMPLATE = '''from mlx_raclate.utils.utils import load
from mlx_raclate.utils.token_classification import (
    postprocess_token_classification_output,
    viterbi_transition_biases_from_calibration,
)

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
    max_length=max_length,
    return_offsets_mapping=True,
)
offset_mapping = tokens.pop("offset_mapping")

# Run inference
outputs = model(
    input_ids=tokens["input_ids"],
    attention_mask=tokens["attention_mask"],
    return_dict=True
)

# Get predictions
logits = outputs["logits"]
id2label = model.config.id2label
transition_biases = viterbi_transition_biases_from_calibration(
    getattr(model, "viterbi_calibration", None)
)
processed = postprocess_token_classification_output(
    logits=logits,
    probabilities=outputs["probabilities"],
    id2label=id2label,
    texts=texts,
    offsets=offset_mapping.tolist(),
    transition_biases=transition_biases,
)

# Process and print grouped spans
for i, text in enumerate(texts):
    print(f"Text: {{text}}")
    print("Grouped spans:")
    for span in processed["grouped_spans"][i]:
        print(f"  {{span['entity_group']}}: {{span['word']!r}} [{{span['start']}}, {{span['end']}}] score={{span['score']:.3f}}")
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
