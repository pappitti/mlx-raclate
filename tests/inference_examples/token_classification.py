import mlx.core as mx
from typing import List, Dict, Any, Optional

from mlx_raclate.utils.token_classification import (
    decode_token_classification_batch,
    viterbi_decode_bioes_batch,
)


def _ordered_label_names(id2label: Dict[Any, str], num_labels: int) -> List[str]:
    ordered = []
    for index in range(num_labels):
        if str(index) in id2label:
            ordered.append(id2label[str(index)])
        else:
            ordered.append(id2label[index])
    return ordered

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
    label_names = None
    decoded_label_ids = None

    if id2label is not None:
        label_names = _ordered_label_names(id2label, logits.shape[-1])
        special_tokens_mask = None
        if offset_mapping is not None:
            special_tokens_mask = [
                [int(start) >= int(end) for start, end in sequence_offsets.tolist()]
                for sequence_offsets in offset_mapping
            ]
        decoded_label_ids = viterbi_decode_bioes_batch(
            emissions=logits.tolist(),
            label_names=label_names,
            special_tokens_mask=special_tokens_mask,
        )
    
    # Get predictions
    predictions = []
    grouped_labels = []
    grouped_scores = []
    for i in range(logits.shape[0]):
        if decoded_label_ids is None:
            token_prediction_ids = mx.argmax(logits[i], axis=-1).tolist()
        else:
            token_prediction_ids = decoded_label_ids[i]
        
        pred_list = []
        label_list = []
        score_list = []
        for j, pred_idx in enumerate(token_prediction_ids):
            token = tokenizer.decode([tokens["input_ids"][i][j].item()])
            label = id2label[str(pred_idx)] if id2label else str(pred_idx)
            score = float(probabilities[i, j, pred_idx].item())
            pred_list.append({
                "token": token,
                "label": label,
                "label_id": pred_idx,
                "score": score,
            })
            label_list.append(label)
            score_list.append(score)
        
        predictions.append(pred_list)
        grouped_labels.append(label_list)
        grouped_scores.append(score_list)

    grouped_spans = None
    if offset_mapping is not None:
        grouped_spans = decode_token_classification_batch(
            texts=texts,
            offsets=offset_mapping.tolist(),
            labels=grouped_labels,
            scores=grouped_scores,
        )
    
    return {
        "predictions": predictions,
        "grouped_spans": grouped_spans,
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

# Process and print grouped spans
for i, text in enumerate(texts):
    print(f"Text: {{text}}")
    labels = [id2label[str(index)] for index in range(logits.shape[-1])]
    offsets = tokenizer._tokenizer(
        [text],
        return_tensors="mlx",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )["offset_mapping"][0].tolist()

    from mlx_raclate.utils.token_classification import decode_bioes_spans, viterbi_decode_bioes_ids
    decoded_ids = viterbi_decode_bioes_ids(
        logits[i].tolist(),
        labels,
        special_tokens_mask=[start >= end for start, end in offsets],
    )
    decoded_labels = [labels[pred_idx] for pred_idx in decoded_ids]
    scores = [float(outputs["probabilities"][i, j, pred_idx].item()) for j, pred_idx in enumerate(decoded_ids)]
    spans = decode_bioes_spans(text, offsets, decoded_labels, scores)

    print("Grouped spans:")
    for span in spans:
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
