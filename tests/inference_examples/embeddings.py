from typing import List, Dict, Any, Optional

def run_inference(
    model_path: str,
    texts: List[str],
    model_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run embeddings extraction inference.
    
    Args:
        model_path: HuggingFace model ID or local path
        texts: List of texts to embed
        model_config: Additional model configuration
        
    Returns:
        Dictionary containing:
        - embeddings: Tensor of shape [batch_size, hidden_size]
        - hidden_size: Dimension of embeddings
    """
    from mlx_raclate.utils.utils import load
    
    config = model_config or {}
    
    # Load model and tokenizer
    model, tokenizer = load(
        model_path,
        model_config=config,
        pipeline="embeddings"
    )
    
    max_length = getattr(model.config, "max_position_embeddings", 512)
    
    # Tokenize
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
    )
    
    embeddings = outputs["embeddings"]
    
    return {
        "embeddings": embeddings,
        "hidden_size": embeddings.shape[-1],
    }


_EXAMPLE_CODE_TEMPLATE = '''import mlx.core as mx
from mlx_raclate.utils.utils import load

# Load model and tokenizer
model, tokenizer = load(
    "{model_path}",
    pipeline="embeddings"
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
)

# Get normalized embeddings
embeddings = outputs["embeddings"]  # Shape: [batch_size, hidden_size]

print(f"Embeddings shape: {{embeddings.shape}}")

# Compute cosine similarity between embeddings
def cosine_similarity(a, b):
    return mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b))

print("\\nCosine Similarity Matrix:")
for i in range(len(texts)):
    for j in range(len(texts)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"  {{i}} vs {{j}}: {{float(sim):.4f}}")
'''


def get_example_code(
    model_path: str = "{{MODEL_PATH}}",
    texts: Optional[List[str]] = None,
) -> str:

    if texts is None:
        texts = [
            "I like grapes",
            "I like fruits",
            "The weather is nice today",
        ]
    
    return _EXAMPLE_CODE_TEMPLATE.format(
        model_path=model_path,
        texts=repr(texts),
    ).strip()


if __name__ == "__main__":
    print("Example code for model card:")
    print("-" * 40)
    print(get_example_code("answerdotai/ModernBERT-base"))
