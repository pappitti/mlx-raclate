from typing import List, Dict, Any, Optional

def run_inference(
    model_path: str,
    queries: List[str],
    documents: List[str],
    use_late_interaction: bool = False,
    model_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run sentence similarity inference.
    
    Args:
        model_path: HuggingFace model ID or local path
        queries: List of query texts
        documents: List of document texts to compare against
        use_late_interaction: Use MaxSim (ColBERT-style) instead of cosine similarity
        model_config: Additional model configuration
        
    Returns:
        Dictionary containing:
        - similarities: Similarity matrix [num_queries, num_documents]
    """
    from mlx_raclate.utils.utils import load
    
    if "ColBERT" in model_path:
        use_late_interaction = True
    config = {"use_late_interaction": use_late_interaction}
    
    # Merge user config with defaults
    if model_config:
        config.update(model_config)
    
    # Load model and tokenizer
    model, tokenizer = load(
        model_path,
        model_config=config,
        pipeline="sentence-similarity"
    )
    
    max_length = getattr(model.config, "max_position_embeddings", 512)
    
    # Tokenize queries and documents
    query_tokens = tokenizer._tokenizer(
        queries,
        return_tensors="mlx",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    doc_tokens = tokenizer._tokenizer(
        documents,
        return_tensors="mlx",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    # Run inference
    outputs = model(
        input_ids=query_tokens["input_ids"],
        reference_input_ids=doc_tokens["input_ids"],
        attention_mask=query_tokens["attention_mask"],
        reference_attention_mask=doc_tokens["attention_mask"],
    )
    
    return {
        "similarities": outputs["similarities"]
    }


_DENSE_EXAMPLE_CODE_TEMPLATE = '''import mlx.core as mx
from mlx_raclate.utils.utils import load

# Load model and tokenizer
model, tokenizer = load(
    "{model_path}",
    pipeline="sentence-similarity"
)

# Prepare queries and documents
queries = {queries}
documents = {documents}

# Tokenize
max_length = getattr(model.config, "max_position_embeddings", 512)

query_tokens = tokenizer._tokenizer(
    queries,
    return_tensors="mlx",
    padding=True,
    truncation=True,
    max_length=max_length
)

doc_tokens = tokenizer._tokenizer(
    documents,
    return_tensors="mlx",
    padding=True,
    truncation=True,
    max_length=max_length
)

# Run inference - computes similarity matrix automatically
outputs = model(
    input_ids=query_tokens["input_ids"],
    reference_input_ids=doc_tokens["input_ids"],
    attention_mask=query_tokens["attention_mask"],
    reference_attention_mask=doc_tokens["attention_mask"],
)

# Get cosine similarity matrix
similarities = outputs["similarities"]  # Shape: [num_queries, num_documents]

# Print results
print("Cosine Similarity Matrix:")
for i, query in enumerate(queries):
    print(f"Query: {{query}}")
    for j, doc in enumerate(documents):
        print(f"  vs '{{doc[:50]}}...': {{similarities[i, j]:.4f}}")
'''

_LATE_INTERACTION_EXAMPLE_CODE_TEMPLATE = '''import mlx.core as mx
from mlx_raclate.utils.utils import load

# Load ColBERT-style model with late interaction enabled
model, tokenizer = load(
    "{model_path}",
    model_config={{"use_late_interaction": True}},
    pipeline="sentence-similarity"
)

# Prepare queries and documents
queries = {queries}
documents = {documents}

# Tokenize
max_length = getattr(model.config, "max_position_embeddings", 512)

query_tokens = tokenizer._tokenizer(
    queries,
    return_tensors="mlx",
    padding=True,
    truncation=True,
    max_length=max_length
)

doc_tokens = tokenizer._tokenizer(
    documents,
    return_tensors="mlx",
    padding=True,
    truncation=True,
    max_length=max_length
)

# Run inference with MaxSim scoring
outputs = model(
    input_ids=query_tokens["input_ids"],
    reference_input_ids=doc_tokens["input_ids"],
    attention_mask=query_tokens["attention_mask"],
    reference_attention_mask=doc_tokens["attention_mask"],
)

# Get MaxSim scores
similarities = outputs["similarities"]

# Print results
print("MaxSim Scores (Late Interaction):")
for i, query in enumerate(queries):
    print(f"Query: {{query}}")
    for j, doc in enumerate(documents):
        print(f"  vs '{{doc[:50]}}...': {{similarities[i, j]:.4f}}")
'''


def get_example_code(
    model_path: str = "{{MODEL_PATH}}",
    queries: Optional[List[str]] = None,
    documents: Optional[List[str]] = None,
    use_late_interaction: bool = False,
) -> str:
    
    if queries is None:
        queries = ["What is MLX?", "How does Apple Silicon work?"]
    
    if documents is None:
        documents = [
            "MLX is an array framework for machine learning on Apple Silicon.",
            "Apple Silicon uses ARM architecture with unified memory.",
            "Python is a popular programming language.",
        ]
    
    template = _LATE_INTERACTION_EXAMPLE_CODE_TEMPLATE if use_late_interaction else _DENSE_EXAMPLE_CODE_TEMPLATE
    
    return template.format(
        model_path=model_path,
        queries=repr(queries),
        documents=repr(documents),
    ).strip()


if __name__ == "__main__":
    # Print example code for model card
    print("Example code for dense similarity model card:")
    print("-" * 40)
    print(get_example_code("nomic-ai/modernbert-embed-base"))
    print()
    print("Example code for late interaction model card:")
    print("-" * 40)
    print(get_example_code("LiquidAI/LFM2-ColBERT-350M", use_late_interaction=True))
