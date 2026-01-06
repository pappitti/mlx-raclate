from typing import Dict, List, Any, Optional
import random

# ============================================================================
# DUMMY DATASET GENERATORS
# ============================================================================

def generate_classification_dataset(
    n_samples: int = 20,
    n_classes: int = 3,
    text_prefix: str = "Sample text",
    seed: int = 42,
) -> Dict[str, List[Any]]:
    """
    Generate a dummy text classification dataset.
    
    Args:
        n_samples: Number of samples to generate
        n_classes: Number of classes
        text_prefix: Prefix for generated texts
        seed: Random seed for reproducibility
        
    Returns:
        Dict with "text" and "label" keys, each mapping to a list of values
        (HF dataset batch format)
    """
    random.seed(seed)
    return {
        "text": [f"{text_prefix} {i}: This is a sample for classification testing." for i in range(n_samples)],
        "label": [i % n_classes for i in range(n_samples)],
    }


def generate_regression_dataset(
    n_samples: int = 20,
    seed: int = 42,
) -> Dict[str, List[Any]]:
    """
    Generate a dummy regression dataset.
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        Dict with "text" and "label" keys, each mapping to a list of values
        (HF dataset batch format)
    """
    random.seed(seed)
    return {
        "text": [f"Sample {i}: This is a sample for regression testing." for i in range(n_samples)],
        "label": [random.uniform(0.0, 1.0) for i in range(n_samples)],
    }


def generate_similarity_dataset(
    n_samples: int = 20,
    seed: int = 42,
) -> Dict[str, List[Any]]:
    """
    Generate a dummy sentence similarity dataset (scored pairs).
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        Dict with "text", "text_pair", and "label" keys
        (HF dataset batch format)
    """
    random.seed(seed)
    pairs = [
        ("I like cats", "I love cats"),
        ("The weather is nice", "It's a beautiful day"),
        ("Python is great", "Python is a good language"),
        ("MLX runs on Apple Silicon", "Apple Silicon powers MLX"),
        ("Hello world", "Goodbye world"),
    ]
    
    return {
        "text": [pairs[i % len(pairs)][0] + f" (sample {i})" for i in range(n_samples)],
        "text_pair": [pairs[i % len(pairs)][1] + f" (sample {i})" for i in range(n_samples)],
        "label": [random.uniform(0.5, 1.0) if i % 2 == 0 else random.uniform(0.0, 0.5) for i in range(n_samples)],
    }


def generate_triplet_dataset(
    n_samples: int = 20,
    seed: int = 42,
) -> Dict[str, List[Any]]:
    """
    Generate a dummy triplet dataset (anchor, positive, negative).
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        Dict with "text", "text_pair", and "negative" keys
        (HF dataset batch format)
    """
    random.seed(seed)
    triplets = [
        ("What is MLX?", "MLX is an array framework", "The weather is nice today"),
        ("How does Python work?", "Python is an interpreted language", "I like pizza"),
        ("What is machine learning?", "ML is a subset of AI", "The cat sat on the mat"),
    ]
    
    return {
        "text": [triplets[i % len(triplets)][0] + f" ({i})" for i in range(n_samples)],
        "text_pair": [triplets[i % len(triplets)][1] for i in range(n_samples)],
        "negative": [triplets[i % len(triplets)][2] for i in range(n_samples)],
    }


def generate_mlm_dataset(
    n_samples: int = 20,
    seed: int = 42,
) -> Dict[str, List[Any]]:
    """
    Generate a dummy masked language modeling dataset.
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        Dict with "text" key (HF dataset batch format)
    """
    random.seed(seed)
    templates = [
        "The capital of France is Paris and it has many tourists.",
        "Machine learning is a subset of artificial intelligence.",
        "Apple Silicon provides excellent performance for ML workloads.",
        "Python is one of the most popular programming languages.",
        "Transformers have revolutionized natural language processing.",
    ]
    
    return {
        "text": [templates[i % len(templates)] + f" (Variation {i})" for i in range(n_samples)],
    }


def generate_ner_dataset(
    n_samples: int = 20,
    n_tags: int = 5,
    seed: int = 42,
) -> Dict[str, List[Any]]:
    """
    Generate a dummy NER/token classification dataset.
    
    Args:
        n_samples: Number of samples to generate
        n_tags: Number of unique tags (e.g., B-PER, I-PER, O, etc.)
        seed: Random seed for reproducibility
        
    Returns:
        Dict with "text" (list of token lists) and "labels" (list of label lists)
        Note: 'text' contains pre-tokenized inputs as expected by DataCollatorForTokenClassification
        (HF dataset batch format)
    """
    random.seed(seed)
    sample_sentences = [
        ["John", "works", "at", "Apple", "in", "California", "."],
        ["The", "Eiffel", "Tower", "is", "in", "Paris", "."],
        ["Microsoft", "was", "founded", "by", "Bill", "Gates", "."],
    ]
    
    # Simple tag mapping: O=0, B-PER=1, I-PER=2, B-ORG=3, B-LOC=4
    tag_patterns = [
        [1, 0, 0, 3, 0, 4, 0],  # John works at Apple in California .
        [0, 4, 4, 0, 0, 4, 0],  # The Eiffel Tower is in Paris .
        [3, 0, 0, 0, 1, 2, 0],  # Microsoft was founded by Bill Gates .
    ]
    
    return {
        "text": [sample_sentences[i % len(sample_sentences)] for i in range(n_samples)],
        "labels": [[t % n_tags for t in tag_patterns[i % len(tag_patterns)]] for i in range(n_samples)],
    }


# ============================================================================
# ID2LABEL GENERATORS
# ============================================================================

def generate_id2label(n_classes: int = 3) -> Dict[str, str]:
    """Generate id2label mapping for classification."""
    label_names = ["negative", "neutral", "positive", "very_positive", "very_negative"]
    return {str(i): label_names[i % len(label_names)] for i in range(n_classes)}


def generate_label2id(n_classes: int = 3) -> Dict[str, int]:
    """Generate label2id mapping for classification."""
    id2label = generate_id2label(n_classes)
    return {v: int(k) for k, v in id2label.items()}


def generate_ner_id2label(n_tags: int = 5) -> Dict[str, str]:
    """Generate id2label mapping for NER."""
    tags = ["O", "B-PER", "I-PER", "B-ORG", "B-LOC", "I-ORG", "I-LOC"]
    return {str(i): tags[i % len(tags)] for i in range(n_tags)}


# ============================================================================
# TEST INPUT GENERATORS
# ============================================================================

def get_sample_texts(n: int = 4) -> List[str]:
    """Get sample texts for inference testing."""
    texts = [
        "What is MLX?",
        "How does Apple Silicon work?",
        "I really enjoyed this movie!",
        "The weather is terrible today.",
        "Machine learning is fascinating.",
        "Python is a great programming language.",
    ]
    return texts[:n]


def get_sample_documents(n: int = 5) -> List[str]:
    """Get sample documents for similarity testing."""
    docs = [
        "MLX is an array framework for machine learning on Apple Silicon.",
        "Apple Silicon uses ARM architecture with unified memory.",
        "Python is a popular programming language for data science.",
        "The movie was absolutely fantastic and I loved every minute.",
        "Today's weather forecast predicts rain and thunderstorms.",
    ]
    return docs[:n]
