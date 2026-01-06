"""
Model Registry for mlx-raclate tests.

This module provides a central registry of HuggingFace endpoints for testing each
(model_family, pipeline) combination. The registry is used by both inference tests
and training tests.

To add custom test endpoints, use the `register_endpoint` function or pass
additional endpoints via pytest fixtures.
"""

from typing import Dict, Tuple, Optional, List, Set
from dataclasses import dataclass, field
from mlx_raclate.utils.utils import PIPELINES

# Model families supported by the library
MODEL_FAMILIES = [
    "modernbert",
    "qwen3", 
    "gemma3",
    "t5gemma",
    "lfm2",
]

# Pipelines that support training
TRAINABLE_PIPELINES = [
    "text-classification",
    "sentence-similarity",
    "masked-lm",
    "token-classification",
]

# Base models for each family (used for training tests when no pretrained head exists)
BASE_MODELS: Dict[str, str] = {
    "modernbert": "answerdotai/ModernBERT-base",
    "qwen3": "Qwen/Qwen3-Embedding-0.6B",
    "gemma3": "google/embeddinggemma-300m",
    "t5gemma": "google/t5gemma-s-s-ul2",
    "lfm2": "LiquidAI/LFM2-350M",
}

@dataclass
class EndpointConfig:
    """Configuration for a model endpoint."""
    endpoint: str
    model_family: str
    pipeline: str
    model_config: Dict = field(default_factory=dict)
    notes: str = ""

# Curated registry of tested HuggingFace endpoints
# Extracted from examples/ and README
_ENDPOINT_REGISTRY: Dict[Tuple[str, str], EndpointConfig] = {}

def _init_registry():
    """Initialize the registry with curated endpoints from examples."""
    
    # ModernBERT endpoints
    _register("modernbert", "embeddings", "answerdotai/ModernBERT-base")
    _register("modernbert", "text-classification", "NousResearch/Minos-v1")
    _register("modernbert", "sentence-similarity", "nomic-ai/modernbert-embed-base")
    _register("modernbert", "sentence-transformers", "tasksource/ModernBERT-base-embed")
    _register("modernbert", "masked-lm", "answerdotai/ModernBERT-base")
    _register("modernbert", "zero-shot-classification", "answerdotai/ModernBERT-Large-Instruct")
    
    # Qwen3 endpoints
    _register("qwen3", "embeddings", "Qwen/Qwen3-Embedding-0.6B")
    _register("qwen3", "text-classification", "TerenceLau/galahad-classifier-0.6B",
              notes="Used for testing, model quality varies")
    _register("qwen3", "sentence-similarity", "Qwen/Qwen3-Embedding-0.6B")
    
    # Gemma3 endpoints (sensitive to input formatting)
    _register("gemma3", "embeddings", "google/embeddinggemma-300m",
              notes="Extremely sensitive to input formatting")
    _register("gemma3", "sentence-similarity", "google/embeddinggemma-300m",
              notes="Requires specific input format: 'task: search result | query: ...'")
    
    # T5Gemma endpoints (encoder-only)
    _register("t5gemma", "embeddings", "google/t5gemma-s-s-ul2")
    _register("t5gemma", "sentence-similarity", "google/t5gemma-b-b-ul2")
    
    # LFM2 endpoints
    _register("lfm2", "embeddings", "LiquidAI/LFM2-350M")
    _register("lfm2", "sentence-similarity", "LiquidAI/LFM2-ColBERT-350M",
              model_config={"use_late_interaction": True},
              notes="Late interaction (ColBERT-style) model")


def _register(
    model_family: str, 
    pipeline: str, 
    endpoint: str,
    model_config: Dict = None,
    notes: str = ""
):
    """Internal helper to register an endpoint."""
    _ENDPOINT_REGISTRY[(model_family, pipeline)] = EndpointConfig(
        endpoint=endpoint,
        model_family=model_family,
        pipeline=pipeline,
        model_config=model_config or {},
        notes=notes,
    )


def register_endpoint(
    model_family: str,
    pipeline: str,
    endpoint: str,
    model_config: Dict = None,
    notes: str = ""
):
    """
    Register a custom endpoint for testing.
    
    Use this to add endpoints not in the curated list.
    
    Args:
        model_family: One of MODEL_FAMILIES
        pipeline: One of PIPELINES  
        endpoint: HuggingFace model ID or local path
        model_config: Additional model configuration
        notes: Optional notes about the endpoint
    """
    if model_family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family: {model_family}. Must be one of {MODEL_FAMILIES}")
    if pipeline not in PIPELINES:
        raise ValueError(f"Unknown pipeline: {pipeline}. Must be one of {PIPELINES}")
    
    _register(model_family, pipeline, endpoint, model_config, notes)


def get_endpoint(model_family: str, pipeline: str) -> Optional[EndpointConfig]:
    """
    Get the endpoint configuration for a (model_family, pipeline) combination.
    
    Returns None if no endpoint is registered for this combination.
    """
    return _ENDPOINT_REGISTRY.get((model_family, pipeline))


def get_all_endpoints() -> Dict[Tuple[str, str], EndpointConfig]:
    """Get all registered endpoints."""
    return _ENDPOINT_REGISTRY.copy()


def get_inference_test_cases() -> List[Tuple[str, str, EndpointConfig]]:
    """
    Get all test cases for inference testing.
    
    Returns list of (model_family, pipeline, endpoint_config) tuples.
    """
    return [
        (model_family, pipeline, config)
        for (model_family, pipeline), config in _ENDPOINT_REGISTRY.items()
    ]


def get_untested_combinations() -> List[Tuple[str, str]]:
    """
    Get all (model_family, pipeline) combinations that don't have a test endpoint.
    
    This is useful for flagging coverage gaps.
    """
    untested = []
    for model_family in MODEL_FAMILIES:
        for pipeline in PIPELINES:
            if (model_family, pipeline) not in _ENDPOINT_REGISTRY:
                untested.append((model_family, pipeline))
    return untested


def get_coverage_report() -> str:
    """
    Generate a coverage report showing which combinations are tested.
    
    Returns a formatted string suitable for printing or logging.
    """
    lines = ["=" * 60, "Model Registry Coverage Report", "=" * 60, ""]
    
    # Create a matrix
    header = "Pipeline".ljust(25) + "".join(fam[:8].center(10) for fam in MODEL_FAMILIES)
    lines.append(header)
    lines.append("-" * len(header))
    
    for pipeline in PIPELINES:
        row = pipeline.ljust(25)
        for fam in MODEL_FAMILIES:
            if (fam, pipeline) in _ENDPOINT_REGISTRY:
                row += "✓".center(10)
            else:
                row += "✗".center(10)
        lines.append(row)
    
    lines.append("")
    lines.append("=" * 60)
    
    untested = get_untested_combinations()
    if untested:
        lines.append(f"⚠️  {len(untested)} untested combinations:")
        for fam, pipe in untested:
            lines.append(f"   - ({fam}, {pipe})")
    else:
        lines.append("✓ All combinations have test endpoints")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


# Initialize the registry on module import
_init_registry()


if __name__ == "__main__":
    # Print coverage report when run directly
    print(get_coverage_report())
