"""
Pytest Configuration and Shared Fixtures for mlx-raclate tests.

This module provides:
- Session-scoped fixtures for model caching
- Pytest command line options for controlling test execution
- Shared utilities for all test modules
"""

import pytest
from typing import Dict

from .model_registry import (
    BASE_MODELS,
    get_coverage_report,
)
from .test_fixtures import (
    generate_classification_dataset,
    generate_regression_dataset,
    generate_similarity_dataset,
    generate_triplet_dataset,
    generate_mlm_dataset,
    generate_ner_dataset,
    generate_id2label,
    generate_label2id,
    get_sample_texts,
    get_sample_documents,
)

# ============================================================================
# PYTEST OPTIONS
# ============================================================================

def pytest_addoption(parser):
    """Add custom command line options for test control."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests that require model downloads",
    )
    parser.addoption(
        "--model-family",
        action="store",
        default=None,
        help="Run tests only for a specific model family (e.g., 'modernbert')",
    )
    parser.addoption(
        "--pipeline",
        action="store",
        default=None,
        help="Run tests only for a specific pipeline (e.g., 'text-classification')",
    )


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: mark test as slow (requires model download)")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    
    # Print coverage report at start
    if config.getoption("verbose") >= 1:
        print("\n" + get_coverage_report())


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    # Handle --run-slow option
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    # Handle --model-family filter
    model_family_filter = config.getoption("--model-family")
    if model_family_filter:
        skip_family = pytest.mark.skip(reason=f"filtered to model_family={model_family_filter}")
        for item in items:
            if hasattr(item, "callspec") and "model_family" in item.callspec.params:
                if item.callspec.params["model_family"] != model_family_filter:
                    item.add_marker(skip_family)
    
    # Handle --pipeline filter
    pipeline_filter = config.getoption("--pipeline")
    if pipeline_filter:
        skip_pipeline = pytest.mark.skip(reason=f"filtered to pipeline={pipeline_filter}")
        for item in items:
            if hasattr(item, "callspec") and "pipeline" in item.callspec.params:
                if item.callspec.params["pipeline"] != pipeline_filter:
                    item.add_marker(skip_pipeline)


# ============================================================================
# DATASET FIXTURES
# ============================================================================

@pytest.fixture
def classification_dataset():
    """Dummy classification dataset for training tests."""
    return generate_classification_dataset(n_samples=20, n_classes=3)


@pytest.fixture
def regression_dataset():
    """Dummy regression dataset for training tests."""
    return generate_regression_dataset(n_samples=20)


@pytest.fixture
def similarity_dataset():
    """Dummy similarity dataset (scored pairs) for training tests."""
    return generate_similarity_dataset(n_samples=20)


@pytest.fixture
def triplet_dataset():
    """Dummy triplet dataset for training tests."""
    return generate_triplet_dataset(n_samples=20)


@pytest.fixture
def mlm_dataset():
    """Dummy MLM dataset for training tests."""
    return generate_mlm_dataset(n_samples=20)


@pytest.fixture
def ner_dataset():
    """Dummy NER dataset for training tests."""
    return generate_ner_dataset(n_samples=20, n_tags=5)


@pytest.fixture
def label_mappings():
    """Label mappings for classification."""
    return {
        "id2label": generate_id2label(3),
        "label2id": generate_label2id(3),
    }


# ============================================================================
# INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_texts():
    """Sample texts for inference testing."""
    return get_sample_texts(4)


@pytest.fixture
def sample_documents():
    """Sample documents for similarity testing."""
    return get_sample_documents(5)


# ============================================================================
# MODEL FIXTURES (Session-scoped for caching)
# ============================================================================

@pytest.fixture(scope="session")
def loaded_models_cache():
    """
    Session-scoped cache for loaded models.
    
    This prevents downloading the same model multiple times during a test session.
    Models are cached by (model_path, pipeline) tuple.
    """
    return {}


@pytest.fixture
def model_loader(loaded_models_cache):
    """
    Factory fixture for loading models with caching.
    
    Usage:
        model, tokenizer = model_loader("NousResearch/Minos-v1", "text-classification")
    """
    def _load(model_path: str, pipeline: str, model_config: Dict = None, train: bool = False):
        from mlx_raclate.utils.utils import load
        
        cache_key = (model_path, pipeline, frozenset((model_config or {}).items()), train)
        
        if cache_key not in loaded_models_cache:
            loaded_models_cache[cache_key] = load(
                model_path,
                model_config=model_config or {},
                pipeline=pipeline,
                train=train,
            )
        
        return loaded_models_cache[cache_key]
    
    return _load


# ============================================================================
# HELPER FIXTURES
# ============================================================================

@pytest.fixture
def base_model_for():
    """Get base model for a given model family."""
    def _get(model_family: str) -> str:
        if model_family not in BASE_MODELS:
            raise ValueError(f"Unknown model family: {model_family}")
        return BASE_MODELS[model_family]
    return _get
