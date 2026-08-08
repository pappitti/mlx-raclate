"""Collection options and filters shared by the mlx-raclate test suites."""

import pytest

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
