# How to Run Tests
## Fast Tests (No Model Downloads)
```uv run pytest tests/ -v -m "not slow"
```
## All Tests (With Model Downloads)
```
uv run pytest tests/ -v --run-slow
```
## Filter by Model Family
```
uv run pytest tests/ -v --run-slow --model-family modernbert
```
## Filter by Pipeline
```
uv run pytest tests/ -v --run-slow --pipeline text-classification
```
## Training Only Tests, for certain models and pipelines
```
uv run pytest tests/test_training.py -v --run-slow -k "gemma3-token-classification or gemma3-masked-lm"
```
## Print Coverage Report
```
uv run python -c "from tests.model_registry import get_coverage_report; print(get_coverage_report())"
```