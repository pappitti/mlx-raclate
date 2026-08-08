# Test suites

The publish gate is isolated under `tests/release/`. Pytest can collect that
directory without importing architecture-specific regression modules or
discovering download-backed model tests. MLX-backed release checks require Apple
silicon with Metal available.

## Release gate

Install the development environment, including the optional training and serving
dependencies, then run:

```shell
uv run pytest tests/release -q
python3 scripts/check_release.py
```

This is the lean, no-download gate for a proposed release. It covers the public
CLI, model registry and loader contract, Hub path handling, tokenizer behavior,
generated model-card code, serving behavior, token-classification utilities, and
fast trainer/checkpoint contracts. The artifact check then builds the wheel and
source distribution and validates their metadata, contents, entry point, extras,
and clean-install behavior.

## Other suites

Focused model behavior, checkpoint compatibility, and architecture regressions:

```shell
uv run pytest tests/models -q
```

Other focused algorithm regressions, without repository recipes:

```shell
uv run pytest tests/test_*.py -m "not recipe" -q
```

Repository data-preparation recipes:

```shell
uv run pytest tools/tests/test_dataset_preparation.py -q
```

All tests that do not download checkpoints:

```shell
uv run pytest tests -m "not slow" -q
```

Download-backed model integration tests:

```shell
uv run pytest tests/integration --run-slow
```

Filter the downloaded inference/training matrix:

```shell
uv run pytest tests/integration --run-slow --model-family modernbert
uv run pytest tests/integration --run-slow --pipeline text-classification
```

Everything, including model downloads:

```shell
uv run pytest tests --run-slow
```

Verify only the isolated release collection:

```shell
uv run pytest tests/release --collect-only -q
```

Print the endpoint coverage report explicitly:

```shell
uv run python -c "from tests.model_registry import get_coverage_report; print(get_coverage_report())"
```

## File classification

| Test file | Primary classification | Purpose |
|---|---|---|
| `release/test_cli.py` | Essential release contract | Installed command routing and lazy help paths. |
| `release/test_hub.py` | Essential release contract | Local/Hub resolution and download error contract. |
| `release/test_model_card_utils.py` | Essential release contract | Executes generated MLX and Transformers snippets with fakes. |
| `release/test_model_dtype.py` | Essential release contract | Loader defaults, class routing, dtype, and quantized loading. |
| `release/test_registry.py` | Essential release contract | Import-light capability registry and supported class matrix. |
| `release/test_server_token_classification.py` | Essential release contract | Mocked HTTP/token-classification serving contract. |
| `release/test_token_classification_utils.py` | Essential release contract | Public decoding, metrics, Viterbi, and calibration behavior. |
| `release/test_tokenizer_utils.py` | Essential release contract | Wrapper behavior and Mistral regex compatibility. |
| `release/test_training.py` | Essential release contract | Fast dataset, collator, checkpoint, resume, and trainer argument behavior. |
| `integration/test_inference_models.py` | Slow integration | Published checkpoint inference matrix and endpoint catalog. |
| `integration/test_model_card_checkpoint.py` | Slow integration | Generated model-card code against a published checkpoint. |
| `integration/test_training_models.py` | Slow integration | Minimal training step across supported model/pipeline pairs. |
| `test_calibrate_ner_viterbi.py` | Useful focused unit/regression | Calibration search, cache, writer, and one-step CLI behavior. |
| `test_evaluate_ner_models.py` | Useful focused unit/regression | NER evaluation discovery and semantic label remapping. |
| `test_gradient_checkpointing.py` | Useful focused unit/regression | Shared checkpointing behavior and gradient equivalence. |
| `test_multi_precision_optimizer.py` | Useful focused unit/regression | FP32 master-state optimizer correctness. |
| `test_runtime_efficiency.py` | Useful focused unit/regression | Allocation-sensitive pooling and MLM collation regressions. |
| `models/test_model_behavior_contracts.py` | Useful focused unit/regression | Forward, pooling, projection, masking, head-loss, and sanitization contracts across embedding models. |
| `models/test_bidirectional_encoder_contracts.py` | Useful focused unit/regression | Bidirectional encoder routing, masks, heads, and checkpoint reload behavior. |
| `models/test_checkpoint_compatibility.py` | Useful focused unit/regression | Remote-code model forward behavior, checkpoint-layout sanitization, and pipeline-head loading. |

`fixtures.py`, `model_registry.py`, and `conftest.py` are support modules, not
test modules.

The recipe-owned `tools/tests/test_dataset_preparation.py` suite is a useful
focused regression for entity alignment and chunking, but is intentionally kept
outside the installed-package release suite.
