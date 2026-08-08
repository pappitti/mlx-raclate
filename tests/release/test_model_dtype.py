import json
from dataclasses import dataclass
from inspect import signature
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_raclate.models.base import BaseModelArgs, RaclateBaseModel
from mlx_raclate.utils.utils import (
    _align_checkpoint_dtypes,
    _determine_model_dtype,
    _get_classes,
    _prepare_quantized_modules,
    _resolve_model_dtype,
    load,
    load_model,
)


pytestmark = pytest.mark.release


@dataclass
class _QuantizedArgs(BaseModelArgs):
    input_dims: int = 32
    output_dims: int = 4


class _QuantizedModel(RaclateBaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.input_dims, config.output_dims, bias=False)


def test_canonical_dtype_overrides_legacy_torch_dtype():
    config = {"dtype": "bfloat16", "torch_dtype": "float16"}

    assert _determine_model_dtype(config, {}) == mx.bfloat16


def test_invalid_dtype_is_rejected():
    with pytest.raises(ValueError, match="Unsupported model dtype"):
        _resolve_model_dtype({"dtype": "bfloat"}, {})


def test_auto_dtype_uses_dominant_checkpoint_dtype():
    weights = {
        "large": mx.ones((16,), dtype=mx.bfloat16),
        "small": mx.ones((1,), dtype=mx.float32),
    }

    policy = _resolve_model_dtype({"dtype": "auto"}, weights)

    assert policy.dtype == mx.bfloat16
    assert policy.name == "bfloat16"
    assert policy.config_name == "auto"
    assert policy.cast_checkpoint is False


def test_quantized_checkpoint_ignores_legacy_preconversion_dtype():
    config = {
        "torch_dtype": "bfloat16",
        "quantization": {"group_size": 32, "bits": 4},
    }
    weights = {"linear.scales": mx.ones((4, 1), dtype=mx.float16)}

    policy = _resolve_model_dtype(config, weights)

    assert policy.dtype == mx.float16
    assert policy.source == "checkpoint"
    assert policy.cast_checkpoint is False


def test_align_checkpoint_dtypes_uses_destination_parameter_dtype():
    model = nn.Linear(2, 2)
    model.set_dtype(mx.bfloat16)
    weights = {
        "weight": mx.ones((2, 2), dtype=mx.float32),
        "bias": mx.ones((2,), dtype=mx.float32),
        "metadata": mx.array([0, 1], dtype=mx.int32),
    }

    result = _align_checkpoint_dtypes(weights, model, cast_floating=True)
    mx.eval(result)

    assert result is weights
    assert result["weight"].dtype == mx.bfloat16
    assert result["bias"].dtype == mx.bfloat16
    assert result["metadata"].dtype == mx.int32


def test_quantized_modules_are_prepared_before_weight_loading():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 4, bias=False)

    source = TinyModel()
    nn.quantize(source, group_size=32, bits=4)
    weights = dict(tree_flatten(source.parameters()))
    destination = TinyModel()

    _prepare_quantized_modules(
        destination,
        weights,
        {"group_size": 32, "bits": 4},
    )
    destination.load_weights(list(weights.items()))

    assert isinstance(destination.linear, nn.QuantizedLinear)


def test_load_model_quantizes_structure_before_strict_loading(tmp_path):
    source = _QuantizedModel(_QuantizedArgs())
    nn.quantize(source, group_size=32, bits=4)
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        dict(tree_flatten(source.parameters())),
    )
    with open(tmp_path / "config.json", "w") as config_file:
        json.dump(
            {
                "model_type": "test",
                "dtype": "auto",
                "input_dims": 32,
                "output_dims": 4,
                "quantization": {"group_size": 32, "bits": 4},
            },
            config_file,
        )

    model, config = load_model(
        tmp_path,
        get_model_classes=lambda **_: (_QuantizedModel, _QuantizedArgs),
        pipeline="embeddings",
    )

    assert isinstance(model.linear, nn.QuantizedLinear)
    assert config["dtype"] == "float32"
    assert model.config.dtype == "float32"


def test_loader_configuration_defaults_are_not_mutable():
    assert signature(load_model).parameters["model_config"].default is None
    assert signature(load).parameters["model_config"].default is None
    assert signature(load).parameters["tokenizer_config"].default is None


def test_load_rejects_unsupported_adapter_path_before_model_resolution():
    with pytest.raises(NotImplementedError, match="Adapter loading"):
        load("unused", adapter_path="unused-adapter")


def test_loader_rejects_placeholder_head_before_importing_it():
    with pytest.raises(ValueError, match="not implemented"):
        _get_classes({"model_type": "colbert_zero"}, pipeline="masked-lm")


@pytest.mark.parametrize(
    ("sentence_transformer", "expected_pipeline"),
    [
        (False, "embeddings"),
        (True, "sentence-transformers"),
    ],
)
def test_loader_defaults_pipeline_before_checkpoint_io(
    tmp_path,
    sentence_transformer,
    expected_pipeline,
):
    with open(tmp_path / "config.json", "w") as config_file:
        json.dump({"model_type": "test"}, config_file)
    if sentence_transformer:
        with open(
            tmp_path / "config_sentence_transformers.json",
            "w",
        ) as sentence_transformer_file:
            json.dump({}, sentence_transformer_file)

    class PipelineResolved(Exception):
        pass

    def capture_pipeline(**kwargs):
        assert kwargs["pipeline"] == expected_pipeline
        raise PipelineResolved

    with pytest.raises(PipelineResolved):
        load_model(tmp_path, get_model_classes=capture_pipeline)


def test_resize_token_embeddings_preserves_model_dtype():
    class EmbeddingModel(RaclateBaseModel):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(4, 3)
            self.embedding.set_dtype(mx.bfloat16)
            self.config = SimpleNamespace(vocab_size=4)

        def get_input_embeddings(self):
            return self.embedding

        def set_input_embeddings(self, value):
            self.embedding = value

    model = EmbeddingModel()
    resized = model.resize_token_embeddings(6)
    mx.eval(resized.parameters())

    assert resized.weight.dtype == mx.bfloat16
    assert model.config.vocab_size == 6
