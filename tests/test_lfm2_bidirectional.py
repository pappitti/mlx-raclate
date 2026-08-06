import json
from dataclasses import asdict

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_raclate.models.lfm2_bidirectional import (
    Lfm2BidirectionalModel,
    Model,
    ModelArgs,
    ModelForMaskedLM,
    NonCausalShortConv,
)
from mlx_raclate.utils.utils import (
    _get_classes,
    _get_pipeline_from_config,
    _remap_lfm2_bidirectional_model_type,
    load_model,
)


def _tiny_config(**overrides):
    values = {
        "vocab_size": 32,
        "hidden_size": 4,
        "block_dim": 4,
        "block_ff_dim": 8,
        "block_multiple_of": 1,
        "block_auto_adjust_ff_dim": False,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "layer_types": ["conv", "full_attention"],
        "conv_L_cache": 3,
    }
    values.update(overrides)
    return ModelArgs(**values)


def test_lfm2_bidirectional_config_routing():
    config = {
        "model_type": "lfm2",
        "architectures": ["Lfm2BidirectionalForMaskedLM"],
        "auto_map": {
            "AutoModelForMaskedLM": (
                "modeling_lfm2_bidirectional.Lfm2BidirectionalForMaskedLM"
            )
        },
    }

    _remap_lfm2_bidirectional_model_type(config)

    assert config["model_type"] == "lfm2_bidirectional"
    assert _get_pipeline_from_config(config["architectures"][0]) == "masked-lm"
    model_class, args_class = _get_classes(config, pipeline="masked-lm")
    assert model_class is ModelForMaskedLM
    assert args_class is ModelArgs


def test_lfm2_bidirectional_base_architecture_infers_embeddings():
    config = {
        "model_type": "lfm2",
        "architectures": ["Lfm2BidirectionalModel"],
    }
    _remap_lfm2_bidirectional_model_type(config)

    assert config["model_type"] == "lfm2_bidirectional"
    assert _get_pipeline_from_config(config["architectures"][0]) == "embeddings"


def test_explicit_lfm2_model_type_override_disables_bidirectional_remap():
    config = {
        "model_type": "lfm2",
        "architectures": ["Lfm2BidirectionalModel"],
    }
    _remap_lfm2_bidirectional_model_type(config, explicit_model_type=True)

    assert config["model_type"] == "lfm2"


def test_noncausal_shortconv_uses_centered_context():
    conv = NonCausalShortConv(
        _tiny_config(
            hidden_size=1,
            block_dim=1,
            num_attention_heads=1,
            num_key_value_heads=1,
        ),
        layer_idx=0,
    )
    conv.in_proj.weight = mx.ones((3, 1))
    conv.conv.weight = mx.ones((1, 3, 1))
    conv.out_proj.weight = mx.ones((1, 1))

    output = conv(mx.array([[[1.0], [2.0], [4.0]]]))
    expected = mx.array([[[5.0], [42.0], [80.0]]])

    assert float(mx.max(mx.abs(output - expected)).item()) < 1e-6


def test_bidirectional_attention_mask_is_padding_only():
    model = Lfm2BidirectionalModel(_tiny_config())
    attention_mask = mx.array([[1, 1, 0]], dtype=mx.int32)

    updated_mask = model._update_attention_mask(attention_mask, dtype=mx.float32)

    assert updated_mask.shape == (1, 1, 1, 3)
    assert float(updated_mask[0, 0, 0, 0].item()) == 0.0
    assert float(updated_mask[0, 0, 0, 1].item()) == 0.0
    assert float(updated_mask[0, 0, 0, 2].item()) < -1e8


def test_lfm2_embedding_uses_normalized_cls_pooling():
    model = Model(_tiny_config())
    input_ids = mx.array([[1, 2, 0]], dtype=mx.int32)
    attention_mask = mx.array([[1, 1, 0]], dtype=mx.int32)

    outputs = model(input_ids, attention_mask=attention_mask)

    assert outputs["embeddings"].shape == (1, 4)
    norm = mx.sqrt(mx.sum(outputs["embeddings"] ** 2, axis=-1))
    assert float(mx.max(mx.abs(norm - 1.0)).item()) < 1e-5


def test_lfm2_bidirectional_masked_lm_forward_and_sanitize():
    config = _tiny_config(vocab_size=16)
    model = ModelForMaskedLM(config)
    input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    labels = mx.array([[-100, 2, -100]], dtype=mx.int32)

    outputs = model(input_ids, labels=labels)

    assert outputs["logits"].shape == (1, 3, 16)
    assert outputs["loss"] is not None

    sanitized = model.sanitize(
        {
            "lfm2.embed_tokens.weight": mx.zeros((16, 4)),
            "lfm2.layers.0.conv.conv.weight": mx.zeros((4, 1, 3)),
        }
    )
    assert "model.embed_tokens.weight" in sanitized
    assert "lm_head.weight" not in sanitized
    assert sanitized["model.layers.0.conv.conv.weight"].shape == (4, 3, 1)


def _to_hf_backbone_weights(model, prefix):
    weights = {}
    for key, value in tree_flatten(model.parameters()):
        if not key.startswith("model."):
            continue
        key = key.removeprefix("model.")
        if key.endswith("conv.conv.weight"):
            value = value.transpose(0, 2, 1)
        weights[f"{prefix}{key}"] = value
    return weights


def test_encoder_checkpoint_routes_and_loads_without_model_override(tmp_path):
    source = ModelForMaskedLM(_tiny_config(vocab_size=16))
    weights = _to_hf_backbone_weights(source, prefix="lfm2.")
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)

    config = asdict(source.config)
    config.update(
        {
            "model_type": "lfm2",
            "architectures": ["Lfm2BidirectionalForMaskedLM"],
            "auto_map": {
                "AutoModelForMaskedLM": (
                    "modeling_lfm2_bidirectional.Lfm2BidirectionalForMaskedLM"
                )
            },
            "dtype": "float32",
        }
    )
    (tmp_path / "config.json").write_text(json.dumps(config))

    loaded, loaded_config = load_model(tmp_path)

    assert isinstance(loaded, ModelForMaskedLM)
    assert loaded_config["model_type"] == "lfm2_bidirectional"


def test_embedding_sentence_transformer_routes_and_loads(tmp_path):
    source = Model(_tiny_config(vocab_size=16))
    weights = _to_hf_backbone_weights(source, prefix="")
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)

    config = asdict(source.config)
    config.update(
        {
            "model_type": "lfm2",
            "architectures": ["Lfm2BidirectionalModel"],
            "auto_map": {
                "AutoModel": "modeling_lfm2_bidirectional.Lfm2BidirectionalModel"
            },
            "dtype": "float32",
        }
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "config_sentence_transformers.json").write_text(
        json.dumps({"model_type": "SentenceTransformer"})
    )
    (tmp_path / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer",
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Pooling",
                    "type": "sentence_transformers.models.Pooling",
                },
            ]
        )
    )

    loaded, loaded_config = load_model(tmp_path)

    assert loaded.__class__.__name__ == "ModelForSentenceTransformers"
    assert loaded_config["model_type"] == "lfm2_bidirectional"
