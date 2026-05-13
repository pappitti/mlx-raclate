import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_raclate.models.openai_privacy_filter import ModelArgs, ModelForTokenClassification
from mlx_raclate.utils.utils import _initialize_head_weights, _verify_weights


def _tiny_config() -> ModelArgs:
    return ModelArgs(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        bidirectional_left_context=2,
        bidirectional_right_context=1,
        num_labels=5,
        classifier_bias=True,
    )


def test_openai_privacy_filter_tiny_forward():
    model = ModelForTokenClassification(_tiny_config())

    input_ids = mx.array([[1, 2, 3, 4], [5, 6, 0, 0]], dtype=mx.int32)
    attention_mask = mx.array([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=mx.int32)
    labels = mx.array([[0, 1, 2, 3], [0, 1, -100, -100]], dtype=mx.int32)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    assert outputs["logits"].shape == (2, 4, 5)
    assert outputs["probabilities"].shape == (2, 4, 5)
    assert outputs["loss"] is not None


def test_openai_privacy_filter_sanitize_hf_attention_weights():
    config = _tiny_config()
    model = ModelForTokenClassification(config)

    q = mx.ones((config.num_attention_heads * config.head_dim, config.hidden_size), dtype=mx.float32)
    k = mx.full((config.num_key_value_heads * config.head_dim, config.hidden_size), 2.0, dtype=mx.float32)
    v = mx.full((config.num_key_value_heads * config.head_dim, config.hidden_size), 3.0, dtype=mx.float32)
    gate_up = mx.zeros((config.num_local_experts, config.hidden_size, 2 * config.intermediate_size), dtype=mx.float32)
    down = mx.zeros((config.num_local_experts, config.intermediate_size, config.hidden_size), dtype=mx.float32)

    sanitized = model.sanitize(
        {
            "model.layers.0.self_attn.q_proj.weight": q,
            "model.layers.0.self_attn.k_proj.weight": k,
            "model.layers.0.self_attn.v_proj.weight": v,
            "model.layers.0.input_layernorm.weight": mx.ones((config.hidden_size,), dtype=mx.float32),
            "model.layers.0.post_attention_layernorm.weight": mx.ones((config.hidden_size,), dtype=mx.float32),
            "model.layers.0.mlp.router.weight": mx.zeros((config.num_local_experts, config.hidden_size), dtype=mx.float32),
            "model.layers.0.mlp.experts.gate_up_proj": gate_up,
            "model.layers.0.mlp.experts.down_proj": down,
            "score.weight": mx.zeros((config.num_labels, config.hidden_size), dtype=mx.float32),
        }
    )

    assert "model.layers.0.self_attn.q_proj.weight" in sanitized
    assert "model.layers.0.self_attn.k_proj.weight" in sanitized
    assert "model.layers.0.self_attn.v_proj.weight" in sanitized
    assert float(sanitized["model.layers.0.self_attn.q_proj.weight"][0, 0].item()) == 1.0
    assert float(sanitized["model.layers.0.self_attn.k_proj.weight"][0, 0].item()) == 2.0
    assert float(sanitized["model.layers.0.self_attn.v_proj.weight"][0, 0].item()) == 3.0
    assert "model.layers.0.input_layernorm.weight" in sanitized
    assert "model.layers.0.post_attention_layernorm.weight" in sanitized
    assert "model.layers.0.mlp.router.weight" in sanitized
    assert "model.layers.0.mlp.experts.gate_up_proj" in sanitized
    assert "model.layers.0.mlp.experts.down_proj" in sanitized
    assert "score.weight" in sanitized
    assert "score.bias" in sanitized


def test_openai_privacy_filter_sanitize_targets_existing_parameters():
    model = ModelForTokenClassification(_tiny_config())
    param_names = {name for name, _ in tree_flatten(model.parameters())}

    sanitized = model.sanitize(
        {
            "model.layers.0.input_layernorm.weight": mx.ones((32,), dtype=mx.float32),
            "model.layers.0.mlp.router.bias": mx.zeros((4,), dtype=mx.float32),
            "score.weight": mx.zeros((5, 32), dtype=mx.float32),
            "score.bias": mx.zeros((5,), dtype=mx.float32),
            "classifier.weight": mx.zeros((5, 32), dtype=mx.float32),
        }
    )

    assert set(sanitized) <= param_names


def test_openai_privacy_filter_sanitize_hf_layout_loads_strictly():
    model = ModelForTokenClassification(_tiny_config())
    hf_weights = {
        name: mx.zeros(value.shape, dtype=value.dtype)
        for name, value in tree_flatten(model.parameters())
    }
    hf_weights["model.rotary_emb.inv_freq"] = mx.zeros((4,), dtype=mx.float32)

    sanitized = model.sanitize(hf_weights)
    model.load_weights(list(sanitized.items()))


def test_loader_helpers_treat_score_as_pipeline_head():
    model = ModelForTokenClassification(_tiny_config())
    weights = dict(tree_flatten(model.parameters()))
    weights.pop("score.weight")
    weights.pop("score.bias")

    with pytest.raises(ValueError, match="pipeline head"):
        _verify_weights(model, weights, train_mode=False)

    _verify_weights(model, weights, train_mode=True)
    _initialize_head_weights(model, weights, model.config, target_dtype=mx.float32)

    assert "score.weight" in weights
    assert "score.bias" in weights
