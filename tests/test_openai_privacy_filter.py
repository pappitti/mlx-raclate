import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_raclate.models.openai_privacy_filter import (
    ModelArgs,
    ModelForSequenceClassification,
    ModelForTokenClassification,
    OpenAIPrivacyFilterModel,
)
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


def test_openai_privacy_filter_update_attention_mask_requires_dtype():
    model = OpenAIPrivacyFilterModel(_tiny_config())
    attention_mask = mx.array([[1, 1, 1, 0]], dtype=mx.int32)

    with pytest.raises(ValueError, match="dtype must be provided"):
        model._update_attention_mask(attention_mask, dtype=None)


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


def test_openai_privacy_filter_sequence_sanitize_targets_existing_parameters():
    model = ModelForSequenceClassification(_tiny_config())
    param_names = {name for name, _ in tree_flatten(model.parameters())}

    sanitized = model.sanitize(
        {
            "model.layers.0.input_layernorm.weight": mx.ones((32,), dtype=mx.float32),
            "score.weight": mx.zeros((5, 32), dtype=mx.float32),
            "score.bias": mx.zeros((5,), dtype=mx.float32),
            "classifier.weight": mx.zeros((5, 32), dtype=mx.float32),
            "classifier.bias": mx.zeros((5,), dtype=mx.float32),
        }
    )

    assert set(sanitized) <= param_names
    assert "score.weight" in sanitized
    assert "score.bias" not in sanitized


def test_openai_privacy_filter_sanitize_hf_layout_loads_strictly():
    model = ModelForTokenClassification(_tiny_config())
    hf_weights = {
        name: mx.zeros(value.shape, dtype=value.dtype)
        for name, value in tree_flatten(model.parameters())
    }
    hf_weights["model.rotary_emb.inv_freq"] = mx.zeros((4,), dtype=mx.float32)

    sanitized = model.sanitize(hf_weights)
    model.load_weights(list(sanitized.items()))


def test_openai_privacy_filter_sanitize_split_expert_layout_loads_strictly():
    config = _tiny_config()
    model = ModelForTokenClassification(config)
    hf_weights = {
        name: mx.zeros(value.shape, dtype=value.dtype)
        for name, value in tree_flatten(model.parameters())
    }

    expert_prefix = "model.layers.0.mlp.experts"
    for suffix in ("gate_up_proj", "gate_up_proj_bias", "down_proj", "down_proj_bias"):
        hf_weights.pop(f"{expert_prefix}.{suffix}")

    hf_weights[f"{expert_prefix}.gate_proj.weight"] = mx.zeros(
        (config.num_local_experts, config.hidden_size, config.intermediate_size),
        dtype=mx.float32,
    )
    hf_weights[f"{expert_prefix}.up_proj.weight"] = mx.zeros(
        (config.num_local_experts, config.hidden_size, config.intermediate_size),
        dtype=mx.float32,
    )
    hf_weights[f"{expert_prefix}.gate_proj.bias"] = mx.zeros(
        (config.num_local_experts, config.intermediate_size),
        dtype=mx.float32,
    )
    hf_weights[f"{expert_prefix}.up_proj.bias"] = mx.zeros(
        (config.num_local_experts, config.intermediate_size),
        dtype=mx.float32,
    )
    hf_weights[f"{expert_prefix}.down_proj.weight"] = mx.zeros(
        (config.num_local_experts, config.intermediate_size, config.hidden_size),
        dtype=mx.float32,
    )
    hf_weights[f"{expert_prefix}.down_proj.bias"] = mx.zeros(
        (config.num_local_experts, config.hidden_size),
        dtype=mx.float32,
    )

    sanitized = model.sanitize(hf_weights)
    assert f"{expert_prefix}.gate_up_proj" in sanitized
    assert f"{expert_prefix}.gate_up_proj_bias" in sanitized
    assert f"{expert_prefix}.down_proj" in sanitized
    assert f"{expert_prefix}.down_proj_bias" in sanitized
    model.load_weights(list(sanitized.items()))


def test_openai_privacy_filter_sanitize_block_layout_loads_strictly():
    config = _tiny_config()
    model = ModelForTokenClassification(config)
    hf_weights = {
        name: mx.zeros(value.shape, dtype=value.dtype)
        for name, value in tree_flatten(model.parameters())
    }

    layer_prefix = "model.layers.0"
    block_prefix = "block.0"
    for suffix in (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.o_proj.weight",
        "self_attn.o_proj.bias",
        "self_attn.sinks",
        "mlp.router.weight",
        "mlp.router.bias",
        "mlp.experts.gate_up_proj",
        "mlp.experts.gate_up_proj_bias",
        "mlp.experts.down_proj",
        "mlp.experts.down_proj_bias",
    ):
        hf_weights.pop(f"{layer_prefix}.{suffix}")
    hf_weights.pop("model.embed_tokens.weight")
    hf_weights.pop("model.norm.weight")
    hf_weights.pop("score.weight")
    hf_weights.pop("score.bias")

    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim
    hf_weights["embedding.weight"] = mx.zeros((config.vocab_size, config.hidden_size), dtype=mx.float32)
    hf_weights["norm.scale"] = mx.ones((config.hidden_size,), dtype=mx.float32)
    hf_weights["unembedding.weight"] = mx.zeros((config.num_labels, config.hidden_size), dtype=mx.float32)
    hf_weights["unembedding.bias"] = mx.zeros((config.num_labels,), dtype=mx.float32)
    hf_weights[f"{block_prefix}.attn.qkv.weight"] = mx.zeros(
        (q_dim + 2 * kv_dim, config.hidden_size),
        dtype=mx.float32,
    )
    hf_weights[f"{block_prefix}.attn.qkv.bias"] = mx.zeros((q_dim + 2 * kv_dim,), dtype=mx.float32)
    hf_weights[f"{block_prefix}.attn.out.weight"] = mx.zeros((config.hidden_size, q_dim), dtype=mx.float32)
    hf_weights[f"{block_prefix}.attn.out.bias"] = mx.zeros((config.hidden_size,), dtype=mx.float32)
    hf_weights[f"{block_prefix}.attn.sinks"] = mx.zeros((config.num_attention_heads,), dtype=mx.float32)
    hf_weights[f"{block_prefix}.attn.norm.scale"] = mx.ones((config.hidden_size,), dtype=mx.float32)
    hf_weights[f"{block_prefix}.mlp.norm.scale"] = mx.ones((config.hidden_size,), dtype=mx.float32)
    hf_weights[f"{block_prefix}.mlp.gate.weight"] = mx.zeros(
        (config.num_local_experts, config.hidden_size),
        dtype=mx.float32,
    )
    hf_weights[f"{block_prefix}.mlp.gate.bias"] = mx.zeros((config.num_local_experts,), dtype=mx.float32)
    hf_weights[f"{block_prefix}.mlp.swiglu.weight"] = mx.zeros(
        (config.num_local_experts, config.hidden_size, 2 * config.intermediate_size),
        dtype=mx.float32,
    )
    hf_weights[f"{block_prefix}.mlp.swiglu.bias"] = mx.zeros(
        (config.num_local_experts, 2 * config.intermediate_size),
        dtype=mx.float32,
    )
    hf_weights[f"{block_prefix}.mlp.out.weight"] = mx.zeros(
        (config.num_local_experts, config.intermediate_size, config.hidden_size),
        dtype=mx.float32,
    )
    hf_weights[f"{block_prefix}.mlp.out.bias"] = mx.zeros(
        (config.num_local_experts, config.hidden_size),
        dtype=mx.float32,
    )

    sanitized = model.sanitize(hf_weights)
    assert f"{layer_prefix}.self_attn.q_proj.weight" in sanitized
    assert f"{layer_prefix}.self_attn.k_proj.weight" in sanitized
    assert f"{layer_prefix}.self_attn.v_proj.weight" in sanitized
    assert "model.embed_tokens.weight" in sanitized
    assert "model.norm.weight" in sanitized
    assert "score.weight" in sanitized
    assert "score.bias" in sanitized
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
