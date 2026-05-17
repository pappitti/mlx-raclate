import inspect

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_raclate.models.bidirectional_pplx_qwen3 import (
    ModelArgs as PPLXArgs,
    Model as PPLXModel,
    ModelForMaskedLM as PPLXForMaskedLM,
    ModelForTokenClassification as PPLXForTokenClassification,
)
from mlx_raclate.models.colbert_zero import (
    ModelArgs as ColBERTArgs,
    ModelForSentenceSimilarity as ColBERTModel,
)
from mlx_raclate.models.granite_embedding import (
    ModelArgs as GraniteArgs,
    Model as GraniteModel,
)
# from dev_tests.jina_embeddings_v5 import (
#     ModelArgs as JinaArgs,
#     Model as JinaModel,
# )
from mlx_raclate.models.lateon import (
    ModelArgs as LateOnArgs,
    ModelForSentenceSimilarity as LateOnModel,
    PyLateDense,
)
from mlx_raclate.models.modernbert import (
    ModelArgs as ModernBertArgs,
    ModernBertMLP,
    ModelForMaskedLM as ModernBertForMaskedLM,
    ModelForTokenClassification as ModernBertForTokenClassification,
)
from mlx_raclate.models.gemma3_text import (
    ModelArgs as Gemma3Args,
    ModelForMaskedLM as Gemma3ForMaskedLM,
    ModelForTokenClassification as Gemma3ForTokenClassification,
)
from mlx_raclate.models.t5gemma_encoder import (
    ModelArgs as T5GemmaArgs,
    ModelForMaskedLM as T5GemmaForMaskedLM,
    ModelForTokenClassification as T5GemmaForTokenClassification,
)
from mlx_raclate.models.neobert import (
    ModelArgs as NeoBERTArgs,
    Model as NeoBERTModel,
    NeoBERTModel as NeoBERTBackbone,
    ModelForMaskedLM as NeoBERTForMaskedLM,
    ModelForSentenceSimilarity as NeoBERTForSentenceSimilarity,
)
from mlx_raclate.utils.utils import _is_lateon_dense_layout


def test_modernbert_mlp_uses_configured_silu_activation():
    config = ModernBertArgs(
        hidden_size=4,
        intermediate_size=3,
        hidden_activation="silu",
    )
    mlp = ModernBertMLP(config)
    hidden_states = mx.array(
        [[[0.5, -1.0, 2.0, -0.25], [1.5, 0.25, -0.5, 1.0]]],
        dtype=mx.float32,
    )

    x = mlp.Wi(hidden_states)
    input, gate = (
        x[:, :, : config.intermediate_size],
        x[:, :, config.intermediate_size :],
    )
    expected = mlp.Wo(nn.silu(input) * gate)
    actual = mlp(hidden_states)

    assert float(mx.max(mx.abs(actual - expected)).item()) < 1e-6


def _assert_sparse_masked_lm_loss_handles_ignore_index(model):
    model.train()
    input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    attention_mask = mx.array([[1, 1, 1]], dtype=mx.int32)

    labels = mx.array([[-100, 2, -100]], dtype=mx.int32)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    assert outputs["loss"] is not None
    assert float(outputs["loss"].item()) > 0.0

    ignored_labels = mx.array([[-100, -100, -100]], dtype=mx.int32)
    ignored_outputs = model(input_ids, attention_mask=attention_mask, labels=ignored_labels)
    assert float(ignored_outputs["loss"].item()) == 0.0


def _assert_token_classification_loss_handles_ignore_index(model):
    input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    attention_mask = mx.array([[1, 1, 1]], dtype=mx.int32)

    labels = mx.array([[0, -100, 1]], dtype=mx.int32)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    assert outputs["loss"] is not None
    assert float(outputs["loss"].item()) > 0.0

    ignored_labels = mx.array([[-100, -100, -100]], dtype=mx.int32)
    ignored_outputs = model(input_ids, attention_mask=attention_mask, labels=ignored_labels)
    assert float(ignored_outputs["loss"].item()) == 0.0


def test_sparse_masked_lm_loss_handles_ignore_index_across_models():
    configs = [
        (
            ModernBertForMaskedLM,
            ModernBertArgs(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=24,
                num_hidden_layers=1,
                num_attention_heads=4,
                global_attn_every_n_layers=1,
                local_attention=4,
            ),
        ),
        (
            Gemma3ForMaskedLM,
            Gemma3Args(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=4,
                layer_types=["full_attention"],
                use_bidirectional_attention=True,
            ),
        ),
        (
            T5GemmaForMaskedLM,
            T5GemmaArgs(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=4,
                layer_types=["full_attention"],
            ),
        ),
        (
            PPLXForMaskedLM,
            PPLXArgs(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=4,
            ),
        ),
    ]

    for model_cls, config in configs:
        _assert_sparse_masked_lm_loss_handles_ignore_index(model_cls(config))


def test_token_classification_loss_handles_ignore_index_across_models():
    id2label = {0: "O", 1: "B-X", 2: "I-X"}
    configs = [
        (
            ModernBertForTokenClassification,
            ModernBertArgs(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=24,
                num_hidden_layers=1,
                num_attention_heads=4,
                global_attn_every_n_layers=1,
                local_attention=4,
                id2label=id2label,
            ),
        ),
        (
            Gemma3ForTokenClassification,
            Gemma3Args(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=4,
                layer_types=["full_attention"],
                use_bidirectional_attention=True,
                id2label=id2label,
            ),
        ),
        (
            T5GemmaForTokenClassification,
            T5GemmaArgs(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=4,
                layer_types=["full_attention"],
                id2label=id2label,
            ),
        ),
        (
            PPLXForTokenClassification,
            PPLXArgs(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=4,
                id2label=id2label,
            ),
        ),
    ]

    for model_cls, config in configs:
        _assert_token_classification_loss_handles_ignore_index(model_cls(config))


def test_granite_tiny_forward_and_sanitize():
    config = GraniteArgs(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        global_attn_every_n_layers=1,
        local_attention=4,
    )
    model = GraniteModel(config)

    outputs = model(
        mx.array([[1, 2, 3, 0]], dtype=mx.int32),
        attention_mask=mx.array([[1, 1, 1, 0]], dtype=mx.int32),
    )

    assert outputs["embeddings"].shape == (1, 16)

    sanitized = model.sanitize(
        {
            "embeddings.tok_embeddings.weight": mx.zeros((64, 16)),
            "layers.0.mlp.Wi.weight": mx.zeros((64, 16)),
            "final_norm.weight": mx.ones((16,)),
        }
    )
    assert "model.embeddings.tok_embeddings.weight" in sanitized
    assert "model.layers.0.mlp.Wi.weight" in sanitized
    assert "model.final_norm.weight" in sanitized


def test_colbert_tiny_forward_projection_and_sanitize():
    config = ColBERTArgs(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=4,
        global_attn_every_n_layers=1,
        local_attention=4,
        colbert_dim=8,
        use_late_interaction=True,
    )
    model = ColBERTModel(config)

    outputs = model(
        input_ids=mx.array([[1, 2, 3, 0]], dtype=mx.int32),
        reference_input_ids=mx.array([[1, 4, 5, 0]], dtype=mx.int32),
        attention_mask=mx.array([[1, 1, 1, 0]], dtype=mx.int32),
        reference_attention_mask=mx.array([[1, 1, 1, 0]], dtype=mx.int32),
    )

    assert outputs["embeddings"].shape == (1, 4, 8)
    assert outputs["similarities"].shape == (1, 1)

    sanitized = model.sanitize(
        {
            "embeddings.tok_embeddings.weight": mx.zeros((64, 16)),
            "1_Dense.linear.weight": mx.zeros((8, 16)),
        }
    )
    assert "model.embeddings.tok_embeddings.weight" in sanitized
    assert "dense.0.weight" in sanitized


def test_lateon_tiny_forward_projection_and_sanitize():
    dense_layers = [
        {
            "in_features": 16,
            "out_features": 32,
            "bias": False,
            "activation_function": "identity",
            "use_residual": True,
        },
        {
            "in_features": 32,
            "out_features": 16,
            "bias": False,
            "activation_function": "identity",
            "use_residual": True,
        },
        {
            "in_features": 16,
            "out_features": 8,
            "bias": False,
            "activation_function": "identity",
            "use_residual": False,
        },
    ]
    config = LateOnArgs(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=4,
        global_attn_every_n_layers=1,
        local_attention=4,
        colbert_dim=8,
        use_late_interaction=True,
        pylate_dense_layers=dense_layers,
    )
    model = LateOnModel(config)

    outputs = model(
        input_ids=mx.array([[1, 2, 3, 0]], dtype=mx.int32),
        reference_input_ids=mx.array([[1, 4, 5, 0]], dtype=mx.int32),
        attention_mask=mx.array([[1, 1, 1, 0]], dtype=mx.int32),
        reference_attention_mask=mx.array([[1, 1, 1, 0]], dtype=mx.int32),
    )

    assert outputs["embeddings"].shape == (1, 4, 8)
    assert outputs["similarities"].shape == (1, 1)

    sanitized = model.sanitize(
        {
            "embeddings.tok_embeddings.weight": mx.zeros((64, 16)),
            "1_Dense.linear.weight": mx.zeros((32, 16)),
            "1_Dense.residual.weight": mx.zeros((32, 16)),
            "2_Dense.linear.weight": mx.zeros((16, 32)),
            "2_Dense.residual.weight": mx.zeros((16, 32)),
            "3_Dense.linear.weight": mx.zeros((8, 16)),
        }
    )
    assert "model.embeddings.tok_embeddings.weight" in sanitized
    assert "dense.0.linear.weight" in sanitized
    assert "dense.0.residual.weight" in sanitized
    assert "dense.1.linear.weight" in sanitized
    assert "dense.1.residual.weight" in sanitized
    assert "dense.2.linear.weight" in sanitized


def test_lateon_dense_residual_without_projection():
    layer = PyLateDense(
        in_features=4,
        out_features=4,
        bias=False,
        activation_function="identity",
        use_residual=True,
    )
    layer.linear.weight = mx.zeros((4, 4))
    token_embeddings = mx.ones((1, 2, 4))

    outputs = layer(token_embeddings)

    assert float(mx.max(mx.abs(outputs - token_embeddings)).item()) < 1e-6


def test_lateon_dense_accepts_hf_activation_name():
    layer = PyLateDense(
        in_features=4,
        out_features=4,
        activation_function="torch.nn.modules.linear.Identity",
    )

    assert layer.activation_name == "identity"


def test_lateon_dense_layout_detection():
    assert _is_lateon_dense_layout(
        [
            {"in_features": 768, "out_features": 1536, "use_residual": True},
            {"in_features": 1536, "out_features": 768, "use_residual": True},
            {"in_features": 768, "out_features": 128, "use_residual": False},
        ]
    )
    assert not _is_lateon_dense_layout(
        [{"in_features": 768, "out_features": 128, "use_residual": False}]
    )


def test_pplx_uses_padding_only_attention_and_mean_pooling():
    config = PPLXArgs(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    model = PPLXModel(config)
    attention_mask = mx.array([[1, 1, 0]], dtype=mx.int32)

    updated_mask = model.model._update_attention_mask(attention_mask, dtype=mx.float32)
    assert updated_mask.shape == (1, 1, 1, 3)
    assert float(updated_mask[0, 0, 0, 0].item()) == 0.0
    assert float(updated_mask[0, 0, 0, 1].item()) == 0.0
    assert float(updated_mask[0, 0, 0, 2].item()) < -1e8

    outputs = model(
        mx.array([[1, 2, 0]], dtype=mx.int32),
        attention_mask=attention_mask,
    )
    assert outputs["embeddings"].shape == (1, 16)


# def test_jina_tiny_forward_truncates_last_token_embedding():
#     config = JinaArgs(
#         vocab_size=64,
#         hidden_size=16,
#         intermediate_size=32,
#         num_hidden_layers=1,
#         num_attention_heads=4,
#         num_key_value_heads=2,
#         head_dim=4,
#         truncate_dim=8,
#     )
#     model = JinaModel(config)

#     outputs = model(
#         mx.array([[1, 2, 0]], dtype=mx.int32),
#         attention_mask=mx.array([[1, 1, 0]], dtype=mx.int32),
#         task="retrieval",
#         prompt_name="document",
#     )

#     assert outputs["embeddings"].shape == (1, 8)


# def test_jina_task_is_selected_at_load_time():
#     config = JinaArgs(
#         vocab_size=64,
#         hidden_size=16,
#         intermediate_size=32,
#         num_hidden_layers=1,
#         num_attention_heads=4,
#         num_key_value_heads=2,
#         head_dim=4,
#         task="retrieval",
#     )
#     model = JinaModel(config)

#     with pytest.raises(ValueError, match="merged into the model at load time"):
#         model(
#             mx.array([[1, 2, 0]], dtype=mx.int32),
#             attention_mask=mx.array([[1, 1, 0]], dtype=mx.int32),
#             task="classification",
#         )


# def test_jina_sanitize_merges_selected_lora_adapter():
#     config = JinaArgs(
#         vocab_size=64,
#         hidden_size=16,
#         intermediate_size=32,
#         num_hidden_layers=1,
#         num_attention_heads=4,
#         num_key_value_heads=2,
#         head_dim=4,
#         adapter_config={"peft_type": "LORA", "r": 2, "lora_alpha": 4},
#     )
#     model = JinaModel(config)

#     sanitized = model.sanitize(
#         {
#             "layers.0.self_attn.q_proj.weight": mx.zeros((16, 16)),
#             "adapter.base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": mx.ones((2, 16)),
#             "adapter.base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": mx.ones((16, 2)),
#         }
#     )

#     merged = sanitized["model.layers.0.self_attn.q_proj.weight"]
#     assert merged.shape == (16, 16)
#     assert float(merged[0, 0].item()) == 4.0


# def test_load_adapter_weights_sets_config_and_prefixes_keys(tmp_path):
#     adapter_dir = tmp_path / "retrieval"
#     adapter_dir.mkdir()
#     (adapter_dir / "adapter_config.json").write_text(
#         json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 4})
#     )
#     mx.save_safetensors(
#         str(adapter_dir / "adapter_model.safetensors"),
#         {"base_model.model.layers.0.self_attn.q_proj.lora_A.weight": mx.ones((2, 16))},
#     )

#     weights = {}
#     config = {"task_names": ["retrieval", "classification"]}
#     _load_adapter_weights(weights, config, adapter_dir, required=True)

#     assert config["adapter_config"]["peft_type"] == "LORA"
#     assert config["task"] == "retrieval"
#     assert "adapter.base_model.model.layers.0.self_attn.q_proj.lora_A.weight" in weights


# def test_load_adapter_weights_rejects_task_mismatch(tmp_path):
#     adapter_dir = tmp_path / "classification"
#     adapter_dir.mkdir()
#     (adapter_dir / "adapter_config.json").write_text(
#         json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 4})
#     )
#     mx.save_safetensors(
#         str(adapter_dir / "adapter_model.safetensors"),
#         {"base_model.model.layers.0.self_attn.q_proj.lora_A.weight": mx.ones((2, 16))},
#     )

#     with pytest.raises(ValueError, match="model_config requested task"):
#         _load_adapter_weights(
#             {},
#             {"task_names": ["retrieval", "classification"], "task": "retrieval"},
#             adapter_dir,
#             required=True,
#         )


def test_neobert_tiny_forward_and_sanitize():
    config = NeoBERTArgs(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_length=8,
    )
    model = NeoBERTModel(config)

    outputs = model(
        mx.array([[1, 2, 3, 0]], dtype=mx.int32),
        attention_mask=mx.array([[1, 1, 1, 0]], dtype=mx.int32),
    )

    assert outputs["embeddings"].shape == (1, 16)
    assert outputs["last_hidden_state"].shape == (1, 4, 16)

    sanitized = model.sanitize(
        {
            "encoder.weight": mx.zeros((64, 16)),
            "transformer_encoder.0.qkv.weight": mx.zeros((48, 16)),
            "layer_norm.weight": mx.ones((16,)),
            "decoder.weight": mx.zeros((64, 16)),
            "freqs_cis": mx.zeros((8, 8)),
        }
    )
    assert "model.encoder.weight" in sanitized
    assert "model.transformer_encoder.0.qkv.weight" in sanitized
    assert "model.layer_norm.weight" in sanitized
    assert "decoder.weight" not in sanitized
    assert "freqs_cis" not in sanitized


def test_neobert_masked_lm_and_sentence_similarity_heads():
    config = NeoBERTArgs(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_length=8,
    )
    input_ids = mx.array([[1, 2, 3, 0]], dtype=mx.int32)
    attention_mask = mx.array([[1, 1, 1, 0]], dtype=mx.int32)

    mlm = NeoBERTForMaskedLM(config)
    mlm_outputs = mlm(input_ids, attention_mask=attention_mask)
    assert mlm_outputs["logits"].shape == (1, 4, 64)

    logits = mx.array(
        [
            [
                [1.0, 0.0, -1.0, 0.5],
                [0.1, -0.2, 0.3, 0.4],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=mx.float32,
    )
    labels = mx.array([[-100, 2, -100]], dtype=mx.int32)
    expected = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1])[1:2],
        mx.array([2], dtype=mx.int32),
        reduction="mean",
    )
    actual = mlm._compute_loss(logits, labels)
    assert abs(float((actual - expected).item())) < 1e-6

    ignored_labels = mx.array([[-100, -100, -100]], dtype=mx.int32)
    ignored_loss = mlm._compute_loss(logits, ignored_labels)
    assert float(ignored_loss.item()) == 0.0

    sanitized = mlm.sanitize(
        {
            "model.encoder.weight": mx.zeros((64, 16)),
            "decoder.weight": mx.zeros((64, 16)),
            "decoder.bias": mx.zeros((64,)),
        }
    )
    assert "model.encoder.weight" in sanitized
    assert "decoder.weight" in sanitized
    assert "decoder.bias" in sanitized

    similarity = NeoBERTForSentenceSimilarity(config)
    sim_outputs = similarity(
        input_ids=input_ids,
        reference_input_ids=input_ids,
        attention_mask=attention_mask,
        reference_attention_mask=attention_mask,
    )
    assert sim_outputs["embeddings"].shape == (1, 16)
    assert sim_outputs["similarities"].shape == (1, 1)


def test_neobert_backbone_uses_input_ids_for_position_ids_and_not_inputs_embeds():
    config = NeoBERTArgs(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_length=8,
    )
    backbone = NeoBERTBackbone(config)
    input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    attention_mask = mx.ones_like(input_ids)

    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=mx.array([[2, 3, 4]], dtype=mx.int32),
    )

    assert outputs["last_hidden_state"].shape == (1, 3, 16)

    assert "inputs_embeds" not in inspect.signature(backbone.__call__).parameters
    with pytest.raises(TypeError):
        backbone(
            input_ids=None,
            inputs_embeds=backbone.encoder(input_ids),
            attention_mask=attention_mask,
        )
