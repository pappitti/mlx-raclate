from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import (
    RaclateBaseModel,
    compute_similarity_and_loss,
    mean_pooling,
    normalize_embeddings,
)
from .modernbert import ModelArgs as ModernBertArgs, ModernBertModel


DEFAULT_LATEON_DENSE_LAYERS = [
    {
        "in_features": 768,
        "out_features": 1536,
        "bias": False,
        "activation_function": "identity",
        "use_residual": True,
    },
    {
        "in_features": 1536,
        "out_features": 768,
        "bias": False,
        "activation_function": "identity",
        "use_residual": True,
    },
    {
        "in_features": 768,
        "out_features": 128,
        "bias": False,
        "activation_function": "identity",
        "use_residual": False,
    },
]


@dataclass
class ModelArgs(ModernBertArgs):
    architectures: List[str] = field(default_factory=lambda: ["ModernBertModel"])
    model_type: str = "lateon"
    colbert_dim: int = 128
    use_late_interaction: bool = True
    pylate_dense_layers: List[Dict[str, Any]] = field(
        default_factory=lambda: [dict(layer) for layer in DEFAULT_LATEON_DENSE_LAYERS]
    )


class PyLateDense(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_function: str = "identity",
        use_residual: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_residual = use_residual
        self.activation_name = self._normalize_activation_name(activation_function)
        self.activation = self._get_activation(self.activation_name)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if use_residual and in_features != out_features:
            self.residual = nn.Linear(in_features, out_features, bias=False)

    @staticmethod
    def _normalize_activation_name(activation_function: str) -> str:
        activation = (activation_function or "identity").lower()
        if "identity" in activation:
            return "identity"
        if "gelu" in activation:
            return "gelu"
        if "silu" in activation or "swish" in activation:
            return "silu"
        raise ValueError(f"Unsupported PyLate Dense activation: {activation_function}")

    @staticmethod
    def _get_activation(activation_name: str):
        if activation_name == "identity":
            return nn.Identity()
        if activation_name == "gelu":
            return nn.GELU()
        if activation_name == "silu":
            return nn.silu
        raise ValueError(f"Unsupported PyLate Dense activation: {activation_name}")

    def __call__(self, token_embeddings: mx.array) -> mx.array:
        projected = self.activation(self.linear(token_embeddings))
        if self.use_residual:
            residual = token_embeddings
            if self.in_features != self.out_features:
                residual = self.residual(token_embeddings)
            projected = projected + residual
        return projected


def _sanitize_modernbert_backbone(weights):
    sanitized_weights = {}
    for key, value in weights.items():
        if "position_ids" in key:
            continue
        if key.startswith("model."):
            sanitized_weights[key] = value
        elif key.startswith(("embeddings.", "layers.", "final_norm.")):
            sanitized_weights[f"model.{key}"] = value
    return sanitized_weights


class Model(RaclateBaseModel):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model = ModernBertModel(config)
        self.dense = [
            PyLateDense(
                in_features=layer.get("in_features"),
                out_features=layer.get("out_features"),
                bias=layer.get("bias", False),
                activation_function=layer.get(
                    "activation_function",
                    "identity",
                ),
                use_residual=layer.get("use_residual", False),
            )
            for layer in config.pylate_dense_layers
        ]

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ):
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape
            attention_mask = mx.ones(
                (batch_size, seq_len),
                dtype=self.model.embeddings.tok_embeddings.weight.dtype,
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        last_hidden_state = outputs["last_hidden_state"]
        token_embeddings = last_hidden_state
        for layer in self.dense:
            token_embeddings = layer(token_embeddings)

        if self.config.use_late_interaction:
            embeddings = normalize_embeddings(token_embeddings)
            embeddings = embeddings * attention_mask[..., None]
        else:
            # Pooling based on config
            if self.config.classifier_pooling == "cls":
                embeddings = token_embeddings[:, 0]
            elif self.config.classifier_pooling == "mean":
                embeddings = normalize_embeddings(mean_pooling(token_embeddings, attention_mask))

        if not return_dict:
            return (embeddings, last_hidden_state)
        return {
            "embeddings": embeddings,
            "last_hidden_state": last_hidden_state,
        }


    def sanitize(self, weights):
        sanitized_weights = _sanitize_modernbert_backbone(weights)
        for key, value in weights.items():
            if "position_ids" in key:
                continue
            ## TBC
            for layer_idx in range(len(self.dense)):
                module_prefix = f"{layer_idx + 1}_Dense."
                dense_prefix = f"dense.{layer_idx}."
                if key == f"{module_prefix}linear.weight":
                    sanitized_weights[f"{dense_prefix}linear.weight"] = value
                elif key == f"{module_prefix}linear.bias":
                    sanitized_weights[f"{dense_prefix}linear.bias"] = value
                elif key == f"{module_prefix}residual.weight":
                    sanitized_weights[f"{dense_prefix}residual.weight"] = value
                elif key.startswith(dense_prefix):
                    sanitized_weights[key] = value
        return sanitized_weights


class ModelForSentenceSimilarity(RaclateBaseModel):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model = ModernBertModel(config)
        self.dense = [
            PyLateDense(
                in_features=layer.get("in_features"),
                out_features=layer.get("out_features"),
                bias=layer.get("bias", False),
                activation_function=layer.get(
                    "activation_function",
                    "identity",
                ),
                use_residual=layer.get("use_residual", False),
            )
            for layer in config.pylate_dense_layers
        ]

    def _call_model(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        last_hidden_state = outputs["last_hidden_state"]
        token_embeddings = last_hidden_state
        for layer in self.dense:
            token_embeddings = layer(token_embeddings)

        if self.config.use_late_interaction:
            embeddings = normalize_embeddings(token_embeddings)
            embeddings = embeddings * attention_mask[..., None]
        else:
            # Pooling based on config
            if self.config.classifier_pooling == "cls":
                embeddings = token_embeddings[:, 0]
            elif self.config.classifier_pooling == "mean":
                embeddings = normalize_embeddings(mean_pooling(token_embeddings, attention_mask))

        if not return_dict:
            return (embeddings, last_hidden_state)
        return {
            "embeddings": embeddings,
            "last_hidden_state": last_hidden_state,
        }

    def __call__(
        self,
        input_ids,
        reference_input_ids: Optional[mx.array] = None,
        negative_input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        reference_attention_mask: Optional[mx.array] = None,
        negative_attention_mask: Optional[mx.array] = None,
        similarity_scores: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        return_dict: Optional[bool] = True,
        reference_position_ids: Optional[mx.array] = None,
        negative_position_ids: Optional[mx.array] = None,
    ):
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape
            attention_mask = mx.ones(
                (batch_size, seq_len),
                dtype=self.model.embeddings.tok_embeddings.weight.dtype,
            )
        
        batch_outputs = self._call_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        embeddings = batch_outputs["embeddings"]

        loss = None
        similarities = None
        if reference_input_ids is not None:
            ref_outputs = self._call_model(
                input_ids=reference_input_ids,
                attention_mask=reference_attention_mask,
                position_ids=reference_position_ids,
                return_dict=True,
            )
            similarities, loss = compute_similarity_and_loss(
                self.config,
                input_ids,
                embeddings,
                ref_outputs["embeddings"],
                self._call_model,
                similarity_scores,
                negative_input_ids,
                negative_attention_mask,
                negative_position_ids,
            )

        if not return_dict:
            return (loss, similarities, embeddings)
        return {
            "loss": loss,
            "similarities": similarities,
            "embeddings": embeddings,
        }
    
    def sanitize(self, weights):
        sanitized_weights = _sanitize_modernbert_backbone(weights)
        for key, value in weights.items():
            if "position_ids" in key:
                continue
            ## TBC
            for layer_idx in range(len(self.dense)):
                module_prefix = f"{layer_idx + 1}_Dense."
                dense_prefix = f"dense.{layer_idx}."
                if key == f"{module_prefix}linear.weight":
                    sanitized_weights[f"{dense_prefix}linear.weight"] = value
                elif key == f"{module_prefix}linear.bias":
                    sanitized_weights[f"{dense_prefix}linear.bias"] = value
                elif key == f"{module_prefix}residual.weight":
                    sanitized_weights[f"{dense_prefix}residual.weight"] = value
                elif key.startswith(dense_prefix):
                    sanitized_weights[key] = value
        return sanitized_weights


class ModelForSentenceTransformers(ModelForSentenceSimilarity):
    pass

class ModelForMaskedLM(RaclateBaseModel):
    def __init__(self, config: ModelArgs):
        raise NotImplementedError("Masked LM head is not implemented for Lateon yet.")
    
class ModelForSequenceClassification(RaclateBaseModel):
    def __init__(self, config: ModelArgs):
        raise NotImplementedError("Sequence Classification head is not implemented for Lateon yet.")
    
class ModelForTokenClassification(RaclateBaseModel):
    def __init__(self, config: ModelArgs):
        raise NotImplementedError("Token Classification head is not implemented for Lateon yet.")
