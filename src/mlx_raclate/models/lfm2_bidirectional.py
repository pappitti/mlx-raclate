from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import (
    RaclateBaseModel,
    compute_similarity_and_loss,
    mean_pooling,
    normalize_embeddings,
)
from .lfm2 import (
    Lfm2DecoderLayer,
    Lfm2Model,
    ModelArgs as Lfm2Args,
    ShortConv,
    _sanitize_backbone,
)


@dataclass
class ModelArgs(Lfm2Args):
    architectures: List[str] = field(
        default_factory=lambda: ["Lfm2BidirectionalModel"]
    )
    model_type: str = "lfm2_bidirectional"
    is_causal: bool = False
    use_cache: bool = False
    tie_word_embeddings: bool = True
    decoder_bias: bool = False
    classifier_dropout: float = 0.0
    classifier_bias: bool = False
    sparse_prediction: bool = True
    sparse_pred_ignore_index: int = -100
    pooling_mode: Literal["cls", "mean"] = "cls"


class NonCausalShortConv(ShortConv):
    """LFM2 depthwise short convolution with centered, non-causal padding."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__(args, layer_idx)
        self.conv.padding = self.L_cache // 2

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        # LiquidAI's eager/SDPA path supplies a 4D additive attention mask, for
        # which padding-state masking is intentionally a no-op. Retain support
        # for a direct 2D padding mask for parity with their alternate path.
        if mask is not None and len(mask.shape) == 2:
            x = mx.where(mask[..., None] != 0, x, 0)

        BCx = self.in_proj(x)
        B, C, projected_x = mx.split(BCx, 3, axis=-1)
        Bx = B * projected_x

        conv_out = self.conv(Bx)
        target_length = Bx.shape[-2]
        if conv_out.shape[-2] > target_length:
            conv_out = conv_out[:, :target_length, :]
        elif conv_out.shape[-2] < target_length:
            missing = target_length - conv_out.shape[-2]
            zeros = mx.zeros(
                (conv_out.shape[0], missing, conv_out.shape[-1]),
                dtype=conv_out.dtype,
            )
            conv_out = mx.concatenate([conv_out, zeros], axis=-2)

        return self.out_proj(C * conv_out)


class Lfm2BidirectionalDecoderLayer(Lfm2DecoderLayer):
    short_conv_class = NonCausalShortConv


class Lfm2BidirectionalModel(Lfm2Model):
    """LFM2 backbone with bidirectional attention and non-causal short-conv."""

    decoder_layer_class = Lfm2BidirectionalDecoderLayer

    def _update_attention_mask(
        self,
        attention_mask: Optional[mx.array] = None,
        dtype=None,
    ) -> Optional[mx.array]:
        if attention_mask is None:
            return None

        # A broadcastable padding-only additive mask lets every non-padding
        # query attend both left and right without allocating an L x L tensor.
        padding_mask = attention_mask[:, None, None, :]
        return mx.where(padding_mask == 0, -1e9, 0.0).astype(dtype)


def _sanitize_bidirectional_backbone(weights: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize both bare AutoModel and ``lfm2.*`` masked-LM checkpoints."""
    normalized = {}
    for key, value in weights.items():
        if key.startswith("lfm2."):
            key = key.removeprefix("lfm2.")
        normalized[key] = value
    return _sanitize_backbone(normalized)


def _pool_embeddings(
    config: ModelArgs,
    last_hidden_state: mx.array,
    attention_mask: mx.array,
) -> mx.array:
    if config.use_late_interaction:
        embeddings = normalize_embeddings(last_hidden_state)
        return embeddings * attention_mask[..., None]
    if config.pooling_mode == "cls":
        embeddings = last_hidden_state[:, 0]
    elif config.pooling_mode == "mean":
        embeddings = mean_pooling(last_hidden_state, attention_mask)
    else:
        raise ValueError(f"Unsupported LFM2 bidirectional pooling mode: {config.pooling_mode}")
    return normalize_embeddings(embeddings)


class Model(RaclateBaseModel):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Lfm2BidirectionalModel(config)
        self.hf_transformers_arch = "Lfm2BidirectionalModel"

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def __call__(
        self,
        input_ids: mx.array,
        position_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ):
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        last_hidden_state = outputs["last_hidden_state"]
        embeddings = _pool_embeddings(
            self.config,
            last_hidden_state,
            attention_mask,
        )

        if not return_dict:
            return (embeddings, last_hidden_state)
        return {
            "embeddings": embeddings,
            "last_hidden_state": last_hidden_state,
        }

    def sanitize(self, weights):
        return {
            key: value
            for key, value in _sanitize_bidirectional_backbone(weights).items()
            if key.startswith("model.")
        }


class ModelForSentenceSimilarity(Model):
    def _call_model(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        return_dict=True,
    ):
        return super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=return_dict,
        )

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
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)

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
            if reference_attention_mask is None:
                reference_attention_mask = mx.ones(
                    reference_input_ids.shape,
                    dtype=mx.int32,
                )
            reference_outputs = self._call_model(
                input_ids=reference_input_ids,
                attention_mask=reference_attention_mask,
                position_ids=reference_position_ids,
                return_dict=True,
            )
            similarities, loss = compute_similarity_and_loss(
                self.config,
                input_ids,
                embeddings,
                reference_outputs["embeddings"],
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


class ModelForSentenceTransformers(ModelForSentenceSimilarity):
    pass


class TiedLMHead(nn.Module):
    """Output projection that reads the embedding weight without duplicating it."""

    def __init__(self, weight_getter):
        super().__init__()
        self.weight_getter = weight_getter

    @property
    def weight(self):
        return self.weight_getter()

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return mx.matmul(hidden_states, self.weight.T)


class ModelForMaskedLM(RaclateBaseModel):
    """LFM2.5 bidirectional encoder with the checkpoint's tied MLM head."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Lfm2BidirectionalModel(config)
        if config.tie_word_embeddings:
            self.lm_head = TiedLMHead(lambda: self.model.embed_tokens.weight)
        else:
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )
        self.hf_transformers_arch = "Lfm2BidirectionalForMaskedLM"

    def tie_weights(self):
        if self.config.tie_word_embeddings and not isinstance(self.lm_head, TiedLMHead):
            self.lm_head = TiedLMHead(lambda: self.model.embed_tokens.weight)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
        self.tie_weights()

    def set_output_embeddings(self, value):
        self.config.tie_word_embeddings = False
        self.lm_head = value

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ):
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        last_hidden_state = outputs["last_hidden_state"]
        logits = self.lm_head(last_hidden_state)

        loss = None
        if labels is not None:
            flat_labels = labels.reshape(-1)
            flat_logits = logits.reshape(-1, logits.shape[-1])
            ignore_index = self.config.sparse_pred_ignore_index
            valid_mask = flat_labels != ignore_index
            safe_labels = mx.where(valid_mask, flat_labels, 0)
            token_losses = nn.losses.cross_entropy(
                flat_logits,
                safe_labels,
                reduction="none",
            )
            valid_weights = valid_mask.astype(token_losses.dtype)
            loss = mx.sum(token_losses * valid_weights) / mx.maximum(
                mx.sum(valid_weights),
                1.0,
            )

        if not return_dict:
            return (loss, logits)
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": None,
        }

    def sanitize(self, weights):
        sanitized = {
            key: value
            for key, value in _sanitize_bidirectional_backbone(weights).items()
            if key.startswith("model.")
        }

        lm_head_weight = weights.get("lm_head.weight")
        embedding_weight = sanitized.get("model.embed_tokens.weight")
        if embedding_weight is None and lm_head_weight is not None:
            sanitized["model.embed_tokens.weight"] = lm_head_weight
        if not self.config.tie_word_embeddings:
            if lm_head_weight is not None:
                sanitized["lm_head.weight"] = lm_head_weight
            elif embedding_weight is not None:
                sanitized["lm_head.weight"] = embedding_weight

        return sanitized


class ModelForTokenClassification(RaclateBaseModel):
    """Bidirectional LFM2 backbone with a per-token classification head."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        if config.is_causal:
            raise ValueError(
                "ModelForTokenClassification requires bidirectional attention."
            )

        self.config = config
        self.model_type = config.model_type
        self.num_labels = config.num_labels
        self.model = Lfm2BidirectionalModel(config)
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.score = nn.Linear(
            config.hidden_size,
            config.num_labels,
            bias=config.classifier_bias,
        )
        self.hf_transformers_arch = "Lfm2BidirectionalForTokenClassification"

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def _compute_loss(self, logits: mx.array, labels: mx.array) -> mx.array:
        flat_logits = logits.reshape(-1, self.num_labels)
        flat_labels = labels.reshape(-1)
        ignore_index = self.config.sparse_pred_ignore_index
        valid_mask = flat_labels != ignore_index
        safe_labels = mx.where(valid_mask, flat_labels, 0)
        token_losses = nn.losses.cross_entropy(
            flat_logits,
            safe_labels,
            reduction="none",
        )
        valid_weights = valid_mask.astype(token_losses.dtype)
        return mx.sum(token_losses * valid_weights) / mx.maximum(
            mx.sum(valid_weights),
            1.0,
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ):
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        last_hidden_state = outputs["last_hidden_state"]
        logits = self.score(self.dropout(last_hidden_state))
        probabilities = mx.softmax(logits, axis=-1)
        loss = self._compute_loss(logits, labels) if labels is not None else None

        if not return_dict:
            return (loss, logits, None)
        return {
            "loss": loss,
            "logits": logits,
            "probabilities": probabilities,
            "hidden_states": None,
        }

    def sanitize(self, weights):
        return {
            key: value
            for key, value in _sanitize_bidirectional_backbone(weights).items()
            if key.startswith(("model.", "score."))
        }
