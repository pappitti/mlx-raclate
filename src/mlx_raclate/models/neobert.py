from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Literal, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs,
    RaclateBaseModel,
    RotaryEmbedding,
    SwiGLU,
    compute_similarity_and_loss,
    mean_pooling,
    normalize_embeddings,
)


@dataclass
class ModelArgs(BaseModelArgs):
    architectures: List[str] = field(default_factory=lambda: ["NeoBERTLMHead"])
    classifier_init_range: float = 0.02
    decoder_bias: bool = True
    decoder_init_range: float = 0.02
    dim_head: Optional[int] = None
    embedding_init_range: float = 0.02
    hidden_size: int = 768
    intermediate_size: int = 3072
    max_length: int = 4096
    model_type: str = "neobert"
    norm_eps: float = 1e-5
    num_attention_heads: int = 12
    num_hidden_layers: int = 28
    pad_token_id: int = 0
    rope_theta: float = 10000.0
    vocab_size: int = 30522

    # Pipeline args.
    classifier_pooling: Literal["cls", "mean"] = "mean"
    classifier_dropout: float = 0.1
    classifier_bias: bool = True
    sparse_prediction: bool = True
    sparse_pred_ignore_index: int = -100
    is_regression: Optional[bool] = None
    label2id: Optional[Dict[str, int]] = None
    id2label: Optional[Dict[Any, str]] = None
    pipeline_config: Optional[Dict[str, Any]] = None
    problem_type: Optional[str] = None
    use_late_interaction: bool = False

    def __post_init__(self):
        if self.dim_head is None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError("hidden_size must be divisible by num_attention_heads")
            self.dim_head = self.hidden_size // self.num_attention_heads

    @property
    def num_labels(self) -> int:
        if self.is_regression:
            return 1

        if self.pipeline_config and self.pipeline_config.get("binary_sigmoid", False):
            return 1

        if self.id2label is None:
            raise ValueError(
                "id2label mapping must be provided for categorical classification. "
                "For regression or binary classification with sigmoid output, "
                "set is_regression=True or binary_sigmoid=True in pipeline_config."
            )

        return len(self.id2label)


# TBC
def _sanitize_neobert_weights(
    weights: Dict[str, Any],
    include_decoder: bool = False,
    include_classifier: bool = False,
) -> Dict[str, Any]:
    sanitized = {}

    for key, value in weights.items():
        if "position_ids" in key or "freqs_cis" in key:
            continue
        if key.endswith("rotary_emb.inv_freq") or key.endswith("rotary_emb._freqs"):
            continue

        if key.startswith("model."):
            sanitized[key] = value
            continue

        if key.startswith(("encoder.", "transformer_encoder.", "layer_norm.")):
            sanitized[f"model.{key}"] = value
            continue

        if include_decoder and key.startswith("decoder."):
            sanitized[key] = value
            continue

        if include_classifier and key.startswith(("dense.", "classifier.")):
            sanitized[key] = value
            continue

    return sanitized


class EncoderBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.dim_head)
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(
            config.hidden_size,
            config.hidden_size * 3,
            bias=False,
        )
        self.wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        multiple_of = 8
        intermediate_size = int(2 * config.intermediate_size / 3)
        intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
        self.ffn = SwiGLU(
            config.hidden_size,
            intermediate_size,
            config.hidden_size,
            bias=False,
        )

        self.attention_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.rotary_emb = (
            RotaryEmbedding(self.head_dim, base=config.rope_theta)
        )

    def _att_block(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.qkv(hidden_states).reshape(
            batch_size,
            seq_len,
            self.num_heads,
            3 * self.head_dim,
        )
        queries, keys, values = mx.split(qkv, 3, axis=-1)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        queries = self.rotary_emb(queries, position_ids=position_ids)
        keys = self.rotary_emb(keys, position_ids=position_ids)

        attn_output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=attention_mask,
        )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size,
            seq_len,
            self.config.hidden_size,
        )
        return self.wo(attn_output)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        attn_output = self._att_block(
            self.attention_norm(hidden_states),
            attention_mask,
            position_ids,
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states


class NeoBERTModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.encoder = nn.Embedding(config.vocab_size, config.hidden_size)
        self.transformer_encoder = [
            EncoderBlock(config) for _ in range(config.num_hidden_layers)
        ]
        self.layer_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.encoder

    def set_input_embeddings(self, value):
        self.encoder = value

    def _update_attention_mask(
        self,
        attention_mask: mx.array,
        dtype: mx.Dtype,
    ) -> Optional[mx.array]:
        padding_mask = attention_mask[:, None, None, :]
        return mx.where(padding_mask == 1, 0.0, -1e9).astype(dtype)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ):
        hidden_states = self.encoder(input_ids)
        model_dtype = hidden_states.dtype

        attention_mask = self._update_attention_mask(
            attention_mask,
            dtype=model_dtype,
        )

        all_hidden_states = [] if output_hidden_states else None
        for layer in self.transformer_encoder:
            hidden_states = layer(hidden_states, attention_mask, position_ids)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        hidden_states = self.layer_norm(hidden_states)

        if not return_dict:
            return (hidden_states, all_hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
        }


class Model(RaclateBaseModel):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model = NeoBERTModel(config)
        
        # no transformers arch for NeoBert

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
        

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape
            attention_mask = mx.ones(
                (batch_size, seq_len),
                dtype=self.model.encoder.weight.dtype,
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        last_hidden_state = outputs["last_hidden_state"]

        if self.config.use_late_interaction:
            embeddings = normalize_embeddings(last_hidden_state)
            embeddings = embeddings * attention_mask[..., None]
        elif self.config.classifier_pooling == "cls":
            embeddings = normalize_embeddings(last_hidden_state[:, 0])
        else:
            pooled = mean_pooling(last_hidden_state, attention_mask)
            embeddings = normalize_embeddings(pooled)

        if not return_dict:
            return (embeddings, last_hidden_state)

        return {
            "embeddings": embeddings,
            "last_hidden_state": last_hidden_state,
            "hidden_states": outputs.get("hidden_states"),
        }

    # TBC
    def sanitize(self, weights):
        sanitized = _sanitize_neobert_weights(weights)
        return {
            key: value
            for key, value in sanitized.items()
            if key.startswith("model.")
        }


class ModelForSentenceSimilarity(RaclateBaseModel):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model = NeoBERTModel(config)

        # no transformers arch for NeoBert
    
    def _call_model(
        self,
        input_ids,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,   
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        last_hidden_state = outputs["last_hidden_state"]

        if self.config.classifier_pooling == "cls":
            embeddings = normalize_embeddings(last_hidden_state[:, 0])
        else:
            pooled = mean_pooling(last_hidden_state, attention_mask)
            embeddings = normalize_embeddings(pooled)

        if not return_dict:
            return (embeddings, last_hidden_state)

        return {
            "embeddings": embeddings,
            "last_hidden_state": last_hidden_state,
            "hidden_states": outputs.get("hidden_states"),
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
                dtype=self.model.encoder.weight.dtype,
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
            reference_embeddings = ref_outputs["embeddings"]

            similarities, loss = compute_similarity_and_loss(
                self.config,
                input_ids,
                embeddings,
                reference_embeddings,
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


class ModelForMaskedLM(RaclateBaseModel):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model = NeoBERTModel(config)
        self.decoder = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=config.decoder_bias,
        )
        # no transformers arch for NeoBert

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, new_embeddings):
        self.decoder = new_embeddings

    def _compute_loss(self, logits: mx.array, labels: mx.array) -> mx.array:
        if self.config.sparse_prediction:
            flat_labels = labels.reshape(-1)
            flat_predictions = logits.reshape(-1, logits.shape[-1])

            # Filter out non-masked tokens
            ignore_index = getattr(self.config, "sparse_pred_ignore_index", -100)
            mask_tokens = flat_labels != ignore_index

            safe_labels = mx.where(mask_tokens, flat_labels, 0)
            token_losses = nn.losses.cross_entropy(
                flat_predictions,
                safe_labels,
                reduction="none",
            )
            valid_weights = mask_tokens.astype(token_losses.dtype)
            return mx.sum(token_losses * valid_weights) / mx.maximum(
                mx.sum(valid_weights),
                1.0,
            )

        return nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="mean",
        )

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape
            attention_mask = mx.ones(
                (batch_size, seq_len),
                dtype=self.model.encoder.weight.dtype,
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        last_hidden_state = outputs["last_hidden_state"]
        logits = self.decoder(last_hidden_state)

        loss = None
        if labels is not None:
            loss = self._compute_loss(logits, labels)

        if not return_dict:
            return (loss, logits, outputs.get("hidden_states"))

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.get("hidden_states"),
        }

    def sanitize(self, weights):
        return _sanitize_neobert_weights(weights, include_decoder=True)


class ModelForSequenceClassification(RaclateBaseModel):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.is_regression = config.is_regression
        self.model = NeoBERTModel(config)
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.classifier_bias,
        )
        self.classifier = nn.Linear(
            config.hidden_size,
            config.num_labels,
            bias=config.classifier_bias,
        )
        
        # no transformers arch for NeoBert

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def _process_outputs(self, logits: mx.array) -> mx.array:
        if self.is_regression:
            return logits
        if self.num_labels == 1:
            return mx.sigmoid(logits)
        return mx.softmax(logits, axis=-1)

    def _compute_loss(self, logits: mx.array, labels: mx.array) -> mx.array:
        if self.is_regression:
            return nn.losses.mse_loss(logits.squeeze(), labels.squeeze())
        if self.num_labels == 1:
            return nn.losses.binary_cross_entropy(mx.sigmoid(logits), labels)
        return nn.losses.cross_entropy(
            logits.reshape(-1, self.num_labels),
            labels.reshape(-1),
        )

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape
            attention_mask = mx.ones(
                (batch_size, seq_len),
                dtype=self.model.encoder.weight.dtype,
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        last_hidden_state = outputs["last_hidden_state"]
        pooled = last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        pooled = mx.tanh(self.dense(pooled))
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        probabilities = self._process_outputs(logits)

        loss = None
        if labels is not None:
            loss = self._compute_loss(logits, labels)

        if not return_dict:
            return (loss, probabilities, outputs.get("hidden_states"))

        return {
            "loss": loss,
            "logits": logits,
            "probabilities": probabilities,
            "hidden_states": outputs.get("hidden_states"),
        }

    def sanitize(self, weights):
        return _sanitize_neobert_weights(weights, include_classifier=True)


class ModelForTokenClassification(RaclateBaseModel):
    """ 
    Untested - Not sure if checkpoints already exist for this architecture. 
    Suggesting this one, that's close to the SequenceClassification architecture
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.model = NeoBERTModel(config)
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.classifier = nn.Linear(
            config.hidden_size,
            config.num_labels,
            bias=config.classifier_bias,
        )
        
        # no transformers arch for NeoBert

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def _compute_loss(self, logits: mx.array, labels: mx.array) -> mx.array:
        flat_logits = logits.reshape(-1, self.config.num_labels)
        flat_labels = labels.reshape(-1)
        valid_mask = flat_labels != -100
        safe_labels = mx.where(valid_mask, flat_labels, 0)
        token_losses = nn.losses.cross_entropy(
            flat_logits,
            safe_labels,
            reduction="none",
        )
        valid_weights = valid_mask.astype(token_losses.dtype)
        return mx.sum(token_losses * valid_weights) / mx.maximum(mx.sum(valid_weights), 1.0)

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape
            attention_mask = mx.ones(
                (batch_size, seq_len),
                dtype=self.model.encoder.weight.dtype,
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        last_hidden_state = outputs["last_hidden_state"]
        pooled = self.dropout(last_hidden_state)
        logits = self.classifier(pooled)
        probabilities = mx.softmax(logits, axis=-1)

        loss = None
        if labels is not None:
            loss = self._compute_loss(logits, labels)

        if not return_dict:
            return (loss, logits, outputs.get("hidden_states"))

        return {
            "loss": loss,
            "logits": logits,
            "probabilities": probabilities,
            "hidden_states": outputs.get("hidden_states"),
        }
    
    def sanitize(self, weights):
        return _sanitize_neobert_weights(weights, include_classifier=True)