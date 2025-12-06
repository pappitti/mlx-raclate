from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Literal
import re

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.gemma3_text import ModelArgs, RMSNorm, TransformerBlock

from .base import (
    BaseModelArgs,
    last_token_pooling,
    normalize_embeddings,
    compute_similarity,
)

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    head_dim: int
    tie_word_embeddings: bool
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    attention_bias: Optional[bool] = False
    attention_dropout: Optional[float] = 0.0
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    hidden_act: Optional[str] = "silu"
    max_window_layers: Optional[int] = 28
    architectures: List[str] = field(default_factory=lambda: ["Qwen3Model"])

    initializer_range: Optional[float] = (
        0.02  # Only needed in case of initializing weights
    )

    ### pipeline args
    decoder_bias=True,
    classifier_pooling: Literal["cls", "mean"] = "cls"
    classifier_dropout=0.0 
    classifier_bias=False
    sparse_prediction=True ### True seems a more appropriate value for MLM
    sparse_pred_ignore_index=-100 
    is_regression: Optional[bool] = None
    label2id: Optional[Dict[str, int]] = None
    id2label: Optional[Dict[int, str]] = None
    pipeline_config: Optional[Dict[str, Any]] = None  # for Sequence Classification

    @property
    def num_labels(self) -> int:
        """
        Number of labels is determined by:
        - For zero-shot classification: length of label_candidates
        - For regression or binary with sigmoid: 1
        - For classification: length of id2label mapping
        """
        
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


class Gemma3Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TransformerBlock(config, i)
            for i in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _update_attention_mask(self, attention_mask: Optional[mx.array] = None):
        """
        Creates a causal mask and combines it with the padding mask.
        """
        dtype = attention_mask.dtype
        B, L = attention_mask.shape

        causal_mask = mx.triu(mx.full((L, L), -1e9, dtype), k=1)

        if attention_mask is not None:
            # Reshape padding mask from (B, L) to (B, 1, 1, L) to be broadcastable
            padding_mask = attention_mask[:, None, None, :]
            additive_padding_mask = mx.where(padding_mask == 0, -1e9, 0.0).astype(dtype)

            causal_mask = causal_mask + additive_padding_mask

        return causal_mask.astype(dtype)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        attention_mask = self._update_attention_mask(
            attention_mask,
        )

        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(input_ids)
        h *= mx.array(
            self.config.hidden_size**0.5, self.embed_tokens.weight.dtype
        ).astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            j = self.config.sliding_window_pattern
            full_mask = create_attention_mask(h, cache[j - 1 : j])
            sliding_window_mask = create_attention_mask(h, cache)

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            is_global = (
                i % self.config.sliding_window_pattern
                == self.config.sliding_window_pattern - 1
            )

            local_mask = mask
            if mask is None and is_global:
                local_mask = full_mask
            elif mask is None:
                local_mask = sliding_window_mask

            h = layer(h, local_mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Gemma3Model(config)
        self.dense = [
            nn.Linear(config.hidden_size, config.hidden_size * 4, bias=False),
            nn.Linear(config.hidden_size * 4, config.hidden_size, bias=False),
        ]

    def get_extended_attention_mask(self, attention_mask, input_shape):
        if attention_mask.ndim == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.ndim == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = mx.repeat(
                extended_attention_mask, attention_mask.shape[-1], -2
            )

        else:
            raise ValueError(
                f"Wrong shape for attention_mask (shape {attention_mask.shape})"
            )
        return extended_attention_mask

    def __call__(
        self,
        inputs: mx.array,
        attention_mask: Optional[mx.array] = None,
    ):

        if attention_mask is None:
            attention_mask = mx.ones(inputs.shape)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, inputs.shape
        )

        out = self.model(inputs, extended_attention_mask)

        for dense in self.dense:
            out = dense(out)

        # normalized features
        text_embeds = mean_pooling(out, attention_mask)
        text_embeds = normalize_embeddings(text_embeds)

        return BaseModelOutput(
            last_hidden_state=out,
            text_embeds=text_embeds,
            pooler_output=None,
        )

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "linear" not in k and "dense" not in k:
                new_key = f"model.{k}" if not k.startswith("model") else k
                sanitized_weights[new_key] = v
            elif "dense" not in k:
                key_id = "0" if v.shape[0] > v.shape[1] else "1"
                new_key = re.sub(r"\d+_Dense\.linear", f"dense.{key_id}", k)
                sanitized_weights[new_key] = v
            else:
                sanitized_weights[k] = v

        return sanitized_weights

    @property
    def layers(self):
        return self.model.layers