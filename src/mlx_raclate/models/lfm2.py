from functools import cache
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal

import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs,
    last_token_pooling,
    mean_pooling,
    normalize_embeddings,
    compute_similarity_and_loss,
    RaclateBaseModel,
)

"""
Not using cache in this implementation given 
the model is intended to be used for embedding and classification tasks.
"""

@dataclass
class ModelArgs(BaseModelArgs):
    architectures: List[str] = field(default_factory=lambda: ["Lfm2Model"])
    block_auto_adjust_ff_dim: bool = False
    block_dim: int = 1024
    block_ff_dim: int = 6656
    block_ffn_dim_multiplier: float = 1.0
    block_mlp_init_scale: Optional[float] = None
    block_multiple_of: int = 256
    block_norm_eps: float = 1e-5 # where to use this?
    block_use_swiglu: bool = True # where to use this?
    block_use_xavier_init: bool = True # where to use this?
    bos_token_id: int = 1
    conv_bias: bool = False
    conv_L_cache: int = 3
    conv_dim : int = 1024 # where to use this?
    conv_dim_out : int = 1024 # where to use this?
    conv_use_xavier_init: bool = True # where to use this?
    eos_token_id: int = 7
    full_attn_idxs: Optional[List[int]] = None
    hidden_size: int = 1024
    initializer_range: Optional[float] = (
        0.02  # Only needed in case of initializing weights
    )
    layer_types: Optional[List[str]] = None 
    max_position_embeddings: int = 128000
    model_type: str = "lfm2"
    norm_eps: float = 1e-05
    num_attention_heads: int = 16
    num_hidden_layers: int = 16
    num_key_value_heads: int = 8
    out_features: int = 128 # classifier output features
    pad_token_id: int = 0
    rope_theta: float = 1000000.0
    vocab_size: int = 65536
    
    ### pipeline args
    decoder_bias=True,
    classifier_dropout=0.0 
    classifier_bias=False
    sparse_prediction=True ### True seems a more appropriate value for MLM
    sparse_pred_ignore_index=-100 
    is_regression: Optional[bool] = None
    label2id: Optional[Dict[str, int]] = None
    id2label: Optional[Dict[int, str]] = None
    pipeline_config: Optional[Dict[str, Any]] = None  # for Sequence Classification
    use_late_interaction: bool = False 

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


def _sanitize_backbone(weights: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardizes keys for the Gemma3 Backbone. 
    Prefixes generic keys with 'model.' and handles basic mapping.
    """
    sanitized = {}
    for k, v in weights.items():
        # Skip unrelated heads that might be in the checkpoint
        if any(x in k for x in ["lm_head", "classifier"]):
            # We don't automatically map these; specific models handle them if needed
            continue
            
        if "position_ids" in k:
            # Remove unused position_ids
            continue

        if "linear" not in k and "dense" not in k:
            new_key = f"model.{k}" if not k.startswith("model") else k
            if "conv.weight" in new_key:
                if v.shape[-1] > v.shape[1]:
                    v = v.transpose(0, 2, 1)
            sanitized[new_key] = v
        else:
            sanitized[k] = v
            
    return sanitized

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.hidden_size // n_heads

        self.scale = head_dim**-0.5

        self.q_layernorm = nn.RMSNorm(head_dim, eps=args.norm_eps)
        self.k_layernorm = nn.RMSNorm(head_dim, eps=args.norm_eps)

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = nn.RoPE(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_layernorm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_layernorm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = self.rope(queries)
        keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)

class ShortConv(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        layer_idx: int,
    ):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.L_cache = args.conv_L_cache
        self.bias = args.conv_bias

        self.conv = nn.Conv1d(
            in_channels=args.hidden_size,
            out_channels=args.hidden_size,
            kernel_size=self.L_cache,
            groups=args.hidden_size,
            bias=self.bias,
        )
        self.in_proj = nn.Linear(args.hidden_size, 3 * args.hidden_size, bias=self.bias)
        self.out_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=self.bias)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None
    ):
        BCx = self.in_proj(x)
        B, C, x = mx.split(BCx, 3, axis=-1)
        Bx = B * x
        if mask is not None:
            Bx = mx.where(mask[..., None], Bx, 0)

        state = mx.zeros(
            (Bx.shape[0], self.L_cache - 1, self.args.hidden_size), dtype=Bx.dtype
        )

        Bx = mx.concatenate([state, Bx], axis=-2)
        conv_out = self.conv(Bx)

        y = C * conv_out
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_dim: int,
        multiple_of: int,
        auto_adjust_ff_dim: bool,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        if auto_adjust_ff_dim:
            ff_dim = int(2 * ff_dim / 3)
            if ffn_dim_multiplier is not None:
                ff_dim = int(ffn_dim_multiplier * ff_dim)
            ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, ff_dim, bias=False)
        self.w3 = nn.Linear(dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Lfm2DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        if args.full_attn_idxs :
            self.is_attention_layer = layer_idx in args.full_attn_idxs
        elif args.layer_types:
            self.is_attention_layer = args.layer_types[layer_idx] == "full_attention"
        else:
            raise ValueError("Either full_attn_idxs or layer_types must be provided in ModelArgs")

        if self.is_attention_layer:
            self.self_attn = Attention(args)
        else:
            self.conv = ShortConv(args, layer_idx)
        
        self.feed_forward = MLP(
            dim=args.block_dim,
            ff_dim=args.block_ff_dim,
            multiple_of=args.block_multiple_of,
            auto_adjust_ff_dim=args.block_auto_adjust_ff_dim,
            ffn_dim_multiplier=args.block_ffn_dim_multiplier,
        )

        self.operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:

        if self.is_attention_layer:
            r = self.self_attn(self.operator_norm(x), mask=mask, cache=cache)
        else:
            r = self.conv(
                self.operator_norm(x),
                mask=mask,
                cache=cache,
            )
        h = x + r
        out = h + self.feed_forward(self.ffn_norm(h))
        return (out,)

class Lfm2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Lfm2DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]

        self.embedding_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

        self.conv_idx = 0
        if args.full_attn_idxs:
            for i in range(args.num_hidden_layers):
                if i in args.full_attn_idxs:
                    self.conv_idx += 1
                else:
                    break
        elif args.layer_types:
            for i in range(args.num_hidden_layers):
                if args.layer_types[i] != "full_attention":
                    self.conv_idx += 1
                else:
                    break
        else:
            raise ValueError("Either full_attn_idxs or layer_types must be provided in ModelArgs")

        self.hf_transformers_arch = "Lfm2Model"

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _update_attention_mask(self, attention_mask: Optional[mx.array] = None, dtype=None):
        """
        Creates a causal mask and combines it with the padding mask.
        """
 
        B, L = attention_mask.shape

        causal_mask = mx.triu(mx.full((L, L), -1e9, dtype), k=1)

        if attention_mask is not None:
            # Reshape padding mask from (B, L) to (B, 1, 1, L) to be broadcastable
            padding_mask = attention_mask[:, None, None, :]
            additive_padding_mask = mx.where(padding_mask == 0, -1e9, 0.0).astype(dtype)

            causal_mask = causal_mask + additive_padding_mask

        return causal_mask.astype(dtype)
    
    def _create_ssm_mask(self, h, cache=None):
        if cache and hasattr(cache, "make_mask"):
            return cache.make_mask(h.shape[1])
        return None

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ):

        hidden_states = self.embed_tokens(input_ids)
        model_dtype = hidden_states.dtype

        cache = [None] * len(self.layers)

        attn_mask = self._update_attention_mask(attention_mask, dtype=model_dtype)
        conv_mask = self._create_ssm_mask(hidden_states, cache[self.conv_idx])

        for layer, c in zip(self.layers, cache):
            mask = attn_mask if layer.is_attention_layer else conv_mask
            layer_outputs = layer(hidden_states, mask, cache=c)
            hidden_states = layer_outputs[0]

        hidden_states = self.embedding_norm(hidden_states)

        return {
            "last_hidden_state": hidden_states,
        }


class Model(RaclateBaseModel):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Lfm2Model(config)

    def __call__(
        self,
        input_ids: mx.array, 
        position_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None, 
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ):

        if attention_mask is None:
            batch_size, seq_len = input_ids.shape
            attention_mask = mx.ones(
                (batch_size, seq_len),
                dtype=self.model.embed_tokens.weight.dtype,
            )

        out = self.model(input_ids, attention_mask)

        last_hidden_state = (
            out["last_hidden_state"] if isinstance(out, dict) else out[0]
        )

        # LFM2 is a causal model, so we use last token pooling for embeddings
        text_embeds = last_token_pooling(last_hidden_state, attention_mask)
        text_embeds = normalize_embeddings(text_embeds)

        if not return_dict:
            return (text_embeds, last_hidden_state) 

        return {
            "embeddings": text_embeds, # normalized embeddings
            "last_hidden_state": last_hidden_state,
        }


    def sanitize(self, weights):

        return _sanitize_backbone(weights)


class ModelForSentenceSimilarity(RaclateBaseModel):
    """
    Computes similarity scores between input sequences and reference sentences.
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Lfm2Model(config)
        self.dense = [
            nn.Linear(config.block_dim, config.out_features, bias=False),
        ]

    def _call_model(self, input_ids, attention_mask=None, return_dict=True):
        out = self.model(input_ids, attention_mask)
        last_hidden_state = (
            out["last_hidden_state"] if isinstance(out, dict) else out[0]
        )

        for dense in self.dense:
            last_hidden_state = dense(last_hidden_state)

        # text_embeds = normalize_embeddings(last_hidden_state)
        if self.config.use_late_interaction:
            text_embeds = normalize_embeddings(last_hidden_state)
            # Keep unpooled for ColBERT style
            # Mask padding tokens to avoid them affecting MaxSim
            if attention_mask is not None:
                text_embeds = text_embeds * attention_mask[..., None]
        else:
            # Standard dense retrieval: Mean Pooling
            text_embeds = mean_pooling(last_hidden_state, attention_mask)
            text_embeds = normalize_embeddings(text_embeds)


        if not return_dict:
            return (text_embeds, last_hidden_state) 

        return {
            "embeddings": text_embeds, # normalized embeddings
            "last_hidden_state": last_hidden_state,
        }
    
    def __call__(
        self,
        input_ids,
        reference_input_ids : Optional[mx.array] = None,  # Shape: [num_references, seq_len]
        negative_input_ids : Optional[mx.array] = None,  # Shape: [num_negatives, seq_len]
        attention_mask: Optional[mx.array] = None,
        reference_attention_mask: Optional[mx.array] = None,
        negative_attention_mask: Optional[mx.array] = None,
        similarity_scores: Optional[mx.array] = None,  # Shape: [batch_size, num_references]
        position_ids: Optional[mx.array] = None,
        return_dict: Optional[bool] = True,
    ):
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape
            attention_mask = mx.ones(
                (batch_size, seq_len),
                dtype=self.model.embed_tokens.weight.dtype,
            )

        # Get embeddings for input batch
        batch_outputs = self._call_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        embeddings = batch_outputs["embeddings"]  # [batch_size, hidden_size]

        loss = None
        similarities = None
        if reference_input_ids is not None:
        
            # Get embeddings for reference sentences
            ref_outputs = self._call_model(
                input_ids=reference_input_ids,
                attention_mask=reference_attention_mask,
                return_dict=True
            )
            reference_embeddings = ref_outputs["embeddings"]  # [num_references, hidden_size]

            similarities, loss = compute_similarity_and_loss(
                self.config,
                input_ids,
                embeddings,
                reference_embeddings,
                self._call_model,
                similarity_scores,
                negative_input_ids,
                negative_attention_mask
            )
            
        if not return_dict:
            return (loss, similarities, embeddings)
            
        return {
            "loss": loss,
            "similarities": similarities,  # [batch_size, num_references]
            "embeddings": embeddings,  # [batch_size, hidden_size]
        }
    
    def sanitize(self, weights):
        return _sanitize_backbone(weights)

class ModelForSentenceTransformers(ModelForSentenceSimilarity):
    """
    Extends ModelForSentenceSimilarity to provide embeddings for input sequences.
    This class sanitizes typical sentence transformers weights to align with the T5Gemma model.
    """
    def __init__(self, config: ModelArgs):
        super().__init__(config)

    def sanitize(self, weights):
        """Convert sentence transformer weights to T5Gemma format."""
        sanitized = _sanitize_backbone(weights)
        
        sanitized_weights = {}
        for k, v in sanitized.items():
            if "1_Dense.linear" in k:
                new_key = k.replace("1_Dense.linear", "dense.0")
                sanitized_weights[new_key] = v
            else:
                sanitized_weights[k] = v
        return sanitized_weights
    
class ModelForSequenceClassification(RaclateBaseModel):
    """
    Computes sequence classification probabilities for input sequences.

    NOTE : regression and binary classification not tested.
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.is_regression = config.is_regression
        
        self.model = Lfm2Model(config)

        # No HF transformers architecture SequenceClassification typically only as a score layer
        self.score = nn.Linear(
            config.hidden_size, 
            config.num_labels, 
            bias=False
        ) 

        # No HF transformers architecture for LFM2 and SequenceClassification
    
    def _process_outputs(self, logits: mx.array) -> mx.array:
        """Apply the appropriate activation function to the logits."""
        if self.is_regression:
            return logits  # No activation for regression
        elif self.num_labels == 1:
            return mx.sigmoid(logits)  # Binary classification
        else:
            # Using softmax for multi-class classification
            return mx.softmax(logits, axis=-1)

    def _compute_loss(self, logits: mx.array, labels: mx.array) -> mx.array:
        """Compute the appropriate loss based on label characteristics."""
        if self.is_regression:
            return nn.losses.mse_loss(logits.squeeze(), labels.squeeze())
        elif self.num_labels == 1:
            return nn.losses.binary_cross_entropy(mx.sigmoid(logits), labels)
        else:
            return nn.losses.cross_entropy(
                logits.reshape(-1, self.num_labels),
                labels.reshape(-1)
            )

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None, ### need this?
        labels: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Dict:
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape
            attention_mask = mx.ones(
                (batch_size, seq_len),
                dtype=self.model.embed_tokens.weight.dtype,
            )

        outputs = self.model(
            input_ids, 
            attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        last_hidden_state = (
            outputs["last_hidden_state"] if isinstance(outputs, dict) else outputs[0]
        )

        # pooling for AR models such as LFM2 leverages the last token
        pooled = last_token_pooling(last_hidden_state, attention_mask)

        ### The HF architecture for SequenceClassification typically only has a score layer
        logits = self.score(pooled)

        processed_logits = self._process_outputs(logits)

        loss = None
        if labels is not None :
            loss = self._compute_loss(logits, labels)

        if not return_dict:
            return [loss, processed_logits, outputs[1:]]

        return {
            "loss": loss,
            "probabilities": processed_logits,
            "hidden_states": outputs.get("hidden_states", None),
        }
    
    def sanitize(self, weights):
        return _sanitize_backbone(weights)
    

# TokenClassification and MaskedLM not implemented for now AR models such as LFM2
# Attempting to train pretrained weights would be catastrophic 