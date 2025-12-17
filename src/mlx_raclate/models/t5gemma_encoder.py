from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Literal
import re
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs,
    mean_pooling,
    normalize_embeddings,
    compute_similarity_and_loss,
    RaclateBaseModel,
)

@dataclass
class ModelArgs(BaseModelArgs):
    architectures: List[str] = field(default_factory=lambda: ["T5GemmaForConditionalGeneration"])
    attention_bias: Optional[bool] = False
    attention_dropout: Optional[float] = 0.0
    attn_logit_softcapping: Optional[float] = None # Not supported with sdpa
    bos_token_id: Optional[int] = None
    dropout_rate: float = 0.0
    eos_token_id: Optional[List[int]] = None
    final_logit_softcapping: Optional[float] = None # Not supported with sdpa
    head_dim: int = 64
    hidden_activation: Optional[str] = "gelu_pytorch_tanh"
    hidden_size: int = 768
    initializer_range: Optional[float] = 0.02  # Only needed in case of initializing weights
    intermediate_size: int = 2048
    is_causal: bool = False
    is_encoder_decoder: bool = False
    layer_types: List[str] = field(default_factory=list) 
    max_position_embeddings: int = 8192
    model_type: str = "t5gemma"
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    num_key_value_heads: int = 12
    query_pre_attn_scalar: float = 64
    rms_norm_eps: float = 1.0e-6
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    sliding_window: int = 4096
    vocab_size: int = 256000
   

    ### pipeline args
    decoder_bias=True,
    classifier_dropout_rate: float = 0.0 
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
    Standardizes keys for the T5Gemma Encoder Backbone. 
    Drops Decoder weights and handles basic mapping.
    """
    sanitized_weights = {}
    for k, v in weights.items():
        # Skip unused buffers or position IDs if present
        if "position_ids" in k or "rotary_emb.inv_freq" in k:
            continue

        # Skip decoder weights
        if k.startswith("model.decoder."):
            continue

        # if the model uses shared embeddings, map them correctly
        if k == "shared.weight":
            if "encoder.embed_tokens.weight" not in weights:
                sanitized_weights["model.embed_tokens.weight"] = v
            continue

        # Process Encoder weights
        if k.startswith("model.encoder."):
            new_k =  k.replace("model.encoder.", "model.")
        # handle non-prefix keys but must keep 
        # "score" for classification and "head" and "decoder" for masked lm
        elif not k.startswith("model.") and not k.startswith("score.") and not k.startswith("head.") and not k.startswith("decoder."):
            new_k = f"model.{k}"
        else:
            new_k = k

        sanitized_weights[new_k] = v

    return sanitized_weights

class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.repeats = n_heads // n_kv_heads
        self.head_dim = head_dim = args.head_dim
        self.layer_idx = layer_idx

        self.scale = args.query_pre_attn_scalar**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        layer_type = args.layer_types[layer_idx] if args.layer_types else None
        self.is_sliding = layer_type == "sliding_window"

        base = (
            args.rope_theta
        )

        self.rope = nn.RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=base,
        )
        
        # softcapping support
        self.attn_logit_softcapping = args.attn_logit_softcapping


    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None
    ) -> mx.array:
        B, L, _ = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = self.rope(queries)
        keys = self.rope(keys)

        if self.attn_logit_softcapping is None:
            output = mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=self.scale, mask=mask
            )
        else:
            queries = queries * self.scale
            
            attn_weights = mx.matmul(queries, keys.transpose(0, 1, 3, 2))
            cap = self.attn_logit_softcapping
            attn_weights = mx.tanh(attn_weights / cap) * cap
            
            if mask is not None:
                attn_weights = attn_weights + mask

            attn_weights = mx.softmax(attn_weights, axis=-1)
            output = mx.matmul(attn_weights, values)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))
    
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.pre_self_attn_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_self_attn_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.dropout = nn.Dropout(args.dropout_rate)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None
    ) -> mx.array:
        r = x
        h = self.self_attn(self.pre_self_attn_layernorm(x), mask)
        h = self.post_self_attn_layernorm(h)
        h = r + self.dropout(h)
        r = h
        h= self.mlp(self.pre_feedforward_layernorm(h))
        out = self.post_feedforward_layernorm(h)
        out = r + self.dropout(out)
        return (out,)

class T5GemmaEncoder(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout_rate)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _update_attention_mask(self, attention_mask: Optional[mx.array] = None, dtype=None):
        """
        Creates a causal mask and combines it with the padding mask.
        """

        B, L = attention_mask.shape
        window_size = self.config.sliding_window
        indices = mx.arange(L)
        row = indices[:, None]
        col = row.T

        if not self.config.is_causal:
            mask_base = mx.zeros((L, L), dtype=mx.bool_) # All False (visible)
            
            # Sliding Window Logic for Bidirectional:
            # Valid if abs(row - col) < window
            # Mask if distance >= window
            dist = mx.abs(row - col)
            mask_window_violator = dist >= window_size
        
        # In practice, T5Gemma Encoder is always non-causal but we keep the code 
        # from other models for consistency
        else:
            # Causal: Standard triangular mask
            mask_future = col > row
            mask_base = mask_future
            
            # Sliding Window Logic for Causal:
            # Valid if row - col < window (and not future)
            # Mask if (row - col) >= window
            mask_past = (row - col) >= window_size
            mask_window_violator = mask_past

        global_mask = mx.where(mask_base, -1e9, 0.0).astype(dtype)
        sliding_mask_bool = mask_base | mask_window_violator
        sliding_mask = mx.where(sliding_mask_bool, -1e9, 0.0).astype(dtype)

        # Padding Mask
        if attention_mask is not None:
            # Reshape padding mask from (B, L) to (B, 1, 1, L) to be broadcastable
            padding_mask = attention_mask[:, None, None, :]
            additive_padding = mx.where(padding_mask == 0, -1e9, 0.0).astype(dtype)

            global_mask = global_mask + additive_padding
            sliding_mask = sliding_mask + additive_padding

        return global_mask, sliding_mask

    def __call__(
        self, 
        input_ids: mx.array, 
        attention_mask: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = False,
        position_ids: Optional[mx.array] = None,
        return_dict: Optional[bool] = True
    ):
        hidden_states = self.embed_tokens(input_ids)
        model_dtype = hidden_states.dtype

        # normalizer
        hidden_states *= mx.array(self.config.hidden_size**0.5, model_dtype)

        hidden_states = self.dropout(hidden_states)

        global_mask, sliding_window_mask = self._update_attention_mask(
            attention_mask,
            dtype=model_dtype
        )

        for i, layer in enumerate(self.layers):
            is_global = self.config.layer_types[i] == "full_attention"
            layer_mask = global_mask if is_global else sliding_window_mask
            layer_outputs = layer(hidden_states, layer_mask)
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return {
            "last_hidden_state": hidden_states,
        }
    

class T5GemmaClassificationHead(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(p=config.classifier_dropout_rate)
        self.out_proj = nn.Linear(
            config.hidden_size, config.num_labels
        )
        self.soft_cap = config.final_logit_softcapping

    def __call__(self, hidden_states: mx.array) -> mx.array:
        logits = self.out_proj(self.dropout(hidden_states))
        if self.soft_cap is not None:
            logits = mx.tanh(logits / self.soft_cap) * self.soft_cap
        return logits


class Model(RaclateBaseModel):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = T5GemmaEncoder(config)

         # transformer architecture name for compatibility
        self.hf_transformers_arch = ""

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

        # normalized features
        text_embeds = mean_pooling(last_hidden_state, attention_mask)
        text_embeds = normalize_embeddings(text_embeds)

        if not return_dict:
            return (text_embeds, last_hidden_state) 

        return {
            "embeddings": text_embeds, # normalized embeddings
            "last_hidden_state": last_hidden_state,
        }

    def sanitize(self, weights):
        sanitized = _sanitize_backbone(weights)

        sanitized_weights = {}
        for k, v in sanitized.items():
            if not k.startswith("model."):
                continue
            sanitized_weights[k] = v

        return sanitized_weights
   
    
class ModelForSentenceSimilarity(RaclateBaseModel):
    """
    Computes similarity scores between input sequences and reference sentences.
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = T5GemmaEncoder(config)

    def _call_model(
        self,
        input_ids: mx.array, 
        position_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None, 
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ):
        out = self.model(input_ids, attention_mask)
        last_hidden_state = (
            out["last_hidden_state"] if isinstance(out, dict) else out[0]
        )

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
            position_ids=position_ids,
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
                position_ids=position_ids, ### ?
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
                negative_attention_mask,
            )
         
        if not return_dict:
            return (loss, similarities, embeddings)
            
        return {
            "loss": loss,
            "similarities": similarities,  # [batch_size, num_references]
            "embeddings": embeddings,  # [batch_size, hidden_size]
        }
    
    def sanitize(self, weights):
        sanitized = _sanitize_backbone(weights)

        sanitized_weights = {}
        for k, v in sanitized.items():
            if not k.startswith("model."):
                continue
            sanitized_weights[k] = v

        return sanitized_weights

class ModelForSentenceTransformers(ModelForSentenceSimilarity):
    """
    Extends ModelForSentenceSimilarity to provide embeddings for input sequences.
    This class sanitizes typical sentence transformers weights to align with the T5Gemma model.
    """
    def __init__(self, config: ModelArgs):
        super().__init__(config)

    
class ModelForSequenceClassification(RaclateBaseModel):
    """
    Computes sequence classification probabilities for input sequences.
    Sanitization aligns typical BERT weights with HF's Qwen3ForSequenceClassification architecture.

    NOTE : regression and binary classification not tested.
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.is_regression = config.is_regression
        
        self.model = T5GemmaEncoder(config)

        ### The HF architecture Gemma3ForSequenceClassification 
        ### does not have head and drop
        #### and uses 'score' as the final layer name
        # self.head = Gemma3PredictionHead(config)
        # self.drop = nn.Dropout(p=config.classifier_dropout)

        self.score = T5GemmaClassificationHead(config)

        self.hf_transformers_arch = "T5GemmaForSequenceClassification"
    
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

        # normalized features
        text_embeds = mean_pooling(last_hidden_state, attention_mask)

        ### The HF architecture T5GemmaForSequenceClassification 
        logits = self.score(text_embeds)

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

        sanitized = _sanitize_backbone(weights)

        sanitized_weights = {}
        for k, v in sanitized.items():
            if not k.startswith("model.") and not k.startswith("score."):
                continue
            sanitized_weights[k] = v

        return sanitized_weights
    
class ModelForMaskedLM(RaclateBaseModel):
    """
    Computes masked language modeling (MLM) loss for input sequences.
    """
    def __init__(self, config : ModelArgs):
        super().__init__()
        self.config = config
        if not config.is_causal:
            raise ValueError("ModelForMaskedLM requires bidirectional attention.")
        self.model = T5GemmaEncoder(config)
        self.head = T5GemmaClassificationHead(config) 
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=config.decoder_bias
        )

        # transformers has no MaskedLM class for T5Gemma

        # We explicitly call tie_weights to ensure logic is set up, 
        # though standard loading overwrites this unless sanitized correctly.
        self.tie_weights()
    
    def tie_weights(self):
        self.decoder.weight = self.model.embed_tokens.weight
    
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def get_output_embeddings(self):
        return self.decoder
    
    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
        self.tie_weights()  # Re-tie weights after setting new embeddings
    
    def set_output_embeddings(self, new_embeddings):
        self.decoder = new_embeddings
        self.tie_weights()  # Re-tie weights after setting new decoder
        
    def __call__(
        self,
        input_ids,
        attention_mask: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Dict:
        
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape 
            attention_mask = mx.ones((batch_size, seq_len)) ###  updated via _update_attention_mask() in the model

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        last_hidden_state = outputs["last_hidden_state"] if return_dict else outputs[0]
        logits = self.head(last_hidden_state)  
        logits = self.decoder(logits)
        
        loss = None
        if self.training and labels is not None :  
            if getattr(self.config, "sparse_prediction", False):
                # Flatten labels and predictions
                flat_labels = labels.reshape(-1)
                flat_predictions = logits.reshape(-1, logits.shape[-1])
                
                # Filter out non-masked tokens
                ignore_index = getattr(self.config, "sparse_pred_ignore_index", -100)
                mask_tokens = flat_labels != ignore_index
                
                # Only compute loss on masked tokens
                masked_predictions = flat_predictions[mask_tokens]
                masked_labels = flat_labels[mask_tokens]
                
                loss = nn.losses.cross_entropy(
                    masked_predictions,
                    masked_labels,
                    reduction='mean'
                )
            else:
                # Standard loss computation on all tokens
                loss = nn.losses.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    reduction='mean'
                )
            
        if not return_dict:
            return [loss, logits, outputs[1:]]
            
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.get("hidden_states", None),
        }
    
    def sanitize(self, weights):

        sanitized_weights = _sanitize_backbone(weights)

        # Specific adjustments for MLM
        final_weights = {}
        for k, v in sanitized_weights.items():
            if not k.startswith("model.") and not k.startswith("head.") and not k.startswith("decoder."):
                continue
                
            # Handle Weight Tying for loading:
            if k == "model.embed_tokens.weight" and "decoder.weight" not in weights:
                final_weights["decoder.weight"] = v
                
            final_weights[k] = v

        return final_weights


class ModelForTokenClassification(RaclateBaseModel):
    """
    Computes token classification probabilities for input sequences.

    NOTE: untested for now
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config       
        self.num_labels = config.num_labels

        self.model = T5GemmaEncoder(config)
        self.score = T5GemmaClassificationHead(config)

        self.hf_transformers_arch = "T5GemmaForTokenClassification"
    
    def __call__(
        self,
        input_ids,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Dict:
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape
            attention_mask = mx.ones((batch_size, seq_len))

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs["last_hidden_state"] if return_dict else outputs[0]
        
        # Apply prediction head, dropout, and classification layer to each token

        logits = self.score(last_hidden_state)

        # Process logits for inference
        processed_logits = mx.softmax(logits, axis=-1)

        loss = None
        if labels is not None:
            # Compute token classification loss
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, self.num_labels),
                labels.reshape(-1)
            )

        if not return_dict:
            return [loss, processed_logits, outputs[1:]]

        return {
            "loss": loss,
            "probabilities": processed_logits,
            "hidden_states": outputs.get("hidden_states", None),
        }
    
    def sanitize(self, weights):

        sanitized = _sanitize_backbone(weights)

        sanitized_weights = {}
        for k, v in sanitized.items():
            if not k.startswith("model.") and not k.startswith("score."):
                continue
            sanitized_weights[k] = v

        return sanitized_weights