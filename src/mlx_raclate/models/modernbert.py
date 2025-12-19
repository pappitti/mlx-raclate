from dataclasses import dataclass, field
from typing import Optional, Dict, Literal, Any, List

import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs, 
    RaclateBaseModel, 
    compute_similarity_and_loss, 
    mean_pooling, 
    normalize_embeddings
)

""" NOTE : This implementation of ModernBERT excludes all features related to Flash Attention 2, padded/unpadded handling"""

@dataclass
class ModelArgs(BaseModelArgs):
    architectures: List[str] = field(default_factory=lambda: ["ModernBertModel"])
    attention_bias: bool = False
    attention_dropout : float =0.0
    bos_token_id: int = 50281
    cls_token_id: int = 50281
    embedding_dropout : float = 0.0
    eos_token_id : int = 50282
    global_attn_every_n_layers : int = 3
    global_rope_theta : float = 160000.0
    hidden_size: int = 768
    initializer_range : float = 0.02 
    initializer_cutoff_factor: float = 2.0 # relevant for MLX?
    intermediate_size: int = 1152
    local_attention : int =128
    local_rope_theta: float = 10000
    max_position_embeddings: int = 8192
    mlp_bias: bool = False
    mlp_dropout : float = 0.0
    model_type: str = "modernbert"
    norm_bias : bool = False
    norm_eps: float = 1e-05
    num_attention_heads: int = 12
    num_hidden_layers: int = 22
    output_hidden_states: bool = False 
    pad_token_id: int = 50283
    sep_token_id: int = 50282
    vocab_size: int = 50368

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


class ModernBertEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config: ModelArgs):
        super().__init__() 
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias) 
        self.drop = nn.Dropout(p=config.embedding_dropout)

    def __call__(self, input_ids):
        embeddings = self.tok_embeddings(input_ids)
        embeddings = self.norm(embeddings)
        embeddings = self.drop(embeddings)
        return embeddings


class ModernBertMLP(nn.Module):
    """Applies the GLU at the end of each ModernBERT layer.

    Compared to the default BERT architecture, this block replaces class BertIntermediate`
    and class SelfOutput with a single module that has similar functionality.
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.Wi = nn.Linear(config.hidden_size, config.intermediate_size *2, bias=config.mlp_bias)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=config.mlp_dropout)
        self.Wo = nn.Linear(int(config.intermediate_size), config.hidden_size, bias=config.mlp_bias)

    def __call__(self, hidden_states):
        x = self.Wi(hidden_states)
        
        split_dim = x.shape[-1] // 2
        input, gate = x[:, :, :split_dim], x[:, :, split_dim:] # gate : https://arxiv.org/pdf/2002.05202v1
        return self.Wo(self.drop(self.act(input) * gate))


class ModernBertAttention(nn.Module):
    """Performs multi-headed self attention on a batch of unpadded sequences.
    For now, only supports the Scaled Dot-Product Attention (SDPA) implementation.
    """
    def __init__(self, config: ModelArgs, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by num_attention_heads ({config.num_attention_heads})"
            )
        
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.Wqkv = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=config.attention_bias)

        if layer_id % config.global_attn_every_n_layers != 0:
            self.local_attention = (config.local_attention // 2, config.local_attention // 2)
        else:
            self.local_attention = (-1, -1)

        rope_theta = config.global_rope_theta
        if self.local_attention != (-1, -1) and config.local_rope_theta is not None:
            rope_theta = config.local_rope_theta

        self.rotary_emb = nn.RoPE(dims=self.head_dim, base=rope_theta)

        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.out_drop = nn.Dropout(p=config.attention_dropout) if config.attention_dropout > 0.0 else nn.Identity()
        self.pruned_heads = set()
        
    def __call__(
            self, 
            hidden_states, 
            attention_mask = None,
            sliding_window_mask = None,
            **kwargs
        ):
        qkv = self.Wqkv(hidden_states)
        bs = hidden_states.shape[0]
        qkv = mx.reshape(qkv, (bs, -1, 3, self.num_heads, self.head_dim))

        # Get attention outputs using SDPA
        qkv = mx.transpose(
            qkv, [0, 3, 2, 1, 4]
        )  # [batch_size, nheads, 3, seqlen, headdim]
        query, key, value = mx.split(
            qkv, indices_or_sections=3, axis=2
        )  # each [batch_size, nheads, 1, seqlen, headdim]
        query = query.squeeze(2)  # [batch_size, nheads, seqlen, headdim]
        key = key.squeeze(2)  # [batch_size, nheads, seqlen, headdim]
        value = value.squeeze(2)  # [batch_size, nheads, seqlen, headdim]

        # Applying rotary embeddings
        query = self.rotary_emb(query)
        key = self.rotary_emb(key)
        
        # Handling local attention if needed
        if self.local_attention != (-1, -1):
            attention_mask = sliding_window_mask

        # Computing attention using MLX's SDPA
        scale = query.shape[-1] ** -0.5
        attn_output = mx.fast.scaled_dot_product_attention(
            query, key, value,
            scale=scale,
            mask=attention_mask
        )
        
        # Reshaping and apply output projection
        attn_output = mx.transpose(attn_output, [0, 2, 1, 3])
        attn_output = mx.reshape(attn_output, (bs, -1, self.all_head_size))
        
        # Applying output projection and dropout
        hidden_states = self.Wo(attn_output)
        hidden_states = self.out_drop(hidden_states)

        return (hidden_states,)


class ModernBertEncoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.attn = ModernBertAttention(config=config, layer_id=layer_id)
        self.mlp = ModernBertMLP(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

    def __call__(
            self, 
            hidden_states , 
            attention_mask =None, 
            sliding_window_mask = None,
            position_ids  = None,
    ):
        normalized_hidden_states = self.attn_norm(hidden_states)
        attention_output = self.attn( 
            normalized_hidden_states, 
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
        )
        hidden_states = hidden_states + attention_output[0]
        mlp_output = self.mlp(self.mlp_norm(hidden_states))
        hidden_states = hidden_states + mlp_output

        return (hidden_states,)  


class ModernBertModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = [
            ModernBertEncoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.gradient_checkpointing = False 

    def get_input_embeddings(self) -> ModernBertEmbeddings:
        return self.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value

    def _update_attention_mask(self, attention_mask, model_dtype): #TODO: move to base.py ??

        batch_size, seq_len = attention_mask.shape
        neg_inf = -1e4
        
        additive_mask = mx.where(attention_mask == 1, 0.0, neg_inf)
        additive_mask = additive_mask[:, None, None, :]  # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
        
        # Create the causal mask for global attention
        global_attention_mask = mx.broadcast_to(additive_mask, (batch_size, 1, seq_len, seq_len))
        
        # Create position indices for sliding window
        rows = mx.arange(seq_len)
        rows = rows[None, :]  # (1, seq_len)
        # Calculate position-wise distances
        distance = mx.abs(rows - rows.T)  # (seq_len, seq_len)
        
        # Create sliding window mask using mx.where
        window_mask = mx.where(
            distance <= (self.config.local_attention // 2),
            mx.ones_like(distance),
            mx.zeros_like(distance)
        )
        
        # Expand dimensions using None indexing
        window_mask = window_mask[None, None, :, :]  # (1, 1, seq_len, seq_len)
            
        # Broadcast to match batch size
        window_mask = mx.broadcast_to(window_mask, global_attention_mask.shape)
        
        # Create sliding window attention mask
        # Replace non-window positions with large negative value
        sliding_window_mask = mx.where(
            window_mask,
            global_attention_mask,
            neg_inf # if not broadcasted for some reason : neg_inf * mx.ones_like(global_attention_mask)
        )

        # Convert to model_dtype for scaled_dot_product_attention
        global_attention_mask = global_attention_mask.astype(model_dtype)
        sliding_window_mask = sliding_window_mask.astype(model_dtype)
    
        return global_attention_mask, sliding_window_mask

    def __call__(
        self, 
        input_ids, 
        attention_mask = None, # (batch_size, seq_len) see below
        sliding_window_mask = None,
        position_ids = None,
        output_hidden_states = False,
        return_dict = True,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        batch_size, seq_len = input_ids.shape[:2]

        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len)) ### updated with _update_attention_mask() below

        hidden_states = self.embeddings(input_ids)
        model_dtype = hidden_states.dtype

        # get attention mask and sliding window mask
        attention_mask, sliding_window_mask = self._update_attention_mask(
            attention_mask=attention_mask,
            model_dtype=model_dtype
        )

        all_hidden_states = () if output_hidden_states else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
            )
            
            hidden_states = layer_outputs[0]
        
        hidden_states = self.final_norm(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)
        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
        }


### below are the classes for specific pipelines
class Model(RaclateBaseModel):
    """
    Computes embeddings for input sequences using a ModernBERT model.

    Note : sanitization is a hack to align with other models here while downloading weights 
    with the maskedlm config from HF (original modelBert model).
    """ 
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model = ModernBertModel(config)

        # no transformer architecture for embedding model

    def __call__(
        self, 
        input_ids : mx.array, 
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ):
        
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape 
            attention_mask = mx.ones(
                (batch_size, seq_len),
                dtype=self.model.embeddings.tok_embeddings.weight.dtype) ### updated via _update_attention_mask() in the model

        # Get embeddings and encoder outputs as before
        encoder_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states, 
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs["last_hidden_state"] if isinstance(encoder_outputs, dict) else encoder_outputs[0] 
        
        # Pooling based on config
        if self.config.classifier_pooling == "cls":
            pooled = last_hidden_state[:, 0]
        elif self.config.classifier_pooling == "mean":                
            pooled = mean_pooling(last_hidden_state, attention_mask)

        text_embeds = normalize_embeddings(pooled)

        if not return_dict:
            return (text_embeds, last_hidden_state) 

        return {
            "embeddings": text_embeds, # normalized embeddings
            "last_hidden_state": last_hidden_state,
        }
    
    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            if k in ["head.norm.weight", "head.dense.weight", "decoder.bias"]:
                continue
            else:
                sanitized_weights[k] = v
        return sanitized_weights


class ModelForSentenceSimilarity(RaclateBaseModel):
    """
    Handles:
    1. Inference: Generates embeddings and similarity scores (cosine similarity or MaxSim if late interaction is used).
    2. Training (Standard): (Sentence1, Sentence2, Score) -> MSE/Cosine Loss.
    3. Training (Triplets): (Anchor, Positive, Negative) -> MNRL with Hard Negatives (Cross-entropy Loss).
    """
    def __init__(self, config : ModelArgs):
        super().__init__()
        self.config = config
        self.model = ModernBertModel(config)

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
            # Pooling based on config
            if self.config.classifier_pooling == "cls":
                pooled = last_hidden_state[:, 0]
            elif self.config.classifier_pooling == "mean":                
                pooled = mean_pooling(last_hidden_state, attention_mask)
            text_embeds = normalize_embeddings(pooled)

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
                dtype=self.model.embeddings.tok_embeddings.weight.dtype) ### updated via _update_attention_mask() in the model
            
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
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            if not k.startswith("model."):
                continue
            else:
                sanitized_weights[k] = v
        return sanitized_weights

class ModelForSentenceTransformers(ModelForSentenceSimilarity):
    """
    Extends ModelForSentenceSimilarity.
    Handles:
    1. Inference: Generates embeddings and similarity scores (cosine similarity or MaxSim if late interaction is used).
    2. Training (Standard): (Sentence1, Sentence2, Score) -> MSE/Cosine Loss.
    3. Training (Triplets): (Anchor, Positive, Negative) -> MNRL with Hard Negatives (Cross-entropy Loss).
    This class sanitizes typical sentence transformers weights to align with the ModernBERT model.
    """
    def __init__(self, config: ModelArgs):
        super().__init__(config)

    def sanitize(self, weights):
        """Convert sentence transformer weights to ModernBERT format."""
        sanitized_weights = {}
        
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            if not k.startswith("model."):
                new_key = "model." + k
            else:
                new_key = k
            sanitized_weights[new_key] = v
        return sanitized_weights
    

class ModernBertPredictionHead(nn.Module):
    def __init__(self, config : ModelArgs):
        super().__init__()
        self.dense = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        ) ### current HF checkpoint does not have bias for the dense layer
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )

    def __call__(self, hidden_states):
        return self.norm(self.act(self.dense(hidden_states)))


class ModelForMaskedLM(RaclateBaseModel):
    """
    Computes masked language modeling (MLM) loss for input sequences.
    """
    def __init__(self, config : ModelArgs):
        super().__init__()
        self.config = config
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config) ## no bias for this in the current HF checkpoint
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=config.decoder_bias
        )

        # transformer architecture name for compatibility
        self.hf_transformers_arch = "ModernBertForMaskedLM" 

        # Tie weights ### does not seem to work (sanitizing the weights to enforce weight tying)
        self.tie_weights()
    
    def tie_weights(self):
        embedding_layer = self.model.get_input_embeddings()
        self.decoder.weight = embedding_layer.weight
    
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
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            if k == "model.embeddings.tok_embeddings.weight":
                ### going around the weight tying issue. TODO : improve this
                sanitized_weights["decoder.weight"] = v
                sanitized_weights[k] = v
            else:
                sanitized_weights[k] = v
        return sanitized_weights
    

class ModelForSequenceClassification(RaclateBaseModel):
    """
    Computes sequence classification probabilities for input sequences.
    Sanitization aligns typical BERT weights with the ModernBERT model.

    NOTE : binary classification not tested.
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.is_regression = config.is_regression
        
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.drop = nn.Dropout(p=config.classifier_dropout)
        self.classifier = nn.Linear(
            config.hidden_size, 
            config.num_labels, 
        ) 

        # transformer architecture name for compatibility
        self.hf_transformers_arch = "ModernBertForSequenceClassification"
    
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
            attention_mask = mx.ones((batch_size, seq_len))

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs["last_hidden_state"] if return_dict else outputs[0]

        # Pooling strategy
        if self.config.classifier_pooling == "cls":
            pooled = last_hidden_state[:, 0]
        elif self.config.classifier_pooling == "mean":
            pooled = mean_pooling(last_hidden_state, attention_mask)

        # Apply head, dropout and classifier
        pooled = self.head(pooled)
        pooled = self.drop(pooled)
        logits = self.classifier(pooled)

        # Process logits for inference
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
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            if k in ["decoder.bias"]:
                ### this is the hack
                continue
            elif k.startswith("bert"):
                # Handle legacy BERT naming if needed
                new_k = k.replace("bert.", "model.")
                sanitized_weights[new_k] = v
            else:
                sanitized_weights[k] = v
        return sanitized_weights
    
class ModelForTokenClassification(RaclateBaseModel):
    """
    Computes token classification probabilities for input sequences.

    NOTE: untested for now
    TODO : https://huggingface.co/disham993/electrical-ner-ModernBERT-base
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config       
        self.num_labels = config.num_labels

        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.drop = nn.Dropout(p=config.classifier_dropout)
        self.classifier = nn.Linear(
            config.hidden_size, 
            config.num_labels, 
            # bias=config.classifier_bias
        ) 

        # transformer architecture name for compatibility
        self.hf_transformers_arch = "ModernBertForTokenClassification"


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
        sequence_output = self.head(last_hidden_state)
        sequence_output = self.drop(sequence_output)
        logits = self.classifier(sequence_output)

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
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            else:
                sanitized_weights[k] = v
        return sanitized_weights