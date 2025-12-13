# Copyright © 2023-2024 Apple Inc.
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Literal

import mlx.core as mx
import mlx.nn as nn
from .base import (
    BaseModelArgs,
    RaclateBaseModel,
    last_token_pooling,
    normalize_embeddings,
    compute_similarity,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        assert config.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads

        head_dim = config.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(dims=head_dim, base=config.rope_theta)

    def __call__(
        self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        B, L, D = hidden_states.shape

        queries, keys, values = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )

        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )

        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = self.rope(queries)
        keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=attention_mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        hidden_states = self.o_proj(output)

        return (hidden_states,)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.config = config

    def __call__(
        self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        attention_output = self.self_attn(
            self.input_layernorm(hidden_states), attention_mask
        )
        hidden_states = hidden_states + attention_output[0]
        mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = mlp_output + hidden_states
        return (hidden_states,)


class Qwen3Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TransformerBlock(config=config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

        attention_mask = self._update_attention_mask(
            attention_mask=attention_mask,
            dtype=model_dtype
        )

        for layer in self.layers:
            layer_outputs = layer(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        return {
            "last_hidden_state": hidden_states,
        }

# Not used for now
class Qwen3PredictionHead(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(
            config.hidden_size, config.hidden_size, config.classifier_bias
        )
        self.act = nn.GELU(approx="precise")
        self.norm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.norm(self.act(self.dense(hidden_states)))


class Model(RaclateBaseModel):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type # not used for now (placeholder)
        self.model = Qwen3Model(config)

        # transformer architecture name for compatibility
        self.hf_transformers_arch = "Qwen3ForCausalLM"

    def __call__(
            self, 
            input_ids: mx.array, 
            position_ids: Optional[mx.array] = None,
            attention_mask: Optional[mx.array] = None, 
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = True,
    ) -> Dict:
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

        # pooling for AR models such as Qwen3 leverages the last token
        pooled_embeddings = last_token_pooling(last_hidden_state, attention_mask)
        text_embeds = normalize_embeddings(pooled_embeddings)

        if not return_dict:
            return (text_embeds, last_hidden_state) 

        return {
            "embeddings": text_embeds, # normalized embeddings
            "last_hidden_state": last_hidden_state,
        }

    def sanitize(self, weights):
        # no need for lm_head.weight in Qwen3 for embedding models
        sanitized_weights = {}
        for k, v in weights.items():
            # Filter out the language model head, which is not used for embeddings.
            if "lm_head.weight" in k:
                continue

            new_key = f"model.{k}"
            sanitized_weights = {}

        for key, value in weights.items():
            # Skip language model head weights (not used for embeddings)
            if "lm_head.weight" in key:
                continue

            # Handle different checkpoint formats
            new_key = key

            # Map common parameter naming patterns
            if key.startswith("transformer."):
                # Some checkpoints use "transformer." prefix
                new_key = key.replace("transformer.", "model.")
            elif key.startswith("model."):
                # Already has correct prefix
                new_key = key
            elif not key.startswith("model.") and "." in key:
                # Add model prefix for transformer parameters
                new_key = f"model.{key}"
            else:
                # Keep as is for other parameters
                new_key = key

            sanitized_weights[new_key] = value

        return sanitized_weights


class ModelForSentenceSimilarity(Model):
    """
    Computes similarity scores between input sequences and reference sentences.
    """
    def __init__(self, config):
        super().__init__(config)
    
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
        # Get embeddings for input batch
        batch_outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True
        )
        embeddings = batch_outputs["embeddings"]  # [batch_size, hidden_size]

        loss = None

        if reference_input_ids is not None:
        
            # Get embeddings for reference sentences
            ref_outputs = super().__call__(
                input_ids=reference_input_ids,
                attention_mask=reference_attention_mask,
                position_ids=position_ids, ### ?
                return_dict=True
            )
            reference_embeddings = ref_outputs["embeddings"]  # [num_references, hidden_size]

            # MSE loss between computed similarities and target scores
            if similarity_scores is not None:
                assert reference_embeddings.shape[0] == input_ids.shape[0], "Number of references must match batch size for paired training"
                assert similarity_scores.shape[0] == input_ids.shape[0], "Number of similarity scores must match batch size for paired training"
                # No matmul here, we only care about Query i vs Ref i
                pairwise_sims = mx.sum(embeddings * reference_embeddings, axis=-1)
                    
                # Ensure scores match shape
                if len(similarity_scores.shape) > 1:
                    similarity_scores = similarity_scores.flatten()
                    
                loss = nn.losses.mse_loss(pairwise_sims, similarity_scores)
                similarities = pairwise_sims
            
            # Cross-entropy loss [for triplet training with hard negatives]
            else:
                if negative_input_ids is not None:
                    assert reference_embeddings.shape[0] == input_ids.shape[0], "Number of references must match batch size for paired training"
                    assert negative_input_ids.shape[0] == input_ids.shape[0], "Number of negatives must match batch size for triplet training"
                    # Embed Negative
                    neg_outputs = super().__call__(
                        input_ids=negative_input_ids, 
                        attention_mask=negative_attention_mask, 
                        return_dict=True
                    )
                    neg_embeddings = neg_outputs["embeddings"]
                    
                    # Stack Candidates: [Positives, Negatives]
                    candidates = mx.concatenate([reference_embeddings, neg_embeddings], axis=0) # Shape: [2 * batch, hidden]
                    similarities = compute_similarity(embeddings, candidates)
                
                else:
                    similarities = compute_similarity(embeddings, reference_embeddings)

                scale = 20.0
                scores = similarities * scale
                
                labels = mx.arange(embeddings.shape[0])
                
                loss = nn.losses.cross_entropy(scores, labels)
    
        else:
            similarities = None
            loss = None
            
        if not return_dict:
            return (loss, similarities, embeddings)
            
        return {
            "loss": loss,
            "similarities": similarities,  # [batch_size, num_references]
            "embeddings": embeddings,  # [batch_size, hidden_size]
        }

class ModelForSentenceTransformers(ModelForSentenceSimilarity):
    """
    Extends ModelForSentenceSimilarity to provide embeddings for input sequences.
    This class sanitizes typical sentence transformers weights to align with the Qwen3 model.
    """
    def __init__(self, config: ModelArgs):
        super().__init__(config)

    def sanitize(self, weights):
        """Convert sentence transformer weights to Qwen3 format."""
        sanitized_weights = {}
        
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            else:
                new_key = "model." + k
                sanitized_weights[new_key] = v
        return sanitized_weights

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
        
        self.model = Qwen3Model(config)

        ### The HF architecture Qwen3ForSequenceClassification 
        ### does not have head and drop
        #### and uses 'score' as the final layer name
        # self.head = Qwen3PredictionHead(config)
        # self.drop = nn.Dropout(p=config.classifier_dropout)

        self.score = nn.Linear(
            config.hidden_size, 
            config.num_labels, 
            bias=False
        ) 

        self.hf_transformers_arch = "Qwen3ForSequenceClassification"
    
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

        # pooling for AR models such as Qwen3 leverages the last token
        pooled = last_token_pooling(last_hidden_state, attention_mask)

        ### The HF architecture Qwen3ForSequenceClassification 
        ### does not have head and drop
        #### and uses 'score' as the final layer name
        # pooled = self.head(pooled)
        # pooled = self.drop(pooled)
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
        # TBC (can't be tested without checkpoints)
        print("Sanitizing weights for ModelForSequenceClassification...")
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            elif "lm_head" in k:
                continue
            # elif "score" in k:
            #     continue # TBC (the only checkpoint I have access to has both classifier and score layers)

            new_key = k

            # Ensure everything maps to self.model.[transformer_parts]
            if k.startswith("model."):
                # Standard HF Qwen format: model.layers...
                new_key = k 
            elif k.startswith("transformer."):
                # Alternative format
                new_key = k.replace("transformer.", "model.")
            elif not k.startswith("score") and not k.startswith("classifier"):
                # If weights are "flat" (e.g. layers.0...), prefix with model.
                new_key = f"model.{k}"
            
            # elif k.startswith("classifier."):
            #     print(f"Sanitizing classifier key: {k}")
            #     # replace classifier with score to match HF naming convention 
            #     # (ensuring compatibility with Qwen3ForSequenceClassification)
            #     new_key = k.replace("classifier.", "score.")
            #     print(f"Sanitized classifier key to: {new_key}")

            
            sanitized_weights[new_key] = v  
            
        return sanitized_weights

# TokenClassification and MaskedLM not implemented for now AR models such as Qwen3
# Attempting to train pretrained weights would be catastrophic 