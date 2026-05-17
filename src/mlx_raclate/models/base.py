import inspect
from dataclasses import dataclass
import math
import mlx.core as mx
import mlx.nn as nn
from typing import Any, Dict, Optional, Tuple

class RaclateBaseModel(nn.Module):
    """Base class for Raclate models."""
    def __init__(self):
        super().__init__()
    
    def get_hf_transformers_arch(self):
        return self.hf_transformers_arch if hasattr(self, "hf_transformers_arch") else None

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if new_num_tokens != config.vocab_size.
        """
        old_embeddings = self.get_input_embeddings()
        if old_embeddings is None:
            raise ValueError("Model does not support get_input_embeddings")

        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.weight.shape
        
        if new_num_tokens == old_num_tokens:
            return old_embeddings

        # Create new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        
        # Initialize new weights (e.g. Normal)
        new_embeddings.weight = mx.random.normal(shape=(new_num_tokens, old_embedding_dim)) * 0.02
        
        # We copy up to the min size to handle both expansion and shrinking (though usually expansion)
        n = min(old_num_tokens, new_num_tokens)
       
        # Combine old relevant weights with new random weights for the extension
        combined_weight = mx.concatenate([
            old_embeddings.weight[:n],
            new_embeddings.weight[n:]
        ], axis=0) if new_num_tokens > old_num_tokens else old_embeddings.weight[:n]
        
        new_embeddings.weight = combined_weight

        self.set_input_embeddings(new_embeddings)
        
        # Update config if present
        if hasattr(self, "config"):
            self.config.vocab_size = new_num_tokens
            
        return new_embeddings

@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )



class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dims: int,
        freqs: Optional[mx.array] = None,
        base: Optional[float] = None,
        scale: float = 1.0,
        mscale: float = 1.0,
        traditional: bool = True,
    ):
        super().__init__()
        self.dims = dims
        self._freqs = None if freqs is None else (freqs,)
        self.base = base
        self.scale = scale
        self.mscale = mscale
        self.traditional = traditional

    def _default_freqs(self) -> mx.array:
        if self.base is None:
            raise ValueError("base must be set when explicit position_ids are used")
        return self.base ** (mx.arange(0, self.dims, 2, dtype=mx.float32) / self.dims)

    def _apply_with_position_ids(
        self,
        hidden_states: mx.array,
        position_ids: mx.array,
    ) -> mx.array:
        if self.dims % 2 != 0:
            raise ValueError("RoPE dimensions must be even when explicit position_ids are used")

        freqs = self._freqs[0] if self._freqs is not None else self._default_freqs()
        inv_freqs = 1.0 / freqs
        positions = position_ids.astype(mx.float32) * self.scale
        angles = mx.expand_dims(positions, -1) * inv_freqs

        if len(position_ids.shape) == 1:
            while len(angles.shape) < len(hidden_states.shape):
                angles = mx.expand_dims(angles, 0)
        elif len(position_ids.shape) == 2 and len(hidden_states.shape) == 4:
            angles = angles[:, None, :, :]
        else:
            while len(angles.shape) < len(hidden_states.shape):
                angles = mx.expand_dims(angles, 1)

        cos = mx.cos(angles).astype(hidden_states.dtype)
        sin = mx.sin(angles).astype(hidden_states.dtype)

        rotary_states = hidden_states[..., : self.dims]
        pass_states = hidden_states[..., self.dims :]
        if self.traditional:
            even_states = rotary_states[..., 0::2]
            odd_states = rotary_states[..., 1::2]

            rotated = mx.stack(
                [
                    even_states * cos - odd_states * sin,
                    even_states * sin + odd_states * cos,
                ],
                axis=-1,
            ).reshape(rotary_states.shape)
        else:
            half_dims = self.dims // 2
            first_half = rotary_states[..., :half_dims]
            second_half = rotary_states[..., half_dims:self.dims]
            rotated = mx.concatenate(
                [
                    first_half * cos - second_half * sin,
                    second_half * cos + first_half * sin,
                ],
                axis=-1,
            )

        if pass_states.shape[-1] == 0:
            return rotated
        return mx.concatenate([rotated, pass_states], axis=-1)

    def __call__(
        self,
        hidden_states: mx.array,
        offset: int = 0,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        x = hidden_states
        if self.mscale != 1.0:
            scaled = x[..., : self.dims] * self.mscale
            x = (
                scaled
                if self.dims == x.shape[-1]
                else mx.concatenate([scaled, x[..., self.dims :]], axis=-1)
            )
        if position_ids is not None:
            return self._apply_with_position_ids(x, position_ids)
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=self.base,
            scale=self.scale,
            offset=offset,
            freqs=None if self._freqs is None else self._freqs[0],
        )


class SwiGLU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        bias: bool = False,
    ):
        super().__init__()
        self.w12 = nn.Linear(input_dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, output_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.w12(x), 2, axis=-1)
        return self.w3(nn.silu(gate) * up)

def compute_similarity(query_embeddings: mx.array, reference_embeddings: mx.array) -> mx.array:
        """Computes cosine similarity between query embeddings and reference embeddings.
        
        Args:
            query_embeddings: Shape [batch_size, hidden_size]
                These are the embeddings we want to classify/compare - already normalized
            reference_embeddings: Shape [num_references, hidden_size]
                These are our label descriptions or comparison sentences - already normalized
            
        Returns:
            Similarity matrix of shape [batch_size, num_references]
            Each row contains similarities between one query and all references
        """
        # Compute similarities - results in [batch_size, num_references]
        # Each row contains similarities between one input and all references
        similarities = mx.matmul(query_embeddings, reference_embeddings.T)
        
        return similarities

def mean_pooling(token_embeddings: mx.array, attention_mask: mx.array):
    input_mask_expanded = mx.expand_dims(attention_mask, -1)
    input_mask_expanded = mx.broadcast_to(
        input_mask_expanded, token_embeddings.shape
    ).astype(mx.float32)
    
    sum_embeddings = mx.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = mx.maximum(mx.sum(input_mask_expanded, axis=1), 1e-9)
    
    return sum_embeddings / sum_mask


def last_token_pooling(
    last_hidden_states: mx.array, attention_mask: Optional[mx.array] = None
) -> mx.array:
    """
    Last token pooling, compatible with MLX compilation/grad

    Args:
        last_hidden_states: Hidden states from the model, shape (batch_size, seq_len, hidden_size)
        attention_mask: Attention mask, shape (batch_size, seq_len). If None, uses last position.

    Returns:
        Pooled embeddings, shape (batch_size, hidden_size)
    """
    if attention_mask is None:
        return last_hidden_states[:, -1]

    B, S, _ = last_hidden_states.shape
    indices = mx.arange(S)

    # Only keep the unpadded tokens
    masked_indices = indices * attention_mask.astype(indices.dtype)

    # Find the last valid index (max index) for each batch item
    last_token_indices = masked_indices.max(axis=1)

    batch_indices = mx.arange(B)
    
    # Select specific [batch, token] pairs
    return last_hidden_states[batch_indices, last_token_indices]


def normalize_embeddings(embeddings, p=2, axis=-1, keepdims=True, eps=1e-9):
    return embeddings / mx.maximum(
        mx.linalg.norm(embeddings, ord=p, axis=axis, keepdims=keepdims), eps
    )

def compute_late_interaction_scores(Q, D):
    """
    MaxSim: sum_i(max_j(Q_i . D_j))
    Args:
        Q: Query embeddings [B_q, L_q, Dim]
        D: Doc embeddings [B_d, L_d, Dim] 
    Note: If calculating loss with in-batch negatives, shapes might vary.
    """
    # (B, L_q, Dim) @ (B, Dim, L_d) -> (B, L_q, L_d)
    # This assumes pairwise (Query[i] vs Doc[i]).
    sim_matrix = Q @ D.transpose(0, 2, 1)
    max_scores = mx.max(sim_matrix, axis=-1) # (B, L_q)
    return mx.sum(max_scores, axis=-1) # (B,)


def compute_similarity_and_loss(
    config,
    input_ids: mx.array,
    embeddings: mx.array,
    reference_embeddings: mx.array,
    call_model : callable,
    similarity_scores: Optional[mx.array],
    negative_input_ids: Optional[mx.array] = None,
    negative_attention_mask: Optional[mx.array] = None,
    negative_position_ids: Optional[mx.array] = None,
):
    # MSE loss between computed similarities and target scores
    if similarity_scores is not None:
        assert reference_embeddings.shape[0] == input_ids.shape[0], "Number of references must match batch size for paired training"
        assert similarity_scores.shape[0] == input_ids.shape[0], "Number of similarity scores must match batch size for paired training"
        if config.use_late_interaction:
            pairwise_sims = compute_late_interaction_scores(embeddings, reference_embeddings)
        else:
            # No matmul here, we only care about Query i vs Ref i
            pairwise_sims = mx.sum(embeddings * reference_embeddings, axis=-1)
            
        # Ensure scores match shape
        if len(similarity_scores.shape) > 1:
            similarity_scores = similarity_scores.flatten()
            
        loss = nn.losses.mse_loss(pairwise_sims, similarity_scores)
        similarities = pairwise_sims
    
    # Cross-entropy loss [for triplet training with hard negatives]
    else:
        if config.use_late_interaction:
            # Q: [B, L, D], C: [2B, L, D] (if negatives exist)
            if negative_input_ids is not None:
                neg_outputs = call_model(
                    input_ids=negative_input_ids,
                    attention_mask=negative_attention_mask,
                    position_ids=negative_position_ids,
                    return_dict=True,
                )
                neg_embeddings = neg_outputs["embeddings"]
                candidates = mx.concatenate([reference_embeddings, neg_embeddings], axis=0)
            else:
                candidates = reference_embeddings

            # Manual Broadcasting for Late Interaction cross-batch
            Q_broad = embeddings[:, None, :, :]
            C_broad = candidates[None, :, :, :].transpose(0, 1, 3, 2)
            
            sim_matrix = Q_broad @ C_broad
            
            # Max over Doc length, Sum over Query length
            scores = mx.sum(mx.max(sim_matrix, axis=-1), axis=-1)
            similarities = scores # [B, C]

        else:
            if negative_input_ids is not None:
                assert reference_embeddings.shape[0] == input_ids.shape[0], "Number of references must match batch size for paired training"
                assert negative_input_ids.shape[0] == input_ids.shape[0], "Number of negatives must match batch size for triplet training"
                # Embed Negative
                neg_outputs = call_model(
                    input_ids=negative_input_ids, 
                    attention_mask=negative_attention_mask, 
                    position_ids=negative_position_ids,
                    return_dict=True
                )
                neg_embeddings = neg_outputs["embeddings"]
                
                # Stack Candidates: [Positives, Negatives]
                candidates = mx.concatenate([reference_embeddings, neg_embeddings], axis=0) # Shape: [2 * batch, hidden]

            else:
                candidates = reference_embeddings 
                
            similarities = compute_similarity(embeddings, candidates)

        scale = 20.0
        scores = similarities * scale
        
        labels = mx.arange(embeddings.shape[0])
        
        loss = nn.losses.cross_entropy(scores, labels)

    return similarities, loss
