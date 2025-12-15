import inspect
from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn
from typing import Optional

class RaclateBaseModel(nn.Module):
    """Base class for Raclate models."""
    def __init__(self):
        super().__init__()
    
    def get_hf_transformers_arch(self):
        return self.hf_transformers_arch if hasattr(self, "hf_transformers_arch") else None

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
    negative_attention_mask: Optional[mx.array] = None
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
                neg_outputs = call_model(negative_input_ids, negative_attention_mask)
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
