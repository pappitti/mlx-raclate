import inspect
from dataclasses import dataclass
import mlx.core as mx
from typing import Optional


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
    Last token pooling implementation

    Args:
        last_hidden_states: Hidden states from the model, shape (batch_size, seq_len, hidden_size)
        attention_mask: Attention mask, shape (batch_size, seq_len). If None, uses last position.

    Returns:
        Pooled embeddings, shape (batch_size, hidden_size)
    """
    if attention_mask is None:
        return last_hidden_states[:, -1]

    # Check if we have left padding (all sequences end with valid tokens)
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        # Find the last valid token position for each sequence
        sequence_lengths = attention_mask.sum(axis=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[mx.arange(batch_size), sequence_lengths]


def normalize_embeddings(embeddings, p=2, axis=-1, keepdims=True, eps=1e-9):
    return embeddings / mx.maximum(
        mx.linalg.norm(embeddings, ord=p, axis=axis, keepdims=keepdims), eps
    )
