################################################################
# THIS IS A PLACEHOLDER FILE FOR THE GRANITE EMBEDDING MODEL.
# BY DEFAULT GRANITE MODELS SHOULD USE MODERNBERT
# SEE utils/utils.py to route granite models to this file
###################################################################

from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx

from .base import (
    RaclateBaseModel,
    compute_similarity_and_loss,
    normalize_embeddings,
    mean_pooling,
)
from .modernbert import (
    ModelArgs as ModernBertArgs,
    ModernBertModel,
)


@dataclass
class ModelArgs(ModernBertArgs):
    architectures: List[str] = field(default_factory=lambda: ["ModernBertModel"])
    model_type: str = "granite_embedding"
    hidden_activation: str = "silu"
    hidden_size: int = 384
    intermediate_size: int = 1536
    num_hidden_layers: int = 12
    vocab_size: int = 180000
    classifier_pooling: str = "cls"


def _sanitize_modernbert_backbone(weights):
    sanitized_weights = {}
    for key, value in weights.items():
        if "position_ids" in key:
            continue
        if key.startswith("model."):
            sanitized_weights[key] = value
        elif key.startswith(("embeddings.", "layers.", "final_norm.")):
            sanitized_weights[f"model.{key}"] = value
    return sanitized_weights


class Model(RaclateBaseModel):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model = ModernBertModel(config)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ):
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape
            attention_mask = mx.ones(
                (batch_size, seq_len),
                dtype=self.model.embeddings.tok_embeddings.weight.dtype,
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        last_hidden_state = outputs["last_hidden_state"]

        # Pooling based on config
        if self.config.classifier_pooling == "cls":
            pooled = last_hidden_state[:, 0]
        elif self.config.classifier_pooling == "mean":                
            pooled = mean_pooling(last_hidden_state, attention_mask)

        embeddings = normalize_embeddings(pooled)

        if not return_dict:
            return (embeddings, last_hidden_state)

        return {
            "embeddings": embeddings,
            "last_hidden_state": last_hidden_state,
            "hidden_states": outputs.get("hidden_states"),
        }

    def sanitize(self, weights):
        return _sanitize_modernbert_backbone(weights)


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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        last_hidden_state = (
            outputs["last_hidden_state"] if isinstance(outputs, dict) else outputs[0]
        )

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
                dtype=self.model.embeddings.tok_embeddings.weight.dtype)
            
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
            similarities, loss = compute_similarity_and_loss(
                self.config,
                input_ids,
                embeddings,
                ref_outputs["embeddings"],
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

    def sanitize(self, weights):
        return _sanitize_modernbert_backbone(weights)


class ModelForSentenceTransformers(ModelForSentenceSimilarity):
    pass

class ModelForSequenceClassification(RaclateBaseModel):
    def __init__(self, config : ModelArgs):
        raise NotImplementedError("Granite does not support sequence classification yet.")
    
class ModelForMaskedLM(RaclateBaseModel):
    def __init__(self, config : ModelArgs):
        raise NotImplementedError("Granite does not support masked language modeling yet.")
    
class ModelForTokenClassification(RaclateBaseModel):
    def __init__(self, config : ModelArgs):
        raise NotImplementedError("Granite does not support token classification yet.")
