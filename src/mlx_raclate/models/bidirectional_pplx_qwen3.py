from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import (
    RaclateBaseModel,
    compute_similarity_and_loss,
    mean_pooling,
    normalize_embeddings,
)
from .qwen3 import (
    ModelArgs as Qwen3Args,
    Qwen3Model,
    _sanitize_backbone,
)


@dataclass
class ModelArgs(Qwen3Args):
    architectures: List[str] = field(default_factory=lambda: ["PPLXQwen3Model"])
    model_type: str = "bidirectional_pplx_qwen3"
    vocab_size: int = 151936
    rope_theta: float = 1000000.0
    use_bidirectional_attention: bool = True


class PPLXQwen3Model(Qwen3Model):
    def _update_attention_mask(
        self,
        attention_mask: Optional[mx.array] = None,
        dtype=None,
    ):
        if attention_mask is None:
            return None

        padding_mask = attention_mask[:, None, None, :]
        return mx.where(padding_mask == 0, -1e9, 0.0).astype(dtype)


class Model(RaclateBaseModel):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model = PPLXQwen3Model(config)
        self.hf_transformers_arch = "PPLXQwen3Model"

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)


    def __call__(
        self,
        input_ids: mx.array,
        position_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) :
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape
            attention_mask = mx.ones(
                (batch_size, seq_len),
                dtype=self.model.embed_tokens.weight.dtype,
            )
        
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        last_hidden_state = outputs["last_hidden_state"]
        embeddings = normalize_embeddings(mean_pooling(last_hidden_state, attention_mask))

        if not return_dict:
            return (embeddings, last_hidden_state)
        return {
            "embeddings": embeddings,
            "last_hidden_state": last_hidden_state,
        }

    def sanitize(self, weights):
        sanitized_weights = _sanitize_backbone(weights)
        return {
            key: value
            for key, value in sanitized_weights.items()
            if key.startswith("model.")
        }


# This is a placeholder that will work to trian new models
# but there is no standard so the ecosystem may settle on something different. 
# The main point is to have a separate head for classification that can be added on top of the backbone, but the details may change.
class PPLXQwen3ClassificationHead(nn.Module):
    """Classification head for PPLXQwen3Model outputs, inspired by T5Gemma's architecture but simplified for PPLX."""
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(
            p=getattr(config, "classifier_dropout_rate", config.classifier_dropout)
        )
        self.out_proj = nn.Linear(
            config.hidden_size, config.num_labels
        )
        self.soft_cap = getattr(config, "final_logit_softcapping", None)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        logits = self.out_proj(self.dropout(hidden_states))
        if self.soft_cap is not None:
            logits = mx.tanh(logits / self.soft_cap) * self.soft_cap
        return logits
    
class PPLXQwen3PredictionHead(nn.Module):
    """Prediction head for PPLXQwen3Model outputs, used for masked language modeling, inspired by ModernBert."""
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(
            config.hidden_size,
            bias=getattr(config, "norm_bias", False),
            eps=getattr(config, "norm_eps", config.rms_norm_eps),
        )


    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.layer_norm(self.act(self.dense(hidden_states)))


class ModelForSentenceSimilarity(RaclateBaseModel):
    """
    Computes similarity scores between input sequences and reference sentences.
    """
    def __init__(self, config : ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type # not used for now (placeholder)
        self.model = PPLXQwen3Model(config)

    def _call_model(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        return_dict=True,
    ):
        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
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
            # Standard causal model retrieval: Last Token Pooling
            text_embeds = normalize_embeddings(mean_pooling(last_hidden_state, attention_mask))

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
                dtype=self.model.embed_tokens.weight.dtype,
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
        sanitized_weights = _sanitize_backbone(weights)
        return {
            key: value
            for key, value in sanitized_weights.items()
            if key.startswith("model.")
        }


class ModelForSentenceTransformers(ModelForSentenceSimilarity):
    pass

class ModelForSequenceClassification(RaclateBaseModel):
    """
    Computes sequence classification probabilities for input sequences.

    NOTE : not tested as no models exist. need to train one first.
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.is_regression = config.is_regression
        
        self.model = PPLXQwen3Model(config)

        #No HF transformers architecture; SequenceClassification typically only as a score layer
        self.score = PPLXQwen3ClassificationHead(config)

        # No HF transformers architecture for PPLX and SequenceClassification
    
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
    ) :
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

        # pooling for PPLX is just mean pooling, no CLS token
        pooled = mean_pooling(last_hidden_state, attention_mask)

        logits = self.score(pooled)

        processed_logits = self._process_outputs(logits)

        loss = None
        if labels is not None :
            loss = self._compute_loss(logits, labels)

        if not return_dict:
            return [loss, processed_logits, outputs[1:]]

        return {
            "loss": loss,
            "logits": logits,
            "probabilities": processed_logits,
            "hidden_states": outputs.get("hidden_states", None),
        }
    
    def sanitize(self, weights):
        
        return _sanitize_backbone(weights)
    

class ModelForMaskedLM(RaclateBaseModel):
    """
    Computes masked language modeling (MLM) loss for input sequences.
    """
    def __init__(self, config : ModelArgs):
        super().__init__()
        self.config = config
        if getattr(config, "is_causal", False):
            raise ValueError("ModelForMaskedLM requires bidirectional attention.")
        self.model = PPLXQwen3Model(config)
        self.head = PPLXQwen3PredictionHead(config) 
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=config.decoder_bias
        )

        # transformers has no arch for any PPLX models

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
    ):
        
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
                
                safe_labels = mx.where(mask_tokens, flat_labels, 0)
                token_losses = nn.losses.cross_entropy(
                    flat_predictions,
                    safe_labels,
                    reduction='none'
                )
                valid_weights = mask_tokens.astype(token_losses.dtype)
                loss = mx.sum(token_losses * valid_weights) / mx.maximum(
                    mx.sum(valid_weights),
                    1.0,
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

        self.model = PPLXQwen3Model(config)
        self.score = PPLXQwen3ClassificationHead(config)
    
    def __call__(
        self,
        input_ids,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) :
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
            flat_logits = logits.reshape(-1, self.num_labels)
            flat_labels = labels.reshape(-1)
            valid_mask = flat_labels != -100
            safe_labels = mx.where(valid_mask, flat_labels, 0)
            token_losses = nn.losses.cross_entropy(
                flat_logits,
                safe_labels,
                reduction="none",
            )
            valid_weights = valid_mask.astype(token_losses.dtype)
            loss = mx.sum(token_losses * valid_weights) / mx.maximum(
                mx.sum(valid_weights),
                1.0,
            )

        if not return_dict:
            return [loss, processed_logits, outputs[1:]]

        return {
            "loss": loss,
            "logits": logits,
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
