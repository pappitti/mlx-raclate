from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, RaclateBaseModel, mean_pooling, normalize_embeddings


OPENAI_PRIVACY_FILTER_SPAN_LABELS = (
	"O",
	"account_number",
	"private_address",
	"private_date",
	"private_email",
	"private_person",
	"private_phone",
	"private_url",
	"secret",
)

OPENAI_PRIVACY_FILTER_NER_LABELS = ("O",) + tuple(
	f"{prefix}-{label}"
	for label in OPENAI_PRIVACY_FILTER_SPAN_LABELS
	if label != "O"
	for prefix in ("B", "I", "E", "S")
)


@dataclass
class ModelArgs(BaseModelArgs):
	architectures: List[str] = field(
		default_factory=lambda: ["OpenAIPrivacyFilterForTokenClassification"]
	)
	attention_bias: bool = True
	attention_dropout: float = 0.0
	attention_probs_dropout_prob: Optional[float] = None
	bidirectional_context: bool = True
	bidirectional_left_context: Optional[int] = None
	bidirectional_right_context: Optional[int] = None
	bos_token_id: Optional[int] = None
	classifier_dropout: float = 0.0
	classifier_bias: bool = True
	eos_token_id: Optional[int] = 199999
	experts_per_token: Optional[int] = None
	
	head_dim: int = 64
	hidden_size: int = 640
	initializer_range: float = 0.02
	initial_context_length: int = 4096
	intermediate_size: int = 640
	layer_norm_eps: Optional[float] = None
	max_position_embeddings: int = 131072
	model_type: str = "openai_privacy_filter"
	num_attention_heads: int = 14
	num_experts: Optional[int] = None
	num_experts_per_tok: int = 4
	num_key_value_heads: int = 2
	num_hidden_layers: int = 8
	num_local_experts: int = 128
	output_router_logits: bool = False
	pad_token_id: Optional[int] = 199999
	
	rope_ntk_alpha: Optional[float] = None
	rope_ntk_beta: Optional[float] = None
	rope_parameters: Optional[Dict[str, Any]] = None
	rope_scaling_factor: Optional[float] = None
	rope_theta: float = 150000.0
	router_aux_loss_coef: float = 0.001
	rms_norm_eps: float = 1e-5
	sliding_window: int = 128
	swiglu_limit: float = 7.0
	tie_word_embeddings: bool = False
	use_cache: bool = False
	vocab_size: int = 200064

	# Pipeline args.
	num_labels: int = len(OPENAI_PRIVACY_FILTER_NER_LABELS)
	label2id: Optional[Dict[str, int]] = None
	id2label: Optional[Dict[str, str]] = None
	pipeline_config: Optional[Dict[str, Any]] = None
	use_late_interaction: bool = False
	is_regression: Optional[bool] = None

	def __post_init__(self):
		if self.num_experts is not None:
			self.num_local_experts = self.num_experts

		if self.experts_per_token is not None:
			self.num_experts_per_tok = self.experts_per_token

		if self.num_key_value_heads is None:
			self.num_key_value_heads = self.num_attention_heads

		if self.head_dim is None:
			self.head_dim = self.hidden_size // self.num_attention_heads

		if self.attention_probs_dropout_prob is not None:
			self.attention_dropout = self.attention_probs_dropout_prob

		if self.layer_norm_eps is not None:
			self.rms_norm_eps = self.layer_norm_eps

		if self.id2label is not None:
			self.num_labels = len(self.id2label)
			if self.label2id is None:
				self.label2id = {
					label: int(index)
					for index, label in self.id2label.items()
				}

		self.rope_parameters = self._build_rope_parameters()
		self.rope_theta = float(self.rope_parameters.get("rope_theta", self.rope_theta))

	def _build_rope_parameters(self) -> Dict[str, Any]:
		params = dict(self.rope_parameters or {})
		params.setdefault("rope_theta", self.rope_theta)

		if "rope_type" not in params:
			params["rope_type"] = "yarn"

		if params["rope_type"] == "yarn":
			params.setdefault(
				"factor",
				self.rope_scaling_factor
				or (self.max_position_embeddings / max(self.initial_context_length, 1)),
			)
			params.setdefault("beta_fast", self.rope_ntk_beta or 32.0)
			params.setdefault("beta_slow", self.rope_ntk_alpha or 1.0)
			params.setdefault(
				"original_max_position_embeddings",
				self.initial_context_length,
			)
			params.setdefault("truncate", False)

		return params

	@property
	def context_window(self) -> Tuple[int, int]:
		if self.bidirectional_left_context is not None or self.bidirectional_right_context is not None:
			left = self.bidirectional_left_context
			right = self.bidirectional_right_context
			if left is None and right is not None:
				left = right + 1
			if right is None and left is not None:
				right = max(left - 1, 0)
			return int(left or 0), int(right or 0)

		# HF privacy-filter configs store the one-sided context as `sliding_window`.
		left = int(self.sliding_window)
		right = max(left - 1, 0)
		return left, right


def _find_correction_dim(
	num_rotations: float,
	dim: int,
	base: float,
	max_position_embeddings: int,
) -> float:
	return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _find_correction_range(
	low_rot: float,
	high_rot: float,
	dim: int,
	base: float,
	max_position_embeddings: int,
) -> Tuple[float, float]:
	low = _find_correction_dim(low_rot, dim, base, max_position_embeddings)
	high = _find_correction_dim(high_rot, dim, base, max_position_embeddings)
	low = math.floor(low)
	high = math.ceil(high)
	return max(low, 0.0), min(high, dim - 1.0)


def _linear_ramp_factor(start: float, stop: float, dim: int) -> mx.array:
	if start == stop:
		stop += 0.001
	linear = (mx.arange(dim, dtype=mx.float32) - start) / (stop - start)
	return mx.clip(linear, 0.0, 1.0)


def _yarn_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
	if scale <= 1:
		return 1.0
	return 0.1 * mscale * math.log(scale) + 1.0


def _get_rope_parameters(config: ModelArgs) -> Tuple[mx.array, float]:
	rope_parameters = dict(config.rope_parameters or {})
	rope_type = rope_parameters.get("rope_type", "default")
	base = float(rope_parameters.get("rope_theta", config.rope_theta))
	dim = int(config.head_dim)
	freq_extra = base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)

	if rope_type == "yarn":
		factor = float(rope_parameters.get("factor", 1.0))
		attention_factor = rope_parameters.get("attention_factor")
		if attention_factor is None:
			attention_factor = _yarn_mscale(
				factor,
				float(rope_parameters.get("mscale", 1.0)),
			) / _yarn_mscale(
				factor,
				float(rope_parameters.get("mscale_all_dim", 0.0)),
			)

		beta_fast = float(rope_parameters.get("beta_fast", 32.0))
		beta_slow = float(rope_parameters.get("beta_slow", 1.0))
		original_max = int(
			rope_parameters.get(
				"original_max_position_embeddings",
				config.initial_context_length,
			)
		)

		freq_inter = factor * freq_extra

		low, high = _find_correction_range(
			beta_fast,
			beta_slow,
			dim,
			base,
			original_max,
		)
		freq_mask = 1.0 - _linear_ramp_factor(low, high, dim // 2)
		freqs = (freq_inter * freq_extra) / (
			freq_inter * freq_mask + freq_extra * (1.0 - freq_mask)
		)
		return freqs.astype(mx.float32), float(attention_factor)

	if rope_type == "linear":
		factor = float(rope_parameters.get("factor", 1.0))
		freq_extra = freq_extra * factor

	return freq_extra.astype(mx.float32), 1.0


class OpenAIPrivacyFilterRMSNorm(nn.Module):
	def __init__(self, dims: int, eps: float = 1e-5):
		super().__init__()
		self.weight = mx.ones((dims,))
		self.eps = eps

	def __call__(self, hidden_states: mx.array) -> mx.array:
		hidden_states_f32 = hidden_states.astype(mx.float32)
		variance = mx.mean(hidden_states_f32 * hidden_states_f32, axis=-1, keepdims=True)
		normalized = hidden_states_f32 * mx.rsqrt(variance + self.eps)
		return (normalized * self.weight.astype(mx.float32)).astype(hidden_states.dtype)


class OpenAIPrivacyFilterRotaryEmbedding(nn.Module):
	def __init__(self, config: ModelArgs):
		super().__init__()
		self.dims = config.head_dim
		freqs, self.mscale = _get_rope_parameters(config)
		self._freqs = (freqs,)

	def __call__(
		self,
		hidden_states: mx.array,
		offset: int = 0,
	) -> mx.array:
		x = hidden_states
		if self.mscale != 1.0:
			scaled = x[..., : self.dims] * self.mscale
			x = (
				scaled
				if self.dims == x.shape[-1]
				else mx.concatenate([scaled, x[..., self.dims :]], axis=-1)
			)
		return mx.fast.rope(
			x,
			self.dims,
			traditional=True,
			base=None,
			scale=1.0,
			offset=offset,
			freqs=self._freqs[0],
		)


class OpenAIPrivacyFilterAttention(nn.Module):
	def __init__(self, config: ModelArgs):
		super().__init__()
		self.num_attention_heads = config.num_attention_heads
		self.num_key_value_heads = config.num_key_value_heads
		self.head_dim = config.head_dim
		self.q_proj = nn.Linear(
			config.hidden_size,
			self.num_attention_heads * self.head_dim,
			bias=config.attention_bias,
		)
		self.k_proj = nn.Linear(
			config.hidden_size,
			self.num_key_value_heads * self.head_dim,
			bias=config.attention_bias,
		)
		self.v_proj = nn.Linear(
			config.hidden_size,
			self.num_key_value_heads * self.head_dim,
			bias=config.attention_bias,
		)
		self.o_proj = nn.Linear(
			self.num_attention_heads * self.head_dim,
			config.hidden_size,
			bias=config.attention_bias,
		)
		self.sliding_window = int(config.sliding_window)
		self.chunk_size = max(2 * self.sliding_window + 1, 128)
		self.scaling = config.head_dim**-0.25
		self.sinks = mx.zeros((self.num_attention_heads,))
		self.rope = OpenAIPrivacyFilterRotaryEmbedding(config)

	def _chunked_attention(
		self,
		query_states: mx.array,
		key_states: mx.array,
		value_states: mx.array,
		attention_mask: Optional[mx.array],
	) -> mx.array:
		sequence_length = query_states.shape[2]
		outputs: List[mx.array] = []

		for start in range(0, sequence_length, self.chunk_size):
			end = min(start + self.chunk_size, sequence_length)
			span_start = max(0, start - self.sliding_window)
			span_end = min(sequence_length, end + self.sliding_window + 1)

			query_chunk = query_states[:, :, start:end, :]
			key_chunk = key_states[:, :, span_start:span_end, :]
			value_chunk = value_states[:, :, span_start:span_end, :]

			mask_chunk = None
			if attention_mask is not None:
				if len(attention_mask.shape) == 4:
					mask_chunk = attention_mask[:, :, start:end, span_start:span_end]
				else:
					key_mask = attention_mask[:, None, None, span_start:span_end].astype(mx.bool_)
					mask_chunk = mx.where(
						key_mask,
						mx.array(0.0, dtype=query_states.dtype),
						mx.array(-mx.inf, dtype=query_states.dtype),
					)

			outputs.append(
				mx.fast.scaled_dot_product_attention(
					query_chunk,
					key_chunk,
					value_chunk,
					scale=1.0,
					mask=mask_chunk,
					sinks=self.sinks,
				)
			)

		return mx.concatenate(outputs, axis=2)

	def __call__(
		self,
		hidden_states: mx.array,
		attention_mask: Optional[mx.array] = None,
	) -> Tuple[mx.array, None]:
		batch_size, sequence_length, _ = hidden_states.shape

		query_states = self.q_proj(hidden_states)
		key_states = self.k_proj(hidden_states)
		value_states = self.v_proj(hidden_states)

		query_states = query_states.reshape(
			batch_size,
			sequence_length,
			self.num_attention_heads,
			self.head_dim,
		).transpose(0, 2, 1, 3)
		key_states = key_states.reshape(
			batch_size,
			sequence_length,
			self.num_key_value_heads,
			self.head_dim,
		).transpose(0, 2, 1, 3)
		value_states = value_states.reshape(
			batch_size,
			sequence_length,
			self.num_key_value_heads,
			self.head_dim,
		).transpose(0, 2, 1, 3)

		query_states = self.rope(query_states)
		key_states = self.rope(key_states)
		query_states = query_states * self.scaling
		key_states = key_states * self.scaling
		attn_output = self._chunked_attention(
			query_states,
			key_states,
			value_states,
			attention_mask,
		)
		attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
			batch_size,
			sequence_length,
			-1,
		)
		return self.o_proj(attn_output), None


class OpenAIPrivacyFilterExperts(nn.Module):
	def __init__(self, config: ModelArgs):
		super().__init__()
		self.top_k = config.num_experts_per_tok
		self.alpha = 1.702
		self.limit = config.swiglu_limit
		self.gate_up_proj = mx.zeros(
			(config.num_local_experts, config.hidden_size, 2 * config.intermediate_size)
		)
		self.gate_up_proj_bias = mx.zeros((config.num_local_experts, 2 * config.intermediate_size))
		self.down_proj = mx.zeros((config.num_local_experts, config.intermediate_size, config.hidden_size))
		self.down_proj_bias = mx.zeros((config.num_local_experts, config.hidden_size))

	def _apply_gate(self, gate_up: mx.array) -> mx.array:
		gate, up = mx.split(gate_up, 2, axis=-1)
		gate = mx.minimum(gate, self.limit)
		up = mx.clip(up, -self.limit, self.limit)
		glu = gate * mx.sigmoid(gate * self.alpha)
		return (up + 1.0) * glu

	def __call__(self, hidden_states: mx.array, router_indices: mx.array, router_scores: mx.array) -> mx.array:
		next_states = mx.zeros(hidden_states.shape, dtype=mx.float32)

		for slot in range(self.top_k):
			expert_indices = router_indices[:, slot]
			weights = router_scores[:, slot : slot + 1].astype(mx.float32)
			gate_up_weight = self.gate_up_proj[expert_indices].astype(mx.float32)
			gate_up = mx.matmul(
				hidden_states[:, None, :].astype(mx.float32),
				gate_up_weight,
			).squeeze(1)
			gate_up = gate_up + self.gate_up_proj_bias[expert_indices].astype(mx.float32)
			gated_output = self._apply_gate(gate_up)
			down_weight = self.down_proj[expert_indices].astype(mx.float32)
			expert_output = mx.matmul(
				gated_output[:, None, :].astype(mx.float32),
				down_weight,
			).squeeze(1)
			expert_output = expert_output + self.down_proj_bias[expert_indices].astype(mx.float32)
			next_states = next_states + expert_output * weights

		return next_states
	

class OpenAIPrivacyFilterTopKRouter(nn.Module):
	def __init__(self, config: ModelArgs):
		super().__init__()
		self.top_k = config.num_experts_per_tok
		self.weight = mx.zeros((config.num_local_experts, config.hidden_size))
		self.bias = mx.zeros((config.num_local_experts,))

	def __call__(self, hidden_states: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
		logits = mx.matmul(
			hidden_states.astype(mx.float32),
			self.weight.astype(mx.float32).T,
		) + self.bias.astype(mx.float32) 
		sorted_indices = mx.argsort(logits, axis=-1)[..., -self.top_k :]
		top_values = mx.take_along_axis(logits, sorted_indices, axis=-1)
		router_scores = mx.softmax(top_values, axis=-1) / self.top_k
		return logits, router_scores, sorted_indices
		

class OpenAIPrivacyFilterMLP(nn.Module):
	def __init__(self, config: ModelArgs):
		super().__init__()
		self.num_experts = config.num_experts_per_tok
		self.router = OpenAIPrivacyFilterTopKRouter(config)
		self.experts = OpenAIPrivacyFilterExperts(config)

	def __call__(self, hidden_states: mx.array) -> Tuple[mx.array, mx.array]:
		batch_size, sequence_length, hidden_dim = hidden_states.shape
		flat_states = hidden_states.reshape(-1, hidden_dim)
		router_logits, router_scores, router_indices = self.router(flat_states)
		next_states = self.experts(flat_states, router_indices, router_scores)
		next_states = (next_states * self.num_experts).reshape(batch_size, sequence_length, hidden_dim) # mlx-embeddings uses  y = y * mx.expand_dims(weights, axis=-1) return y.sum(axis=-2)
		return next_states.astype(hidden_states.dtype), router_logits.reshape(batch_size, sequence_length, -1)


class OpenAIPrivacyFilterEncoderLayer(nn.Module):
	def __init__(self, config: ModelArgs):
		super().__init__()
		self.self_attn = OpenAIPrivacyFilterAttention(config)
		self.mlp = OpenAIPrivacyFilterMLP(config)
		self.input_layernorm= OpenAIPrivacyFilterRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
		self.post_attention_layernorm = OpenAIPrivacyFilterRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

	def __call__(
		self,
		x: mx.array,
		attention_mask: Optional[mx.array] = None,
	) -> Tuple[mx.array, mx.array]:

		h = self.input_layernorm(x)
		h, _ = self.self_attn(
			h,
			attention_mask=attention_mask,
		)
		x = x + h

		h = self.post_attention_layernorm(x)
		h, router_logits = self.mlp(h)
		hidden_states = x + h
		return hidden_states, router_logits


class OpenAIPrivacyFilterModel(nn.Module):
	def __init__(self, config: ModelArgs):
		super().__init__()
		self.config = config
		self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
		self.layers = [OpenAIPrivacyFilterEncoderLayer(config) for _ in range(config.num_hidden_layers)]
		self.norm = OpenAIPrivacyFilterRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
		self.sliding_window = config.sliding_window

	def get_input_embeddings(self) -> nn.Embedding:
		return self.embed_tokens

	def set_input_embeddings(self, value):
		self.embed_tokens = value

	def _update_attention_mask(self, attention_mask: mx.array = None, dtype = None) -> mx.array:
		B, L = attention_mask.shape
		window_size = self.sliding_window

		padding_mask = attention_mask[:, None, None, :]
		additive_padding = mx.where(padding_mask == 0, -1e9, 0.0).astype(dtype)

		indices = mx.arange(L)
		row = indices[:, None]
		col = row.T

		mask_base = mx.zeros((L, L), dtype=mx.bool_) # All False (visible)

		# Sliding Window Logic for Bidirectional:
		# Valid if abs(row - col) <= window
		# Mask if distance > window
		dist = mx.abs(row - col)
		mask_window_violator = dist > window_size # should this be window_size // 2 for bidirectional (not the case for gemma3, but yes for modernBert)

		sliding_mask_bool = mask_base | mask_window_violator
		sliding_mask = mx.where(sliding_mask_bool, -1e9, 0.0).astype(dtype)

		sliding_mask = sliding_mask[None, None, :, :] + additive_padding # does this broadcasting work correctly? should additive_padding be added inside the loop instead?
		return sliding_mask

	def __call__(
		self,
		input_ids: mx.array,
		attention_mask: Optional[mx.array] = None,
		position_ids: Optional[mx.array] = None,
		output_hidden_states: bool = False,
		output_router_logits: Optional[bool] = None,
		return_dict: bool = True,
	):
		hidden_states = self.embed_tokens(input_ids)
		model_dtype = hidden_states.dtype

		attention_mask = self._update_attention_mask(
			attention_mask, 
			dtype=model_dtype
		)

		capture_router_logits = self.config.output_router_logits if output_router_logits is None else output_router_logits

		hidden_state_list = [] if output_hidden_states else None
		router_logits_list = [] if capture_router_logits else None

		for layer in self.layers:
			if hidden_state_list is not None:
				hidden_state_list.append(hidden_states)
			hidden_states, router_logits = layer(
				hidden_states,
				attention_mask=attention_mask,
			)
			if router_logits_list is not None:
				router_logits_list.append(router_logits)

		hidden_states = self.norm(hidden_states)

		if hidden_state_list is not None:
			hidden_state_list.append(hidden_states)

		if not return_dict:
			return hidden_states, hidden_state_list, router_logits_list

		return {
			"last_hidden_state": hidden_states,
			"hidden_states": hidden_state_list,
			"router_logits": router_logits_list,
		}


def _sanitize_weights(weights: Dict[str, Any], include_head: bool) -> Dict[str, Any]:
	sanitized: Dict[str, Any] = {}

	for key, value in weights.items():
		if "position_ids" in key or key.endswith("rotary_emb.inv_freq"):
			continue

		if key.startswith("model."):
			sanitized[key] = value
			continue

		if include_head and key.startswith("score."):
			sanitized[key] = value
			continue

		if include_head and key.startswith("classifier."):
			sanitized[f"score.{key.removeprefix('classifier.')}"] = value
			continue

		if include_head and key.startswith("unembedding."):
			sanitized[f"score.{key.removeprefix('unembedding.')}"] = value
			continue

	if include_head and "score.weight" in sanitized and "score.bias" not in sanitized:
		sanitized["score.bias"] = mx.zeros((sanitized["score.weight"].shape[0],), dtype=sanitized["score.weight"].dtype)

	return sanitized


class Model(RaclateBaseModel):
	def __init__(self, config: ModelArgs):
		super().__init__()
		self.config = config
		self.model = OpenAIPrivacyFilterModel(config)
		
        # no transformer architecture for embedding model

	def get_input_embeddings(self) -> nn.Embedding:
		return self.model.get_input_embeddings()

	def set_input_embeddings(self, value):
		self.model.set_input_embeddings(value)

	def __call__(
		self,
		input_ids: mx.array,
		attention_mask: Optional[mx.array] = None,
		position_ids: Optional[mx.array] = None,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		if attention_mask is None:
			batch_size, seq_len = input_ids.shape
			attention_mask = mx.ones(
                (batch_size, seq_len),
                dtype=self.model.embed_tokens.weight.dtype,
            )
	
		outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			output_hidden_states=output_hidden_states,
			return_dict=True,
		)
		last_hidden_state = outputs["last_hidden_state"]
		
		pooled = mean_pooling(last_hidden_state, attention_mask)
		embeddings = normalize_embeddings(pooled)

		if not return_dict:
			return embeddings, last_hidden_state

		return {
			"embeddings": embeddings,
			"last_hidden_state": last_hidden_state,
			"hidden_states": outputs.get("hidden_states"),
		}

	def sanitize(self, weights):
		sanitized = _sanitize_weights(weights, include_head=False)
		return {
			key: value
			for key, value in sanitized.items()
			if key.startswith("model.")
		}


class ModelForTokenClassification(RaclateBaseModel):
	def __init__(self, config: ModelArgs):
		super().__init__()
		self.config = config
		self.model = OpenAIPrivacyFilterModel(config)
		self.dropout = nn.Dropout(p=config.classifier_dropout)
		self.score = nn.Linear(
			config.hidden_size,
			config.num_labels,
			bias=config.classifier_bias,
		)
		self.hf_transformers_arch = "OpenAIPrivacyFilterForTokenClassification"

	def get_input_embeddings(self) -> nn.Embedding:
		return self.model.get_input_embeddings()

	def set_input_embeddings(self, value):
		self.model.set_input_embeddings(value)

	def _compute_loss(self, logits: mx.array, labels: mx.array) -> mx.array:
		flat_logits = logits.reshape(-1, self.config.num_labels)
		flat_labels = labels.reshape(-1)
		valid_mask = flat_labels != -100
		safe_labels = mx.where(valid_mask, flat_labels, 0)
		token_losses = nn.losses.cross_entropy(
			flat_logits,
			safe_labels,
			reduction="none",
		)
		valid_weights = valid_mask.astype(token_losses.dtype)
		return mx.sum(token_losses * valid_weights) / mx.maximum(mx.sum(valid_weights), 1.0)

	def __call__(
		self,
		input_ids: mx.array,
		attention_mask: Optional[mx.array] = None,
		position_ids: Optional[mx.array] = None,
		labels: Optional[mx.array] = None,
		output_hidden_states: bool = False,
		output_router_logits: Optional[bool] = None,
		return_dict: bool = True,
	):
		
		if attention_mask is None:
			batch_size, seq_len = input_ids.shape
			attention_mask = mx.ones(
                (batch_size, seq_len),
                dtype=self.model.embed_tokens.weight.dtype,
            )
			
		outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			output_hidden_states=output_hidden_states,
			output_router_logits=output_router_logits,
			return_dict=True,
		)
		last_hidden_state = outputs["last_hidden_state"]
		logits = self.score(self.dropout(last_hidden_state))
		probabilities = mx.softmax(logits, axis=-1)

		loss = None
		if labels is not None:
			loss = self._compute_loss(logits, labels)

		if not return_dict:
			return loss, logits, probabilities

		return {
			"loss": loss,
			"logits": logits,
			"probabilities": probabilities,
			"hidden_states": outputs.get("hidden_states"),
			"router_logits": outputs.get("router_logits"),
		}

	def sanitize(self, weights):
		return _sanitize_weights(weights, include_head=True)
