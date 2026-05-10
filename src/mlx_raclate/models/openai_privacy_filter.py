from dataclasses import dataclass, field
import math
import re
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
			params["rope_type"] = "yarn" if self.rope_scaling_factor else "default"

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
	truncate: bool,
) -> Tuple[float, float]:
	low = _find_correction_dim(low_rot, dim, base, max_position_embeddings)
	high = _find_correction_dim(high_rot, dim, base, max_position_embeddings)
	if truncate:
		low = math.floor(low)
		high = math.ceil(high)
	return max(low, 0.0), min(high, dim - 1.0)


def _linear_ramp_factor(start: float, stop: float, dim: int) -> mx.array:
	if start == stop:
		stop += 0.001
	linear = (mx.arange(dim, dtype=mx.float32) - start) / (stop - start)
	return mx.clip(linear, 0.0, 1.0)


def _get_rope_parameters(config: ModelArgs) -> Tuple[mx.array, float]:
	rope_parameters = dict(config.rope_parameters or {})
	rope_type = rope_parameters.get("rope_type", "default")
	base = float(rope_parameters.get("rope_theta", config.rope_theta))
	dim = int(config.head_dim)

	if rope_type == "yarn":
		factor = float(rope_parameters.get("factor", 1.0))
		attention_factor = rope_parameters.get("attention_factor")
		if attention_factor is None:
			attention_factor = 1.0 if factor <= 1 else 0.1 * math.log(factor) + 1.0

		beta_fast = float(rope_parameters.get("beta_fast", 32.0))
		beta_slow = float(rope_parameters.get("beta_slow", 1.0))
		original_max = int(
			rope_parameters.get(
				"original_max_position_embeddings",
				config.initial_context_length,
			)
		)
		truncate = bool(rope_parameters.get("truncate", False))

		positions = mx.arange(0, dim, 2, dtype=mx.float32) / dim
		pos_freqs = base ** positions
		inv_freq_extrapolation = 1.0 / pos_freqs
		inv_freq_interpolation = 1.0 / (factor * pos_freqs)

		low, high = _find_correction_range(
			beta_fast,
			beta_slow,
			dim,
			base,
			original_max,
			truncate,
		)
		extrapolation_factor = 1.0 - _linear_ramp_factor(low, high, dim // 2)
		inv_freq = (
			inv_freq_interpolation * (1.0 - extrapolation_factor)
			+ inv_freq_extrapolation * extrapolation_factor
		)
		return inv_freq.astype(mx.float32), float(attention_factor)

	positions = mx.arange(0, dim, 2, dtype=mx.float32) / dim
	inv_freq = 1.0 / (base ** positions)

	if rope_type == "linear":
		factor = float(rope_parameters.get("factor", 1.0))
		inv_freq = inv_freq / factor

	return inv_freq.astype(mx.float32), 1.0


def _apply_rotary_pos_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
	x_even = x[..., ::2]
	x_odd = x[..., 1::2]
	rotated_even = x_even * cos - x_odd * sin
	rotated_odd = x_even * sin + x_odd * cos
	return mx.stack([rotated_even, rotated_odd], axis=-1).reshape(x.shape)


def _repeat_kv(hidden_states: mx.array, repeats: int) -> mx.array:
	if repeats == 1:
		return hidden_states
	batch, num_heads, seq_len, head_dim = hidden_states.shape
	hidden_states = mx.expand_dims(hidden_states, axis=2)
	hidden_states = mx.broadcast_to(
		hidden_states,
		(batch, num_heads, repeats, seq_len, head_dim),
	)
	return hidden_states.reshape(batch, num_heads * repeats, seq_len, head_dim)


class OpenAIPrivacyFilterRMSNorm(nn.Module):
	def __init__(self, dims: int, eps: float = 1e-5):
		super().__init__()
		self.scale = mx.ones((dims,))
		self.eps = eps

	def __call__(self, hidden_states: mx.array) -> mx.array:
		hidden_states_f32 = hidden_states.astype(mx.float32)
		variance = mx.mean(hidden_states_f32 * hidden_states_f32, axis=-1, keepdims=True)
		normalized = hidden_states_f32 * mx.rsqrt(variance + self.eps)
		return (normalized * self.scale.astype(mx.float32)).astype(hidden_states.dtype)


class OpenAIPrivacyFilterRotaryEmbedding(nn.Module):
	def __init__(self, config: ModelArgs):
		super().__init__()
		self.freqs = _get_rope_parameters(config)

	def __call__(
		self,
		hidden_states: mx.array,
		position_ids: mx.array,
	) -> Tuple[mx.array, mx.array]:
		if len(position_ids.shape) == 1:
			position_ids = mx.expand_dims(position_ids, axis=0)

		inv_freq, attention_factor = self.freqs
		freqs = position_ids.astype(mx.float32)[..., None] * inv_freq[None, None, :]
		cos = (mx.cos(freqs) * attention_factor).astype(hidden_states.dtype)
		sin = (mx.sin(freqs) * attention_factor).astype(hidden_states.dtype)
		return cos[:, None, :, :], sin[:, None, :, :]


class OpenAIPrivacyFilterAttention(nn.Module):
	def __init__(self, config: ModelArgs):
		super().__init__()
		self.num_heads = config.num_attention_heads
		self.num_key_value_heads = config.num_key_value_heads
		self.head_dim = config.head_dim
		self.num_key_value_groups = self.num_heads // self.num_key_value_heads
		self.left_context, self.right_context = config.context_window
		self.chunk_size = max(self.left_context + self.right_context + 1, 128)
		qkv_dim = self.num_heads * self.head_dim + 2 * self.num_key_value_heads * self.head_dim

		self.qkv = nn.Linear(config.hidden_size, qkv_dim, bias=config.attention_bias)
		self.out = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
		self.sinks = mx.zeros((self.num_heads,))
		self.scale = config.head_dim**-0.25

	def _chunked_attention(
		self,
		query_states: mx.array,
		key_states: mx.array,
		value_states: mx.array,
		attention_mask: Optional[mx.array],
	) -> mx.array:
		batch_size, num_heads, sequence_length, _ = query_states.shape
		outputs: List[mx.array] = []
		all_positions = mx.arange(sequence_length)
		key_token_mask = attention_mask.astype(mx.bool_) if attention_mask is not None else None

		for start in range(0, sequence_length, self.chunk_size):
			end = min(start + self.chunk_size, sequence_length)
			span_start = max(0, start - self.left_context)
			span_end = min(sequence_length, end + self.right_context)

			query_chunk = query_states[:, :, start:end, :]
			key_chunk = key_states[:, :, span_start:span_end, :]
			value_chunk = value_states[:, :, span_start:span_end, :]

			query_positions = all_positions[start:end]
			key_positions = all_positions[span_start:span_end]
			allowed_window = (key_positions[None, :] >= (query_positions[:, None] - self.left_context))
			allowed_window = allowed_window & (key_positions[None, :] <= (query_positions[:, None] + self.right_context))
			allowed_window = allowed_window[None, None, :, :]

			logits = mx.matmul(query_chunk.astype(mx.float32), key_chunk.astype(mx.float32).transpose(0, 1, 3, 2))

			if key_token_mask is not None:
				allowed_window = allowed_window & key_token_mask[:, None, None, span_start:span_end]

			logits = mx.where(allowed_window, logits, -1e9)

			sink_logits = mx.broadcast_to(
				self.sinks.astype(mx.float32).reshape(1, num_heads, 1, 1),
				(batch_size, num_heads, end - start, 1),
			)
			combined_logits = mx.concatenate([logits, sink_logits], axis=-1)
			combined_logits = combined_logits - mx.max(combined_logits, axis=-1, keepdims=True)
			probs = mx.softmax(combined_logits, axis=-1)
			scores = probs[..., :-1].astype(value_chunk.dtype)
			attn_output = mx.matmul(scores.astype(mx.float32), value_chunk.astype(mx.float32))

			if attention_mask is not None:
				query_token_mask = attention_mask[:, start:end].astype(mx.float32)
				attn_output = attn_output * query_token_mask[:, None, :, None]

			outputs.append(attn_output.astype(query_states.dtype))

		return mx.concatenate(outputs, axis=2)

	def __call__(
		self,
		hidden_states: mx.array,
		# position_embeddings: Tuple[mx.array, mx.array],
		attention_mask: Optional[mx.array] = None,
	) -> Tuple[mx.array, None]:
		batch_size, sequence_length, _ = hidden_states.shape
		q_len = self.num_heads * self.head_dim
		kv_len = self.num_key_value_heads * self.head_dim

		qkv = self.qkv(hidden_states)
		query_states, key_states, value_states = mx.split(qkv, [q_len, q_len + kv_len], axis=-1)

		query_states = query_states.reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
		key_states = key_states.reshape(batch_size, sequence_length, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
		value_states = value_states.reshape(batch_size, sequence_length, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

		cos, sin = position_embeddings
		query_states = _apply_rotary_pos_emb(query_states, cos, sin) * self.scale
		key_states = _apply_rotary_pos_emb(key_states, cos, sin) * self.scale
		key_states = _repeat_kv(key_states, self.num_key_value_groups)
		value_states = _repeat_kv(value_states, self.num_key_value_groups)

		attn_output = self._chunked_attention(query_states, key_states, value_states, attention_mask)
		attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)
		return self.out(attn_output), None


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
		self.gate = OpenAIPrivacyFilterTopKRouter(config)
		self.experts = OpenAIPrivacyFilterExperts(config)

	def __call__(self, hidden_states: mx.array) -> Tuple[mx.array, mx.array]:
		batch_size, sequence_length, hidden_dim = hidden_states.shape
		flat_states = hidden_states.reshape(-1, hidden_dim)
		router_logits, router_scores, router_indices = self.gate(flat_states)
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
		# position_embeddings: Tuple[mx.array, mx.array],
		attention_mask: Optional[mx.array] = None,
	) -> Tuple[mx.array, mx.array]:

		h = self.input_layernorm(x)
		h, _ = self.self_attn(
			h,
			# position_embeddings=position_embeddings,
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
		# self.rotary_emb = OpenAIPrivacyFilterRotaryEmbedding(config) # TBC
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
		# Valid if abs(row - col) < window
		# Mask if distance >= window
		dist = mx.abs(row - col)
		mask_window_violator = dist >= window_size # should this be window_size // 2 for bidirectional (not the case for gemma3, but yes for modernBert)

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

		# if position_ids is None:  # TBC
		# 	position_ids = mx.arange(seq_len, dtype=mx.int32)[None, :]

		capture_router_logits = self.config.output_router_logits if output_router_logits is None else output_router_logits

		# position_embeddings = self.rotary_emb(hidden_states, position_ids) # TBC

		hidden_state_list = [] if output_hidden_states else None
		router_logits_list = [] if capture_router_logits else None

		for layer in self.layers:
			if hidden_state_list is not None:
				hidden_state_list.append(hidden_states)
			hidden_states, router_logits = layer(
				hidden_states,
				# position_embeddings=position_embeddings, # TBC
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


def _remap_openmed_mlx_key(key: str) -> Optional[str]:
	"""Map OpenMed/mlx-embeddings privacy-filter keys to this module tree."""
	body = key[6:] if key.startswith("model.") else key

	if body.startswith("unembedding."):
		return body

	if not body.startswith(("embedding.", "block.", "norm.")):
		return None

	if body == "embedding.weight":
		return "model.embed_tokens.weight"

	if body == "norm.scale":
		return "model.norm.scale"

	match = re.match(r"^block\.(\d+)\.attn\.norm\.scale$", body)
	if match:
		return f"model.layers.{match.group(1)}.input_layernorm.scale"

	match = re.match(r"^block\.(\d+)\.mlp\.norm\.scale$", body)
	if match:
		return f"model.layers.{match.group(1)}.post_attention_layernorm.scale"

	match = re.match(r"^block\.(\d+)\.attn\.out\.(weight|bias)$", body)
	if match:
		return f"model.layers.{match.group(1)}.self_attn.out.{match.group(2)}"

	match = re.match(r"^block\.(\d+)\.attn\.sinks$", body)
	if match:
		return f"model.layers.{match.group(1)}.self_attn.sinks"

	match = re.match(r"^block\.(\d+)\.attn\.qkv\.(weight|bias)$", body)
	if match:
		return f"model.layers.{match.group(1)}.self_attn.qkv.{match.group(2)}"

	match = re.match(r"^block\.(\d+)\.mlp\.(?:gate|router)\.(weight|bias)$", body)
	if match:
		return f"model.layers.{match.group(1)}.mlp.gate.{match.group(2)}"

	match = re.match(r"^block\.(\d+)\.mlp\.swiglu\.(weight|bias)$", body)
	if match:
		suffix = "_bias" if match.group(2) == "bias" else ""
		return f"model.layers.{match.group(1)}.mlp.experts.gate_up_proj{suffix}"

	match = re.match(r"^block\.(\d+)\.mlp\.out\.(weight|bias)$", body)
	if match:
		suffix = "_bias" if match.group(2) == "bias" else ""
		return f"model.layers.{match.group(1)}.mlp.experts.down_proj{suffix}"

	return f"model.{body}"


def _sanitize_weights(weights: Dict[str, Any], include_head: bool) -> Dict[str, Any]:
	sanitized: Dict[str, Any] = {}
	qkv_parts: Dict[Tuple[str, str], Dict[str, Any]] = {}

	for key, value in weights.items():
		if "position_ids" in key or key.endswith("rotary_emb.inv_freq"):
			continue

		mapped_key = _remap_openmed_mlx_key(key)
		if mapped_key is not None:
			if include_head or not mapped_key.startswith("unembedding."):
				sanitized[mapped_key] = value
			continue

		if include_head and key.startswith("unembedding."):
			sanitized[key] = value
			continue

		if key == "model.embed_tokens.weight":
			sanitized[key] = value
			continue

		if key == "model.norm.scale":
			sanitized[key] = value
			continue

		if key == "model.norm.weight":
			sanitized["model.norm.scale"] = value
			continue

		match = re.match(r"model\.layers\.(\d+)\.input_layernorm\.weight$", key)
		if match:
			sanitized[f"model.layers.{match.group(1)}.input_layernorm.scale"] = value
			continue

		match = re.match(r"model\.layers\.(\d+)\.post_attention_layernorm\.weight$", key)
		if match:
			sanitized[f"model.layers.{match.group(1)}.post_attention_layernorm.scale"] = value
			continue

		match = re.match(r"model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$", key)
		if match:
			sanitized[f"model.layers.{match.group(1)}.self_attn.out.{match.group(2)}"] = value
			continue

		match = re.match(r"model\.layers\.(\d+)\.self_attn\.sinks$", key)
		if match:
			sanitized[key] = value
			continue

		match = re.match(r"model\.layers\.(\d+)\.self_attn\.qkv_proj\.(weight|bias)$", key)
		if match:
			sanitized[f"model.layers.{match.group(1)}.self_attn.qkv.{match.group(2)}"] = value
			continue

		match = re.match(r"model\.layers\.(\d+)\.self_attn\.(qkv|out)\.(weight|bias)$", key)
		if match:
			sanitized[key] = value
			continue

		match = re.match(r"model\.layers\.(\d+)\.self_attn\.(q|k|v)_proj\.(weight|bias)$", key)
		if match:
			layer_key = match.group(1)
			attr = match.group(3)
			qkv_parts.setdefault((layer_key, attr), {})[match.group(2)] = value
			continue

		match = re.match(r"model\.layers\.(\d+)\.mlp\.router\.(weight|bias)$", key)
		if match:
			sanitized[f"model.layers.{match.group(1)}.mlp.gate.{match.group(2)}"] = value
			continue

		match = re.match(r"model\.layers\.(\d+)\.mlp\.gate\.(weight|bias)$", key)
		if match:
			sanitized[key] = value
			continue

		match = re.match(r"model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj(?:\.weight)?$", key)
		if match:
			sanitized[f"model.layers.{match.group(1)}.mlp.experts.gate_up_proj"] = value
			continue

		match = re.match(r"model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj_bias$", key)
		if match:
			sanitized[key] = value
			continue

		match = re.match(r"model\.layers\.(\d+)\.mlp\.experts\.down_proj(?:\.weight)?$", key)
		if match:
			sanitized[f"model.layers.{match.group(1)}.mlp.experts.down_proj"] = value
			continue

		match = re.match(r"model\.layers\.(\d+)\.mlp\.experts\.down_proj_bias$", key)
		if match:
			sanitized[key] = value
			continue

		if include_head and key in {"score.weight", "classifier.weight"}:
			sanitized["unembedding.weight"] = value
			continue

		if include_head and key in {"score.bias", "classifier.bias"}:
			sanitized["unembedding.bias"] = value
			continue

	for (layer_key, attr), parts in qkv_parts.items():
		if all(name in parts for name in ("q", "k", "v")):
			sanitized[f"model.layers.{layer_key}.self_attn.qkv.{attr}"] = mx.concatenate(
				[parts["q"], parts["k"], parts["v"]],
				axis=0,
			)

	if include_head and "unembedding.weight" in sanitized and "unembedding.bias" not in sanitized:
		sanitized["unembedding.bias"] = mx.zeros((sanitized["unembedding.weight"].shape[0],), dtype=sanitized["unembedding.weight"].dtype)

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
		self.unembedding = nn.Linear( # self.score?
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
		logits = self.unembedding(self.dropout(last_hidden_state))
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
