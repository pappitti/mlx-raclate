# Copyright © 2024 Apple Inc.
from typing import Callable, Dict, Union

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

EMBEDDING_LAYER_NAMES = {
    "gemma3_text": ["embed_tokens"],
    "lfm2": ["embed_tokens"],
    "lfm2_bidirectional": ["embed_tokens"],
    "neobert": ["encoder"],
    "qwen3": ["embed_tokens"],
    "t5gemma_encoder": ["embed_tokens"],
    "modernbert": ["embeddings"]
}


class MultiPrecisionAdamW(opt.Optimizer):
    """AdamW with FP32 master weights and moments for low-precision models."""

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        bias_correction: bool = False,
    ):
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.bias_correction = bias_correction

    def update(self, model: nn.Module, gradients: dict):
        if not self._initialized:
            self.init(model.trainable_parameters())
        super().update(model, gradients)

    def init_single(self, parameter: mx.array, state: dict):
        master = parameter.astype(mx.float32)
        state["master"] = master
        state["m"] = mx.zeros(parameter.shape, dtype=mx.float32)
        state["v"] = mx.zeros(parameter.shape, dtype=mx.float32)

    def apply_single(
        self,
        gradient: mx.array,
        parameter: mx.array,
        state: dict,
    ) -> mx.array:
        gradient = gradient.astype(mx.float32)
        learning_rate = self.learning_rate.astype(mx.float32)
        master = state["master"]
        m = state["m"]
        v = state["v"]
        beta1, beta2 = self.betas

        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * mx.square(gradient)
        decayed_master = master * (1 - learning_rate * self.weight_decay)

        if self.bias_correction:
            numerator = learning_rate * m / (1 - beta1**self.step)
            denominator = mx.sqrt(v / (1 - beta2**self.step)) + self.eps
        else:
            numerator = learning_rate * m
            denominator = mx.sqrt(v) + self.eps

        master = decayed_master - numerator / denominator
        state["master"] = master
        state["m"] = m
        state["v"] = v
        return master.astype(parameter.dtype)
    
def build_schedule(schedule_config: Dict):
    """
    Build a learning rate schedule from the given config.
    Adapted for MLX Optimizers.
    """
    name = schedule_config["name"]
    arguments = schedule_config["arguments"]
    initial_lr = arguments[0]
    
    # Create the main schedule function
    if name == "constant":
        # Create a callable that ignores the step and returns the LR
        def bound_schedule_fn(_):
            return initial_lr
    else:
        # For cosine_decay, linear_schedule, etc.
        schedule_fn = getattr(opt, name)
        bound_schedule_fn = schedule_fn(*arguments)
    
    # Check for warmup
    warmup_steps = schedule_config.get("warmup_steps", 0)
    
    if warmup_steps > 0:
        warmup_init = schedule_config.get("warmup_init", 0.0)
        
        # Linear warmup: 0 -> initial_lr
        warmup_fn = opt.linear_schedule(
            warmup_init, initial_lr, warmup_steps
        )
        
        return opt.join_schedules(
            [warmup_fn, bound_schedule_fn], [warmup_steps]
        )
    else:
        return bound_schedule_fn

def dequantize(model: nn.Module) -> nn.Module:
    """
    Dequantize the quantized linear layers in the model.

    Args:
        model (nn.Module): The model with quantized linear layers.

    Returns:
        nn.Module: The model with dequantized layers.
    """
    de_quantize_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            bias = "bias" in module
            weight = module.weight
            weight = mx.dequantize(
                weight,
                module.scales,
                module.biases,
                module.group_size,
                module.bits,
            ).astype(mx.float16)
            output_dims, input_dims = weight.shape
            linear = nn.Linear(input_dims, output_dims, bias=bias)
            linear.weight = weight
            if bias:
                linear.bias = module.bias
            de_quantize_layers.append((name, linear))
        if isinstance(module, nn.QuantizedEmbedding):
            weight = mx.dequantize(
                module.weight,
                module.scales,
                module.biases,
                module.group_size,
                module.bits,
            ).astype(mx.float16)
            num_embeddings, dims = weight.shape
            emb = nn.Embedding(num_embeddings, dims)
            emb.weight = weight
            de_quantize_layers.append((name, emb))

    if len(de_quantize_layers) > 0:
        model.update_modules(tree_unflatten(de_quantize_layers))
    return model


def nparams(module):
    if hasattr(module, "bits"):
        n = 0 if not hasattr(module, "bias") else module.bias.size
        return n + module.weight.size * 32 // module.bits
    return sum(v.size for _, v in tree_flatten(module.parameters()))


def print_trainable_parameters(model):
    leaf_modules = tree_flatten(
        model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)
    )
    total_p = sum(nparams(m) for _, m in leaf_modules) / 10**6
    trainable_p = (
        sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    )
    print(
        f"Trainable parameters: {(trainable_p * 100 / total_p):.3f}% "
        f"({trainable_p:.3f}M/{total_p:.3f}M)"
    )
