# Copyright © 2023-2024 Apple Inc.

import copy
import glob
import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Type, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_reduce

from mlx_raclate.registry import (
    PIPELINES as PIPELINES,
    infer_pipeline_from_architecture,
    resolve_model_class,
)
from mlx_raclate.tuner.utils import nparams

from .hub import ModelNotFoundError as ModelNotFoundError
from .hub import get_model_path as get_model_path
from .tokenizer_utils import TokenizerWrapper, load_tokenizer

# Map accepted model-config spellings to a canonical name and MLX dtype.
STR_TO_DTYPE = {
    "float32": ("float32", mx.float32),
    "fp32": ("float32", mx.float32),
    "float16": ("float16", mx.float16),
    "fp16": ("float16", mx.float16),
    "half": ("float16", mx.float16),
    "bfloat16": ("bfloat16", mx.bfloat16),
    "bf16": ("bfloat16", mx.bfloat16),
}

@dataclass(frozen=True)
class ModelDTypePolicy:
    name: str
    config_name: str
    dtype: mx.Dtype
    cast_checkpoint: bool
    source: str


def _canonical_dtype(dtype: mx.Dtype) -> str:
    for canonical, candidate in STR_TO_DTYPE.values():
        if dtype == candidate:
            return canonical
    raise ValueError(f"Unsupported floating-point dtype in checkpoint: {dtype}")


def _infer_checkpoint_dtype(loaded_weights: dict) -> tuple[mx.Dtype, bool]:
    dtype_sizes = {}
    for value in loaded_weights.values():
        if not mx.issubdtype(value.dtype, mx.floating):
            continue
        _canonical_dtype(value.dtype)
        dtype_sizes[value.dtype] = dtype_sizes.get(value.dtype, 0) + value.size

    if not dtype_sizes:
        return mx.float32, False
    return max(dtype_sizes, key=dtype_sizes.get), len(dtype_sizes) > 1


def _resolve_model_dtype(config: dict, loaded_weights: dict) -> ModelDTypePolicy:
    """Resolve model-config dtype, preferring canonical dtype over torch_dtype."""
    if config.get("dtype") is not None:
        requested = config["dtype"]
        source = "dtype"
    elif config.get("quantization") is not None:
        # Quantized MLX configs often retain the pre-conversion torch_dtype;
        # the packed checkpoint is the reliable source for auxiliary tensors.
        requested = "auto"
        source = "checkpoint"
    elif config.get("torch_dtype") is not None:
        requested = config["torch_dtype"]
        source = "torch_dtype"
    else:
        requested = "auto"
        source = "checkpoint"

    if not isinstance(requested, str):
        raise TypeError(
            "Model config dtype must be a string such as 'bfloat16' or 'auto', "
            f"got {type(requested).__name__}."
        )

    normalized = requested.lower()
    if normalized == "auto":
        inferred, is_mixed = _infer_checkpoint_dtype(loaded_weights)
        canonical = _canonical_dtype(inferred)
        return ModelDTypePolicy(
            name=canonical,
            config_name="auto" if is_mixed else canonical,
            dtype=inferred,
            cast_checkpoint=False,
            source="checkpoint",
        )

    if normalized not in STR_TO_DTYPE:
        supported = ", ".join(sorted(STR_TO_DTYPE))
        raise ValueError(
            f"Unsupported model dtype '{requested}'. Supported values: auto, {supported}."
        )

    canonical, dtype = STR_TO_DTYPE[normalized]
    return ModelDTypePolicy(
        name=canonical,
        config_name=canonical,
        dtype=dtype,
        cast_checkpoint=True,
        source=source,
    )


def _determine_model_dtype(config: dict, loaded_weights: dict) -> mx.Dtype:
    """Backward-compatible dtype-only view of the resolved model policy."""
    return _resolve_model_dtype(config, loaded_weights).dtype


def _align_checkpoint_dtypes(
    weights: dict,
    model: nn.Module,
    *,
    cast_floating: bool,
) -> dict:
    """Align explicit floating checkpoint overrides to destination parameters."""
    if not cast_floating:
        return weights

    parameters = dict(tree_flatten(model.parameters()))
    for key, value in weights.items():
        destination = parameters.get(key)
        if destination is None or not mx.issubdtype(value.dtype, mx.floating):
            continue
        if not mx.issubdtype(destination.dtype, mx.floating):
            continue
        if value.dtype != destination.dtype:
            weights[key] = value.astype(destination.dtype)
    return weights


def _validate_explicit_model_dtype(
    model: nn.Module,
    dtype_policy: ModelDTypePolicy,
) -> None:
    if not dtype_policy.cast_checkpoint:
        return

    mismatched = [
        f"{key}={value.dtype}"
        for key, value in tree_flatten(model.parameters())
        if mx.issubdtype(value.dtype, mx.floating)
        and value.dtype != dtype_policy.dtype
    ]
    if mismatched:
        raise RuntimeError(
            f"Model dtype policy '{dtype_policy.name}' was not applied to: "
            + ", ".join(mismatched[:5])
        )


def _get_pipeline_from_config(arch: Optional[str]) -> Optional[str]:
    """
    Retrieve the pipeline type based on the model configuration.

    Args:
        arch: first item of architectures from the model configuration.

    Returns:
        str: The pipeline type.
    """
    return infer_pipeline_from_architecture(arch)


def _uses_lfm2_bidirectional_architecture(config: dict) -> bool:
    architectures = config.get("architectures") or []
    if any("Lfm2Bidirectional" in architecture for architecture in architectures):
        return True

    auto_map = config.get("auto_map") or {}
    return any(
        "Lfm2Bidirectional" in str(mapped_class)
        or "modeling_lfm2_bidirectional" in str(mapped_class)
        for mapped_class in auto_map.values()
    )


def _remap_lfm2_bidirectional_model_type(
    config: dict,
    explicit_model_type: bool = False,
) -> None:
    if (
        not explicit_model_type
        and config.get("model_type") == "lfm2"
        and _uses_lfm2_bidirectional_architecture(config)
    ):
        config["model_type"] = "lfm2_bidirectional"


def _get_classes(config: dict, pipeline: Optional[str] = "masked-lm"):
    """
    Retrieve the model and model args classes based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    model_type = config.get("model_type")
    if not model_type:
        raise ValueError("Model config is missing the required 'model_type' field.")

    reference = resolve_model_class(model_type, pipeline)
    architecture = importlib.import_module(reference.module)
    try:
        return (
            getattr(architecture, reference.model_class),
            getattr(architecture, reference.args_class),
        )
    except AttributeError as error:
        raise RuntimeError(
            f"Runtime registry entry for {model_type!r}/{pipeline!r} does not "
            f"match module {reference.module!r}."
        ) from error


def _initialize_head_weights(model: nn.Module, loaded_weights: dict, config: Any):
    """
    If we are in training mode and missing head weights, we generate them 
    using the specific distribution required (e.g., Normal 0.02) rather 
    than relying on default initialization.
    """
    # Flattens the model so we know the shape and dtype of every expected parameter
    model_params = dict(tree_flatten(model.parameters()))
    
    # Keywords that identify a 'Head' or 'Classifier' layer in your architectures
    head_keywords = ["classifier", "score", "head", "decoder", "dense", "unembedding"]
    
    initializer_range = getattr(config, "initializer_range", 0.02)
    
    initialized_count = 0
    
    for key, param in model_params.items():
        # If the parameter is missing from the loaded checkpoint
        if key not in loaded_weights:
            # And it belongs to a prediction head
            if any(x in key for x in head_keywords):
                parameter_dtype = param.dtype
                
                # Initialize Biases to Zero
                if "bias" in key:
                    print(f"[INFO] Initializing missing bias {key} to Zeros ({parameter_dtype})")
                    loaded_weights[key] = mx.zeros(param.shape, dtype=parameter_dtype)
                
                # 2. Initialize Weights
                elif "weight" in key:
                    # Norm weights (Gamma) should be 1.0
                    if "norm" in key:
                         print(f"[INFO] Initializing missing normalization weight {key} to Ones ({parameter_dtype})")
                         loaded_weights[key] = mx.ones(param.shape, dtype=parameter_dtype)
                    # Other weights to Normal (std=0.02)
                    else:
                        print(f"[INFO] Initializing missing weight {key} with Normal(0.0, {initializer_range})  ({parameter_dtype})")
                        loaded_weights[key] = mx.random.normal(
                            param.shape, 
                            scale=initializer_range, 
                            dtype=parameter_dtype
                        )
                    
                initialized_count += 1

    if initialized_count > 0:
        print(f"[INFO] Explicitly initialized {initialized_count} missing parameters for transfer learning.")


def _verify_weights(model: nn.Module, loaded_weights: dict, train_mode: bool):
    """
    Ensures safety. 
    - Inference: CRASH if head weights are missing.
    - Training: PASS (we will initialize them next).
    """
    model_params = dict(tree_flatten(model.parameters()))
    missing_keys = [k for k in model_params.keys() if k not in loaded_weights]
    extra_keys = [k for k in loaded_weights.keys() if k not in model_params]
    
    head_keywords = ["classifier", "score", "head", "decoder", "dense", "unembedding"]
    missing_head_keys = [k for k in missing_keys if any(x in k for x in head_keywords)]

    if missing_head_keys:
        if not train_mode:
            # CRASH: User wants inference but loaded a base model
            raise ValueError(
                f"Weights missing for pipeline head: {missing_head_keys[:3]}...\n"
                f"You are trying to run Inference using a checkpoint that lacks the "
                f"classifier/decoder layers (likely a base model).\n"
                f"Set `train=True` if you intend to finetune this model."
                f" Extra keys found in loaded weights: {extra_keys[:3]}..."
            )


def compute_bits_per_weight(model):
    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
    )
    leaf_modules = tree_flatten(
        model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)
    )
    model_params = sum(nparams(m) for _, m in leaf_modules)
    return model_bytes * 8 / model_params


def load_config(model_path: Path) -> dict:
    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found in {model_path}")
        raise
    return config


def _load_sentence_transformer_dense_layers(
    model_path: Path,
    sentence_transformer_modules: list,
) -> list:
    dense_layers = []
    for module in sentence_transformer_modules:
        module_type = module.get("type", "")
        module_path = module.get("path", "")
        if "Dense" not in module_type or not module_path:
            continue

        config_file = model_path / module_path / "config.json"
        if not config_file.exists():
            continue

        with open(config_file, "r") as f:
            layer_config = json.load(f)
        layer_config["path"] = module_path
        dense_layers.append(layer_config)
    return dense_layers


def _load_sentence_transformer_pooling_mode(
    model_path: Path,
    sentence_transformer_modules: list,
) -> Optional[str]:
    """Return a supported pooling mode declared by a Pooling module."""
    for module in sentence_transformer_modules:
        module_type = module.get("type", "")
        module_path = module.get("path", "")
        if "Pooling" not in module_type or not module_path:
            continue

        config_file = model_path / module_path / "config.json"
        if not config_file.exists():
            continue

        with open(config_file, "r") as f:
            pooling_config = json.load(f)

        pooling_mode = pooling_config.get("pooling_mode")
        if pooling_mode in {"cls", "mean"}:
            return pooling_mode

        enabled_modes = [
            mode
            for config_key, mode in (
                ("pooling_mode_cls_token", "cls"),
                ("pooling_mode_mean_tokens", "mean"),
            )
            if pooling_config.get(config_key) is True
        ]
        if len(enabled_modes) == 1:
            return enabled_modes[0]

    return None


def _is_lateon_dense_layout(dense_layers: list) -> bool:
    expected = [
        (768, 1536, True),
        (1536, 768, True),
        (768, 128, False),
    ]
    if len(dense_layers) != len(expected):
        return False

    for layer, (in_features, out_features, use_residual) in zip(dense_layers, expected):
        if int(layer.get("in_features", -1)) != in_features:
            return False
        if int(layer.get("out_features", -1)) != out_features:
            return False
        if bool(layer.get("use_residual", False)) != use_residual:
            return False
    return True

def _prepare_quantized_modules(
    model: nn.Module,
    weights: dict,
    quantization_config: dict,
) -> None:
    """Create quantized module parameter structures before loading weights."""
    quantization_config = copy.deepcopy(quantization_config)
    layer_overrides = quantization_config.pop("layers", {})
    supported_keys = {"group_size", "bits", "mode", "quantize_input"}

    # Backward compatibility for configs that stored layer paths alongside the
    # global quantization arguments.
    for key in list(quantization_config):
        if key not in supported_keys:
            layer_overrides[key] = quantization_config.pop(key)

    def class_predicate(path, module):
        if f"{path}.scales" not in weights:
            return False
        override = layer_overrides.get(path)
        return override if override is not None else hasattr(module, "to_quantized")

    nn.quantize(
        model,
        **quantization_config,
        class_predicate=class_predicate,
    )


def load_model(
    model_path: Path,
    lazy: bool = False,
    model_config: Optional[dict] = None,
    get_model_classes: Callable[..., Tuple[Type[nn.Module], Type]] = _get_classes,
    pipeline: Optional[str] = None,
    train: bool = False,
) -> Tuple[nn.Module, dict]:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        model_config (dict, optional): Configuration parameters for the model.
            Defaults to an empty dictionary.
        get_model_classes (Callable[[dict], Tuple[Type[nn.Module], Type]], optional):
            A function that returns the model class and model args class given a config.
            Defaults to the _get_classes function.
        pipeline (str, optional): The pipeline type. If None, it will be inferred
            from the model configuration. Defaults to None.
        train (bool, optional): Whether the model is being loaded for training.
            In training model, models can be loaded from a different pipeline and
            some weights can be initialized accordingly. Defaults to False.
    Returns:
        Tuple[nn.Module, dict]: The loaded model and its resolved configuration.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """

    if model_config is None:
        model_config = {}

    # check if model_path/config_sentence_transformers.json exists
    sentence_transformers_config_file = model_path / "config_sentence_transformers.json"
    modules_file = model_path / "modules.json"
    is_sentence_transformer = sentence_transformers_config_file.exists()
    sentence_transformers_config = {}
    sentence_transformer_modules = []

    if is_sentence_transformer:
        with open(sentence_transformers_config_file, "r") as f:
            sentence_transformers_config = json.load(f)

    if modules_file.exists():
        with open(modules_file, "r") as f:
            sentence_transformer_modules = json.load(f)

    config = load_config(model_path)
    if 'is_encoder_decoder' in config and config.get('encoder', None):
        model_type = config['model_type']
        print(f"[INFO] Detected {model_type} model, merging encoder config.")
        # merge encoder config for main models
        encoder_config = config.get('encoder', {})
        encoder_config['model_type'] = model_type + '_encoder'
        config.update(encoder_config)
    explicit_model_type = "model_type" in model_config
    config.update(model_config)
    _remap_lfm2_bidirectional_model_type(config, explicit_model_type)

    if is_sentence_transformer:
        config["sentence_transformers_config"] = sentence_transformers_config
        config["sentence_transformer_modules"] = sentence_transformer_modules
        
        # this feels like a terrible hack
        # the only way to avoid this would be to have something closer
        # to Sentence Transformers with a dynamic architecture that can be inferred from the config, 
        # that may be a future iteration 
        dense_layers = _load_sentence_transformer_dense_layers(
            model_path,
            sentence_transformer_modules,
        )
        if dense_layers:
            config["pylate_dense_layers"] = dense_layers
            config["colbert_dim"] = dense_layers[-1].get(
                "out_features",
                config.get("colbert_dim", 128),
            )

        pooling_mode = _load_sentence_transformer_pooling_mode(
            model_path,
            sentence_transformer_modules,
        )
        if pooling_mode is not None:
            # Architecture wrappers use one of these two names. Supplying both
            # lets the SentenceTransformer module remain authoritative while
            # unknown dataclass fields are ignored by other architectures.
            config["classifier_pooling"] = pooling_mode
            config["pooling_mode"] = pooling_mode

        st_model_type = sentence_transformers_config.get("model_type")
        if not explicit_model_type and st_model_type == "ColBERT":
            if (
                config.get("model_type") == "modernbert"
                and _is_lateon_dense_layout(dense_layers)
            ):
                config["model_type"] = "lateon"
            else:
                config["model_type"] = "colbert_zero"
            config.setdefault("use_late_interaction", True)

        ## Currently a contradiction between pooling strategy in granite's config
        ## and in the sentence_transformers config, 
        ## we can enforece the sentence_transformers config here
        ## or even route to a different model class, 
        ## for now, all this is ignore and config will be passed to modernbert
        
        # if (
        #     not explicit_model_type
        #     and config.get("model_type") == "modernbert"
        #     and config.get("vocab_size") == 180000
        #     and config.get("hidden_size") == 384
        # ):
        #     config["model_type"] = "granite_embedding"
        #     config["classifier_pooling"] = "cls"

    arch = config.get("architectures", None)
    model_arch = None
    if arch is not None:
        model_arch = _get_pipeline_from_config(arch[0])

    if model_arch is not None:
        if pipeline is None:
            pipeline = model_arch
            print(f"[INFO] Using pipeline {pipeline} based on model architecture {model_arch}")
        elif pipeline != model_arch:
                print(
                    f"[INFO] Using pipeline {pipeline} based on user input, ignoring model architecture {model_arch}"
                )

    if is_sentence_transformer:
        if pipeline is None:
            pipeline = "sentence-transformers"
        if pipeline not in ["sentence-transformers", "embeddings", "sentence-similarity"]:
            if not train:
                raise ValueError(
                    f"Pipeline '{pipeline}' cannot be used with a Sentence Transformer model in Inference mode. "
                    f"These models only support embeddings/similarity."
                )
            else:
                print(f"[INFO] Adaptation: Loading Sentence Transformer base into {pipeline} pipeline for training.")
        else:
            pipeline = "sentence-transformers"
            print(f"[INFO] Using pipeline {pipeline} based on Sentence Transformer config file.")
    elif pipeline is None:
        pipeline = "embeddings"
        print("[INFO] No task architecture found; using the embeddings pipeline.")

    # Validate the model/pipeline pair before reading checkpoint tensors. This
    # rejects unimplemented placeholder heads without paying the I/O cost first.
    model_class, model_args_class = get_model_classes(
        config=config,
        pipeline=pipeline,
    )

    weights = {}

    # Sentence Transformer weights may be loaded from subfolders 
    # prefix keys added so sanitize() can identify them
    if is_sentence_transformer and modules_file.exists():
        for module in sentence_transformer_modules:
            sub_path = module.get("path", "")
            module_dir = model_path / sub_path
            
            module_weights = glob.glob(str(module_dir / "model*.safetensors"))
            if not module_weights:
                # Fallback for older naming conventions
                module_weights = glob.glob(str(module_dir / "weight*.safetensors"))
            
            for wf in module_weights:
                sub_weights = mx.load(wf)
                for k, v in sub_weights.items():
                    # prefix the key 'linear.weight' -> '1_Dense.linear.weight'
                    # This allows the regex in sanitize() (r"\d+_Dense\.linear").
                    if sub_path:
                        weights[f"{sub_path}.{k}"] = v
                    else:
                        # Root module (Transformer), load keys as is
                        weights[k] = v

    # Load weights from safetensors at the root of model_path
    # Typically for non-Sentence Transformer models
    if not weights:
        weight_files = glob.glob(str(model_path / "model*.safetensors"))
        if not weight_files:
            # Try weight for back-compat
            weight_files = glob.glob(str(model_path / "weight*.safetensors"))

        if not weight_files:
            logging.error(f"No safetensors found in {model_path}")
            raise FileNotFoundError(f"No safetensors found in {model_path}")

        for wf in weight_files:
            weights.update(mx.load(wf))

    dtype_policy = _resolve_model_dtype(config, weights)
    config["dtype"] = dtype_policy.config_name
    print(
        "[INFO] Model parameter dtype: "
        f"{dtype_policy.dtype} (source: {dtype_policy.source})"
    )

    model_args = model_args_class.from_dict(config)
    
    # Instantiate the model (random init)
    model = model_class(model_args)
    calibration_file = model_path / "viterbi_calibration.json"
    if calibration_file.exists():
        with open(calibration_file) as f:
            model.viterbi_calibration = json.load(f)

    # Use set_dtype to update all floating-point parameters recursively.
    # The default predicate ensures we don't accidentally cast integer params.
    model.set_dtype(dtype_policy.dtype)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    if (quantization := config.get("quantization")) is not None:
        _prepare_quantized_modules(model, weights, quantization)

    weights = _align_checkpoint_dtypes(
        weights,
        model,
        cast_floating=dtype_policy.cast_checkpoint,
    )

    _verify_weights(model, weights, train_mode=train)

    if train:
        _initialize_head_weights(model, weights, model_args)

    model.load_weights(list(weights.items()))

    model.config.dtype = dtype_policy.config_name
    _validate_explicit_model_dtype(model, dtype_policy)

    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model, config


def load(
    path_or_hf_repo: str,
    tokenizer_config: Optional[dict] = None,
    model_config: Optional[dict] = None,
    adapter_path: Optional[str] = None,
    lazy: bool = False,
    pipeline: Optional[str] = None,
    train: bool = False
) -> Tuple[nn.Module, TokenizerWrapper]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (Path): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
            Defaults to an empty dictionary.
        model_config(dict, optional): Configuration parameters specifically for the model.
            Defaults to an empty dictionary.
        adapter_path (str, optional): Reserved for adapter loading, which is not
            currently implemented. Passing a value raises ``NotImplementedError``.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        pipeline (str, optional): The pipeline type. If None, it will be inferred
            from the model configuration. Defaults to None.
        train (bool, optional): Whether the model is being loaded for training.
            In training model, models can be loaded from a different pipeline and
            some weights can be initialized accordingly. Defaults to False.
    Returns:
        Tuple[nn.Module, TokenizerWrapper]: A tuple containing the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    if adapter_path is not None:
        raise NotImplementedError(
            "Adapter loading is not currently supported; adapter_path must be None."
        )

    model_path = get_model_path(path_or_hf_repo)

    model, config = load_model(
        model_path,
        lazy,
        model_config=model_config,
        pipeline=pipeline,
        train=train,
    )
    tokenizer = load_tokenizer(model_path, tokenizer_config)

    return model, tokenizer

def quantize_model(
    model: nn.Module,
    config: dict,
    q_group_size: int = 64,
    q_bits: int = 4,
    quant_predicate: Optional[
        Callable[[str, nn.Module, dict], Union[bool, dict]]
    ] = None,
) -> Tuple:
    """
    Applies quantization to the model weights.

    Args:
        model (nn.Module): The model to be quantized.
        config (dict): Model configuration.
        q_group_size (int): Group size for quantization.
        q_bits (int): Bits per weight for quantization.
        quant_predicate (Callable): A callable that decides how
            to quantize each layer based on the path.
            Accepts the layer `path`, the `module` and the model `config`.
            Returns either a bool to signify quantize/no quantize or
            a dict of quantization parameters to pass to `to_quantized`.

    Returns:
        Tuple: Tuple containing quantized weights and config.
    """
    quantized_config = copy.deepcopy(config)
    quantized_config["quantization"] = {"group_size": q_group_size, "bits": q_bits}

    # Add any custom quantization parameters to the config as we go
    def _class_predicate(p, m):
        bool_or_params = quant_predicate(p, m, config)
        quantized_config["quantization"].setdefault("layers", {})[p] = bool_or_params
        return bool_or_params

    nn.quantize(
        model,
        q_group_size,
        q_bits,
        class_predicate=_class_predicate if quant_predicate else None,
    )
    # support hf model tree #957
    quantized_config["quantization_config"] = quantized_config["quantization"]
    quantized_weights = dict(tree_flatten(model.parameters()))

    bpw = compute_bits_per_weight(model)
    print(f"[INFO] Quantized model with {bpw:.3f} bits per weight.")

    return quantized_weights, quantized_config
