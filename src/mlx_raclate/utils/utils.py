# Copyright © 2023-2024 Apple Inc.

import contextlib
import copy
import glob
import importlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_reduce
from huggingface_hub import snapshot_download
from transformers import PreTrainedTokenizer

# Local imports
from .tokenizer_utils import TokenizerWrapper, load_tokenizer
# Training imports 
from mlx_raclate.tuner.utils import nparams #, load_adapters ### removing adapters for now

PIPELINES = [
    "embeddings",
    "masked-lm", 
    "text-classification", 
    "token-classification",
    "sentence-transformers",
    "zero-shot-classification",
    "sentence-similarity"
]

HF_ARCH_TO_PIPELINE_MAPPING = {
    "ForSequenceClassification": "text-classification",
    "ForMaskedLM": "masked-lm",
    "ForTokenClassification": "token-classification",
}

MODEL_REMAPPING = {
    "mistral": "llama",  # mistral is compatible with llama
    "phi-msft": "phixtral"
}

MAX_FILE_SIZE_GB = 5

class ModelNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def _get_pipeline_from_config(arch : str):
    """
    Retrieve the pipeline type based on the model configuration.

    Args:
        arch: first item of architectures from the model configuration.

    Returns:
        str: The pipeline type.
    """
    if arch is not None:
        for k,v in HF_ARCH_TO_PIPELINE_MAPPING.items():
            if k in arch:
                return v
    return None  


def _get_classes(config: dict, pipeline: Optional[str] = 'masked-lm'):
    """
    Retrieve the model and model args classes based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    if pipeline not in PIPELINES:
        raise ValueError(f"Pipeline {pipeline} not supported. Supported pipelines: {PIPELINES}")

    model_type = config["model_type"]
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    try:
        arch = importlib.import_module(f"mlx_raclate.models.{model_type}")
    except ImportError:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)

    if pipeline == "masked-lm":
        return arch.ModelForMaskedLM, arch.ModelArgs
    
    if pipeline == "text-classification":
        return arch.ModelForSequenceClassification, arch.ModelArgs
    
    if pipeline == "token-classification":
        return arch.ModelForTokenClassification, arch.ModelArgs
    
    if pipeline == "embeddings":
        return arch.Model, arch.ModelArgs
    
    if pipeline == "sentence-transformers":
        return arch.ModelForSentenceTransformers, arch.ModelArgs
    
    if pipeline == "zero-shot-classification":
        return arch.ModelForMaskedLM, arch.ModelArgs
        # using the MaskeLM pipeline for now (see models/modernbert.py comment for class ModelForZeroShotClassification)
        # return arch.ModelForZeroShotClassification, arch.ModelArgs
    
    if pipeline == "sentence-similarity":
        return arch.ModelForSentenceSimilarity, arch.ModelArgs

    ### should not reach here
    return arch.Model, arch.ModelArgs


def _initialize_head_weights(model: nn.Module, loaded_weights: dict, config: Any):
    """
    If we are in training mode and missing head weights, we generate them 
    using the specific distribution required (e.g., Normal 0.02) rather 
    than relying on default initialization.
    """
    # Flattens the model so we know the shape and dtype of every expected parameter
    model_params = dict(tree_flatten(model.parameters()))
    
    # Keywords that identify a 'Head' or 'Classifier' layer in your architectures
    head_keywords = ["classifier", "score", "head", "decoder"]
    
    initializer_range = getattr(config, "initializer_range", 0.02)
    
    initialized_count = 0
    
    for key, param in model_params.items():
        # If the parameter is missing from the loaded checkpoint
        if key not in loaded_weights:
            # And it belongs to a prediction head
            if any(x in key for x in head_keywords):
                
                # 1. Initialize Biases to Zero
                if "bias" in key:
                    print(f"[INFO] Initializing missing bias {key} to Zeros")
                    loaded_weights[key] = mx.zeros_like(param)
                
                # 2. Initialize Weights to Normal (std=0.02)
                # We skip 'norm' weights (Gamma) which should be 1.0, but usually
                # those are covered by default init or exist in the checkpoint.
                elif "weight" in key and "norm" not in key:
                    print(f"[INFO] Initializing missing weight {key} with Normal(0.0, {initializer_range})")
                    loaded_weights[key] = mx.random.normal(
                        param.shape, 
                        scale=initializer_range, 
                        dtype=param.dtype
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
    
    head_keywords = ['classifier', 'score', 'head', 'decoder']
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


def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
        revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        try:
            model_path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    revision=revision,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                        "*.txt",
                    ],
                )
            )
        except:
            raise ModelNotFoundError(
                f"Model not found for path or HF repo: {path_or_hf_repo}.\n"
                "Please make sure you specified the local path or Hugging Face"
                " repo id correctly.\nIf you are trying to access a private or"
                " gated Hugging Face repo, make sure you are authenticated:\n"
                "https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login"
            ) from None
    return model_path


def load_config(model_path: Path) -> dict:
    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found in {model_path}")
        raise
    return config


def load_model(
    model_path: Path,
    lazy: bool = False,
    model_config: dict = {},
    get_model_classes: Callable[[dict], Tuple[Type[nn.Module], Type]] = _get_classes,
    pipeline: Optional[str] = None,
    train: bool = False, 
) -> nn.Module:
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
        nn.Module: The loaded and initialized model.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """

    # check if model_path/config_sentence_transformers.json exists
    is_sentence_transformer= (model_path / "config_sentence_transformers.json").exists()

    config = load_config(model_path)
    if 'is_encoder_decoder' in config and config.get('encoder', None):
        model_type = config['model_type']
        print(f"[INFO] Detected {model_type} model, merging encoder config.")
        # merge encoder config for main models
        encoder_config = config.get('encoder', {})
        encoder_config['model_type'] = model_type + '_encoder'
        config.update(encoder_config)
    
    config.update(model_config)

    arch = config.get("architectures", None)
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

    if is_sentence_transformer :
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

    weights = {}
    modules_file = model_path / "modules.json"

    # Sentence Transformer weights may be loaded from subfolders 
    # prefix keys added so sanitize() can identify them
    if is_sentence_transformer and modules_file.exists():
        with open(modules_file, "r") as f:
            modules = json.load(f)
        
        for module in modules:
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
    model_class, model_args_class = get_model_classes(config=config, pipeline=pipeline)

    model_args = model_args_class.from_dict(config)
    
    # Instantiate the model (random init)
    model = model_class(model_args)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    
    _verify_weights(model, weights, train_mode=train)

    if train:
        _initialize_head_weights(model, weights, model_args)

    model.load_weights(list(weights.items()))

    if (quantization := config.get("quantization", None)) is not None:
        # Handle legacy models which may not have everything quantized
        def class_predicate(p, m):
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            **quantization,
            class_predicate=class_predicate,
        )

    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model, config


def load(
    path_or_hf_repo: str,
    tokenizer_config={},
    model_config={},
    adapter_path: Optional[str] = None, ## for now, disabling adapter loading
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
        adapter_path (str, optional): Path to the LoRA adapters. If provided, applies LoRA layers
            to the model. Default: ``None``.
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
    model_path = get_model_path(path_or_hf_repo)

    model, config = load_model(model_path, lazy, model_config, pipeline=pipeline, train=train)
    ### disabling adapter for encoders
    # if adapter_path is not None:
    #     model = load_adapters(model, adapter_path)
    #     model.eval()
    tokenizer = load_tokenizer(model_path, tokenizer_config)

    return model, tokenizer

def fetch_from_hub(
    model_path: Path, lazy: bool = False
) -> Tuple[nn.Module, dict, PreTrainedTokenizer]:
    model, config = load_model(model_path, lazy)
    tokenizer = load_tokenizer(
        model_path, eos_token_ids=config.get("eos_token_id", None)
    )
    return model, config, tokenizer


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
        quantized_config["quantization"][p] = bool_or_params
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

### Conversion should not be needed if we work with safetensors
# def convert(
#     hf_path: str,
#     mlx_path: str = "mlx_model",
#     quantize: bool = False,
#     q_group_size: int = 64,
#     q_bits: int = 4,
#     dtype: str = "float16",
#     upload_repo: str = None,
#     revision: Optional[str] = None,
#     dequantize: bool = False,
#     quant_predicate: Optional[
#         Callable[[str, nn.Module, dict], Union[bool, dict]]
#     ] = None,
# ):
#     # Check the save path is empty
#     if isinstance(mlx_path, str):
#         mlx_path = Path(mlx_path)

#     if mlx_path.exists():
#         raise ValueError(
#             f"Cannot save to the path {mlx_path} as it already exists."
#             " Please delete the file/directory or specify a new path to save to."
#         )

#     print("[INFO] Loading")
#     model_path = get_model_path(hf_path, revision=revision)
#     model, config, tokenizer = fetch_from_hub(model_path, lazy=True)

#     weights = dict(tree_flatten(model.parameters()))
#     dtype = getattr(mx, dtype)
#     weights = {k: v.astype(dtype) for k, v in weights.items()}

#     if quantize and dequantize:
#         raise ValueError("Choose either quantize or dequantize, not both.")

#     if quantize:
#         print("[INFO] Quantizing")
#         model.load_weights(list(weights.items()))
#         weights, config = quantize_model(
#             model, config, q_group_size, q_bits, quant_predicate=quant_predicate
#         )

#     if dequantize:
#         print("[INFO] Dequantizing")
#         model = dequantize_model(model)
#         weights = dict(tree_flatten(model.parameters()))

#     del model
#     save_weights(mlx_path, weights, donate_weights=True)

#     py_files = glob.glob(str(model_path / "*.py"))
#     for file in py_files:
#         shutil.copy(file, mlx_path)

#     tokenizer.save_pretrained(mlx_path)

#     save_config(config, config_path=mlx_path / "config.json")

#     if upload_repo is not None:
#         upload_to_hub(mlx_path, upload_repo, hf_path)