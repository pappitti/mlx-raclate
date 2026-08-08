"""Runtime architecture capabilities without importing MLX model modules."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional


PIPELINE_CLASS_NAMES = MappingProxyType(
    {
        "embeddings": "Model",
        "masked-lm": "ModelForMaskedLM",
        "text-classification": "ModelForSequenceClassification",
        "token-classification": "ModelForTokenClassification",
        "sentence-transformers": "ModelForSentenceTransformers",
        # Zero-shot classification currently uses the masked-LM implementation.
        "zero-shot-classification": "ModelForMaskedLM",
        "sentence-similarity": "ModelForSentenceSimilarity",
    }
)
PIPELINES = tuple(PIPELINE_CLASS_NAMES)

_EMBEDDING_PIPELINES = (
    "embeddings",
    "sentence-transformers",
    "sentence-similarity",
)


@dataclass(frozen=True)
class ArchitectureCapabilities:
    """Import path and implemented pipeline classes for one model type."""

    module: str
    pipeline_classes: Mapping[str, str]
    args_class: str = "ModelArgs"


@dataclass(frozen=True)
class ModelClassReference:
    """Names needed to import a model implementation lazily."""

    module: str
    model_class: str
    args_class: str


def _architecture(
    model_type: str,
    *additional_pipelines: str,
) -> ArchitectureCapabilities:
    pipelines = _EMBEDDING_PIPELINES + additional_pipelines
    return ArchitectureCapabilities(
        module=f"mlx_raclate.models.{model_type}",
        pipeline_classes=MappingProxyType(
            {pipeline: PIPELINE_CLASS_NAMES[pipeline] for pipeline in pipelines}
        ),
    )


ARCHITECTURES = MappingProxyType(
    {
        "bidirectional_pplx_qwen3": _architecture(
            "bidirectional_pplx_qwen3",
            "masked-lm",
            "text-classification",
            "token-classification",
            "zero-shot-classification",
        ),
        "colbert_zero": _architecture("colbert_zero"),
        "gemma3_text": _architecture(
            "gemma3_text",
            "masked-lm",
            "text-classification",
            "token-classification",
            "zero-shot-classification",
        ),
        "granite_embedding": _architecture("granite_embedding"),
        "lateon": _architecture("lateon"),
        "lfm2": _architecture("lfm2", "text-classification"),
        "lfm2_bidirectional": _architecture(
            "lfm2_bidirectional",
            "masked-lm",
            "token-classification",
            "zero-shot-classification",
        ),
        "modernbert": _architecture(
            "modernbert",
            "masked-lm",
            "text-classification",
            "token-classification",
            "zero-shot-classification",
        ),
        "neobert": _architecture(
            "neobert",
            "masked-lm",
            "text-classification",
            "token-classification",
            "zero-shot-classification",
        ),
        "openai_privacy_filter": _architecture(
            "openai_privacy_filter",
            "text-classification",
            "token-classification",
        ),
        "qwen3": _architecture("qwen3", "text-classification"),
        "t5gemma_encoder": _architecture(
            "t5gemma_encoder",
            "masked-lm",
            "text-classification",
            "token-classification",
            "zero-shot-classification",
        ),
    }
)

_EXACT_ARCHITECTURE_PIPELINES = MappingProxyType(
    {"Lfm2BidirectionalModel": "embeddings"}
)
_ARCHITECTURE_SUFFIX_PIPELINES = MappingProxyType(
    {
        "ForSequenceClassification": "text-classification",
        "ForMaskedLM": "masked-lm",
        "ForTokenClassification": "token-classification",
    }
)


def infer_pipeline_from_architecture(architecture: Optional[str]) -> Optional[str]:
    """Infer a pipeline from a Hugging Face architecture class name."""
    if architecture is None:
        return None
    if architecture in _EXACT_ARCHITECTURE_PIPELINES:
        return _EXACT_ARCHITECTURE_PIPELINES[architecture]
    for suffix, pipeline in _ARCHITECTURE_SUFFIX_PIPELINES.items():
        if suffix in architecture:
            return pipeline
    return None


def resolve_model_class(
    model_type: str, pipeline: Optional[str]
) -> ModelClassReference:
    """Resolve implemented class names for a model type and pipeline."""
    if pipeline not in PIPELINES:
        raise ValueError(
            f"Pipeline {pipeline!r} is not supported. Supported pipelines: "
            f"{', '.join(PIPELINES)}."
        )

    capabilities = ARCHITECTURES.get(model_type)
    if capabilities is None:
        raise ValueError(
            f"Model type {model_type!r} is not supported. Supported model types: "
            f"{', '.join(ARCHITECTURES)}."
        )

    model_class = capabilities.pipeline_classes.get(pipeline)
    if model_class is None:
        supported = ", ".join(capabilities.pipeline_classes)
        raise ValueError(
            f"Pipeline {pipeline!r} is not implemented for model type "
            f"{model_type!r}. Supported pipelines for this model: {supported}."
        )

    return ModelClassReference(
        module=capabilities.module,
        model_class=model_class,
        args_class=capabilities.args_class,
    )
