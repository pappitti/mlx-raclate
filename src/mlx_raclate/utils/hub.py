"""Lightweight local/Hugging Face path resolution helpers."""

from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download


MODEL_ALLOW_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "*.txt",
)
TOKENIZER_ALLOW_PATTERNS = (
    "*.json",
    "*.model",
    "*.py",
    "*.tiktoken",
    "*.txt",
)


class ModelNotFoundError(Exception):
    """Raised when a model path cannot be resolved locally or from the Hub."""


def get_model_path(
    path_or_hf_repo: str,
    revision: Optional[str] = None,
    *,
    allow_patterns: Sequence[str] = MODEL_ALLOW_PATTERNS,
) -> Path:
    """Resolve a local path or download the selected files from the Hub."""
    model_path = Path(path_or_hf_repo)
    if model_path.exists():
        return model_path

    try:
        return Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                revision=revision,
                allow_patterns=list(allow_patterns),
            )
        )
    except Exception:
        raise ModelNotFoundError(
            f"Model not found for path or HF repo: {path_or_hf_repo}.\n"
            "Please make sure you specified the local path or Hugging Face"
            " repo id correctly.\nIf you are trying to access a private or"
            " gated Hugging Face repo, make sure you are authenticated:\n"
            "https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login"
        ) from None
