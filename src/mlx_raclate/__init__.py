"""Public API for :mod:`mlx_raclate`.

Heavy MLX modules are imported only when their functionality is requested so
package metadata and command help remain cheap to inspect.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Any


try:
    __version__ = version("mlx-raclate")
except PackageNotFoundError:  # Source tree without an installed distribution.
    __version__ = "0.0.0+local"


def load(*args: Any, **kwargs: Any):
    """Load a Raclate model and tokenizer.

    This is the stable public wrapper around the implementation in
    :mod:`mlx_raclate.utils.utils`.
    """
    from .utils.utils import load as _load

    return _load(*args, **kwargs)


__all__ = ["__version__", "load"]
