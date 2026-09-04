"""Public package interface.

Model classes are imported lazily so the Windows acquisition environment can
use the configuration and recording utilities without installing PyTorch.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["EEGTransformerEncoder", "EncoderConfig", "WorkloadClassifier"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    models = import_module(".models", __name__)
    return getattr(models, name)
