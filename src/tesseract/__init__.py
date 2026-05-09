"""Typed execution coprocessor for exact language-model computation."""

from importlib import import_module
from types import ModuleType

from . import backbone, critic, vm

__version__ = "2.0.0"

_OPTIONAL_RUNTIME_DEPENDENCIES = {"numpy", "torch"}


def _import_optional_submodule(name: str) -> ModuleType | None:
    try:
        return import_module(name, package=__name__)
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via package import tests
        if exc.name in _OPTIONAL_RUNTIME_DEPENDENCIES:
            return None
        raise


compiler = _import_optional_submodule(".compiler")
evaluation = _import_optional_submodule(".evaluation")

__all__ = [
    "backbone",
    "compiler",
    "vm",
    "critic",
    "evaluation",
    "__version__",
]
