"""TESSERACT compiler–executor language model scaffold."""

from . import backbone, critic, vm

try:
    from . import compiler
except (ModuleNotFoundError, ImportError):  # pragma: no cover - optional runtime dependency path
    compiler = None  # type: ignore[assignment]

__all__ = [
    "backbone",
    "compiler",
    "vm",
    "critic",
]
