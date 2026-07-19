"""Registry of embedding models.

Models register a factory under a short name; :func:`get_model` instantiates one
by name (loading weights lazily). Built-in models are imported on demand so the
registry itself stays torch-free — tests register a deterministic fake model and
never trigger the SigLIP/torch import path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from forge.embed.base import ModelError

if TYPE_CHECKING:
    from forge.embed.base import EmbeddingModel

DEFAULT_MODEL = "siglip-so400m"

# name -> factory(**kwargs) -> EmbeddingModel
_FACTORIES: dict[str, Callable[..., EmbeddingModel]] = {}

# Built-in models and the module that registers them (imported on demand).
_BUILTIN_MODULES = {
    "siglip-so400m": "forge.embed.siglip",
}


def register_model(
    name: str, factory: Callable[..., EmbeddingModel] | None = None
):
    """Register an embedding-model factory under ``name``.

    Usable as a decorator (``@register_model("x")``) or directly
    (``register_model("x", factory)``).
    """

    def _register(f: Callable[..., EmbeddingModel]) -> Callable[..., EmbeddingModel]:
        _FACTORIES[name] = f
        return f

    return _register(factory) if factory is not None else _register


def available_models() -> list[str]:
    """Registered model names (plus built-ins not yet imported)."""
    return sorted(set(_FACTORIES) | set(_BUILTIN_MODULES))


def get_model(name: str | None = None, **kwargs) -> EmbeddingModel:
    """Instantiate the model registered as ``name`` (default: SigLIP).

    Built-in models are imported on first use. ``kwargs`` (e.g. ``device``) are
    passed to the factory.
    """
    name = name or DEFAULT_MODEL
    if name not in _FACTORIES and name in _BUILTIN_MODULES:
        __import__(_BUILTIN_MODULES[name])
    if name not in _FACTORIES:
        raise ModelError(
            f"unknown embedding model '{name}'. Available: {available_models()}"
        )
    return _FACTORIES[name](**kwargs)
