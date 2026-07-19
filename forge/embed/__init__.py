"""Embedding models for Forge (Phase 2 of the catalog).

The embedding *engine*: models that map images/text to vectors, plus frame
sampling and pooling. The catalog (:mod:`forge.catalog`) stores and searches the
vectors — this package knows nothing about it.

    from forge.embed import get_model
    model = get_model("siglip-so400m", device="auto")
    vecs = model.embed_images([frame_hwc_uint8, ...])   # (N, model.dim) float32

Imports are lazy (PEP 562) so importing :mod:`forge` never pulls in torch —
that loads only when a real model is instantiated. Tests register a torch-free
fake model via :func:`register_model`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = [
    "EmbeddingModel",
    "ModelError",
    "select_device",
    "get_model",
    "register_model",
    "available_models",
    "DEFAULT_MODEL",
    "sample_frame_indices",
    "pool",
]

_LAZY = {
    "EmbeddingModel": "forge.embed.base",
    "ModelError": "forge.embed.base",
    "select_device": "forge.embed.base",
    "get_model": "forge.embed.registry",
    "register_model": "forge.embed.registry",
    "available_models": "forge.embed.registry",
    "DEFAULT_MODEL": "forge.embed.registry",
    "sample_frame_indices": "forge.embed.sampling",
    "pool": "forge.embed.sampling",
}

if TYPE_CHECKING:
    from forge.embed.base import EmbeddingModel, ModelError, select_device
    from forge.embed.registry import (
        DEFAULT_MODEL,
        available_models,
        get_model,
        register_model,
    )
    from forge.embed.sampling import pool, sample_frame_indices


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)
