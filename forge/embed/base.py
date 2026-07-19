"""Embedding-model interface and device selection.

An ``EmbeddingModel`` maps images (and, for shared-space models, text) to fixed
dimensional vectors. The catalog stores those vectors and searches them; this
module knows nothing about the catalog. Concrete models (SigLIP, or a fake
deterministic model for tests) live in sibling modules and register themselves
via :mod:`forge.embed.registry`.

Only numpy is imported at module load — torch is imported lazily inside
:func:`select_device` and the model implementations, so importing
``forge.embed`` never requires the ``[embed]`` extra (tests use a torch-free
fake model).
"""

from __future__ import annotations

import abc

import numpy as np

from forge.core.exceptions import ForgeError


class ModelError(ForgeError):
    """Raised for embedding-model problems (unknown model, text unsupported…)."""


def select_device(preference: str = "auto") -> str:
    """Pick a torch device string: ``cuda`` → ``mps`` → ``cpu``.

    ``preference`` other than ``"auto"`` is returned as-is (e.g. to force
    ``cpu``). Apple Silicon Macs resolve to ``mps`` (Metal).
    """
    if preference and preference != "auto":
        return preference
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows to unit length (zero rows left as zero)."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return (x / norm).astype(np.float32)


class EmbeddingModel(abc.ABC):
    """Maps images/text to unit-length vectors of a fixed dimension.

    Implementations return **L2-normalized float32** vectors so cosine
    similarity reduces to a dot product. ``model_id`` combines the model name
    with a checkpoint hash so two different checkpoints are never confused and
    vectors are reproducible across machines.
    """

    #: Short registry name, e.g. "siglip-so400m".
    name: str = "base"
    #: Whether embed_text() is supported (shared image-text space).
    supports_text: bool = False

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Vector dimension D (fixed for this model)."""

    @property
    @abc.abstractmethod
    def checkpoint_hash(self) -> str:
        """Short, stable id of the exact checkpoint (e.g. HF revision sha)."""

    @property
    def model_id(self) -> str:
        return f"{self.name}@{self.checkpoint_hash}"

    @abc.abstractmethod
    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        """Embed a batch of HWC uint8 images → ``(len(images), dim)`` float32."""

    def embed_text(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of strings → ``(len(texts), dim)`` float32."""
        raise ModelError(f"model '{self.name}' does not support text embedding")
