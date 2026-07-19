"""SigLIP embedding model (shared image–text space → text search works).

Default checkpoint ``google/siglip-so400m-patch14-384`` (D=1152). Device is
auto-selected (``cuda`` → ``mps`` → ``cpu``); on Apple Metal we enable the CPU
fallback for any op not implemented on MPS and keep fp32 for numerical safety.
Registered lazily so importing :mod:`forge.embed` never pulls in torch.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from forge.core.exceptions import MissingDependencyError
from forge.embed.base import EmbeddingModel, l2_normalize, select_device
from forge.embed.registry import register_model

_DEFAULT_REPO = "google/siglip-so400m-patch14-384"


def _require() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise MissingDependencyError(
            dependency="torch, transformers",
            feature="embedding models (SigLIP)",
            install_hint="pip install forge-robotics[embed]",
        ) from exc


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Coerce a frame to HWC uint8 RGB for the image processor."""
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(arr.max(initial=0.0)) <= 1.0:
            arr = (arr * 255.0).round()
        arr = arr.clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


class SiglipModel(EmbeddingModel):
    """SigLIP image + text encoder via transformers."""

    name = "siglip-so400m"
    supports_text = True

    def __init__(
        self,
        *,
        repo: str = _DEFAULT_REPO,
        device: str = "auto",
        revision: str | None = None,
    ):
        _require()
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self.device = select_device(device)
        if self.device == "mps":
            # Let ops unimplemented on Metal fall back to CPU instead of erroring.
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        self._repo = repo
        self._processor = AutoProcessor.from_pretrained(repo, revision=revision)
        self._model = (
            AutoModel.from_pretrained(repo, revision=revision).to(self.device).eval()
        )

        commit = getattr(self._model.config, "_commit_hash", None) or revision or "main"
        self._ckpt = hashlib.sha1(f"{repo}@{commit}".encode()).hexdigest()[:12]
        vision_cfg = getattr(self._model.config, "vision_config", None)
        self._dim = int(getattr(vision_cfg, "hidden_size", 0)) or 1152

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def checkpoint_hash(self) -> str:
        return self._ckpt

    @staticmethod
    def _to_tensor(out):
        """Extract the feature tensor across transformers versions.

        transformers >=5 returns a ``BaseModelOutputWithPooling`` (use
        ``pooler_output``); older versions return the tensor directly.
        """
        pooled = getattr(out, "pooler_output", None)
        return pooled if pooled is not None else out

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        from PIL import Image

        pil = [Image.fromarray(_to_uint8_rgb(im)) for im in images]
        inputs = self._processor(images=pil, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            feats = self._to_tensor(self._model.get_image_features(**inputs))
        return l2_normalize(feats.float().cpu().numpy())

    def embed_text(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # SigLIP text tower expects max_length padding (64 tokens).
        inputs = self._processor(
            text=list(texts),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with self._torch.no_grad():
            feats = self._to_tensor(self._model.get_text_features(**inputs))
        return l2_normalize(feats.float().cpu().numpy())


@register_model("siglip-so400m")
def _make_siglip(**kwargs) -> SiglipModel:
    return SiglipModel(**kwargs)
