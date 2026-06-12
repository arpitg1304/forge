"""Mu-law companding tokenizer (numpy only).

Mu-law companding allocates finer resolution near zero, which suits robot
action distributions that concentrate around small deltas. Each dimension is
normalized to ``[-1, 1]`` by its observed max-abs value, companded, then
uniformly quantized in the companded domain.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from forge.tokenize.base import BaseTokenizer
from forge.tokenize.registry import TokenizerRegistry


@TokenizerRegistry.register("mu-law")
class MuLawTokenizer(BaseTokenizer):
    """Mu-law companding + uniform quantization (fine resolution near 0)."""

    def __init__(self, num_bins: int = 256, mu: float = 255.0) -> None:
        super().__init__()
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2")
        self._num_bins = int(num_bins)
        self._mu = float(mu)
        self._scale: NDArray | None = None  # (D,) per-dim max-abs

    @property
    def vocab_size(self) -> int:
        return self._num_bins

    def fit(self, actions: NDArray) -> MuLawTokenizer:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2:
            raise ValueError(f"fit expects a 2D (N, D) corpus, got {actions.shape}")
        scale = np.abs(actions).max(axis=0)
        scale[scale == 0.0] = 1.0  # avoid divide-by-zero on constant dims
        self._scale = scale
        self._fitted = True
        return self

    def _compand(self, xn: NDArray) -> NDArray:
        """Mu-law forward transform on values already in [-1, 1]."""
        return np.sign(xn) * np.log1p(self._mu * np.abs(xn)) / np.log1p(self._mu)

    def _expand(self, y: NDArray) -> NDArray:
        """Inverse mu-law on companded values in [-1, 1]."""
        return np.sign(y) * ((1.0 + self._mu) ** np.abs(y) - 1.0) / self._mu

    def encode(self, actions: NDArray) -> NDArray:
        self._check_fitted()
        assert self._scale is not None
        actions = np.asarray(actions, dtype=np.float64)
        xn = np.clip(actions / self._scale, -1.0, 1.0)
        y = self._compand(xn)  # [-1, 1]
        t = np.floor((y + 1.0) / 2.0 * self._num_bins)
        return np.clip(t, 0, self._num_bins - 1).astype(np.int64)

    def decode(self, tokens: NDArray) -> NDArray:
        self._check_fitted()
        assert self._scale is not None
        tokens = np.asarray(tokens)
        idx = np.clip(tokens, 0, self._num_bins - 1).astype(np.float64)
        y = (idx + 0.5) / self._num_bins * 2.0 - 1.0  # bin centers in [-1, 1]
        xn = self._expand(y)
        return (xn * self._scale).astype(np.float32)

    def get_params(self) -> dict:
        assert self._scale is not None
        return {
            "num_bins": self._num_bins,
            "mu": self._mu,
            "scale": self._scale.tolist(),
        }

    @classmethod
    def from_params(cls, params: dict) -> MuLawTokenizer:
        tok = cls(num_bins=int(params["num_bins"]), mu=float(params["mu"]))
        tok._scale = np.asarray(params["scale"], dtype=np.float64)
        tok._fitted = True
        return tok
