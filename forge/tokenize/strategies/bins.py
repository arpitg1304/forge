"""Per-step binning tokenizers (numpy only).

All three strategies share :class:`_PerStepBinTokenizer`: ``fit`` learns
per-dimension bin edges, ``encode`` is ``np.digitize`` against the interior
edges, and ``decode`` looks up bin centers. They differ only in how the edges
are computed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from forge.tokenize.base import BaseTokenizer
from forge.tokenize.registry import TokenizerRegistry


def _ensure_increasing(edges: NDArray) -> NDArray:
    """Force a strictly increasing edge vector (degenerate/constant dims)."""
    edges = np.asarray(edges, dtype=np.float64).copy()
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges


class _PerStepBinTokenizer(BaseTokenizer):
    """Shared base for per-dimension uniform/percentile/quantile binning."""

    def __init__(self, num_bins: int = 256) -> None:
        super().__init__()
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2")
        self._num_bins = int(num_bins)
        self._edges: NDArray | None = None  # (D, num_bins + 1)

    @property
    def vocab_size(self) -> int:
        return self._num_bins

    def _compute_dim_edges(self, column: NDArray) -> NDArray:
        """Return ``num_bins + 1`` edges for a single dimension's values."""
        raise NotImplementedError

    def fit(self, actions: NDArray) -> _PerStepBinTokenizer:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2:
            raise ValueError(f"fit expects a 2D (N, D) corpus, got {actions.shape}")
        n_dims = actions.shape[1]
        edges = np.empty((n_dims, self._num_bins + 1), dtype=np.float64)
        for d in range(n_dims):
            edges[d] = _ensure_increasing(self._compute_dim_edges(actions[:, d]))
        self._edges = edges
        self._fitted = True
        return self

    def encode(self, actions: NDArray) -> NDArray:
        self._check_fitted()
        actions = np.asarray(actions, dtype=np.float64)
        assert self._edges is not None
        tokens = np.empty(actions.shape, dtype=np.int64)
        for d in range(actions.shape[-1]):
            interior = self._edges[d][1:-1]
            idx = np.digitize(actions[..., d], interior)
            tokens[..., d] = np.clip(idx, 0, self._num_bins - 1)
        return tokens

    def decode(self, tokens: NDArray) -> NDArray:
        self._check_fitted()
        tokens = np.asarray(tokens)
        assert self._edges is not None
        out = np.empty(tokens.shape, dtype=np.float64)
        for d in range(tokens.shape[-1]):
            centers = (self._edges[d][:-1] + self._edges[d][1:]) / 2.0
            idx = np.clip(tokens[..., d], 0, self._num_bins - 1)
            out[..., d] = centers[idx]
        return out.astype(np.float32)

    def get_params(self) -> dict:
        assert self._edges is not None
        return {"num_bins": self._num_bins, "edges": self._edges.tolist()}

    @classmethod
    def from_params(cls, params: dict) -> _PerStepBinTokenizer:
        tok = cls(num_bins=int(params["num_bins"]))
        tok._edges = np.asarray(params["edges"], dtype=np.float64)
        tok._fitted = True
        return tok


@TokenizerRegistry.register("uniform-bins")
class UniformBinTokenizer(_PerStepBinTokenizer):
    """RT-1 style: per-dim min/max split into ``num_bins`` uniform bins."""

    def _compute_dim_edges(self, column: NDArray) -> NDArray:
        lo, hi = float(column.min()), float(column.max())
        if hi <= lo:
            hi = lo + 1.0
        return np.linspace(lo, hi, self._num_bins + 1)


@TokenizerRegistry.register("openvla-bins")
class OpenVLABinTokenizer(_PerStepBinTokenizer):
    """OpenVLA style: 1st-99th percentile range, uniform bins, clip outliers."""

    def _compute_dim_edges(self, column: NDArray) -> NDArray:
        lo, hi = np.percentile(column, [1.0, 99.0])
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            hi = lo + 1.0
        return np.linspace(lo, hi, self._num_bins + 1)


@TokenizerRegistry.register("quantile-bins")
class QuantileBinTokenizer(_PerStepBinTokenizer):
    """Equal-mass bins: each token is roughly equally likely on the corpus."""

    def _compute_dim_edges(self, column: NDArray) -> NDArray:
        qs = np.linspace(0.0, 1.0, self._num_bins + 1)
        return np.quantile(column, qs)
