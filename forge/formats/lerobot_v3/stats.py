"""Per-episode and dataset-wide statistics in the shape LeRobot v3 expects.

LeRobot policies normalize inputs from ``meta/stats.json``: ACT and Diffusion
use ``mean``/``std`` (images included), pi0.5 uses the ``q01``/``q99``
quantiles. A dataset that lacks those keys either fails at training start
(quantiles) or trains on unnormalized images without any error (image stats).

These functions follow the algorithm in ``lerobot.datasets.compute_stats``
(lerobot 0.5–0.7) so that a Forge-written dataset carries the same numbers
LeRobot would have computed itself:

- Quantiles come from a 5000-bin histogram over the episode's values, with
  linear interpolation inside the bin, not from an exact sort.
- Image stats are per channel, scaled to [0, 1], over an evenly spaced sample
  of frames (``estimate_num_samples``), downsampled when the larger side is
  300 px or more. Their arrays have shape ``(C, 1, 1)``.
- Aggregating episodes pools ``mean``/``std`` exactly and keeps a conservative
  envelope of the quantiles: the minimum for q <= 50, the maximum above.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

QUANTILES: tuple[float, ...] = (0.01, 0.10, 0.50, 0.90, 0.99)
QUANTILE_KEYS: tuple[str, ...] = tuple(f"q{int(q * 100):02d}" for q in QUANTILES)
_NUM_BINS = 5000

Stats = dict[str, np.ndarray]


def _histogram_quantile(values: np.ndarray, lo: float, hi: float, q: float) -> float:
    edges = np.linspace(lo - 1e-10, hi + 1e-10, _NUM_BINS + 1)
    hist, _ = np.histogram(values, bins=edges)
    cumsum = np.cumsum(hist)
    target = q * values.shape[0]
    idx = int(np.searchsorted(cumsum, target))
    if idx == 0:
        return float(edges[0])
    if idx >= len(cumsum):
        return float(edges[-1])
    before = cumsum[idx - 1]
    in_bin = cumsum[idx] - before
    if in_bin == 0:
        return float(edges[idx])
    return float(edges[idx] + (target - before) / in_bin * (edges[idx + 1] - edges[idx]))


def _columns_stats(samples: np.ndarray, count: int) -> Stats:
    """Stats over axis 0 of a ``(N, D)`` float64 array."""
    if samples.shape[0] < 2:
        mean = samples.mean(axis=0)
        out: Stats = {
            "min": samples.min(axis=0),
            "max": samples.max(axis=0),
            "mean": mean,
            "std": samples.std(axis=0),
            "count": np.array([count]),
        }
        for key in QUANTILE_KEYS:
            out[key] = mean.copy()
        return out

    lo, hi = samples.min(axis=0), samples.max(axis=0)
    mean = samples.mean(axis=0)
    var = (samples**2).mean(axis=0) - mean**2
    out = {
        "min": lo,
        "max": hi,
        "mean": mean,
        "std": np.sqrt(np.maximum(0.0, var)),
        "count": np.array([count]),
    }
    for q, key in zip(QUANTILES, QUANTILE_KEYS, strict=True):
        out[key] = np.array(
            [_histogram_quantile(samples[:, d], lo[d], hi[d], q) for d in range(samples.shape[1])]
        )
    return out


def vector_stats(values: np.ndarray) -> Stats:
    """Stats for a numeric feature over one episode.

    ``values`` is ``(N,)`` for a scalar column or ``(N, D)`` for a vector
    column. Every stat except ``count`` has shape ``(1,)`` or ``(D,)``.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.shape[0])
    stats = _columns_stats(arr.reshape(n, -1), n)
    if arr.ndim == 1:
        return {k: v if k == "count" else v.reshape(1) for k, v in stats.items()}
    return stats


def estimate_num_samples(n: int, min_samples: int = 100, max_samples: int = 10_000, power: float = 0.75) -> int:
    if n < min_samples:
        min_samples = n
    return max(min_samples, min(int(n**power), max_samples))


def sample_indices(n: int) -> list[int]:
    return np.round(np.linspace(0, n - 1, estimate_num_samples(n))).astype(int).tolist()


def _downsample(img_chw: np.ndarray, target: int = 150, threshold: int = 300) -> np.ndarray:
    _, h, w = img_chw.shape
    if max(h, w) < threshold:
        return img_chw
    factor = int(w / target) if w > h else int(h / target)
    return img_chw[:, ::factor, ::factor]


def image_stats(frames: Sequence[np.ndarray], episode_length: int) -> Stats:
    """Per-channel stats for already-sampled ``(H, W, C)`` uint8 frames.

    Call ``sample_indices(episode_length)`` to pick which frames to pass.
    Values are scaled to [0, 1]; ``count`` is the episode length, as LeRobot
    weights image stats by frames in the episode rather than frames sampled.
    """
    chw = np.stack([_downsample(np.asarray(f).transpose(2, 0, 1)) for f in frames])
    channels = chw.shape[1]
    pixels = chw.transpose(0, 2, 3, 1).reshape(-1, channels).astype(np.float64)
    stats = _columns_stats(pixels, chw.shape[0])
    return {
        k: np.array([episode_length]) if k == "count" else (v / 255.0).reshape(channels, 1, 1)
        for k, v in stats.items()
    }


def aggregate(per_episode: list[dict[str, Stats]]) -> dict[str, Stats]:
    """Pool per-episode stats into dataset-wide stats."""
    out: dict[str, Stats] = {}
    for key in sorted({k for ep in per_episode for k in ep}):
        items = [ep[key] for ep in per_episode if key in ep]
        means = np.stack([s["mean"] for s in items])
        variances = np.stack([s["std"] ** 2 for s in items])
        counts = np.stack([s["count"] for s in items]).astype(np.float64)
        total = counts.sum(axis=0)
        while counts.ndim < means.ndim:
            counts = np.expand_dims(counts, axis=-1)
        mean = (means * counts).sum(axis=0) / total
        var = ((variances + (means - mean) ** 2) * counts).sum(axis=0) / total
        agg: Stats = {
            "min": np.min(np.stack([s["min"] for s in items]), axis=0),
            "max": np.max(np.stack([s["max"] for s in items]), axis=0),
            "mean": mean,
            "std": np.sqrt(var),
            "count": total.astype(np.int64),
        }
        for qk in QUANTILE_KEYS:
            if all(qk in s for s in items):
                stacked = np.stack([s[qk] for s in items])
                agg[qk] = stacked.min(axis=0) if int(qk[1:]) <= 50 else stacked.max(axis=0)
        out[key] = agg
    return out


def to_jsonable(stats: dict[str, Stats]) -> dict[str, dict[str, Any]]:
    return {k: {s: np.asarray(v).tolist() for s, v in d.items()} for k, d in stats.items()}


def from_jsonable(stats: dict[str, dict[str, Any]]) -> dict[str, Stats]:
    return {k: {s: np.asarray(v) for s, v in d.items()} for k, d in stats.items()}
