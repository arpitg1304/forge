"""Frame sampling and pooling for episode-level embeddings.

An episode is embedded per camera by sampling frames at roughly a target rate,
encoding each, and pooling the per-frame vectors into one episode/camera vector.
"""

from __future__ import annotations

import numpy as np

# Cap on frames sampled per episode when fps is unknown or the episode is long,
# to keep embedding cost bounded.
_MAX_SAMPLED_FRAMES = 32


def sample_frame_indices(
    num_frames: int, fps: float | None, *, target_hz: float = 1.0
) -> list[int]:
    """Indices to sample from an episode of ``num_frames`` at ~``target_hz``.

    Uses a stride of ``round(fps / target_hz)`` when fps is known; otherwise
    samples up to ``_MAX_SAMPLED_FRAMES`` evenly-spaced frames. Always returns
    at least one index for a non-empty episode, and is capped at
    ``_MAX_SAMPLED_FRAMES``.
    """
    if num_frames <= 0:
        return []
    if fps and fps > 0 and target_hz > 0:
        stride = max(1, round(fps / target_hz))
        idx = list(range(0, num_frames, stride))
    else:
        idx = list(range(num_frames))

    if len(idx) > _MAX_SAMPLED_FRAMES:
        # Evenly subsample down to the cap.
        picks = np.linspace(0, len(idx) - 1, _MAX_SAMPLED_FRAMES).round().astype(int)
        idx = [idx[i] for i in dict.fromkeys(picks.tolist())]
    return idx


def pool(vectors: np.ndarray, method: str = "mean") -> np.ndarray:
    """Pool per-frame vectors ``(K, D)`` into one ``(D,)`` vector.

    - ``mean``: average of all sampled frames.
    - ``first-mid-last``: average of the first, middle, and last sampled frames
      (cheap, keeps endpoints; good when motion matters more than duration).
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("pool() expects a non-empty (K, D) array")
    if method == "mean":
        return vectors.mean(axis=0)
    if method == "first-mid-last":
        k = vectors.shape[0]
        picks = sorted({0, k // 2, k - 1})
        return vectors[picks].mean(axis=0)
    raise ValueError(f"unknown pooling method: {method!r}")
