"""Perceptual hashes for episode deduplication — numpy only.

Three hashes, all computed on a downscaled grayscale frame (reusing the Tier 0
video path, so no new dependencies — no Pillow, no scipy):

- **aHash** (average): bit = pixel brighter than the frame mean. Cheapest.
- **dHash** (difference/gradient): bit = pixel brighter than its right neighbour.
  Robust to brightness/gamma shifts.
- **pHash** (DCT low-frequency): bit = DCT coefficient above the median of the
  low-frequency block. Most robust to scaling, blur, and compression.

Each returns a flat boolean array of ``hash_size * hash_size`` bits.

References:
    - Krawetz, "Looks Like It" (pHash/aHash) — the dct/average-hash recipes.
    - Buchner, `imagehash` — the canonical dHash/pHash bit definitions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from forge.quality.video.pixel import _LUMA, _to_float_image


def _to_gray(img: NDArray) -> NDArray:
    """Float 0–255 grayscale from a uint8/float, HWC or HW, image."""
    a = _to_float_image(img)
    if a.ndim == 3 and a.shape[2] >= 3:
        return a[..., :3] @ _LUMA
    if a.ndim == 3:
        return a[..., 0]
    return a


def _resize(gray: NDArray, out_h: int, out_w: int) -> NDArray:
    """Nearest-neighbour resize to an exact (out_h, out_w)."""
    h, w = gray.shape
    ys = np.minimum((np.arange(out_h) * (h / out_h)).astype(np.intp), h - 1)
    xs = np.minimum((np.arange(out_w) * (w / out_w)).astype(np.intp), w - 1)
    return gray[ys][:, xs]


def ahash(img: NDArray, hash_size: int = 8) -> NDArray:
    g = _resize(_to_gray(img), hash_size, hash_size)
    return (g > g.mean()).flatten()


def dhash(img: NDArray, hash_size: int = 8) -> NDArray:
    # One extra column so the horizontal gradient yields hash_size**2 bits.
    g = _resize(_to_gray(img), hash_size, hash_size + 1)
    return (g[:, 1:] > g[:, :-1]).flatten()


# Cosine bases for the DCT, cached per matrix size.
_DCT_BASIS: dict[int, NDArray] = {}


def _dct_basis(n: int) -> NDArray:
    basis = _DCT_BASIS.get(n)
    if basis is None:
        idx = np.arange(n)
        # B[k, m] = cos(pi * (2m + 1) * k / (2n)) — DCT-II, unnormalized.
        # Normalization is irrelevant: we threshold by median, which is
        # invariant to a global positive scale.
        basis = np.cos(np.pi * (2 * idx + 1)[None, :] * idx[:, None] / (2 * n))
        _DCT_BASIS[n] = basis
    return basis


def _dct2(g: NDArray) -> NDArray:
    """2-D DCT-II of a square matrix via cached cosine basis (no scipy)."""
    b = _dct_basis(g.shape[0])
    return b @ g @ b.T


def phash(img: NDArray, hash_size: int = 8, highfreq_factor: int = 4) -> NDArray:
    img_size = hash_size * highfreq_factor
    g = _resize(_to_gray(img), img_size, img_size)
    low = _dct2(g)[:hash_size, :hash_size]
    return (low > np.median(low)).flatten()


_HASHERS = {"phash": phash, "dhash": dhash, "ahash": ahash}


def get_hasher(method: str):
    """Return the hash function for ``method`` (phash | dhash | ahash)."""
    try:
        return _HASHERS[method]
    except KeyError:
        raise ValueError(
            f"Unknown hash method '{method}'. Choose from: {', '.join(_HASHERS)}"
        )


def normalized_hamming(a: NDArray, b: NDArray) -> float:
    """Fraction of differing bits between two equal-length boolean hashes."""
    return float(np.count_nonzero(a != b)) / a.size
