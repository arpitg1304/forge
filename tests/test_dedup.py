"""Tests for forge.dedup — perceptual episode deduplication (Tier 0)."""

from pathlib import Path

import numpy as np
import pytest

from forge.core.models import DatasetInfo, Episode, Frame, LazyImage
from forge.dedup import DedupConfig, DedupEngine
from forge.dedup.hashes import (
    ahash,
    dhash,
    get_hasher,
    normalized_hamming,
    phash,
)
from forge.formats.registry import FormatRegistry


# ── Image helpers ────────────────────────────────────────────────


def _img(seed: int, h: int = 96, w: int = 96) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _jpeg_ish(img: np.ndarray, noise: int = 6, seed: int = 99) -> np.ndarray:
    """Mild per-pixel noise — mimics a re-encode of the same content."""
    rng = np.random.default_rng(seed)
    out = img.astype(np.int16) + rng.integers(-noise, noise + 1, size=img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _lazy(img: np.ndarray) -> LazyImage:
    return LazyImage(loader=lambda im=img: im, height=img.shape[0], width=img.shape[1], channels=3)


def make_episode(ep_id: str, imgs_per_frame: list[np.ndarray], cam: str = "cam") -> Episode:
    frames = [Frame(index=i, images={cam: _lazy(im)}) for i, im in enumerate(imgs_per_frame)]
    return Episode(episode_id=ep_id, _frame_loader=lambda f=frames: iter(f))


# ── Hash primitives ──────────────────────────────────────────────


@pytest.mark.parametrize("method", ["phash", "dhash", "ahash"])
def test_hash_identical_is_zero_distance(method):
    h = get_hasher(method)
    img = _img(1)
    assert normalized_hamming(h(img), h(img)) == 0.0


@pytest.mark.parametrize("method", ["phash", "dhash", "ahash"])
def test_hash_bit_length(method):
    bits = get_hasher(method)(_img(1), hash_size=8)
    assert bits.size == 64
    assert bits.dtype == bool


def test_phash_robust_to_mild_noise():
    img = _img(2)
    # A re-encode (mild noise) stays close under pHash...
    assert normalized_hamming(phash(img), phash(_jpeg_ish(img))) < 0.1
    # ...while unrelated content is far.
    assert normalized_hamming(phash(img), phash(_img(3))) > 0.25


def test_get_hasher_rejects_unknown():
    with pytest.raises(ValueError):
        get_hasher("nope")


# ── Engine: signatures & distance ────────────────────────────────


def test_identical_episodes_are_duplicates():
    imgs = [_img(i) for i in range(20)]
    ep_a = make_episode("a", imgs)
    ep_b = make_episode("b", [im.copy() for im in imgs])
    engine = DedupEngine(DedupConfig())
    sa, sb = engine._signature(ep_a), engine._signature(ep_b)
    assert engine._distance(sa, sb) == 0.0


def test_distinct_episodes_are_far():
    engine = DedupEngine(DedupConfig())
    sa = engine._signature(make_episode("a", [_img(i) for i in range(20)]))
    sb = engine._signature(make_episode("b", [_img(i + 1000) for i in range(20)]))
    assert engine._distance(sa, sb) > 0.25


def test_no_shared_camera_is_max_distance():
    engine = DedupEngine(DedupConfig())
    sa = engine._signature(make_episode("a", [_img(i) for i in range(10)], cam="left"))
    sb = engine._signature(make_episode("b", [_img(i) for i in range(10)], cam="right"))
    assert engine._distance(sa, sb) == 1.0


def test_signature_none_without_images():
    frames = [Frame(index=i) for i in range(5)]
    ep = Episode(episode_id="x", _frame_loader=lambda: iter(frames))
    assert DedupEngine(DedupConfig())._signature(ep) is None


# ── Engine: clustering via a mock reader ─────────────────────────


class _MockReader:
    """Episodes provided at registration time."""

    episodes: list[Episode] = []

    def inspect(self, path: Path) -> DatasetInfo:
        return DatasetInfo(path=path, format="mock-dedup", num_episodes=len(self.episodes))

    def read_episodes(self, path: Path):
        yield from self.episodes


@pytest.fixture
def register_mock():
    FormatRegistry._readers["mock-dedup"] = _MockReader
    yield _MockReader
    _MockReader.episodes = []
    del FormatRegistry._readers["mock-dedup"]


def test_clusters_exact_and_reencoded_duplicates(register_mock, tmp_path):
    base = [_img(i) for i in range(20)]
    other = [_img(i + 500) for i in range(20)]
    register_mock.episodes = [
        make_episode("ep_000", base),                          # representative
        make_episode("ep_001", [im.copy() for im in base]),    # exact dup
        make_episode("ep_002", [_jpeg_ish(im) for im in base]),  # re-encoded dup
        make_episode("ep_003", other),                         # unique
    ]
    ds = tmp_path / "ds"
    ds.mkdir()
    result = DedupEngine(DedupConfig()).analyze(ds, source_format="mock-dedup")

    assert result.total_episodes == 4
    assert result.num_unique == 2  # ep_000 (rep) + ep_003
    assert set(result.dropped_ids) == {"ep_001", "ep_002"}
    assert len(result.clusters) == 1
    assert result.clusters[0].representative == "ep_000"


def test_transitive_clustering(register_mock, tmp_path):
    base = [_img(i) for i in range(16)]
    register_mock.episodes = [
        make_episode("ep_000", base),
        make_episode("ep_001", [_jpeg_ish(im, seed=1) for im in base]),
        make_episode("ep_002", [_jpeg_ish(im, seed=2) for im in base]),
    ]
    ds = tmp_path / "ds"
    ds.mkdir()
    result = DedupEngine(DedupConfig()).analyze(ds, source_format="mock-dedup")
    # All three collapse to one cluster, earliest kept.
    assert result.num_unique == 1
    assert result.kept_ids == ["ep_000"]
    assert set(result.dropped_ids) == {"ep_001", "ep_002"}


def test_threshold_zero_keeps_reencodes(register_mock, tmp_path):
    base = [_img(i) for i in range(16)]
    register_mock.episodes = [
        make_episode("ep_000", base),
        make_episode("ep_001", [im.copy() for im in base]),     # exact
        make_episode("ep_002", [_jpeg_ish(im) for im in base]),  # re-encode
    ]
    ds = tmp_path / "ds"
    ds.mkdir()
    result = DedupEngine(DedupConfig(threshold=0.0)).analyze(ds, source_format="mock-dedup")
    # Exact copy still dropped; re-encode survives at threshold 0.
    assert result.dropped_ids == ["ep_001"]
    assert "ep_002" in result.kept_ids


def test_uncomparable_episodes_kept(register_mock, tmp_path):
    frames = [Frame(index=i) for i in range(5)]
    register_mock.episodes = [
        make_episode("ep_000", [_img(i) for i in range(10)]),
        Episode(episode_id="ep_001", _frame_loader=lambda: iter(frames)),  # no images
    ]
    ds = tmp_path / "ds"
    ds.mkdir()
    result = DedupEngine(DedupConfig()).analyze(ds, source_format="mock-dedup")
    assert result.num_uncomparable == 1
    assert set(result.kept_ids) == {"ep_000", "ep_001"}
    assert result.dropped_ids == []
