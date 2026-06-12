"""Tests for the action tokenizer featureset (forge/tokenize)."""

from __future__ import annotations

import numpy as np
import pytest

from forge.core.models import Episode, Frame
from forge.tokenize import (
    ActionTokenizer,
    ComparisonReport,
    TokenizerComparator,
    TokenizerRegistry,
    load_tokenizer,
)
from forge.tokenize.strategies.bins import (
    OpenVLABinTokenizer,
    QuantileBinTokenizer,
    UniformBinTokenizer,
)
from forge.tokenize.strategies.mulaw import MuLawTokenizer

# All built-in per-step strategy names registered in v1.
PER_STEP_STRATEGIES = ["uniform-bins", "openvla-bins", "quantile-bins", "mu-law"]


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def corpus(rng):
    """A synthetic (N, D) action corpus with varied per-dim scales."""
    n, d = 2000, 7
    base = rng.standard_normal((n, d)).astype(np.float32)
    # Give each dim a different scale + offset so per-dim fitting matters.
    scales = np.array([0.1, 1.0, 5.0, 0.5, 2.0, 10.0, 0.01], dtype=np.float32)
    offsets = np.array([0.0, -3.0, 1.0, 0.0, 4.0, 0.0, 0.0], dtype=np.float32)
    return base * scales + offsets


@pytest.fixture
def sample(rng):
    """A held-out (T, D) action sample for reconstruction tests."""
    return (rng.standard_normal((64, 7)) * 2.0).astype(np.float32)


def make_dataset(corpus: np.ndarray, num_episodes: int = 4) -> list[Episode]:
    """Split a corpus into synthetic episodes of equal length."""
    chunks = np.array_split(corpus, num_episodes)
    episodes = []
    for i, chunk in enumerate(chunks):
        frames = [
            Frame(index=j, timestamp=j * 0.1, action=row.astype(np.float32))
            for j, row in enumerate(chunk)
        ]
        episodes.append(
            Episode(episode_id=f"ep_{i}", _frames_cache=frames)
        )
    return episodes


class MockReader:
    """Mock reader that yields pre-built episodes (mirrors test_filter)."""

    def __init__(self, episodes: list[Episode]):
        self._episodes = episodes

    def read_episodes(self, path):
        yield from self._episodes

    def read_metadata(self, path):
        return None

    def can_read(self, path):
        return True


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_builtin_strategies_registered():
    names = TokenizerRegistry.list_strategies()
    for name in PER_STEP_STRATEGIES:
        assert name in names


def test_registry_create_returns_instance():
    tok = TokenizerRegistry.create("uniform-bins", num_bins=128)
    assert isinstance(tok, UniformBinTokenizer)
    assert tok.name == "uniform-bins"
    assert tok.vocab_size == 128


def test_registry_get_returns_class():
    cls = TokenizerRegistry.get("openvla-bins")
    assert cls is OpenVLABinTokenizer


def test_registry_unknown_raises():
    with pytest.raises(Exception):
        TokenizerRegistry.get("does-not-exist")


# --------------------------------------------------------------------------- #
# Protocol / interface conformance
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", PER_STEP_STRATEGIES)
def test_satisfies_protocol(name):
    tok = TokenizerRegistry.create(name)
    assert isinstance(tok, ActionTokenizer)
    assert tok.granularity == "per_step"


@pytest.mark.parametrize("name", PER_STEP_STRATEGIES)
def test_fit_sets_is_fitted(name, corpus):
    tok = TokenizerRegistry.create(name)
    assert tok.is_fitted is False
    out = tok.fit(corpus)
    assert tok.is_fitted is True
    assert out is tok  # fit returns self for chaining


@pytest.mark.parametrize("name", PER_STEP_STRATEGIES)
def test_encode_before_fit_raises(name, sample):
    tok = TokenizerRegistry.create(name)
    with pytest.raises(Exception):
        tok.encode(sample)


# --------------------------------------------------------------------------- #
# Encode / decode behavior
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", PER_STEP_STRATEGIES)
def test_encode_shape_and_dtype(name, corpus, sample):
    tok = TokenizerRegistry.create(name).fit(corpus)
    tokens = tok.encode(sample)
    assert tokens.shape == sample.shape  # per-step: one token per (frame, dim)
    assert np.issubdtype(tokens.dtype, np.integer)


@pytest.mark.parametrize("name", PER_STEP_STRATEGIES)
def test_tokens_within_vocab_bounds(name, corpus, sample):
    tok = TokenizerRegistry.create(name, num_bins=256).fit(corpus)
    # Include extreme out-of-range values to confirm clipping.
    extreme = np.concatenate([sample, np.full((4, 7), 1e6), np.full((4, 7), -1e6)])
    tokens = tok.encode(extreme.astype(np.float32))
    assert tokens.min() >= 0
    assert tokens.max() <= tok.vocab_size - 1


@pytest.mark.parametrize("name", PER_STEP_STRATEGIES)
def test_decode_shape(name, corpus, sample):
    tok = TokenizerRegistry.create(name).fit(corpus)
    recon = tok.decode(tok.encode(sample))
    assert recon.shape == sample.shape
    assert np.issubdtype(recon.dtype, np.floating)


@pytest.mark.parametrize("name", PER_STEP_STRATEGIES)
def test_encode_deterministic(name, corpus, sample):
    tok = TokenizerRegistry.create(name).fit(corpus)
    np.testing.assert_array_equal(tok.encode(sample), tok.encode(sample))


def test_uniform_bins_max_error_within_bin_width():
    """Uniform bins bound the *max* error to ~one bin width per dim."""
    rng = np.random.default_rng(7)
    corpus = (rng.standard_normal((4000, 5)) * 3.0).astype(np.float32)
    tok = UniformBinTokenizer(num_bins=256).fit(corpus)
    recon = tok.decode(tok.encode(corpus))
    rng_per_dim = corpus.max(axis=0) - corpus.min(axis=0)
    tol = 1.5 * rng_per_dim / 256  # half a bin width, with slack
    max_err = np.abs(recon - corpus).max(axis=0)
    assert np.all(max_err <= tol + 1e-6)


@pytest.mark.parametrize("name", PER_STEP_STRATEGIES)
def test_roundtrip_low_error_in_distribution(name, corpus):
    """Every strategy reconstructs the bulk of the corpus with small error.

    Tail outliers can exceed a uniform bin width (openvla clips, quantile/mu-law
    use wide tail bins), so this bounds the *median* per-dim error instead of max.
    """
    tok = TokenizerRegistry.create(name, num_bins=256).fit(corpus)
    recon = tok.decode(tok.encode(corpus))
    rng_per_dim = corpus.max(axis=0) - corpus.min(axis=0)
    median_err = np.median(np.abs(recon - corpus), axis=0)
    assert np.all(median_err <= 0.05 * rng_per_dim + 1e-6)


@pytest.mark.parametrize("name", PER_STEP_STRATEGIES)
def test_more_bins_reduce_error(name, corpus):
    coarse = TokenizerRegistry.create(name, num_bins=16).fit(corpus)
    fine = TokenizerRegistry.create(name, num_bins=512).fit(corpus)
    err_coarse = np.abs(coarse.decode(coarse.encode(corpus)) - corpus).mean()
    err_fine = np.abs(fine.decode(fine.encode(corpus)) - corpus).mean()
    assert err_fine < err_coarse


def test_mulaw_finer_near_zero():
    """Mu-law bin widths should be smaller near 0 than far from 0."""
    # Single dim, symmetric range [-1, 1].
    corpus = np.linspace(-1, 1, 4001).reshape(-1, 1).astype(np.float32)
    tok = MuLawTokenizer(num_bins=256).fit(corpus)
    centers = np.unique(tok.decode(tok.encode(corpus)).ravel())
    centers.sort()
    widths = np.diff(centers)
    # Width of the bins straddling zero vs. bins near the extremes.
    near_zero = widths[len(widths) // 2]
    near_edge = widths[2]
    assert near_zero < near_edge


def test_quantile_uniform_token_distribution():
    """Quantile bins should be roughly equal-mass on the fitting corpus."""
    rng = np.random.default_rng(0)
    corpus = rng.standard_normal((20000, 1)).astype(np.float32)
    tok = QuantileBinTokenizer(num_bins=10).fit(corpus)
    counts = np.bincount(tok.encode(corpus).ravel(), minlength=10)
    frac = counts / counts.sum()
    # Each of 10 bins should hold ~10% of the mass.
    assert np.all(np.abs(frac - 0.1) < 0.03)


# --------------------------------------------------------------------------- #
# Save / load
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", PER_STEP_STRATEGIES)
def test_save_load_roundtrip(name, corpus, sample, tmp_path):
    tok = TokenizerRegistry.create(name, num_bins=128).fit(corpus)
    path = tmp_path / "tok.json"
    tok.save(path)

    loaded = load_tokenizer(path)
    assert loaded.name == tok.name
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.is_fitted
    np.testing.assert_array_equal(loaded.encode(sample), tok.encode(sample))
    np.testing.assert_allclose(loaded.decode(tok.encode(sample)),
                               tok.decode(tok.encode(sample)))


@pytest.mark.parametrize("name", PER_STEP_STRATEGIES)
def test_get_params_from_params_roundtrip(name, corpus, sample):
    tok = TokenizerRegistry.create(name, num_bins=64).fit(corpus)
    params = tok.get_params()
    import json

    json.dumps(params)  # must be JSON-serializable
    rebuilt = type(tok).from_params(params)
    np.testing.assert_array_equal(rebuilt.encode(sample), tok.encode(sample))


# --------------------------------------------------------------------------- #
# Comparator
# --------------------------------------------------------------------------- #


def test_comparator_on_episodes(corpus):
    episodes = make_dataset(corpus, num_episodes=4)
    comparator = TokenizerComparator()
    report = comparator.compare_episodes(episodes, sample=200)

    assert isinstance(report, ComparisonReport)
    assert set(report.stats_by_strategy) >= set(PER_STEP_STRATEGIES)
    for stats in report.stats_by_strategy.values():
        assert stats.mae >= 0.0
        assert stats.mse >= 0.0
        assert stats.tokens_per_step > 0
        assert 0.0 <= stats.vocab_utilization <= 1.0


def test_comparator_subset_of_strategies(corpus):
    episodes = make_dataset(corpus)
    report = TokenizerComparator().compare_episodes(
        episodes, strategies=["uniform-bins", "openvla-bins"], sample=100
    )
    assert set(report.stats_by_strategy) == {"uniform-bins", "openvla-bins"}


def test_comparison_report_json_roundtrip(corpus, tmp_path):
    episodes = make_dataset(corpus)
    report = TokenizerComparator().compare_episodes(episodes, sample=100)
    path = tmp_path / "report.json"
    report.to_json(path)
    loaded = ComparisonReport.from_json(path)
    assert set(loaded.stats_by_strategy) == set(report.stats_by_strategy)


def test_comparator_via_reader(corpus, tmp_path):
    """compare_dataset should drive a reader's read_episodes."""
    episodes = make_dataset(corpus)
    reader = MockReader(episodes)
    report = TokenizerComparator(reader=reader).compare_dataset(
        tmp_path / "fake", sample=100
    )
    assert set(report.stats_by_strategy) >= set(PER_STEP_STRATEGIES)
