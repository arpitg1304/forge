"""Benchmark action tokenizer strategies on a dataset.

:class:`TokenizerComparator` stacks every frame's action into an ``(N, D)``
corpus, fits each strategy, and measures reconstruction error, tokens-per-step
and vocabulary utilization on a held sample — answering "which discretization
fits *my* dataset" without trial-and-error.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from forge.tokenize.models import ComparisonReport, TokenizerStats
from forge.tokenize.registry import TokenizerRegistry

if TYPE_CHECKING:
    from forge.core.models import Episode


class TokenizerComparator:
    """Compare tokenizer strategies on a dataset's action stream."""

    def __init__(self, reader=None) -> None:
        """``reader`` overrides format auto-detection (useful for tests)."""
        self._reader = reader

    # -- public entry points ------------------------------------------------ #

    def compare_dataset(
        self,
        path: str | Path,
        strategies: list[str] | None = None,
        sample: int = 0,
    ) -> ComparisonReport:
        """Resolve a reader for ``path`` and benchmark every strategy."""
        reader = self._reader or self._resolve_reader(path)
        episodes = reader.read_episodes(path)
        return self.compare_episodes(
            episodes, strategies=strategies, sample=sample, dataset_path=str(path)
        )

    def compare_episodes(
        self,
        episodes: Iterable[Episode],
        strategies: list[str] | None = None,
        sample: int = 0,
        dataset_path: str = "<episodes>",
    ) -> ComparisonReport:
        """Benchmark strategies over an iterable of episodes."""
        corpus = self._build_corpus(episodes)
        return self.compare_corpus(
            corpus, strategies=strategies, sample=sample, dataset_path=dataset_path
        )

    def compare_corpus(
        self,
        corpus: NDArray,
        strategies: list[str] | None = None,
        sample: int = 0,
        dataset_path: str = "<corpus>",
    ) -> ComparisonReport:
        """Benchmark strategies on a pre-stacked ``(N, D)`` action corpus."""
        corpus = np.asarray(corpus, dtype=np.float64)
        if corpus.ndim != 2 or corpus.size == 0:
            raise ValueError(f"corpus must be a non-empty 2D array, got {corpus.shape}")

        names = strategies or TokenizerRegistry.list_strategies()
        eval_set = self._eval_subset(corpus, sample)

        report = ComparisonReport(
            dataset_path=dataset_path,
            num_frames=corpus.shape[0],
            action_dim=corpus.shape[1],
            sample_size=eval_set.shape[0],
        )
        for name in names:
            tok = TokenizerRegistry.create(name).fit(corpus)
            report.stats_by_strategy[name] = self._score(name, tok, eval_set)
        return report

    # -- internals ---------------------------------------------------------- #

    @staticmethod
    def _eval_subset(corpus: NDArray, sample: int) -> NDArray:
        """Deterministic evenly-spaced subset of the corpus for scoring."""
        n = corpus.shape[0]
        if sample <= 0 or sample >= n:
            return corpus
        idx = np.linspace(0, n - 1, sample).astype(np.int64)
        return corpus[idx]

    @staticmethod
    def _score(name: str, tok, eval_set: NDArray) -> TokenizerStats:
        tokens = tok.encode(eval_set)
        recon = tok.decode(tokens).astype(np.float64)
        diff = recon - eval_set

        per_dim_rmse = np.sqrt(np.mean(diff**2, axis=0))
        tokens_per_step = tokens.size / eval_set.shape[0]
        floats_per_step = eval_set.shape[1]
        unique = np.unique(tokens).size
        vocab_util = min(1.0, unique / tok.vocab_size)

        return TokenizerStats(
            strategy=name,
            vocab_size=tok.vocab_size,
            granularity=tok.granularity,
            mse=float(np.mean(diff**2)),
            mae=float(np.mean(np.abs(diff))),
            max_abs=float(np.max(np.abs(diff))),
            per_dim_rmse=[float(v) for v in per_dim_rmse],
            tokens_per_step=float(tokens_per_step),
            compression_ratio=float(floats_per_step / tokens_per_step),
            vocab_utilization=float(vocab_util),
        )

    @staticmethod
    def _build_corpus(episodes: Iterable[Episode]) -> NDArray:
        rows: list[NDArray] = []
        for ep in episodes:
            for frame in ep.frames():
                if frame.action is not None:
                    rows.append(np.asarray(frame.action, dtype=np.float64))
        if not rows:
            raise ValueError("no actions found in the provided episodes")
        return np.vstack(rows)

    @staticmethod
    def _resolve_reader(path: str | Path):
        from forge.formats.registry import FormatRegistry

        fmt = FormatRegistry.detect_format(Path(path))
        return FormatRegistry.get_reader(fmt)
