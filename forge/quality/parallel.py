"""Parallel episode analysis for ``forge quality``.

Splits the episode range into contiguous chunks, one per worker, and analyzes
each chunk in its own process (``spawn`` for cross-platform safety). Each worker
iterates the reader once and processes only its slice, so video decode — the
expensive part — fans out across cores.

``QualityConfig``/``VideoQualityConfig`` are plain dataclasses and pickle
directly, as does the returned ``EpisodeQuality`` (its nested results are
dataclasses of plain types), so no manual (de)serialization is needed.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from forge.quality.config import QualityConfig
from forge.quality.models import EpisodeQuality


def _chunk_bounds(total: int, n_chunks: int) -> list[tuple[int, int]]:
    """Split ``range(total)`` into ``n_chunks`` contiguous [start, end) blocks."""
    base, extra = divmod(total, n_chunks)
    bounds = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < extra else 0)
        if size == 0:
            continue
        bounds.append((start, start + size))
        start += size
    return bounds


def _analyze_chunk_worker(
    source_path: str,
    source_format: str,
    start: int,
    end: int,
    config: QualityConfig,
    video_config,
) -> list[tuple[int, EpisodeQuality | None, str | None]]:
    """Analyze episodes whose index is in [start, end) in a worker process."""
    from forge.formats.registry import FormatRegistry
    from forge.quality.analyzer import QualityAnalyzer

    reader = FormatRegistry.get_reader(source_format)
    analyzer = QualityAnalyzer(config=config, video_config=video_config)

    out: list[tuple[int, EpisodeQuality | None, str | None]] = []
    for idx, episode in enumerate(reader.read_episodes(Path(source_path))):
        if idx >= end:
            break
        if idx < start:
            continue
        try:
            out.append((idx, analyzer.analyze_episode(episode), None))
        except Exception as e:  # one bad episode shouldn't kill the chunk
            out.append((idx, None, f"{episode.episode_id}: {e}"))
    return out


def analyze_parallel(
    source_path: str | Path,
    source_format: str,
    total: int,
    config: QualityConfig,
    video_config=None,
    num_workers: int = 4,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[EpisodeQuality], list[str]]:
    """Analyze ``total`` episodes across ``num_workers`` processes.

    Returns ``(episodes_in_order, errors)``.
    """
    n_workers = max(1, min(num_workers, total))
    bounds = _chunk_bounds(total, n_workers)
    ctx = mp.get_context("spawn")

    results: dict[int, EpisodeQuality] = {}
    errors: list[str] = []
    done = 0

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        futures = {
            executor.submit(
                _analyze_chunk_worker,
                str(source_path),
                source_format,
                start,
                end,
                config,
                video_config,
            ): (start, end)
            for start, end in bounds
        }
        for future in as_completed(futures):
            for idx, eq, err in future.result():
                if eq is not None:
                    results[idx] = eq
                if err is not None:
                    errors.append(err)
                done += 1
                if progress_callback:
                    progress_callback(done, total)

    ordered = [results[i] for i in sorted(results)]
    return ordered, errors
