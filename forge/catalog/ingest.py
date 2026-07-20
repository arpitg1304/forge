"""Ingestion pipeline: dataset URIs -> registered, scored episodes.

Per episode: **discover -> content-hash -> skip-if-exists -> register -> score
-> stage**, flushing staged rows to the catalog per batch. Everything reuses
existing Forge internals — the same format readers that power ``forge inspect``
for metadata, and ``QualityAnalyzer.analyze_episode`` (the engine behind
``forge quality``) for scoring. No metadata extraction or scoring is
reimplemented here.

Idempotency & resume: an episode's ``content_hash`` (xxh3 of its proprio stream
plus shape signature) is checked against the catalog before any work; re-running
an ingest over the same sources is a no-op. A crash loses at most the current
uncommitted batch, which the next run re-does — no partial rows are ever
committed (episode + quality commit together, manifest last).
"""

from __future__ import annotations

import struct
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from forge.catalog.schema import QUALITY_METRIC_COLUMNS, SCORER_VERSION

if TYPE_CHECKING:
    from rich.console import Console

    from forge.catalog.catalog import Catalog
    from forge.core.models import Episode


@dataclass
class IngestStats:
    """Outcome of an ingest run."""

    batch_id: str
    ingested: int = 0
    skipped: int = 0
    failed: int = 0
    frames: int = 0
    sources: list[str] = field(default_factory=list)


def _resolve_source(uri: str) -> Path:
    """Resolve a dataset URI (local / hf:// / s3:// / gs://) to a local path."""
    from forge.hub import is_hf_url
    from forge.io import is_remote_uri, localize

    if is_hf_url(uri):
        from forge.hub.download import download_dataset

        return download_dataset(uri)
    if is_remote_uri(uri):
        return localize(uri)
    return Path(uri)


def _resolve_for_ingest(uri: str):
    """Resolve for ingest, streaming remote streamable formats instead of downloading.

    Returns a source usable by the readers — the URI itself for a remote
    ``lerobot-v3``/``zarr`` dataset (metadata + proprio streamed via range
    reads; no video download), or a local path otherwise.
    """
    from forge.formats.registry import FormatRegistry
    from forge.io import is_remote_uri

    if is_remote_uri(uri):
        try:
            fmt = FormatRegistry.detect_format(uri)
        except Exception:
            fmt = None
        if fmt in FormatRegistry.STREAMABLE_FORMATS:
            return uri
    return _resolve_source(uri)


def _episode_content_hash(episode: Episode, frames: list) -> str:
    """xxh3 hash of an episode's content: proprio stream + shape signature.

    Deterministic and per-episode, so the same episode always yields the same
    hash (exact-duplicate skipping). Streams frame proprio (state/action/
    timestamp) rather than raw file bytes — Forge readers expose logical
    episodes, not per-episode byte ranges (multiple episodes often share one
    parquet/video file). Images are intentionally excluded (expensive to decode;
    proprio + shape signature is a strong content fingerprint for teleop data).
    """
    import xxhash

    h = xxhash.xxh3_64()
    # Shape signature disambiguates episodes with little/no proprio.
    h.update(
        struct.pack(
            "<iii",
            len(frames),
            int(episode.state_dim or 0),
            int(episode.action_dim or 0),
        )
    )
    for fr in frames:
        if fr.state is not None:
            h.update(np.ascontiguousarray(fr.state, dtype=np.float32).tobytes())
        if fr.action is not None:
            h.update(np.ascontiguousarray(fr.action, dtype=np.float32).tobytes())
        if fr.timestamp is not None:
            h.update(struct.pack("<d", float(fr.timestamp)))
        h.update(b"\x00")
    return h.hexdigest()


def _f(value) -> float | None:
    """Coerce a metric to a plain float (or None)."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else v


def _episode_row(
    episode: Episode,
    frames: list,
    *,
    content_hash: str,
    source_uri: str,
    source_format: str,
    dataset_info,
    batch_id: str,
    now: datetime,
) -> dict:
    """Build one ``episodes`` row from an Episode and dataset-level fallbacks."""
    num_frames = len(frames)
    fps = episode.fps or (dataset_info.inferred_fps if dataset_info else None) or 0.0
    duration_s = (num_frames / fps) if fps else 0.0
    robot = episode.robot_type or (
        dataset_info.inferred_robot_type if dataset_info else None
    )

    state_dim = episode.state_dim
    action_dim = episode.action_dim
    if state_dim is None and frames and frames[0].state is not None:
        state_dim = int(np.asarray(frames[0].state).shape[-1])
    if action_dim is None and frames and frames[0].action is not None:
        action_dim = int(np.asarray(frames[0].action).shape[-1])

    cameras = [
        {
            "name": name,
            "width": int(cam.width),
            "height": int(cam.height),
            "fps": float(fps),
        }
        for name, cam in (episode.cameras or {}).items()
    ]

    meta = episode.metadata or {}
    extra = {"source_episode_id": str(episode.episode_id)}
    if dataset_info and dataset_info.format_version:
        extra["format_version"] = str(dataset_info.format_version)

    return {
        "episode_id": uuid.uuid4().hex,
        "content_hash": content_hash,
        "source_uri": source_uri,
        "source_format": source_format,
        "robot": robot,
        "task": meta.get("task"),
        "language_instruction": episode.language_instruction,
        "operator_id": meta.get("operator_id"),
        "scene_id": meta.get("scene_id"),
        "num_frames": int(num_frames),
        "duration_s": float(duration_s),
        "fps": float(fps),
        "cameras": cameras,
        "state_dim": int(state_dim) if state_dim is not None else None,
        "action_dim": int(action_dim) if action_dim is not None else None,
        "ingest_batch_id": batch_id,
        "ingested_at": now,
        "extra": extra,
    }


def _quality_row(analyzer, episode: Episode, catalog_episode_id: str, now: datetime) -> dict:
    """Score an episode with the shared QualityAnalyzer and map to a row."""
    eq = analyzer.analyze_episode(episode)
    row = {
        "episode_id": catalog_episode_id,
        "scorer_version": SCORER_VERSION,
        "overall_score": _f(eq.overall_score),
        "flags": list(eq.flags or []),
        "computed_at": now,
    }
    # Mirror EpisodeQuality's flat metric fields onto the schema columns.
    metric_values = {
        "ldlj": eq.ldlj,
        "sparc": eq.sparc,
        "dead_fraction": eq.dead_fraction,
        "gripper_chatter_rate": eq.gripper_chatter_rate,
        "joint_path_length": eq.joint_path_length,
        "overall_saturation": eq.overall_saturation,
        "mean_entropy": eq.mean_entropy,
        "psd_high_fraction": eq.psd_high_fraction,
        "state_action_consistency": eq.state_action_consistency,
        "static_fraction": eq.static.static_fraction if eq.static else None,
        "jitter_ratio": eq.timestamps.jitter_ratio if eq.timestamps else None,
    }
    for col in QUALITY_METRIC_COLUMNS:
        row[col] = _f(metric_values[col])
    return row


def ingest(
    sources: list[str],
    catalog: Catalog,
    *,
    batch_id: str | None = None,
    batch_size: int = 200,
    strict: bool = False,
    console: Console | None = None,
) -> IngestStats:
    """Ingest one or more dataset URIs into ``catalog``.

    Args:
        sources: Dataset URIs (local paths, ``hf://``, ``s3://``, ``gs://``).
        catalog: An open :class:`~forge.catalog.catalog.Catalog`.
        batch_id: Groups this run's committed files; generated if omitted.
        batch_size: Flush staged rows every N new episodes.
        strict: Re-raise per-episode errors instead of counting and continuing.
        console: Optional rich console for a progress bar.
    """
    from forge.formats.registry import FormatRegistry
    from forge.quality import QualityAnalyzer

    batch_id = batch_id or uuid.uuid4().hex
    stats = IngestStats(batch_id=batch_id, sources=list(sources))
    analyzer = QualityAnalyzer()

    # Load existing hashes once; extend as we commit so intra-run dups are
    # skipped too. This is the resume mechanism: already-committed episodes
    # never get re-scored.
    seen_hashes = catalog.episode_hashes()

    staged_eps: list[dict] = []
    staged_q: list[dict] = []

    progress = None
    task = None
    if console is not None:
        try:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
            )

            total = 0
            for src in sources:
                try:
                    path = _resolve_for_ingest(src)
                    fmt = FormatRegistry.detect_format(path)
                    total += FormatRegistry.get_reader(fmt).inspect(path).num_episodes or 0
                except Exception:
                    total = 0
                    break
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Ingesting[/cyan]"),
                BarColumn(),
                MofNCompleteColumn(),
                console=console,
            )
            task = progress.add_task("ingest", total=total or None)
            progress.start()
        except Exception:
            progress = None

    def _flush() -> None:
        if not staged_eps and not staged_q:
            return
        catalog.commit_batch(
            episodes=staged_eps, quality_scores=staged_q, batch_id=batch_id
        )
        catalog._writer.write_checkpoint(
            batch_id,
            {
                "batch_id": batch_id,
                "ingested": stats.ingested,
                "frames": stats.frames,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        staged_eps.clear()
        staged_q.clear()

    try:
        for src in sources:
            try:
                path = _resolve_for_ingest(src)
                fmt = FormatRegistry.detect_format(path)
                reader = FormatRegistry.get_reader(fmt)
                dataset_info = None
                try:
                    dataset_info = reader.inspect(path)
                except Exception:
                    dataset_info = None

                for episode in reader.read_episodes(path):
                    now = datetime.now(timezone.utc)
                    try:
                        frames = episode.load_frames()
                        content_hash = _episode_content_hash(episode, frames)
                        if content_hash in seen_hashes:
                            stats.skipped += 1
                            if progress is not None:
                                progress.advance(task)
                            continue

                        ep_row = _episode_row(
                            episode,
                            frames,
                            content_hash=content_hash,
                            source_uri=src,
                            source_format=fmt,
                            dataset_info=dataset_info,
                            batch_id=batch_id,
                            now=now,
                        )
                        q_row = _quality_row(analyzer, episode, ep_row["episode_id"], now)

                        staged_eps.append(ep_row)
                        staged_q.append(q_row)
                        seen_hashes.add(content_hash)
                        stats.ingested += 1
                        stats.frames += ep_row["num_frames"]

                        if len(staged_eps) >= batch_size:
                            _flush()
                    except Exception:
                        if strict:
                            raise
                        stats.failed += 1
                    finally:
                        episode.clear_cache()
                        if progress is not None:
                            progress.advance(task)
            except Exception:
                if strict:
                    raise
                stats.failed += 1
        _flush()
    finally:
        if progress is not None:
            progress.stop()

    return stats
