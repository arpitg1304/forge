"""Embedding backfill pipeline: catalog episodes -> vectors -> rows.

Reads the episodes already registered in a catalog, re-opens their source
datasets with the existing Forge readers, samples frames, and runs an embedding
model to produce per-camera episode vectors plus an instruction (text) vector.
Rows are keyed back to the catalog by ``content_hash`` (the stable identity) and
committed through the catalog writer.

Idempotent: episodes already embedded for the target ``model_id`` are skipped,
so re-running is a no-op and re-embedding with a new model appends. Reuses the
ingest content-hash, the readers, ``forge.io`` for cloud sources, and the
catalog commit machinery — nothing is reimplemented here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

from forge.catalog.ingest import _episode_content_hash, _resolve_source

if TYPE_CHECKING:
    from rich.console import Console

    from forge.catalog.catalog import Catalog
    from forge.embed.base import EmbeddingModel


@dataclass
class EmbedStats:
    """Outcome of an embed run."""

    model_id: str
    embedded: int = 0
    skipped: int = 0
    failed: int = 0
    vision_rows: int = 0
    text_rows: int = 0
    sources: list[str] = field(default_factory=list)


def _episode_vision_rows(
    model: EmbeddingModel,
    frames: list,
    *,
    cameras: list[str] | None,
    sample_hz: float,
    pooling: str,
    fps: float | None,
) -> list[tuple[str, np.ndarray, int]]:
    """Return (camera, pooled_vector, num_sampled) per camera for one episode."""
    from forge.embed.sampling import pool, sample_frame_indices

    idx = sample_frame_indices(len(frames), fps, target_hz=sample_hz)
    if not idx:
        return []

    # Which cameras appear in the sampled frames.
    present: list[str] = []
    for name in frames[idx[0]].images:
        if cameras is None or name in cameras:
            present.append(name)

    out: list[tuple[str, np.ndarray, int]] = []
    for cam in present:
        images = [
            frames[i].images[cam].load() for i in idx if cam in frames[i].images
        ]
        if not images:
            continue
        vectors = model.embed_images(images)  # (K, D), L2-normalized
        out.append((cam, pool(vectors, pooling), len(images)))
    return out


def embed_catalog(
    catalog: Catalog,
    *,
    model_name: str | None = None,
    device: str = "auto",
    cameras: list[str] | None = None,
    sample_hz: float = 1.0,
    pooling: str = "mean",
    batch_size: int = 64,
    console: Console | None = None,
) -> EmbedStats:
    """Compute and store embeddings for every not-yet-embedded catalog episode."""
    from forge.embed import get_model
    from forge.formats.registry import FormatRegistry

    model = get_model(model_name, device=device)
    model_id = model.model_id
    stats = EmbedStats(model_id=model_id)

    already = catalog.embedded_episode_ids(model_id)
    episodes = catalog.sql(
        "SELECT episode_id, content_hash, source_uri FROM episodes"
    ).to_pylist()

    # content_hash -> catalog episode_id, for the episodes still needing vectors.
    todo: dict[str, str] = {}
    by_source: dict[str, list[str]] = defaultdict(list)
    for row in episodes:
        if row["episode_id"] in already:
            stats.skipped += 1
            continue
        todo[row["content_hash"]] = row["episode_id"]
        by_source[row["source_uri"]].append(row["content_hash"])
    stats.sources = sorted(by_source)

    staged: list[dict] = []

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

            progress = Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Embedding[/cyan]"),
                BarColumn(),
                MofNCompleteColumn(),
                console=console,
            )
            task = progress.add_task("embed", total=len(todo))
            progress.start()
        except Exception:
            progress = None

    def _flush() -> None:
        if staged:
            catalog.add_embeddings(list(staged))
            staged.clear()

    try:
        for source_uri, hashes in by_source.items():
            wanted = set(hashes)
            try:
                path = _resolve_source(source_uri)
                fmt = FormatRegistry.detect_format(path)
                reader = FormatRegistry.get_reader(fmt)
                dataset_info = None
                try:
                    dataset_info = reader.inspect(path)
                except Exception:
                    dataset_info = None

                for episode in reader.read_episodes(path):
                    if not wanted:
                        break
                    now = datetime.now(timezone.utc)
                    try:
                        frames = episode.load_frames()
                        content_hash = _episode_content_hash(episode, frames)
                        if content_hash not in wanted:
                            continue
                        wanted.discard(content_hash)
                        cat_episode_id = todo[content_hash]

                        fps = episode.fps or (
                            dataset_info.inferred_fps if dataset_info else None
                        )
                        rows: list[dict] = []
                        for cam, vec, n in _episode_vision_rows(
                            model,
                            frames,
                            cameras=cameras,
                            sample_hz=sample_hz,
                            pooling=pooling,
                            fps=fps,
                        ):
                            rows.append(
                                _emb_row(
                                    cat_episode_id, model_id, "episode", cam,
                                    pooling, model.dim, vec, n, now,
                                )
                            )
                            stats.vision_rows += 1

                        if episode.language_instruction and model.supports_text:
                            # A text-embedding failure must not discard the
                            # (already computed) vision vectors for this episode.
                            try:
                                tvec = model.embed_text(
                                    [episode.language_instruction]
                                )[0]
                                rows.append(
                                    _emb_row(
                                        cat_episode_id, model_id, "instruction", None,
                                        "none", model.dim, tvec, None, now,
                                    )
                                )
                                stats.text_rows += 1
                            except Exception:
                                pass

                        if rows:
                            staged.extend(rows)
                            stats.embedded += 1
                            if len({r["episode_id"] for r in staged}) >= batch_size:
                                _flush()
                    except Exception:
                        stats.failed += 1
                    finally:
                        episode.clear_cache()
                        if progress is not None:
                            progress.advance(task)
            except Exception:
                stats.failed += 1
        _flush()
    finally:
        if progress is not None:
            progress.stop()

    return stats


def _emb_row(
    episode_id, model_id, level, camera, pooling, dim, vector, n_sampled, now
) -> dict:
    return {
        "episode_id": episode_id,
        "model_id": model_id,
        "level": level,
        "camera": camera,
        "pooling": pooling,
        "dim": int(dim),
        "vector": np.asarray(vector, dtype=np.float32).tolist(),
        "num_frames_sampled": int(n_sampled) if n_sampled is not None else None,
        "computed_at": now,
    }
