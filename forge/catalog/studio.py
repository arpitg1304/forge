"""Forge Studio — a self-contained, themed HTML app for a catalog.

Generates a single shareable HTML file with tabs (Overview · Corpus · Dedup
review · Snapshot) rendered from **real catalog data** and **real video
thumbnails** (extracted via the existing readers, embedded as data URIs). The
design system matches the Forge Studio mockup and the dark theme of
`forge visualize`.

The file is static: the Dedup tab lets you make keep/reject decisions and export
them as `forge curate`-ready commands (a static page can't write back to the
catalog). Thumbnails are capped so generation stays quick.
"""

from __future__ import annotations

import base64
import html
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from forge.catalog.dedup import DEFAULT_THRESHOLD

if TYPE_CHECKING:
    from rich.console import Console

    from forge.catalog.catalog import Catalog


@dataclass
class StudioStats:
    out_path: str
    episodes: int = 0
    pairs: int = 0
    thumbnails: int = 0
    capped: bool = False


def _thumbnail_data_uri(img, size: int = 240) -> str | None:
    """Downscale an HWC uint8 frame to a small JPEG data URI."""
    try:
        import numpy as np
        from PIL import Image

        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = arr.clip(0, 255).astype("uint8")
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        im = Image.fromarray(arr)
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=72)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _extract_thumbnails(
    catalog: Catalog, wanted: dict[str, str], console: Console | None
) -> dict[str, str]:
    """Extract one representative frame per wanted episode.

    ``wanted`` maps content_hash -> catalog episode_id. Groups by source_uri,
    opens each dataset once, and grabs a middle frame's first camera.
    """
    from forge.catalog.ingest import _episode_content_hash, _resolve_source
    from forge.formats.registry import FormatRegistry

    rows = catalog.sql("SELECT content_hash, source_uri FROM episodes").to_pylist()
    by_source: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r["content_hash"] in wanted:
            by_source[r["source_uri"]].add(r["content_hash"])

    thumbs: dict[str, str] = {}
    for source_uri, hashes in by_source.items():
        remaining = set(hashes)
        try:
            path = _resolve_source(source_uri)
            reader = FormatRegistry.get_reader(FormatRegistry.detect_format(path))
            for episode in reader.read_episodes(path):
                if not remaining:
                    break
                try:
                    frames = episode.load_frames()
                    ch = _episode_content_hash(episode, frames)
                    if ch not in remaining:
                        continue
                    remaining.discard(ch)
                    mid = frames[len(frames) // 2]
                    if mid.images:
                        cam = next(iter(mid.images))
                        uri = _thumbnail_data_uri(mid.images[cam].load())
                        if uri:
                            thumbs[wanted[ch]] = uri
                finally:
                    episode.clear_cache()
        except Exception:
            continue
        if console is not None:
            console.print(f"[dim]  thumbnails: {len(thumbs)}/{len(wanted)}[/dim]")
    return thumbs


def _build_data(catalog: Catalog, threshold: float, max_thumbnails: int):
    """Assemble the JSON payload + the episode set to thumbnail."""
    eps = catalog.sql(
        """
        SELECT e.episode_id, e.task, e.robot, e.language_instruction AS instr,
               e.num_frames, e.fps, e.duration_s, e.source_format, e.content_hash,
               q.overall_score AS score, c.label AS label
        FROM episodes e
        LEFT JOIN v_latest_quality q USING(episode_id)
        LEFT JOIN v_curation c USING(episode_id)
        ORDER BY q.overall_score DESC NULLS LAST
        """
    ).to_pylist()

    has_embeddings = bool(catalog.embedding_model_ids())
    pairs = []
    if has_embeddings and _table_has_rows(catalog, "dedup_edges"):
        pairs = catalog.sql(
            f"""
            SELECT d.episode_a, d.episode_b, d.similarity AS sim,
                   ea.task AS task, ea.robot AS robot,
                   qa.overall_score AS qa, qb.overall_score AS qb
            FROM dedup_edges d
            JOIN episodes ea ON ea.episode_id = d.episode_a
            LEFT JOIN v_latest_quality qa ON qa.episode_id = d.episode_a
            LEFT JOIN v_latest_quality qb ON qb.episode_id = d.episode_b
            WHERE d.similarity >= {float(threshold)}
            ORDER BY d.similarity DESC
            """
        ).to_pylist()

    stats = catalog.stats()
    dup_pairs = _table_count(catalog, "dedup_edges")
    labeled = catalog.sql(
        "SELECT label, count(*) n FROM v_curation GROUP BY 1"
    ).to_pylist()

    # Episodes to thumbnail: all pair episodes first, then top-quality, up to cap.
    ordered = []
    for p in pairs:
        ordered += [p["episode_a"], p["episode_b"]]
    ordered += [e["episode_id"] for e in eps]
    seen: set[str] = set()
    thumb_ids: list[str] = []
    for eid in ordered:
        if eid not in seen:
            seen.add(eid)
            thumb_ids.append(eid)
        if len(thumb_ids) >= max_thumbnails:
            break

    hash_by_id = {e["episode_id"]: e["content_hash"] for e in eps}
    wanted = {hash_by_id[i]: i for i in thumb_ids if i in hash_by_id}

    data = {
        "catalog": catalog.root_uri,
        "stats": stats,
        "labeled": {r["label"]: r["n"] for r in labeled},
        "episodes": [
            {
                "id": e["episode_id"],
                "task": e["task"],
                "robot": e["robot"],
                "instr": e["instr"],
                "score": e["score"],
                "frames": e["num_frames"],
                "dur": _fmt_dur(e["duration_s"]),
                "fmt": e["source_format"],
                "label": e["label"],
            }
            for e in eps
        ],
        "pairs": [
            {
                "a": p["episode_a"], "b": p["episode_b"], "sim": p["sim"],
                "task": p["task"], "robot": p["robot"], "qa": p["qa"], "qb": p["qb"],
            }
            for p in pairs
        ],
        "dedup": {"threshold": threshold, "total_pairs": dup_pairs},
        "has_embeddings": has_embeddings,
    }
    return data, wanted


def _table_has_rows(catalog: Catalog, table: str) -> bool:
    return _table_count(catalog, table) > 0


def _table_count(catalog: Catalog, table: str) -> int:
    try:
        return int(catalog.sql(f"SELECT count(*) n FROM {table}").to_pylist()[0]["n"])
    except Exception:
        return 0


def _fmt_dur(seconds) -> str:
    s = int(seconds or 0)
    return f"{s // 60}:{s % 60:02d}"


def generate_studio(
    catalog: Catalog,
    out_path: str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    max_thumbnails: int = 120,
    console: Console | None = None,
) -> StudioStats:
    """Render a Forge Studio HTML file for ``catalog``."""
    data, wanted = _build_data(catalog, threshold, max_thumbnails)
    thumbs = _extract_thumbnails(catalog, wanted, console) if wanted else {}
    for e in data["episodes"]:
        if e["id"] in thumbs:
            e["thumb"] = thumbs[e["id"]]

    payload = json.dumps(data)
    thumb_json = json.dumps(thumbs)
    doc = _HTML.replace("/*__DATA__*/", payload).replace("/*__THUMBS__*/", thumb_json)
    doc = doc.replace("__CATALOG__", html.escape(catalog.root_uri))

    Path(out_path).write_text(doc, encoding="utf-8")
    return StudioStats(
        out_path=out_path,
        episodes=len(data["episodes"]),
        pairs=len(data["pairs"]),
        thumbnails=len(thumbs),
        capped=len(wanted) >= max_thumbnails,
    )


# The template lives in a sibling module to keep this file readable.
from forge.catalog._studio_template import HTML as _HTML  # noqa: E402
