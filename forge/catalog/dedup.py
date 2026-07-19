"""Near-duplicate detection and policy-based curation over a catalog.

`compute_dedup_edges` finds near-duplicate episode pairs from the Phase 2
embeddings (cosine over per-camera vision vectors) and records them as
``dedup_edges`` — *facts*, not verdicts. `curate` then applies a WHERE filter
plus a dedup policy and appends ``curation_labels`` (approved survivors,
rejected dedup losers).

Brute-force all-pairs cosine in numpy is sub-second at lab scale; past tens of
thousands of episodes a blocked/ANN pass is needed (logged, not silently
truncated).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from forge.catalog.catalog import Catalog

# Above this episode count, all-pairs O(N^2) cosine is skipped with a warning
# rather than blowing up memory. (ANN is the documented scale path.)
_MAX_BRUTE_FORCE = 20000

DEFAULT_THRESHOLD = 0.97
DEFAULT_POLICY = "keep-higher-quality"
_POLICIES = ("keep-higher-quality", "keep-longer", "keep-first")


@dataclass
class DedupStats:
    model_id: str
    threshold: float
    episodes: int = 0
    pairs_found: int = 0
    pairs_added: int = 0
    skipped_reason: str | None = None


@dataclass
class CurateStats:
    label: str
    selected: int = 0
    approved: int = 0
    rejected: int = 0
    dedup_policy: str | None = None


def compute_dedup_edges(
    catalog: Catalog,
    *,
    model_id: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    console=None,
) -> DedupStats:
    """Find near-duplicate pairs from episode embeddings; append ``dedup_edges``.

    Similarity between two episodes is the **max cosine over their shared
    cameras** (a dup on any view counts). Idempotent: pairs already recorded for
    this ``model_id`` are not re-added.
    """
    model_id = catalog._resolve_model_id(model_id)
    stats = DedupStats(model_id=model_id, threshold=threshold)

    rows = catalog.sql(
        "SELECT episode_id, camera, vector FROM embeddings "
        f"WHERE model_id = '{model_id.replace(chr(39), chr(39) * 2)}' "
        "AND level = 'episode'"
    ).to_pylist()

    by_cam: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    all_eps: set[str] = set()
    for r in rows:
        ids, vecs = by_cam[r["camera"]]
        ids.append(r["episode_id"])
        vecs.append(r["vector"])
        all_eps.add(r["episode_id"])
    stats.episodes = len(all_eps)

    if stats.episodes > _MAX_BRUTE_FORCE:
        stats.skipped_reason = (
            f"{stats.episodes} episodes exceeds the {_MAX_BRUTE_FORCE} brute-force "
            "cap; ANN-based dedup is a future scale path"
        )
        if console is not None:
            console.print(f"[yellow]Skipped:[/yellow] {stats.skipped_reason}")
        return stats

    # (a, b) canonical -> max cosine across shared cameras.
    pair_sim: dict[tuple[str, str], float] = {}
    for _cam, (ids, vecs) in by_cam.items():
        if len(ids) < 2:
            continue
        v = np.nan_to_num(np.asarray(vecs, dtype=np.float32))  # already L2-normalized
        with np.errstate(all="ignore"):  # silence spurious float32 BLAS warnings
            sim = v @ v.T
        ii, jj = np.where(np.triu(sim >= threshold, k=1))
        for i, j in zip(ii.tolist(), jj.tolist()):
            a, b = ids[i], ids[j]
            if a > b:
                a, b = b, a
            key = (a, b)
            pair_sim[key] = max(pair_sim.get(key, -1.0), float(sim[i, j]))
    stats.pairs_found = len(pair_sim)

    # Skip pairs already recorded for this model.
    existing = {
        (r["episode_a"], r["episode_b"])
        for r in catalog.sql(
            "SELECT episode_a, episode_b FROM dedup_edges "
            f"WHERE model_id = '{model_id.replace(chr(39), chr(39) * 2)}'"
        ).to_pylist()
    }

    now = datetime.now(timezone.utc)
    new_rows = [
        {
            "episode_a": a,
            "episode_b": b,
            "similarity": sim,
            "model_id": model_id,
            "method": "ann-cosine",
            "computed_at": now,
        }
        for (a, b), sim in pair_sim.items()
        if (a, b) not in existing
    ]
    if new_rows:
        catalog.add_dedup_edges(new_rows)
    stats.pairs_added = len(new_rows)
    return stats


def curate(
    catalog: Catalog,
    *,
    where: str | None = None,
    ids: list[str] | None = None,
    label: str = "approved",
    reason: str | None = None,
    labeled_by: str = "policy:curate",
    dedup_threshold: float | None = None,
    dedup_policy: str = DEFAULT_POLICY,
) -> CurateStats:
    """Label a selection, dropping near-dup losers under a policy.

    The selection is either an explicit ``ids`` list (from ``forge search`` or
    Forge Studio) or a SQL ``where`` predicate — or both, in which case ``ids``
    is filtered by the predicate. ``ids`` are filtered to episodes that actually
    exist, so typos/stale ids never create dangling labels.

    Survivors get ``label`` (default ``approved``); if ``dedup_threshold`` is
    given, the dedup losers within the selection get ``rejected``. Appends to the
    ``curation_labels`` log (latest-row-wins), never deleting history.
    """
    if dedup_policy not in _POLICIES:
        raise ValueError(f"unknown dedup policy {dedup_policy!r}; choose {_POLICIES}")

    stats = CurateStats(label=label)

    base = (
        "SELECT e.episode_id FROM episodes e "
        "LEFT JOIN v_latest_quality q USING(episode_id)"
    )
    if ids is not None:
        # Explicit selection: keep order, de-dup, and intersect with the
        # predicate (if any) and with episodes that actually exist.
        matched = {
            r["episode_id"]
            for r in catalog.sql(base + (f" WHERE {where}" if where else "")).to_pylist()
        }
        seen: set[str] = set()
        selected = [
            e for e in ids
            if e in matched and not (e in seen or seen.add(e))
        ]
    else:
        if where:
            base += f" WHERE {where}"
        selected = [r["episode_id"] for r in catalog.sql(base).to_pylist()]
    stats.selected = len(selected)
    if not selected:
        return stats

    losers: set[str] = set()
    if dedup_threshold is not None:
        stats.dedup_policy = dedup_policy
        loser_rows = catalog.sql(
            f"SELECT episode_id FROM v_dup_losers({float(dedup_threshold)}, "
            f"'{dedup_policy}')"
        ).to_pylist()
        all_losers = {r["episode_id"] for r in loser_rows}
        losers = all_losers & set(selected)

    now = datetime.now(timezone.utc)
    rows: list[dict] = []
    for ep in selected:
        if ep in losers:
            rows.append({
                "episode_id": ep,
                "label": "rejected",
                "reason": reason or f"near-duplicate ({dedup_policy})",
                "labeled_by": f"policy:{dedup_policy}",
                "labeled_at": now,
            })
            stats.rejected += 1
        else:
            rows.append({
                "episode_id": ep,
                "label": label,
                "reason": reason,
                "labeled_by": labeled_by,
                "labeled_at": now,
            })
            stats.approved += 1

    catalog.add_curation_labels(rows)
    return stats
