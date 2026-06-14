"""Perceptual deduplication engine.

Finds near-duplicate *episodes* via per-camera keyframe perceptual hashes,
clusters them with union-find, and reports one representative per cluster plus
the episodes that would be dropped. Writing the deduplicated dataset is
delegated to the filter engine (exclude the dropped IDs), so there is exactly
one code path that writes the same source format.

Tier 0: numpy only, CPU, no model. Semantic (CLIP) dedup is a later tier.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from forge.dedup.hashes import get_hasher
from forge.formats.registry import FormatRegistry

logger = logging.getLogger(__name__)


@dataclass
class DedupConfig:
    """Configuration for a dedup operation."""

    method: str = "phash"            # phash | dhash | ahash
    threshold: float = 0.10          # normalized Hamming distance (0 = exact)
    keyframes: int = 16              # frames sampled per episode per camera
    hash_size: int = 8               # hash is hash_size**2 bits
    cameras: list[str] | None = None # restrict to these cameras (None = all)


@dataclass
class DuplicateCluster:
    """A set of near-duplicate episodes sharing one representative."""

    representative: str
    duplicates: list[tuple[str, float]] = field(default_factory=list)  # (id, distance)


@dataclass
class DedupResult:
    """Result of a dedup analysis pass."""

    method: str
    threshold: float
    total_episodes: int = 0
    num_unique: int = 0
    num_duplicates: int = 0
    num_uncomparable: int = 0        # episodes with no usable camera frames
    clusters: list[DuplicateCluster] = field(default_factory=list)
    kept_ids: list[str] = field(default_factory=list)
    dropped_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Keep the lower index as the root so representatives are the
            # earliest episode in dataset order.
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            self.parent[hi] = lo


class DedupEngine:
    """Detects near-duplicate episodes via perceptual hashing."""

    def __init__(self, config: DedupConfig) -> None:
        self.config = config
        self._hasher = get_hasher(config.method)

    def analyze(
        self,
        source: str | Path,
        source_format: str | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> DedupResult:
        """Find duplicate clusters in a dataset (no writing)."""
        source = Path(source)
        result = DedupResult(method=self.config.method, threshold=self.config.threshold)

        if source_format is None:
            source_format = FormatRegistry.detect_format(source)
        reader = FormatRegistry.get_reader(source_format)

        # 1. Build a per-episode signature (per-camera keyframe hashes).
        episode_ids: list[str] = []
        signatures: list[dict[str, NDArray] | None] = []
        for i, episode in enumerate(reader.read_episodes(source)):
            if progress_callback:
                progress_callback("signature", i, 0)
            episode_ids.append(episode.episode_id)
            try:
                signatures.append(self._signature(episode))
            except Exception as e:  # a bad episode should not abort the run
                result.errors.append(f"Signature failed for {episode.episode_id}: {e}")
                signatures.append(None)

        n = len(episode_ids)
        result.total_episodes = n

        # 2. Cluster comparable episodes; uncomparable ones are always kept.
        comparable = [i for i in range(n) if signatures[i] is not None]
        result.num_uncomparable = n - len(comparable)
        if result.num_uncomparable:
            logger.warning(
                "%d episode(s) had no usable camera frames — kept as-is",
                result.num_uncomparable,
            )

        uf = _UnionFind(n)
        thr = self.config.threshold
        for a_pos, i in enumerate(comparable):
            if progress_callback:
                progress_callback("compare", a_pos, len(comparable))
            for j in comparable[a_pos + 1 :]:
                if uf.find(i) == uf.find(j):
                    continue  # already in the same cluster
                if self._distance(signatures[i], signatures[j]) <= thr:
                    uf.union(i, j)

        # 3. Group by root → clusters; representative = earliest index.
        members: dict[int, list[int]] = defaultdict(list)
        for idx in range(n):
            members[uf.find(idx)].append(idx)

        for root in sorted(members):
            group = sorted(members[root])
            rep = episode_ids[group[0]]
            result.kept_ids.append(rep)
            if len(group) == 1:
                continue
            cluster = DuplicateCluster(representative=rep)
            for dup_idx in group[1:]:
                dist = self._distance(signatures[group[0]], signatures[dup_idx])
                cluster.duplicates.append((episode_ids[dup_idx], dist))
                result.dropped_ids.append(episode_ids[dup_idx])
            result.clusters.append(cluster)

        result.num_duplicates = len(result.dropped_ids)
        result.num_unique = len(result.kept_ids)
        return result

    # ── internals ──

    def _signature(self, episode) -> dict[str, NDArray] | None:
        """Per-camera keyframe-hash matrix: {camera: (K, bits) bool}.

        Collects lazy image refs across all frames (cheap — no decode), then
        decodes and hashes only K uniformly-spaced keyframes per camera.
        """
        cams = self.config.cameras
        refs: dict[str, list] = defaultdict(list)
        for frame in episode.frames():
            if not frame.images:
                continue
            for name, lazy in frame.images.items():
                if cams is not None and name not in cams:
                    continue
                refs[name].append(lazy)

        sig: dict[str, NDArray] = {}
        k = self.config.keyframes
        for name, imgs in refs.items():
            nframes = len(imgs)
            if nframes == 0:
                continue
            # Exactly K indices; repeats are fine when the episode is short, so
            # every signature has K aligned rows regardless of episode length.
            idxs = np.linspace(0, nframes - 1, k).round().astype(int)
            rows = []
            for fi in idxs:
                try:
                    rows.append(self._hasher(imgs[fi].load(), self.config.hash_size))
                except Exception:
                    continue
            if rows:
                sig[name] = np.stack(rows)

        return sig or None

    def _distance(self, sig_a: dict[str, NDArray], sig_b: dict[str, NDArray]) -> float:
        """Worst-case normalized Hamming distance across shared cameras.

        Returns 1.0 (maximally far) when the two share no camera, so episodes
        from different camera rigs are never declared duplicates.
        """
        shared = sig_a.keys() & sig_b.keys()
        if not shared:
            return 1.0
        worst = 0.0
        for cam in shared:
            a, b = sig_a[cam], sig_b[cam]
            m = min(a.shape[0], b.shape[0])
            per_frame = np.count_nonzero(a[:m] != b[:m], axis=1) / a.shape[1]
            worst = max(worst, float(per_frame.mean()))
        return worst
