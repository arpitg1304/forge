"""Perceptual deduplication for robotics datasets.

`forge dedup` finds near-duplicate episodes via per-camera keyframe perceptual
hashes (pHash / dHash / aHash) and writes a deduplicated dataset in the same
source format. Tier 0: numpy only, CPU, no model and no new dependencies.

Usage::

    from forge.dedup import DedupEngine, DedupConfig

    result = DedupEngine(DedupConfig(method="phash", threshold=0.10)).analyze("./ds")
    print(result.num_duplicates, "near-duplicates across", len(result.clusters), "clusters")
"""

from forge.dedup.engine import (
    DedupConfig,
    DedupEngine,
    DedupResult,
    DuplicateCluster,
)

__all__ = [
    "DedupEngine",
    "DedupConfig",
    "DedupResult",
    "DuplicateCluster",
]
