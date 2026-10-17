"""Quality metrics for robotics episode data.

Compute episode-level quality scores based on proprioception data:
smoothness (LDLJ), dead actions, gripper chatter, static detection,
timestamp regularity, action saturation, and action diversity.

Usage::

    from forge.quality import QualityAnalyzer, QualityReport

    analyzer = QualityAnalyzer(fps=30.0)
    report = analyzer.analyze_dataset("./bridge_v2")
    print(report.overall_score)
"""

from forge.quality.analyzer import QualityAnalyzer
from forge.quality.config import QualityConfig
from forge.quality.models import EpisodeQuality, QualityEvent, QualityReport

__all__ = [
    "QualityAnalyzer",
    "QualityConfig",
    "QualityReport",
    "EpisodeQuality",
    "QualityEvent",
]
