"""Video (pixel) quality metrics for robotics episodes.

Tier 0 of the video-modality extension: pure pixel statistics computed on a
downscaled grayscale frame — sharpness (blur), exposure, frozen-frame /
encoder-stall detection, and colourfulness. Numpy only; no model, no GPU, and
no new dependencies beyond whatever the dataset reader already uses to decode
frames.

Usage::

    from forge.quality.video import VideoQualityAnalyzer, VideoQualityConfig

    analyzer = VideoQualityAnalyzer(VideoQualityConfig(downscale=128))
    video = analyzer.analyze_episode(episode)   # VideoQuality | None
"""

from forge.quality.video.analyzer import VideoQualityAnalyzer
from forge.quality.video.config import VideoQualityConfig
from forge.quality.video.models import CameraVideoQuality, VideoQuality

__all__ = [
    "VideoQualityAnalyzer",
    "VideoQualityConfig",
    "VideoQuality",
    "CameraVideoQuality",
]
