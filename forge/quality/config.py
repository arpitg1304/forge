"""Configuration for quality metrics.

All tunable thresholds are consolidated here so the scoring system
can be adjusted from a single place.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QualityConfig:
    """Configuration for quality analysis.

    Groups:
        Input: gripper_dim, fps, action_bounds, min_frames
        Metric params: dead_action_eps, static_threshold_percentile, saturation_margin
        Flag thresholds: thresholds that determine when an episode gets flagged
        Scoring: sigmoid parameters for the composite score
    """

    # ── Input / data access ──
    gripper_dim: int = -1
    fps: float = 30.0
    action_bounds: tuple[float, float] | None = None
    min_frames: int = 5

    # ── Metric parameters ──
    dead_action_eps: float = 1e-6
    static_threshold_percentile: float = 95
    saturation_margin: float = 0.05          # 5% when inferring bounds, use 0.01 with --action-bounds

    # ── Flag thresholds (when to flag an episode) ──
    dead_fraction_flag: float = 0.10
    chatter_threshold: float = 2.0         # transitions/sec
    jitter_warn_ratio: float = 0.1         # dt_std / dt_mean
    saturation_flag: float = 0.30          # overall saturation fraction
    saturation_dim_flag: float = 0.05      # per-dimension saturation
    static_fraction_flag: float = 0.50     # fraction of episode that is static
    entropy_low_flag: float = 0.20         # normalized entropy below this = repetitive
    entropy_high_flag: float = 0.95        # normalized entropy above this = random
    ldlj_flag: float = -25.0              # LDLJ below this = jerky

    # ── New metric parameters ──
    psd_low_hz: float = 2.0                # upper edge of low band
    psd_high_hz: float = 8.0               # lower edge of high band ("chatter")
    psd_high_flag: float = 0.20            # >20% of action energy above high_hz = flagged
    state_action_k: int = 5                # neighbors for state-conditioned variance

    # ── Smoothness scoring (sigmoid) ──
    ldlj_sigmoid_center: float = -18.0
    ldlj_sigmoid_scale: float = 4.0

    # ── Entropy scoring (bell curve) ──
    entropy_expected: float = 0.7          # expected normalized entropy
    entropy_width: float = 0.3             # width of the bell curve

    # ── Composite score weights ──
    weight_smoothness: float = 0.25
    weight_dead_actions: float = 0.15
    weight_gripper_health: float = 0.15
    weight_static_detection: float = 0.15
    weight_timestamp_regularity: float = 0.10
    weight_action_saturation: float = 0.10
    weight_action_diversity: float = 0.10
