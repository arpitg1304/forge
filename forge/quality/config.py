"""Configuration for quality metrics.

All tunable thresholds are consolidated here so the scoring system
can be adjusted from a single place.
"""

from __future__ import annotations

import math
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
    gripper_dim: int | None = None
    gripper_dims: tuple[int, ...] | None = None
    fps: float = 30.0
    action_bounds: tuple[float, float] | None = None
    action_limits: tuple[tuple[float, float], ...] | None = None
    min_frames: int = 5

    # States supplied for motion analysis must be positions (not velocities).
    # Scales have one positive value per original state column; thresholds below
    # are in scaled units, so mixed radians/metres need explicit scales.
    state_dims: tuple[int, ...] | None = None
    state_scales: tuple[float, ...] | None = None
    motion_speed_threshold: float = 1e-3
    event_min_duration_s: float = 0.5
    max_gap_factor: float = 2.5
    reversal_min_duration_s: float = 0.1

    # Tracking is opt-in: absolute position commands and explicit (action,state)
    # column pairs with compatible units. Never infer this from vector shape.
    action_semantics: str = "unknown"
    tracking_pairs: tuple[tuple[int, int], ...] | None = None
    tracking_error_threshold: float = 0.05
    tracking_lag_s: float = 0.0

    # ── Metric parameters ──
    dead_action_eps: float = 1e-6
    static_threshold_percentile: float = 95
    saturation_margin: float = 0.05  # 5% when inferring bounds, use 0.01 with --action-bounds

    # ── Flag thresholds (when to flag an episode) ──
    dead_fraction_flag: float = 0.10
    chatter_threshold: float = 2.0  # transitions/sec
    jitter_warn_ratio: float = 0.1  # dt_std / dt_mean
    saturation_flag: float = 0.30  # overall saturation fraction
    saturation_dim_flag: float = 0.05  # per-dimension saturation
    static_fraction_flag: float = 0.50  # fraction of episode that is static
    entropy_low_flag: float = 0.20  # normalized entropy below this = repetitive
    entropy_high_flag: float = 0.95  # normalized entropy above this = random
    ldlj_flag: float = -25.0  # LDLJ below this = jerky

    # ── New metric parameters ──
    psd_low_hz: float = 2.0  # upper edge of low band
    psd_high_hz: float = 8.0  # lower edge of high band ("chatter")
    psd_high_flag: float = 0.20  # >20% of action energy above high_hz = flagged
    state_action_k: int = 5  # neighbors for state-conditioned variance

    # ── Smoothness scoring (sigmoid) ──
    ldlj_sigmoid_center: float = -18.0
    ldlj_sigmoid_scale: float = 4.0

    # ── Entropy scoring (bell curve) ──
    entropy_expected: float = 0.7  # expected normalized entropy
    entropy_width: float = 0.3  # width of the bell curve

    # ── Composite score weights ──
    weight_smoothness: float = 0.25
    weight_dead_actions: float = 0.15
    weight_gripper_health: float = 0.15
    weight_static_detection: float = 0.15
    weight_timestamp_regularity: float = 0.10
    weight_action_saturation: float = 0.10
    weight_action_diversity: float = 0.10

    def __post_init__(self) -> None:
        for name in ("fps", "event_min_duration_s", "max_gap_factor", "reversal_min_duration_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "motion_speed_threshold",
            "tracking_error_threshold",
            "tracking_lag_s",
            "dead_action_eps",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_gap_factor <= 1:
            raise ValueError("max_gap_factor must exceed 1")
        if self.action_semantics not in {"unknown", "position", "velocity", "delta"}:
            raise ValueError("action_semantics must be unknown, position, velocity, or delta")
        if self.tracking_pairs is not None and (
            not self.tracking_pairs or self.action_semantics != "position"
        ):
            raise ValueError("tracking_pairs requires position semantics and at least one pair")
        if self.gripper_dims is not None and self.gripper_dim is not None:
            raise ValueError("Use gripper_dim or gripper_dims, not both")
        if self.state_dims is not None and not self.state_dims:
            raise ValueError("state_dims cannot be empty")
        if self.state_scales is not None and (
            not self.state_scales or any(not math.isfinite(x) or x <= 0 for x in self.state_scales)
        ):
            raise ValueError("state_scales must be finite and positive")
        if self.action_bounds is not None and self.action_limits is not None:
            raise ValueError("Use action_bounds or action_limits, not both")
        limits = (
            self.action_limits
            if self.action_limits is not None
            else (() if self.action_bounds is None else (self.action_bounds,))
        )
        if self.action_limits == ():
            raise ValueError("action_limits cannot be empty")
        for lo, hi in limits:
            if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
                raise ValueError("Action limits must be finite with lower < upper")
