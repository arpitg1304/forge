"""Quality metric functions for robotics episode data.

All functions operate on numpy arrays and return structured results.
No video/image processing — proprioception only.

References:
    - Hogan & Sternad (2009) — LDLJ smoothness metric
    - Balasubramanian et al. (2012, 2015) — SPARC (spectral arc length)
    - Sakr et al. "Consistency Matters" (ACM THRI, 2024) — path length, manipulability
    - Belkhale et al. "DemInf" (2025) — action entropy
    - Liu et al. "SCIZOR" (2025) — static episode detection
    - Kim et al. "OpenVLA" (2024) — dead action filtering
    - Lee et al. "Sentinel" (2024) — STAC (temporal action consistency);
      we adapt the spirit to proprio as state-conditioned action variance
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from forge.quality.config import QualityConfig
from forge.quality.models import StaticResult, TimestampResult


# ── Metric 1: Dead Action Detection ─────────────────────────────


def dead_action_detection(
    actions: NDArray, config: QualityConfig
) -> tuple[float, list[tuple[int, int]]]:
    """Detect timesteps where all action dimensions are zero or constant.

    Args:
        actions: Shape (T, D) action array.
        config: Quality configuration.

    Returns:
        Tuple of (dead_fraction, dead_ranges) where dead_ranges is a list
        of (start_idx, end_idx) tuples marking dead segments.
    """
    eps = config.dead_action_eps

    # Zero actions
    is_zero = np.all(np.abs(actions) < eps, axis=1)

    # Constant actions (same as first timestep)
    is_constant = np.all(np.abs(actions - actions[0:1]) < eps, axis=1)

    is_dead = is_zero | is_constant
    dead_fraction = float(np.mean(is_dead))

    # Find contiguous dead ranges
    dead_ranges = _find_runs(is_dead)

    return dead_fraction, dead_ranges


# ── Metric 2: Log Dimensionless Jerk (LDLJ) ────────────────────


def log_dimensionless_jerk(
    positions: NDArray, dt: float
) -> float | None:
    """Compute LDLJ smoothness metric.

    More negative = jerkier (higher jerk relative to velocity).
    Typical range: -6 to -8 (smooth) to -15 to -25 (jerky).

    Args:
        positions: Shape (T, D) position array (joint angles or EE pose).
        dt: Time step in seconds (1/fps).

    Returns:
        LDLJ score, or None if input too short.
    """
    if len(positions) < 4 or dt <= 0:
        return None

    velocity = np.gradient(positions, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    jerk = np.gradient(acceleration, dt, axis=0)

    t_total = (len(positions) - 1) * dt
    peak_vel = float(np.max(np.linalg.norm(velocity, axis=1)))

    if peak_vel < 1e-10 or t_total < 1e-10:
        return None

    jerk_magnitude_sq = float(np.sum(jerk**2) * dt)

    ldlj_arg = t_total**3 / peak_vel**2 * jerk_magnitude_sq
    ldlj = -np.log(max(ldlj_arg, 1e-10))

    return float(ldlj)


# ── Metric 3: Gripper Chatter Detection ─────────────────────────


def gripper_chatter(
    actions: NDArray, duration: float, config: QualityConfig
) -> tuple[int, float, bool]:
    """Count rapid gripper open/close transitions.

    Args:
        actions: Shape (T, D) action array.
        duration: Episode duration in seconds.
        config: Quality configuration.

    Returns:
        Tuple of (chatter_count, chatter_rate, is_chattery).
    """
    gripper_dim = config.gripper_dim
    gripper = actions[:, gripper_dim]

    # Auto-detect binarization threshold: midpoint of observed range
    g_min, g_max = float(np.min(gripper)), float(np.max(gripper))
    if g_max - g_min < 1e-6:
        # Gripper never moves
        return 0, 0.0, False

    threshold = (g_min + g_max) / 2.0
    binary = (gripper > threshold).astype(np.int32)
    transitions = int(np.sum(np.abs(np.diff(binary))))

    chatter_rate = transitions / max(duration, 1e-6)
    is_chattery = chatter_rate > config.chatter_threshold

    return transitions, chatter_rate, is_chattery


# ── Metric 4: Path Length ────────────────────────────────────────


def path_length(positions: NDArray) -> float:
    """Sum of step-to-step distances in joint or Cartesian space.

    Args:
        positions: Shape (T, D) position array.

    Returns:
        Total path length.
    """
    diffs = np.diff(positions, axis=0)
    step_distances = np.linalg.norm(diffs, axis=1)
    return float(np.sum(step_distances))


# ── Metric 5: Timestamp Regularity ──────────────────────────────


def timestamp_regularity(
    timestamps: NDArray, expected_fps: float
) -> TimestampResult:
    """Analyze timestamp regularity and detect dropped frames.

    Args:
        timestamps: Shape (T,) monotonic timestamp array in seconds.
        expected_fps: Expected recording frequency.

    Returns:
        TimestampResult with regularity statistics.
    """
    dt = np.diff(timestamps)
    dt_mean = float(np.mean(dt))
    dt_std = float(np.std(dt))

    jitter_ratio = dt_std / max(dt_mean, 1e-10)

    expected_dt = 1.0 / expected_fps if expected_fps > 0 else dt_mean
    gap_mask = dt > 2.0 * expected_dt
    gap_locations = np.where(gap_mask)[0].tolist()

    effective_fps = 1.0 / dt_mean if dt_mean > 0 else 0.0

    return TimestampResult(
        dt_mean=dt_mean,
        dt_std=dt_std,
        jitter_ratio=jitter_ratio,
        num_gaps=len(gap_locations),
        gap_locations=gap_locations,
        effective_fps=effective_fps,
    )


# ── Metric 6: Action Range Saturation ───────────────────────────


def action_saturation(
    actions: NDArray, config: QualityConfig
) -> tuple[NDArray, float, list[int]]:
    """Detect timesteps where actions hit their min/max bounds.

    Args:
        actions: Shape (T, D) action array.
        config: Quality configuration.

    Returns:
        Tuple of (saturation_per_dim, overall_saturation, saturated_dims).
    """
    if config.action_bounds is not None:
        a_min = np.full(actions.shape[1], config.action_bounds[0])
        a_max = np.full(actions.shape[1], config.action_bounds[1])
    else:
        a_min = np.min(actions, axis=0)
        a_max = np.max(actions, axis=0)

    range_per_dim = a_max - a_min
    # Skip dimensions with no range
    valid = range_per_dim > 1e-10
    margin = config.saturation_margin * range_per_dim

    at_min = actions <= (a_min + margin)
    at_max = actions >= (a_max - margin)
    saturated = at_min | at_max

    # Zero out invalid dims
    saturated[:, ~valid] = False

    sat_per_dim = np.mean(saturated, axis=0).astype(np.float64)
    # Mean across dims — one saturated gripper dim shouldn't dominate
    overall = float(np.mean(sat_per_dim[valid])) if np.any(valid) else 0.0
    flagged_dims = np.where(sat_per_dim > config.saturation_dim_flag)[0].tolist()

    return sat_per_dim, overall, flagged_dims


# ── Metric 7: Static Episode Detection ──────────────────────────


def static_detection(
    actions: NDArray, fps: float, config: QualityConfig
) -> StaticResult:
    """Detect periods where the robot is not moving.

    Args:
        actions: Shape (T, D) action array.
        fps: Recording frequency (for duration calculation).
        config: Quality configuration.

    Returns:
        StaticResult with static segments and statistics.
    """
    action_norms = np.linalg.norm(actions, axis=1)

    pct_value = np.percentile(action_norms, config.static_threshold_percentile)
    threshold = 0.01 * pct_value if pct_value > 1e-10 else 1e-6

    is_static = action_norms < threshold
    static_fraction = float(np.mean(is_static))

    # Find contiguous static runs
    runs = _find_runs(is_static)
    longest = max((end - start for start, end in runs), default=0)

    # Convert to segments with duration
    dt = 1.0 / fps if fps > 0 else 1.0
    segments = [(s, e, (e - s) * dt) for s, e in runs]

    return StaticResult(
        static_fraction=static_fraction,
        is_mostly_static=static_fraction > config.static_fraction_flag,
        longest_static_run=longest,
        static_segments=segments,
    )


# ── Metric 8: Action Distribution Entropy ───────────────────────


def action_entropy(actions: NDArray) -> tuple[NDArray, float]:
    """Compute Shannon entropy of action distribution per dimension.

    Low entropy = repetitive actions. Very high = random.

    Args:
        actions: Shape (T, D) action array.

    Returns:
        Tuple of (entropy_per_dim, mean_entropy).
    """
    t, d = actions.shape
    entropy_per_dim = np.zeros(d)

    for dim in range(d):
        values = actions[:, dim]
        val_range = np.ptp(values)

        if val_range < 1e-10:
            entropy_per_dim[dim] = 0.0
            continue

        # Sturges' rule for bin count
        n_bins = max(int(np.ceil(np.log2(t) + 1)), 2)
        counts, _ = np.histogram(values, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]

        # Normalized entropy (0-1 range)
        max_entropy = np.log2(n_bins)
        raw_entropy = float(-np.sum(probs * np.log2(probs)))
        entropy_per_dim[dim] = raw_entropy / max_entropy if max_entropy > 0 else 0.0

    mean_entropy = float(np.mean(entropy_per_dim))
    return entropy_per_dim, mean_entropy


# ── Metric 9a: SPARC (Spectral Arc Length) ──────────────────────


def spectral_arc_length(
    positions: NDArray,
    dt: float,
    *,
    pad_level: int = 4,
    fc_max: float = 10.0,
    amp_thresh: float = 0.05,
) -> float | None:
    """Compute SPARC smoothness — Balasubramanian et al. 2012/2015.

    Spectral arc length of the speed-profile magnitude spectrum. More negative
    = jerkier. Complements LDLJ: LDLJ is time-domain (jerk integral), SPARC is
    frequency-domain (spectrum geometry) — disagreement between the two is a
    useful confidence signal.

    Args:
        positions: Shape (T, D) position array (joint angles or EE pose).
        dt: Time step in seconds.
        pad_level: Zero-pad to ``2^(ceil(log2(T)) + pad_level)`` for FFT
            resolution. Default 4 matches the reference implementation.
        fc_max: Hard upper bound on the adaptive cutoff (Hz). Default 10 Hz
            covers typical manipulation; raise for high-bandwidth tasks.
        amp_thresh: Adaptive cutoff = highest frequency where normalized
            spectrum amplitude is still above this threshold. Default 0.05.

    Returns:
        SPARC score, or None if input too short or speed is identically zero.
    """
    if len(positions) < 4 or dt <= 0:
        return None

    velocity = np.gradient(positions, dt, axis=0)
    speed = (
        np.linalg.norm(velocity, axis=1) if velocity.ndim == 2 else np.abs(velocity)
    )
    if speed.size < 4 or float(np.max(speed)) < 1e-12:
        return None

    n = int(2 ** (int(np.ceil(np.log2(len(speed)))) + pad_level))
    spectrum = np.abs(np.fft.rfft(speed, n=n))
    freq = np.fft.rfftfreq(n, d=dt)

    spec_max = float(spectrum.max())
    if spec_max < 1e-12:
        return None
    spec_n = spectrum / spec_max

    # Adaptive cutoff: highest freq still above threshold, capped at fc_max.
    above = np.where(spec_n >= amp_thresh)[0]
    if above.size == 0:
        return None
    fc = float(min(freq[above[-1]], fc_max))
    if fc <= 0:
        return None

    mask = freq <= fc
    f = freq[mask]
    s = spec_n[mask]
    if f.size < 2:
        return None

    # Discretized arc length of the normalized spectrum curve, with the freq
    # axis rescaled by 1/fc so the integral is dimensionless and bounded.
    df = np.diff(f) / fc
    ds = np.diff(s)
    arc = float(np.sum(np.sqrt(df * df + ds * ds)))
    return -arc


# ── Metric 9b: PSD band energy ──────────────────────────────────


def psd_band_energy(
    actions: NDArray,
    fps: float,
    *,
    low_hz: float = 2.0,
    high_hz: float = 8.0,
) -> dict[str, Any] | None:
    """Power-spectral-density of actions split into low / mid / high bands.

    Generalizes ``gripper_chatter`` to every action dim: chatter is just
    excess energy in the high-frequency band. Typical clean teleop sits
    almost entirely in the low band (<2 Hz); excess high-band power
    indicates overcorrection, jitter, or controller resonance.

    Args:
        actions: Shape (T, D) action array.
        fps: Sample rate in Hz.
        low_hz: Upper edge of the low band (default 2 Hz).
        high_hz: Lower edge of the high band (default 8 Hz). Mid band is
            the gap between low_hz and high_hz.

    Returns:
        Dict with overall and per-dim band fractions, or None if fps is
        too low to resolve the requested bands.
    """
    if actions.ndim != 2 or actions.shape[0] < 4:
        return None
    nyquist = float(fps) / 2.0
    if nyquist <= low_hz:
        # Can't resolve the bands the caller asked about.
        return None

    # Demean per-dim to drop the DC bin (mostly position offset, not signal).
    a = actions - actions.mean(axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(a, axis=0)) ** 2
    freq = np.fft.rfftfreq(actions.shape[0], d=1.0 / float(fps))

    low_mask = freq <= low_hz
    high_mask = freq > high_hz
    mid_mask = ~(low_mask | high_mask)

    low_p = spectrum[low_mask].sum(axis=0)
    mid_p = spectrum[mid_mask].sum(axis=0)
    high_p = spectrum[high_mask].sum(axis=0)
    total = low_p + mid_p + high_p
    safe_total = np.where(total > 1e-12, total, 1.0)

    low_frac = low_p / safe_total
    mid_frac = mid_p / safe_total
    high_frac = high_p / safe_total

    return {
        "low_fraction": float(low_frac.mean()),
        "mid_fraction": float(mid_frac.mean()),
        "high_fraction": float(high_frac.mean()),
        "low_fraction_per_dim": low_frac.tolist(),
        "mid_fraction_per_dim": mid_frac.tolist(),
        "high_fraction_per_dim": high_frac.tolist(),
        "low_hz": float(low_hz),
        "high_hz": float(high_hz),
    }


# ── Metric 9c: State-conditioned action variance ────────────────


def state_conditioned_action_variance(
    states: NDArray,
    actions: NDArray,
    *,
    k: int = 5,
) -> tuple[NDArray, float] | tuple[None, None]:
    """Demonstrator self-consistency — proprio cousin of STAC.

    For each frame, find the k nearest-state frames within the same episode
    (excluding the frame itself) and compute the variance of their actions.
    Episode statistic is the mean over per-frame variances.

    High value = the demonstrator picked different actions at very similar
    states. Could be intentional multimodality, a teleop reversal, or sloppy
    teaching. Low value = consistent state→action mapping.

    Args:
        states: Shape (T, D_s) state array.
        actions: Shape (T, D_a) action array.
        k: Neighbors to query (excluding self). Default 5.

    Returns:
        (per_frame_variance, mean_variance) — both None if episode is too
        short or arrays don't align.
    """
    if (
        states.ndim != 2
        or actions.ndim != 2
        or states.shape[0] != actions.shape[0]
        or states.shape[0] < k + 2
    ):
        return None, None

    # Pure-numpy pairwise k-NN. Memory is O(T²); fine for typical episodes
    # (T < 2k frames → <32 MB). For much longer episodes, swap in a KDTree.
    s = states.astype(np.float64, copy=False)
    sq_norm = (s * s).sum(axis=1)
    dist_sq = sq_norm[:, None] + sq_norm[None, :] - 2.0 * (s @ s.T)
    np.fill_diagonal(dist_sq, np.inf)  # never pick self as own neighbor
    nn_idx = np.argpartition(dist_sq, kth=k, axis=1)[:, :k]

    neighbor_actions = actions[nn_idx]  # (T, k, D_a)
    # Variance across the k neighbors per action dim, then mean across dims —
    # gives a single per-frame disagreement scalar.
    var_per_dim = neighbor_actions.var(axis=1)
    per_frame = var_per_dim.mean(axis=1)
    return per_frame, float(per_frame.mean())


# ── Metric 10: Composite Quality Score ──────────────────────────


def composite_score(
    dead_fraction: float | None,
    ldlj_score: float | None,
    is_chattery: bool | None,
    chatter_rate: float | None,
    static_fraction: float | None,
    jitter_ratio: float | None,
    overall_saturation: float | None,
    mean_entropy: float | None,
    config: QualityConfig | None = None,
) -> tuple[float, dict[str, float], list[str]]:
    """Compute weighted composite quality score.

    All thresholds, weights, and scoring parameters are read from config.

    Args:
        Individual metric results (None = not computed, skip).
        config: Quality configuration with all thresholds.

    Returns:
        Tuple of (overall_score 0-10, subscores dict, flags list).
    """
    if config is None:
        config = QualityConfig()

    subscores: dict[str, float] = {}
    flags: list[str] = []
    weights: dict[str, float] = {}

    # Dead actions
    if dead_fraction is not None:
        subscores["dead_actions"] = 1.0 - dead_fraction
        weights["dead_actions"] = config.weight_dead_actions
        if dead_fraction > config.dead_fraction_flag:
            flags.append("dead_actions")

    # Smoothness
    if ldlj_score is not None:
        subscores["smoothness"] = 1.0 / (
            1.0 + np.exp(-(ldlj_score - config.ldlj_sigmoid_center) / config.ldlj_sigmoid_scale)
        )
        weights["smoothness"] = config.weight_smoothness
        if ldlj_score < config.ldlj_flag:
            flags.append("jerky")

    # Gripper health
    if is_chattery is not None:
        if is_chattery and chatter_rate is not None:
            subscores["gripper_health"] = max(0.0, 1.0 - chatter_rate / 5.0)
        else:
            subscores["gripper_health"] = 1.0
        weights["gripper_health"] = config.weight_gripper_health
        if is_chattery:
            flags.append("gripper_chatter")

    # Timestamps
    if jitter_ratio is not None:
        if jitter_ratio < config.jitter_warn_ratio:
            subscores["timestamp_regularity"] = 1.0
        else:
            subscores["timestamp_regularity"] = max(0.0, 1.0 - jitter_ratio)
        weights["timestamp_regularity"] = config.weight_timestamp_regularity
        if jitter_ratio > config.jitter_warn_ratio:
            flags.append("timestamp_jitter")

    # Saturation
    if overall_saturation is not None:
        subscores["action_saturation"] = 1.0 - overall_saturation
        weights["action_saturation"] = config.weight_action_saturation
        if overall_saturation > config.saturation_flag:
            flags.append("saturated")

    # Static
    if static_fraction is not None:
        subscores["static_detection"] = 1.0 - static_fraction
        weights["static_detection"] = config.weight_static_detection
        if static_fraction > config.static_fraction_flag:
            flags.append("mostly_static")

    # Entropy (bell curve: penalize both too low and too high)
    if mean_entropy is not None:
        deviation = abs(mean_entropy - config.entropy_expected) / config.entropy_width
        subscores["action_diversity"] = max(0.0, 1.0 - deviation)
        weights["action_diversity"] = config.weight_action_diversity
        if mean_entropy < config.entropy_low_flag:
            flags.append("low_entropy")
        elif mean_entropy > config.entropy_high_flag:
            flags.append("high_entropy")

    # Weighted average
    if weights:
        total_weight = sum(weights.values())
        score = sum(subscores[k] * weights[k] for k in weights) / total_weight * 10.0
    else:
        score = 0.0

    return float(score), subscores, flags


# ── Helpers ──────────────────────────────────────────────────────


def _find_runs(mask: NDArray) -> list[tuple[int, int]]:
    """Find contiguous True runs in a boolean array.

    Returns list of (start_idx, end_idx) tuples (end exclusive).
    """
    if len(mask) == 0:
        return []

    runs: list[tuple[int, int]] = []
    padded = np.concatenate(([0], mask.astype(np.int32), [0]))
    diffs = np.diff(padded)

    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    for s, e in zip(starts, ends):
        runs.append((int(s), int(e)))

    return runs
