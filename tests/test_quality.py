"""Tests for forge.quality module.

Uses synthetic episode data to verify each metric and the composite score.
"""

import numpy as np
import pytest

from forge.quality.config import QualityConfig
from forge.quality.metrics import (
    action_entropy,
    action_saturation,
    composite_score,
    dead_action_detection,
    gripper_chatter,
    log_dimensionless_jerk,
    path_length,
    psd_band_energy,
    spectral_arc_length,
    state_conditioned_action_variance,
    static_detection,
    timestamp_regularity,
)
from forge.quality.analyzer import QualityAnalyzer
from forge.quality.models import EpisodeQuality, QualityReport


# ── Synthetic Data Generators ────────────────────────────────────


def make_smooth_episode(
    T: int = 150, D: int = 7, fps: float = 30.0
) -> dict:
    """Generate a smooth, high-quality episode.

    Bell-curve velocity profile, no chatter, regular timestamps.
    """
    t = np.linspace(0, 1, T)
    # Smooth sine trajectory per dimension
    actions = np.column_stack(
        [0.5 * np.sin(2 * np.pi * t + phase) for phase in np.linspace(0, np.pi, D)]
    )
    # Gripper: smooth open then close
    actions[:, -1] = np.where(t < 0.5, 0.0, 1.0)

    states = actions * 0.8 + 0.1  # Slightly different from actions
    timestamps = np.arange(T) / fps

    return {
        "episode_id": "smooth_ep",
        "actions": actions,
        "states": states,
        "timestamps": timestamps,
        "fps": fps,
    }


def make_jerky_episode(
    T: int = 150, D: int = 7, fps: float = 30.0
) -> dict:
    """Generate a jerky, low-quality episode.

    Random actions, should have low smoothness score.
    """
    rng = np.random.RandomState(42)
    actions = rng.randn(T, D)
    states = rng.randn(T, D) * 0.5
    timestamps = np.arange(T) / fps

    return {
        "episode_id": "jerky_ep",
        "actions": actions,
        "states": states,
        "timestamps": timestamps,
        "fps": fps,
    }


def make_chattery_episode(
    T: int = 150, D: int = 7, fps: float = 30.0
) -> dict:
    """Generate episode with rapid gripper toggling."""
    t = np.linspace(0, 1, T)
    actions = np.column_stack(
        [0.3 * np.sin(2 * np.pi * t + phase) for phase in np.linspace(0, np.pi, D)]
    )
    # Rapid gripper toggling: 0 → 1 → 0 → 1 every few frames
    gripper = np.zeros(T)
    for i in range(T):
        gripper[i] = 1.0 if (i // 3) % 2 == 0 else 0.0
    actions[:, -1] = gripper

    states = actions * 0.9
    timestamps = np.arange(T) / fps

    return {
        "episode_id": "chattery_ep",
        "actions": actions,
        "states": states,
        "timestamps": timestamps,
        "fps": fps,
    }


def make_static_episode(
    T: int = 150, D: int = 7, fps: float = 30.0
) -> dict:
    """Generate episode that is mostly zeros with brief movement."""
    actions = np.zeros((T, D))
    # Brief movement in the middle (frames 60-80)
    actions[60:80] = np.random.RandomState(42).randn(20, D) * 0.5

    states = np.zeros((T, D))
    states[60:80] = actions[60:80] * 0.8
    timestamps = np.arange(T) / fps

    return {
        "episode_id": "static_ep",
        "actions": actions,
        "states": states,
        "timestamps": timestamps,
        "fps": fps,
    }


def make_dead_episode(
    T: int = 150, D: int = 7, fps: float = 30.0
) -> dict:
    """Generate episode with all-zero actions."""
    actions = np.zeros((T, D))
    states = np.ones((T, D)) * 0.5  # States are constant but non-zero
    timestamps = np.arange(T) / fps

    return {
        "episode_id": "dead_ep",
        "actions": actions,
        "states": states,
        "timestamps": timestamps,
        "fps": fps,
    }


# ── Metric Tests ─────────────────────────────────────────────────


class TestDeadActionDetection:
    def test_all_zeros(self):
        actions = np.zeros((100, 7))
        config = QualityConfig()
        frac, ranges = dead_action_detection(actions, config)
        assert frac == 1.0
        assert len(ranges) == 1
        assert ranges[0] == (0, 100)

    def test_no_dead(self):
        rng = np.random.RandomState(42)
        actions = rng.randn(100, 7)
        config = QualityConfig()
        frac, ranges = dead_action_detection(actions, config)
        assert frac < 0.1

    def test_partial_dead(self):
        # Use varying non-zero values so constant check doesn't flag everything
        rng = np.random.RandomState(42)
        actions = rng.randn(100, 7)
        actions[20:40] = 0.0  # 20% dead
        config = QualityConfig()
        frac, ranges = dead_action_detection(actions, config)
        assert 0.15 < frac < 0.25


class TestLDLJ:
    def test_smooth_trajectory(self):
        t = np.linspace(0, 2 * np.pi, 200)
        positions = np.column_stack([np.sin(t), np.cos(t)])
        score = log_dimensionless_jerk(positions, dt=1 / 30.0)
        assert score is not None
        assert score < -8  # Should be quite smooth

    def test_noisy_trajectory(self):
        # Add noise to a smooth trajectory — should degrade score (more negative)
        t = np.linspace(0, 2 * np.pi, 200)
        smooth = np.column_stack([np.sin(t), np.cos(t)])
        smooth_score = log_dimensionless_jerk(smooth, dt=1 / 30.0)

        rng = np.random.RandomState(42)
        noisy = smooth + rng.randn(200, 2) * 0.3
        noisy_score = log_dimensionless_jerk(noisy, dt=1 / 30.0)

        assert smooth_score is not None
        assert noisy_score is not None
        # More negative = jerkier, noisy should be more negative
        assert noisy_score < smooth_score

    def test_too_short(self):
        positions = np.array([[0, 0], [1, 1]])
        assert log_dimensionless_jerk(positions, dt=1 / 30.0) is None

    def test_static_returns_none(self):
        positions = np.ones((100, 3))
        assert log_dimensionless_jerk(positions, dt=1 / 30.0) is None


class TestGripperChatter:
    def test_no_chatter(self):
        actions = np.zeros((100, 7))
        actions[:, -1] = np.where(np.arange(100) < 50, 0.0, 1.0)  # One transition
        config = QualityConfig()
        count, rate, is_chat = gripper_chatter(actions, 100 / 30.0, config)
        assert count == 1
        assert not is_chat

    def test_high_chatter(self):
        actions = np.zeros((100, 7))
        # Toggle every frame
        actions[:, -1] = np.tile([0.0, 1.0], 50)
        config = QualityConfig()
        count, rate, is_chat = gripper_chatter(actions, 100 / 30.0, config)
        assert count == 99
        assert is_chat

    def test_constant_gripper(self):
        actions = np.zeros((100, 7))
        config = QualityConfig()
        count, rate, is_chat = gripper_chatter(actions, 100 / 30.0, config)
        assert count == 0
        assert not is_chat


class TestPathLength:
    def test_stationary(self):
        positions = np.ones((100, 3))
        assert path_length(positions) == pytest.approx(0.0)

    def test_straight_line(self):
        positions = np.zeros((100, 3))
        positions[:, 0] = np.linspace(0, 10, 100)
        assert path_length(positions) == pytest.approx(10.0, rel=0.01)

    def test_wandering_longer(self):
        rng = np.random.RandomState(42)
        positions = np.cumsum(rng.randn(100, 3), axis=0)
        assert path_length(positions) > 0


class TestTimestampRegularity:
    def test_regular_timestamps(self):
        timestamps = np.arange(100) / 30.0
        result = timestamp_regularity(timestamps, 30.0)
        assert result.jitter_ratio < 0.01
        assert result.num_gaps == 0
        assert result.effective_fps == pytest.approx(30.0, rel=0.01)

    def test_with_gaps(self):
        timestamps = np.arange(100) / 30.0
        # Insert a 10-frame gap at frame 50
        timestamps[50:] += 10 / 30.0
        result = timestamp_regularity(timestamps, 30.0)
        assert result.num_gaps >= 1
        assert 50 in result.gap_locations or 49 in result.gap_locations


class TestActionSaturation:
    def test_no_saturation(self):
        # Gaussian data: most values in center, few at edges
        rng = np.random.RandomState(42)
        actions = rng.randn(500, 7) * 0.3
        config = QualityConfig()
        _, overall, dims = action_saturation(actions, config)
        assert overall < 0.5

    def test_full_saturation(self):
        actions = np.ones((100, 7))
        config = QualityConfig()
        _, overall, dims = action_saturation(actions, config)
        # All values identical → range=0, no valid dims to saturate
        assert overall == 0.0

    def test_known_bounds(self):
        actions = np.ones((100, 7))
        actions[:50] = -1.0
        actions[50:] = 1.0
        config = QualityConfig(action_bounds=(-1.0, 1.0))
        _, overall, dims = action_saturation(actions, config)
        assert overall > 0.9  # All at bounds


class TestStaticDetection:
    def test_all_static(self):
        actions = np.zeros((100, 7))
        config = QualityConfig()
        result = static_detection(actions, 30.0, config)
        assert result.static_fraction > 0.9
        assert result.is_mostly_static

    def test_active(self):
        rng = np.random.RandomState(42)
        actions = rng.randn(100, 7)
        config = QualityConfig()
        result = static_detection(actions, 30.0, config)
        assert result.static_fraction < 0.1
        assert not result.is_mostly_static


class TestActionEntropy:
    def test_constant_actions(self):
        actions = np.ones((100, 7))
        _, mean_ent = action_entropy(actions)
        assert mean_ent == 0.0

    def test_uniform_actions(self):
        rng = np.random.RandomState(42)
        actions = rng.uniform(-1, 1, (1000, 7))
        _, mean_ent = action_entropy(actions)
        assert mean_ent > 0.7  # Should be high entropy


class TestSpectralArcLength:
    def test_smooth_vs_noisy(self):
        t = np.linspace(0, 2 * np.pi, 200)
        smooth = np.column_stack([np.sin(t), np.cos(t)])
        rng = np.random.RandomState(42)
        noisy = smooth + rng.randn(200, 2) * 0.3
        s_smooth = spectral_arc_length(smooth, dt=1 / 30.0)
        s_noisy = spectral_arc_length(noisy, dt=1 / 30.0)
        assert s_smooth is not None and s_noisy is not None
        # More negative = jerkier; noisy should be more negative.
        assert s_noisy < s_smooth

    def test_too_short(self):
        positions = np.array([[0.0, 0.0], [1.0, 1.0]])
        assert spectral_arc_length(positions, dt=1 / 30.0) is None

    def test_static_returns_none(self):
        positions = np.ones((100, 3))
        assert spectral_arc_length(positions, dt=1 / 30.0) is None

    def test_agrees_with_ldlj_direction(self):
        """SPARC and LDLJ should agree on which trajectory is jerkier."""
        t = np.linspace(0, 2 * np.pi, 200)
        smooth = np.column_stack([np.sin(t), np.cos(t)])
        rng = np.random.RandomState(0)
        noisy = smooth + rng.randn(200, 2) * 0.5
        ldlj_d = log_dimensionless_jerk(noisy, dt=1 / 30.0) - log_dimensionless_jerk(
            smooth, dt=1 / 30.0
        )
        sparc_d = spectral_arc_length(noisy, dt=1 / 30.0) - spectral_arc_length(
            smooth, dt=1 / 30.0
        )
        # Both deltas should be negative (noisy worse than smooth).
        assert ldlj_d < 0 and sparc_d < 0


class TestPsdBandEnergy:
    def test_low_band_dominates_smooth(self):
        t = np.linspace(0, 4 * np.pi, 600)
        # 0.5 Hz sinusoid at fs=30 → almost all energy in low band (<2 Hz).
        actions = np.column_stack([np.sin(0.5 * 2 * np.pi * t / (4 * np.pi)) for _ in range(3)])
        bands = psd_band_energy(actions, fps=30.0)
        assert bands is not None
        assert bands["low_fraction"] > 0.9
        assert bands["high_fraction"] < 0.05

    def test_high_band_dominates_chatter(self):
        # Pure 12 Hz signal at fs=30 → energy in high band (>8 Hz).
        t = np.arange(600) / 30.0
        actions = np.column_stack([np.sin(2 * np.pi * 12 * t) for _ in range(3)])
        bands = psd_band_energy(actions, fps=30.0)
        assert bands is not None
        assert bands["high_fraction"] > 0.9
        assert bands["low_fraction"] < 0.05

    def test_per_dim_lists_have_right_shape(self):
        rng = np.random.RandomState(0)
        actions = rng.randn(300, 5)
        bands = psd_band_energy(actions, fps=30.0)
        assert bands is not None
        for key in ("low_fraction_per_dim", "mid_fraction_per_dim", "high_fraction_per_dim"):
            assert len(bands[key]) == 5

    def test_too_short_returns_none(self):
        assert psd_band_energy(np.zeros((3, 4)), fps=30.0) is None

    def test_unresolvable_bands_returns_none(self):
        # fps=2 → nyquist=1, can't resolve a 2 Hz low-band edge.
        assert psd_band_energy(np.zeros((200, 4)), fps=2.0) is None

    def test_high_band_above_nyquist_returns_none(self):
        # fps=10 → nyquist=5 Hz < 8 Hz high-band edge. Chatter aliases into
        # lower bands, so a high_fraction of 0 would be a false negative.
        t = np.arange(200) / 10.0
        actions = np.column_stack([np.sin(2 * np.pi * t)] * 4)
        assert psd_band_energy(actions, fps=10.0) is None


class TestStateConditionedActionVariance:
    def test_identical_states_zero_variance(self):
        # 200 frames of the same state with the same action → all neighbors
        # have identical actions → variance == 0.
        states = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (200, 1))
        actions = np.tile(np.array([0.5, -0.5, 0.0]), (200, 1))
        _, mean_var = state_conditioned_action_variance(states, actions, k=5)
        assert mean_var is not None
        assert mean_var == 0.0

    def test_inconsistent_demo_high_variance(self):
        # Same state distribution but random unrelated actions → high variance.
        rng = np.random.RandomState(0)
        states = rng.randn(200, 4)
        actions = rng.randn(200, 3)
        _, mean_var = state_conditioned_action_variance(states, actions, k=5)
        assert mean_var is not None
        assert mean_var > 0.1

    def test_consistent_lower_than_inconsistent(self):
        rng = np.random.RandomState(1)
        states = rng.randn(200, 4)
        consistent_actions = states[:, :3] * 1.5
        random_actions = rng.randn(200, 3)
        _, v_consistent = state_conditioned_action_variance(states, consistent_actions, k=5)
        _, v_random = state_conditioned_action_variance(states, random_actions, k=5)
        assert v_consistent < v_random

    def test_too_short_returns_none(self):
        states = np.zeros((4, 3))
        actions = np.zeros((4, 2))
        per_frame, mean_var = state_conditioned_action_variance(states, actions, k=5)
        assert per_frame is None and mean_var is None

    def test_mismatched_lengths_returns_none(self):
        states = np.zeros((100, 3))
        actions = np.zeros((99, 2))
        per_frame, mean_var = state_conditioned_action_variance(states, actions, k=5)
        assert per_frame is None and mean_var is None


class TestCompositeScore:
    def test_perfect_scores(self):
        score, subscores, flags = composite_score(
            dead_fraction=0.0,
            ldlj_score=-8.0,  # Smooth trajectory (less negative = smoother)
            is_chattery=False,
            chatter_rate=0.0,
            static_fraction=0.0,
            jitter_ratio=0.02,
            overall_saturation=0.0,
            mean_entropy=0.7,
        )
        assert score > 7.0
        assert len(flags) == 0

    def test_bad_scores(self):
        score, subscores, flags = composite_score(
            dead_fraction=0.5,
            ldlj_score=-3.0,
            is_chattery=True,
            chatter_rate=5.0,
            static_fraction=0.7,
            jitter_ratio=0.5,
            overall_saturation=0.5,
            mean_entropy=0.05,
        )
        assert score < 5.0
        assert len(flags) > 3

    def test_none_values(self):
        score, subscores, flags = composite_score(
            dead_fraction=None,
            ldlj_score=None,
            is_chattery=None,
            chatter_rate=None,
            static_fraction=None,
            jitter_ratio=None,
            overall_saturation=None,
            mean_entropy=None,
        )
        assert score == 0.0


# ── Analyzer Tests ───────────────────────────────────────────────


class TestAnalyzer:
    def test_smooth_episode_high_score(self):
        ep = make_smooth_episode()
        analyzer = QualityAnalyzer()
        eq = analyzer.analyze_episode_arrays(**ep)
        assert eq.overall_score is not None
        assert eq.overall_score > 5.0
        assert eq.num_frames == 150

    def test_jerky_episode_flagged(self):
        """Jerky episode should have low overall score and some flags."""
        ep = make_jerky_episode()
        analyzer = QualityAnalyzer()
        eq = analyzer.analyze_episode_arrays(**ep)
        assert eq.overall_score is not None
        # Random actions produce chatter and/or low scores
        assert eq.overall_score < 8.0
        assert eq.is_flagged

    def test_chattery_episode_flagged(self):
        ep = make_chattery_episode()
        analyzer = QualityAnalyzer()
        eq = analyzer.analyze_episode_arrays(**ep)
        assert eq.is_chattery
        assert "gripper_chatter" in eq.flags

    def test_static_episode_flagged(self):
        ep = make_static_episode()
        analyzer = QualityAnalyzer()
        eq = analyzer.analyze_episode_arrays(**ep)
        assert eq.static is not None
        assert eq.static.is_mostly_static
        assert "mostly_static" in eq.flags

    def test_dead_episode(self):
        ep = make_dead_episode()
        analyzer = QualityAnalyzer()
        eq = analyzer.analyze_episode_arrays(**ep)
        assert eq.dead_fraction is not None
        assert eq.dead_fraction > 0.9

    def test_short_episode_graceful(self):
        analyzer = QualityAnalyzer()
        eq = analyzer.analyze_episode_arrays(
            episode_id="short",
            actions=np.zeros((3, 7)),
            fps=30.0,
        )
        # Should return without crashing, most metrics None
        assert eq.num_frames == 3
        assert eq.ldlj is None

    def test_no_data_graceful(self):
        analyzer = QualityAnalyzer()
        eq = analyzer.analyze_episode_arrays(
            episode_id="empty",
        )
        assert eq.num_frames == 0

    def test_nan_handling(self):
        actions = np.ones((100, 7))
        actions[50] = np.nan
        analyzer = QualityAnalyzer()
        eq = analyzer.analyze_episode_arrays(
            episode_id="nan_ep",
            actions=actions,
            fps=30.0,
        )
        assert "nan_in_actions" in eq.flags
        assert eq.overall_score is not None


class TestQualityReport:
    def test_to_dict(self):
        report = QualityReport(dataset_path="./test", num_episodes=1, overall_score=7.5)
        d = report.to_dict()
        assert d["overall_score"] == 7.5
        assert d["dataset_path"] == "./test"

    def test_to_json(self, tmp_path):
        report = QualityReport(dataset_path="./test", num_episodes=1, overall_score=7.5)
        out_path = tmp_path / "report.json"
        report.to_json(out_path)
        assert out_path.exists()

        import json

        data = json.loads(out_path.read_text())
        assert data["overall_score"] == 7.5
