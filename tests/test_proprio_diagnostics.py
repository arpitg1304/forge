"""Semantic regressions from the October survey, including event alignment."""

import json

import numpy as np
import pytest

from forge.quality import QualityAnalyzer, QualityConfig, QualityReport
from forge.quality.metrics import dead_action_detection


def trajectory(fps=30, seconds=6):
    t = np.arange(int(fps * seconds) + 1) / fps
    q = np.column_stack([1 + t * 0.2, 2 + t * 0.1])
    return t, q


def test_later_nonzero_hold_detected_and_localized():
    t, q = trajectory()
    q[60:] = q[60]
    eq = QualityAnalyzer().analyze_episode_arrays("hold", q, q, t)
    pauses = [e for e in eq.events if e.kind == "motion_pause"]
    assert len(pauses) == 1
    assert pauses[0].start_time_s == pytest.approx(2)
    assert pauses[0].end_time_s == pytest.approx(6)
    assert pauses[0].end_frame == len(t)
    assert eq.static.static_fraction == pytest.approx(4 / 6)
    assert eq.dead_fraction > 0.6
    assert any(e.kind == "held_command" for e in eq.events)


def test_initial_frame_is_not_a_dead_command():
    _, q = trajectory()
    fraction, ranges = dead_action_detection(q, QualityConfig())
    assert fraction == 0 and ranges == []


@pytest.mark.parametrize("fps", [10, 30, 50])
def test_pause_duration_invariant_to_rate_and_position_origin(fps):
    t, q = trajectory(fps)
    q[2 * fps :] = q[2 * fps]
    first = QualityAnalyzer(fps=fps).analyze_episode_arrays("a", q, q, t)
    shifted = QualityAnalyzer(fps=fps).analyze_episode_arrays("b", q + 100, q + 100, t)
    assert first.static.static_fraction == pytest.approx(2 / 3)
    assert shifted.static == first.static


def test_stationary_state_never_falls_back_to_moving_commands():
    t, commands = trajectory()
    state = np.repeat(commands[:1], len(t), axis=0)
    eq = QualityAnalyzer().analyze_episode_arrays("stuck", commands, state, t)
    assert eq.ldlj is None and eq.sparc is None
    assert eq.overall_score is None
    assert eq.static.static_fraction == pytest.approx(1)
    assert "tracking_stall_candidate" not in eq.flags  # No semantic mapping yet.
    eq = QualityAnalyzer(
        action_semantics="position", tracking_pairs=((0, 0), (1, 1))
    ).analyze_episode_arrays("stuck", commands, state, t)
    assert eq.diagnostics["tracking_error_p95"] > 1
    assert eq.diagnostics["tracking_stall_fraction"] > 0.8
    assert "tracking_stall_candidate" in eq.flags


def test_tracking_lag_uses_past_commands():
    t, commands = trajectory()
    states = np.column_stack([1 + np.maximum(0, t - 0.2) * 0.2, 2 + np.maximum(0, t - 0.2) * 0.1])
    eq = QualityAnalyzer(
        action_semantics="position", tracking_pairs=((0, 0), (1, 1)), tracking_lag_s=0.2
    ).analyze_episode_arrays("lag", commands, states, t)
    assert eq.diagnostics["tracking_error_p95"] < 0.01
    assert not any(e.kind == "tracking_stall_candidate" for e in eq.events)


def test_gap_never_becomes_a_pause_or_tracking_event():
    t, q = trajectory()
    t[60:] += 30
    q[:] = 2
    eq = QualityAnalyzer().analyze_episode_arrays("gap", q, q, t)
    assert eq.overall_score is None
    assert any(e.kind == "timestamp_gap" for e in eq.events)
    assert all(e.duration_s < 6 for e in eq.events if e.kind != "timestamp_gap")
    assert eq.diagnostics["time_coverage"] < 0.2


@pytest.mark.parametrize("broken", ["reverse", "duplicate", "nan"])
def test_invalid_timestamps_abstain(broken):
    t, q = trajectory()
    t[20] = {"reverse": t[19] - 1, "duplicate": t[19], "nan": np.nan}[broken]
    eq = QualityAnalyzer().analyze_episode_arrays("bad", q, q, t)
    assert eq.overall_score is None
    assert eq.events == []
    json.dumps(eq.to_dict(), allow_nan=False)


def test_missing_state_does_not_invent_motion():
    t, q = trajectory()
    eq = QualityAnalyzer().analyze_episode_arrays("actions", actions=q, timestamps=t)
    assert eq.static is None and eq.ldlj is None
    assert "net_to_path_ratio" not in eq.diagnostics


def test_range_edges_are_not_hardware_saturation():
    t, q = trajectory()
    eq = QualityAnalyzer().analyze_episode_arrays("edges", q, q, t)
    assert eq.overall_saturation is None
    assert eq.observed_range_edge_fraction is not None
    assert "saturated" not in eq.flags
    eq = QualityAnalyzer(action_limits=((0, 1), (0, 2))).analyze_episode_arrays("limits", q, q, t)
    assert eq.overall_saturation > 0.9
    assert "saturated" in eq.flags


def test_gripper_opt_in_and_two_gripper_channels():
    t, q = trajectory()
    q[:, 0] = np.arange(len(t)) % 2
    eq = QualityAnalyzer().analyze_episode_arrays("none", q, q, t)
    assert eq.is_chattery is None and eq.grippers == {}
    eq = QualityAnalyzer(gripper_dims=(0, 1)).analyze_episode_arrays("both", q, q, t)
    assert eq.grippers["0"]["is_chattery"]
    assert not eq.grippers["1"]["is_chattery"]
    assert eq.is_chattery


def test_reversal_is_diagnostic_not_semantic_retry():
    t, q = trajectory()
    q[:, 0] = np.sin(t * 2 * np.pi)
    q[:, 1] = 2
    eq = QualityAnalyzer().analyze_episode_arrays("cycles", q, q, t)
    assert eq.diagnostics["direction_reversal_count"] >= 10
    assert eq.diagnostics["net_to_path_ratio"] < 0.01
    assert all("retry" not in e.kind for e in eq.events)


def test_roundtrip_preserves_events_status_and_static(tmp_path):
    t, q = trajectory()
    q[60:] = q[60]
    eq = QualityAnalyzer().analyze_episode_arrays("held", q, q, t)
    report = QualityReport("example", num_episodes=1, per_episode=[eq])
    path = tmp_path / "quality.json"
    report.to_json(path)
    loaded = QualityReport.from_json(path).per_episode[0]
    assert loaded.events == eq.events
    assert loaded.static == eq.static
    assert loaded.metric_status == eq.metric_status
    assert loaded.subscores == eq.subscores
    assert loaded.scorer_version == "q-v2-proprio"


def test_legacy_report_loads_without_inventing_intervals():
    report = QualityReport.from_dict(
        {"dataset_path": "old", "per_episode": [{"episode_id": "old", "static_fraction": 0.8}]}
    )
    eq = report.per_episode[0]
    assert eq.scorer_version == "q-v1" and eq.events == []
    assert eq.static.static_fraction == 0.8


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(fps=0),
        dict(tracking_lag_s=-1),
        dict(state_scales=(0,)),
        dict(action_limits=((1, 0),)),
        dict(tracking_pairs=((0, 0),)),
        dict(action_semantics="torque"),
        dict(gripper_dims=(0,), gripper_dim=1),
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        QualityConfig(**kwargs)


def test_misaligned_arrays_rejected():
    t, q = trajectory()
    with pytest.raises(ValueError, match="matching frame counts"):
        QualityAnalyzer().analyze_episode_arrays("bad", q[:-1], q, t)


def test_state_scaling_compensates_for_unit_conversion():
    t, q = trajectory()
    a = QualityAnalyzer().analyze_episode_arrays("radians", q, q, t)
    b = QualityAnalyzer(state_scales=(100, 100)).analyze_episode_arrays("scaled", q, q * 100, t)
    assert b.diagnostics["path_length_scaled"] == pytest.approx(a.diagnostics["path_length_scaled"])
    assert b.ldlj == pytest.approx(a.ldlj, abs=1e-6)
