"""Proprioception-only diagnostics. Events are evidence, not task verdicts.

All motion uses position differences and real elapsed time. Long timestamp gaps
break events; no interpolation bridges missing observations. Tracking requires
explicit absolute-position action/state mappings. Cyclic motion is not a retry.
"""

from __future__ import annotations

import numpy as np

from forge.quality.config import QualityConfig
from forge.quality.models import QualityEvent, StaticResult


def indices(values, width: int, name: str) -> list[int]:
    result = []
    for value in values:
        if not isinstance(value, (int, np.integer)) or not -width <= value < width:
            raise ValueError(f"{name} index {value} is outside {width} columns")
        result.append(int(value) % width)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate columns")
    return result


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def interval_events(mask, timestamps, kind, min_duration, channels=()):
    """A transition run [a,b) covers samples a through b inclusive."""
    out = []
    for a, b in runs(mask):
        if timestamps[b] - timestamps[a] + 1e-9 >= min_duration:
            out.append(
                QualityEvent(
                    kind,
                    int(a),
                    int(b + 1),
                    float(timestamps[a]),
                    float(timestamps[b]),
                    list(channels),
                )
            )
    return out


def diagnose(actions, states, timestamps, config: QualityConfig, fps: float):
    """Return metrics, timestamped events, and applicability descriptions."""
    dt = np.diff(timestamps)
    valid = (dt > 0) & (dt <= config.max_gap_factor / fps)
    observed_s = float(dt[valid].sum())
    metrics = {
        "valid_duration_s": observed_s,
        "time_coverage": observed_s / float(dt.sum()) if dt.sum() > 0 else 0.0,
    }
    events = interval_events(~valid, timestamps, "timestamp_gap", 0.0)
    status = {"tracking": "not_applicable: requires position semantics and explicit tracking_pairs"}
    if actions is not None:
        held = np.all(np.abs(np.diff(actions, axis=0)) <= config.dead_action_eps, axis=1) & valid
        holds = interval_events(held, timestamps, "held_command", config.event_min_duration_s)
        events.extend(holds)
        metrics["held_command_fraction"] = (
            sum(e.duration_s for e in holds) / observed_s if observed_s else None
        )
    static = None
    if states is not None:
        dims = indices(
            config.state_dims if config.state_dims is not None else range(states.shape[1]),
            states.shape[1],
            "state_dims",
        )
        scales = (
            np.ones(states.shape[1])
            if config.state_scales is None
            else np.asarray(config.state_scales)
        )
        if scales.shape != (states.shape[1],):
            raise ValueError("state_scales must have one value per original state column")
        position = states[:, dims] / scales[dims]
        delta = np.diff(position, axis=0)
        speed = np.max(np.abs(delta), axis=1) / dt
        pauses = interval_events(
            (speed <= config.motion_speed_threshold) & valid,
            timestamps,
            "motion_pause",
            config.event_min_duration_s,
            dims,
        )
        events.extend(pauses)
        fraction = sum(e.duration_s for e in pauses) / observed_s if observed_s else 0.0
        static = StaticResult(
            fraction,
            fraction > config.static_fraction_flag,
            max((e.end_frame - e.start_frame for e in pauses), default=0),
            [(e.start_frame, e.end_frame, e.duration_s) for e in pauses],
        )
        metrics["path_length_scaled"] = float(np.linalg.norm(delta[valid], axis=1).sum())
        # A global efficiency ratio is undefined across gaps and for no motion.
        length = metrics["path_length_scaled"]
        metrics["net_to_path_ratio"] = (
            float(np.clip(np.linalg.norm(position[-1] - position[0]) / length, 0, 1))
            if valid.all() and length > 1e-12
            else None
        )
        reversals = 0
        # Require sustained signs, with a deadband; reset the history at gaps.
        for j, dim in enumerate(dims):
            velocity = delta[:, j] / dt
            sign = np.where(np.abs(velocity) > config.motion_speed_threshold, np.sign(velocity), 0)
            previous = 0
            begin = 0
            while begin < len(sign):
                if not valid[begin]:
                    previous = 0
                    begin += 1
                    continue
                end = begin + 1
                while end < len(sign) and valid[end] and sign[end] == sign[begin]:
                    end += 1
                if sign[begin] and dt[begin:end].sum() + 1e-9 >= config.reversal_min_duration_s:
                    if previous and previous != sign[begin]:
                        reversals += 1
                        events.append(
                            QualityEvent(
                                "direction_reversal",
                                begin,
                                end + 1,
                                float(timestamps[begin]),
                                float(timestamps[end]),
                                [dim],
                            )
                        )
                    previous = sign[begin]
                begin = end
        metrics["direction_reversal_count"] = reversals
        status["motion"] = (
            "available: position-state assumption; pauses/reversals are not task failures"
        )
        if actions is not None and config.action_semantics == "position" and config.tracking_pairs:
            ad = indices([p[0] for p in config.tracking_pairs], actions.shape[1], "tracking action")
            sd = indices([p[1] for p in config.tracking_pairs], states.shape[1], "tracking state")
            # Last command available at the configured lag; never select a future sample.
            command_index = (
                np.searchsorted(timestamps, timestamps - config.tracking_lag_s, side="right") - 1
            )
            available = command_index >= 0
            command_index = np.maximum(command_index, 0)
            gaps = np.r_[0, np.cumsum(~valid)]
            available &= gaps == gaps[command_index]
            error = np.abs(actions[command_index][:, ad] - states[:, sd]) / scales[sd]
            residual = np.max(error, axis=1)
            if available.any():
                metrics["tracking_error_mean"] = float(np.mean(residual[available]))
                metrics["tracking_error_p95"] = float(np.percentile(residual[available], 95))
                state_speed = (
                    np.max(np.abs(np.diff(states[:, sd] / scales[sd], axis=0)), axis=1) / dt
                )
                supported = valid & available[:-1] & available[1:]
                mask = (
                    supported
                    & (state_speed <= config.motion_speed_threshold)
                    & (np.minimum(residual[:-1], residual[1:]) > config.tracking_error_threshold)
                )
                stalls = interval_events(
                    mask, timestamps, "tracking_stall_candidate", config.event_min_duration_s, sd
                )
                events.extend(stalls)
                duration = float(dt[supported].sum())
                metrics["tracking_stall_fraction"] = (
                    sum(e.duration_s for e in stalls) / duration if duration else None
                )
                status["tracking"] = (
                    "available: mapped absolute position commands; contact may also cause mismatch"
                )
            else:
                status["tracking"] = "unavailable: lag exceeds observed timeline"
    else:
        status["motion"] = "unavailable: no valid position-state data"
    return metrics, sorted(events, key=lambda e: (e.start_time_s, e.kind)), status, static
