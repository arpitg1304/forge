"""Per-dimension feature names as LeRobot info.json stores them."""

from __future__ import annotations

from typing import Any


def flatten_feature_names(names: Any) -> list[str] | None:
    """Return a feature's ``names`` from info.json as a flat list.

    LeRobot datasets store names in two forms: a flat list
    (``["shoulder_pan.pos", ...]``, what ``LeRobotDataset.create`` writes) and,
    in datasets converted from older versions, a dict of axis name to list
    (``{"motors": ["motor_0", "motor_1"]}``). A dict's lists are joined in
    order. Returns None for missing or unrecognized values.
    """
    if isinstance(names, list) and all(isinstance(n, str) for n in names):
        return list(names)
    if isinstance(names, dict) and names and all(isinstance(v, list) for v in names.values()):
        flat = [str(n) for v in names.values() for n in v]
        return flat or None
    return None


def episode_names(features: dict[str, Any]) -> dict[str, list[str] | None]:
    """``state_names``/``action_names`` keyword arguments for Episode."""
    return {
        "state_names": flatten_feature_names(features.get("observation.state", {}).get("names")),
        "action_names": flatten_feature_names(features.get("action", {}).get("names")),
    }
