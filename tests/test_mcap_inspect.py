"""Targeted tests for inspect.py — heuristics, channel walking, attachments.

Uses the fixture-specific facts captured in tests/fixtures/mcap/INVENTORY.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mcap")

from forge.formats.mcap import generate_config, inspect_mcap  # noqa: E402
from forge.formats.mcap.topic_config import dump_config, load_config  # noqa: E402

FIXTURES = Path(__file__).parent.parent / "sample_data" / "mcap"

if not any(FIXTURES.glob("*.mcap")):
    pytest.skip(
        "MCAP fixtures missing — run `python scripts/download_mcap_fixtures.py`",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Channel inventory facts (sourced from INVENTORY.md)
# ---------------------------------------------------------------------------


def test_inspect_trossen_protobuf_profile_empty() -> None:
    inv = inspect_mcap(FIXTURES / "trossen_transfer_cube.mcap")
    assert inv.profile == ""  # Foxglove protobuf — no ros2 profile
    assert inv.total_messages == 37158
    schemas = {c.schema_name for c in inv.channels}
    assert "schemas.proto.JointState" in schemas
    assert "foxglove.CompressedVideo" in schemas


def test_inspect_supported_ros2_messages_profile() -> None:
    inv = inspect_mcap(FIXTURES / "supported_ros2_messages.mcap")
    assert inv.profile == "ros2"
    schemas = {c.schema_name for c in inv.channels}
    assert "sensor_msgs/msg/JointState" in schemas
    assert "sensor_msgs/msg/Image" in schemas


def test_inspect_r2b_galileo_compressed_zstd() -> None:
    inv = inspect_mcap(FIXTURES / "r2b_galileo.mcap")
    assert inv.profile == "ros2"
    assert "zstd" in inv.compression
    assert inv.has_summary is True


# ---------------------------------------------------------------------------
# generate_config heuristics
# ---------------------------------------------------------------------------


def test_generate_config_trossen_picks_joint_states_as_state() -> None:
    res = generate_config(FIXTURES / "trossen_transfer_cube.mcap")
    assert not res.skipped
    cfg = res.config
    assert cfg is not None
    assert "observation.state" in cfg.fields
    primary_topic = cfg.fields["observation.state"].topic
    assert "joint_states" in primary_topic
    # Both arms present — second one becomes a sibling state field with TODO note.
    assert any("observation.state." in k for k in cfg.fields.keys())
    assert any("# TODO: pick one" in n for n in res.notes)


def test_generate_config_trossen_emits_image_fields_for_videos() -> None:
    res = generate_config(FIXTURES / "trossen_transfer_cube.mcap")
    assert not res.skipped
    cfg = res.config
    assert cfg is not None
    image_fields = {k: v for k, v in cfg.fields.items() if k.startswith("observation.images.")}
    assert len(image_fields) >= 4  # 4 wrist+external video streams in trossen
    for f in image_fields.values():
        # Foxglove CompressedVideo gets encoding="video" hint.
        if "video_compressed" in f.topic:
            assert f.encoding == "video"


def test_generate_config_trossen_surfaces_task_topic() -> None:
    res = generate_config(FIXTURES / "trossen_transfer_cube.mcap")
    assert res.config is not None
    assert res.config.task.topic == "/task_description"


def test_generate_config_supported_ros2_picks_joint_state() -> None:
    res = generate_config(FIXTURES / "supported_ros2_messages.mcap")
    assert not res.skipped
    cfg = res.config
    assert cfg is not None
    assert "observation.state" in cfg.fields
    assert cfg.fields["observation.state"].topic == "/joint_state"


def test_generate_config_r2b_galileo_skips() -> None:
    """r2b_galileo has no JointState/Pose/MultiArray — only camera + IMU + battery.
    There ARE image topics, so generate_config will produce images-only config,
    NOT skip. The sync.primary will be None because there's no observation.state."""
    res = generate_config(FIXTURES / "r2b_galileo.mcap")
    assert not res.skipped
    cfg = res.config
    assert cfg is not None
    assert "observation.state" not in cfg.fields  # no JointState in this file
    assert cfg.sync.primary is None
    image_fields = [k for k in cfg.fields if k.startswith("observation.images.")]
    assert len(image_fields) > 0


def test_generated_config_round_trips_through_yaml(tmp_path: Path) -> None:
    res = generate_config(FIXTURES / "trossen_transfer_cube.mcap")
    assert res.config is not None
    out = tmp_path / "trossen.yaml"
    dump_config(res.config, out)
    reloaded = load_config(out)
    assert set(reloaded.fields) == set(res.config.fields)
