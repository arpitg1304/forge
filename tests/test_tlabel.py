"""Tests for TLabel format reader and writer."""

import json
from pathlib import Path

import pytest


SAMPLE_TLABEL = {
    "schema_version": "0.17.0",
    "sensor": {"name": "GelSight Mini", "type": "vision", "adapter": "gelsight_mini"},
    "capabilities": {"contact": True, "force_magnitude": True, "slip": False},
    "episode": {
        "id": "ep001",
        "task": "pick_and_place",
        "num_frames": 3,
        "success": True,
    },
    "frames": [
        {
            "frame_idx": 0,
            "timestamp_s": 0.0,
            "schema_v2": {
                "contact": 1.0,
                "force_magnitude": 2.3,
                "contact_area": 45.2,
            },
            "confidence": 0.95,
            "manipulation_phase": "approach",
            "is_first": True,
            "is_last": False,
        },
        {
            "frame_idx": 1,
            "timestamp_s": 0.01,
            "schema_v2": {
                "contact": 1.0,
                "force_magnitude": 5.1,
                "contact_area": 62.0,
            },
            "confidence": 0.92,
            "manipulation_phase": "grasp",
            "is_first": False,
            "is_last": False,
        },
        {
            "frame_idx": 2,
            "timestamp_s": 0.02,
            "schema_v2": {
                "contact": 0.0,
                "force_magnitude": 0.0,
                "contact_area": 0.0,
            },
            "confidence": 0.88,
            "manipulation_phase": "release",
            "is_first": False,
            "is_last": True,
        },
    ],
}


@pytest.fixture
def temp_tlabel_file(tmp_path: Path) -> Path:
    """Create a temporary .tlabel file."""
    file_path = tmp_path / "test_episode.tlabel"
    with open(file_path, "w") as f:
        json.dump(SAMPLE_TLABEL, f)
    return file_path


@pytest.fixture
def temp_tlabel_json(tmp_path: Path) -> Path:
    """Create a temporary .json file with TLabel structure."""
    file_path = tmp_path / "test_episode.json"
    with open(file_path, "w") as f:
        json.dump(SAMPLE_TLABEL, f)
    return file_path


class TestTLabelReader:
    """Tests for TLabelReader."""

    def test_can_read_tlabel_file(self, temp_tlabel_file: Path):
        from forge.formats.tlabel.reader import TLabelReader

        assert TLabelReader.can_read(temp_tlabel_file) is True

    def test_can_read_json_with_tlabel_structure(self, temp_tlabel_json: Path):
        from forge.formats.tlabel.reader import TLabelReader

        assert TLabelReader.can_read(temp_tlabel_json) is True

    def test_cannot_read_non_tlabel(self, tmp_path: Path):
        from forge.formats.tlabel.reader import TLabelReader

        # Empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        assert TLabelReader.can_read(empty_dir) is False

        # Non-tlabel json
        other_json = tmp_path / "other.json"
        other_json.write_text('{"not": "tlabel"}')
        assert TLabelReader.can_read(other_json) is False

    def test_detect_version(self, temp_tlabel_file: Path):
        from forge.formats.tlabel.reader import TLabelReader

        version = TLabelReader.detect_version(temp_tlabel_file)
        assert version == "0.17.0"

    def test_inspect(self, temp_tlabel_file: Path):
        from forge.formats.tlabel.reader import TLabelReader

        reader = TLabelReader()
        info = reader.inspect(temp_tlabel_file)

        assert info.format == "tlabel"
        assert info.num_episodes == 1
        assert info.total_frames == 3
        assert info.has_language is True
        assert info.sample_language == "pick_and_place"
        assert info.inferred_fps == 100.0  # 1/0.01
        assert info.inferred_robot_type == "GelSight Mini"

    def test_read_episodes(self, temp_tlabel_file: Path):
        from forge.formats.tlabel.reader import TLabelReader

        reader = TLabelReader()
        episodes = list(reader.read_episodes(temp_tlabel_file))

        assert len(episodes) == 1
        ep = episodes[0]
        assert ep.episode_id == "ep001"
        assert ep.language_instruction == "pick_and_place"
        assert ep.success is True
        assert ep.metadata["sensor"]["name"] == "GelSight Mini"

        frames = list(ep.frames())
        assert len(frames) == 3
        assert frames[0].index == 0
        assert frames[0].timestamp == 0.0
        assert frames[0].is_first is True
        assert frames[0].extras["tlabel.schema_v2"]["contact"] == 1.0
        assert frames[0].extras["tlabel.confidence"] == 0.95

    def test_read_episode_by_id(self, temp_tlabel_file: Path):
        from forge.formats.tlabel.reader import TLabelReader

        reader = TLabelReader()
        ep = reader.read_episode(temp_tlabel_file, "ep001")
        assert ep.episode_id == "ep001"

    def test_read_episode_not_found(self, temp_tlabel_file: Path):
        from forge.core.exceptions import EpisodeNotFoundError
        from forge.formats.tlabel.reader import TLabelReader

        reader = TLabelReader()
        with pytest.raises(EpisodeNotFoundError):
            reader.read_episode(temp_tlabel_file, "nonexistent")


class TestTLabelWriter:
    """Tests for TLabelWriter."""

    def test_write_episode(self, temp_tlabel_file: Path, tmp_path: Path):
        from forge.formats.tlabel.reader import TLabelReader
        from forge.formats.tlabel.writer import TLabelWriter

        # Read
        reader = TLabelReader()
        episodes = list(reader.read_episodes(temp_tlabel_file))
        assert len(episodes) == 1

        # Write
        writer = TLabelWriter()
        out_dir = tmp_path / "output"
        writer.write_episode(episodes[0], out_dir, episode_index=0)

        # Verify output file exists
        tlabel_files = list(out_dir.glob("*.tlabel"))
        assert len(tlabel_files) == 1

        # Verify content
        with open(tlabel_files[0]) as f:
            data = json.load(f)

        assert data["schema_version"] == "0.17.0"
        assert data["episode"]["id"] == "ep001"
        assert data["episode"]["task"] == "pick_and_place"
        assert len(data["frames"]) == 3
        assert data["frames"][0]["schema_v2"]["contact"] == 1.0


class TestTLabelRoundtrip:
    """Test read-write-read roundtrip."""

    def test_roundtrip_preserves_data(self, temp_tlabel_file: Path, tmp_path: Path):
        from forge.formats.tlabel.reader import TLabelReader
        from forge.formats.tlabel.writer import TLabelWriter

        # Read original
        reader = TLabelReader()
        episodes = list(reader.read_episodes(temp_tlabel_file))

        # Write
        writer = TLabelWriter()
        out_dir = tmp_path / "roundtrip"
        writer.write_episode(episodes[0], out_dir, episode_index=0)

        # Read back
        out_files = list(out_dir.glob("*.tlabel"))
        assert len(out_files) == 1
        episodes_back = list(reader.read_episodes(out_files[0]))

        assert len(episodes_back) == 1
        ep = episodes_back[0]
        assert ep.episode_id == "ep001"
        assert ep.language_instruction == "pick_and_place"

        frames = list(ep.frames())
        assert len(frames) == 3
        assert frames[0].extras["tlabel.schema_v2"]["force_magnitude"] == 2.3
        assert frames[1].extras["tlabel.schema_v2"]["contact_area"] == 62.0
