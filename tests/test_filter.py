"""Tests for forge.filter module."""

import json
from pathlib import Path

import numpy as np
import pytest

from forge.core.models import CameraInfo, DatasetInfo, Episode, Frame, LazyImage
from forge.filter.engine import FilterConfig, FilterEngine, FilterResult
from forge.formats.registry import FormatRegistry
from forge.quality.models import EpisodeQuality, QualityReport


# ── Mock Reader/Writer ──────────────────────────────────────────


class MockFilterReader:
    """Mock reader that produces episodes with varying quality characteristics."""

    def __init__(self):
        self._num_episodes = 5

    @property
    def format_name(self) -> str:
        return "mock-filter"

    @classmethod
    def can_read(cls, path: Path) -> bool:
        return (path / ".mock_filter").exists()

    @classmethod
    def detect_version(cls, path: Path) -> str | None:
        return "1.0"

    def inspect(self, path: Path) -> DatasetInfo:
        return DatasetInfo(
            path=path,
            format="mock-filter",
            format_version="1.0",
            num_episodes=self._num_episodes,
            total_frames=self._num_episodes * 30,
            inferred_fps=30.0,
            cameras={},
        )

    def read_episodes(self, path: Path):
        for i in range(self._num_episodes):
            yield self._create_episode(i)

    def read_episode(self, path: Path, episode_id: str) -> Episode:
        idx = int(episode_id.split("_")[1])
        return self._create_episode(idx)

    def _create_episode(self, idx: int) -> Episode:
        def frame_loader():
            for j in range(30):
                yield Frame(
                    index=j,
                    timestamp=j / 30.0,
                    images={},
                    state=np.array([0.1 * idx] * 7, dtype=np.float32),
                    action=np.array([0.01 * idx] * 7, dtype=np.float32),
                )

        return Episode(
            episode_id=f"ep_{idx:03d}",
            cameras={},
            fps=30.0,
            _frame_loader=frame_loader,
        )


class MockFilterWriter:
    """Mock writer that records calls."""

    written_episodes: list[tuple[str, int]] = []
    finalized: bool = False

    def __init__(self):
        MockFilterWriter.written_episodes = []
        MockFilterWriter.finalized = False

    @property
    def format_name(self) -> str:
        return "mock-filter"

    def write_episode(self, episode: Episode, output_path: Path, episode_index: int | None = None) -> None:
        MockFilterWriter.written_episodes.append((episode.episode_id, episode_index))

    def write_dataset(self, episodes, output_path, dataset_info=None) -> None:
        pass

    def finalize(self, output_path: Path, dataset_info: DatasetInfo) -> None:
        MockFilterWriter.finalized = True


@pytest.fixture
def mock_dataset(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "mock_filter_dataset"
    dataset_path.mkdir()
    (dataset_path / ".mock_filter").touch()
    return dataset_path


@pytest.fixture
def register_mock_format():
    """Register mock reader and writer temporarily."""
    FormatRegistry._readers["mock-filter"] = MockFilterReader
    FormatRegistry._writers["mock-filter"] = MockFilterWriter
    yield
    del FormatRegistry._readers["mock-filter"]
    del FormatRegistry._writers["mock-filter"]


@pytest.fixture
def sample_report(tmp_path: Path) -> Path:
    """Create a sample quality report JSON."""
    report = QualityReport(dataset_path="./test")
    for i in range(5):
        # Vary scores: ep_000=9.0, ep_001=7.5, ep_002=5.0, ep_003=3.5, ep_004=8.0
        scores = [9.0, 7.5, 5.0, 3.5, 8.0]
        flags_map = [
            [],
            [],
            ["jerky"],
            ["jerky", "mostly_static"],
            ["saturated"],
        ]
        eq = EpisodeQuality(
            episode_id=f"ep_{i:03d}",
            num_frames=30,
            overall_score=scores[i],
            flags=flags_map[i],
        )
        report.per_episode.append(eq)

    report.num_episodes = 5
    report.overall_score = sum(scores) / len(scores)

    path = tmp_path / "report.json"
    report.to_json(path)
    return path


# ── QualityReport from_json/from_dict ───────────────────────────


class TestQualityReportDeserialization:
    def test_roundtrip(self, tmp_path):
        """to_json then from_json produces equivalent report."""
        original = QualityReport(
            dataset_path="./test",
            num_episodes=2,
            overall_score=7.5,
        )
        original.flags = ["2 episodes with jerky actions"]
        original.flagged_episodes = {"jerky": ["ep_000", "ep_001"]}
        original.per_episode = [
            EpisodeQuality(
                episode_id="ep_000",
                num_frames=100,
                overall_score=6.0,
                flags=["jerky"],
                dead_fraction=0.05,
            ),
            EpisodeQuality(
                episode_id="ep_001",
                num_frames=150,
                overall_score=9.0,
                flags=[],
            ),
        ]

        path = tmp_path / "roundtrip.json"
        original.to_json(path)
        loaded = QualityReport.from_json(path)

        assert loaded.dataset_path == "./test"
        assert loaded.num_episodes == 2
        assert loaded.overall_score == 7.5
        assert len(loaded.per_episode) == 2
        assert loaded.per_episode[0].episode_id == "ep_000"
        assert loaded.per_episode[0].overall_score == 6.0
        assert loaded.per_episode[0].flags == ["jerky"]
        assert loaded.per_episode[0].dead_fraction == 0.05
        assert loaded.per_episode[1].overall_score == 9.0
        assert loaded.flagged_episodes == {"jerky": ["ep_000", "ep_001"]}

    def test_from_dict(self):
        data = {
            "dataset_path": "./data",
            "num_episodes": 1,
            "overall_score": 8.0,
            "computed_at": "2025-01-01T00:00:00",
            "per_episode": [
                {
                    "episode_id": "ep_0",
                    "num_frames": 50,
                    "overall_score": 8.0,
                    "flags": [],
                }
            ],
            "flags": [],
            "flagged_episodes": {},
            "recommendations": [],
            "subscores": {"smoothness": 0.9},
        }
        report = QualityReport.from_dict(data)
        assert report.dataset_path == "./data"
        assert report.num_episodes == 1
        assert report.subscores == {"smoothness": 0.9}
        assert len(report.per_episode) == 1
        assert report.per_episode[0].episode_id == "ep_0"


# ── FilterConfig ────────────────────────────────────────────────


class TestFilterConfig:
    def test_defaults(self):
        config = FilterConfig()
        assert config.min_quality is None
        assert config.exclude_flags is None
        assert config.include_episodes is None
        assert config.exclude_episodes is None
        assert config.from_report is None
        assert config.gripper_dim is None
        assert config.fps == 30.0

    def test_with_values(self):
        config = FilterConfig(
            min_quality=6.0,
            exclude_flags=["jerky", "mostly_static"],
            fps=50.0,
        )
        assert config.min_quality == 6.0
        assert config.exclude_flags == ["jerky", "mostly_static"]
        assert config.fps == 50.0


# ── FilterResult ────────────────────────────────────────────────


class TestFilterResult:
    def test_dry_run_result(self):
        result = FilterResult(
            success=True,
            format="lerobot-v3",
            total_episodes=10,
            episodes_kept=7,
            episodes_excluded=3,
            dry_run=True,
            kept_ids=["ep_0", "ep_1"],
            excluded_ids=["ep_2"],
            exclusion_reasons={"ep_2": ["score 4.0 < min 6.0"]},
        )
        assert result.dry_run
        assert result.output_path is None


# ── FilterEngine ────────────────────────────────────────────────


class TestFilterEngine:
    def test_dry_run_no_filters(self, mock_dataset, register_mock_format):
        """All episodes kept when no filters set."""
        config = FilterConfig()
        engine = FilterEngine(config)
        result = engine.filter(mock_dataset)

        assert result.dry_run
        assert result.success
        assert result.episodes_kept == 5
        assert result.episodes_excluded == 0
        assert len(result.kept_ids) == 5

    def test_dry_run_min_quality_from_report(
        self, mock_dataset, register_mock_format, sample_report
    ):
        """Episodes below min_quality threshold are excluded."""
        config = FilterConfig(min_quality=6.0, from_report=sample_report)
        engine = FilterEngine(config)
        result = engine.filter(mock_dataset)

        assert result.dry_run
        assert result.success
        # ep_000=9.0, ep_001=7.5, ep_002=5.0, ep_003=3.5, ep_004=8.0
        # Below 6.0: ep_002 (5.0) and ep_003 (3.5)
        assert result.episodes_kept == 3
        assert result.episodes_excluded == 2
        assert "ep_002" in result.excluded_ids
        assert "ep_003" in result.excluded_ids
        assert "score 5.0 < min 6.0" in result.exclusion_reasons["ep_002"]
        assert "score 3.5 < min 6.0" in result.exclusion_reasons["ep_003"]

    def test_dry_run_exclude_flags(
        self, mock_dataset, register_mock_format, sample_report
    ):
        """Episodes with matching flags are excluded."""
        config = FilterConfig(
            exclude_flags=["mostly_static"], from_report=sample_report
        )
        engine = FilterEngine(config)
        result = engine.filter(mock_dataset)

        assert result.success
        # Only ep_003 has "mostly_static"
        assert result.episodes_excluded == 1
        assert "ep_003" in result.excluded_ids
        assert "flag: mostly_static" in result.exclusion_reasons["ep_003"]

    def test_dry_run_exclude_multiple_flags(
        self, mock_dataset, register_mock_format, sample_report
    ):
        """Excluding multiple flags catches all matching episodes."""
        config = FilterConfig(
            exclude_flags=["jerky", "saturated"], from_report=sample_report
        )
        engine = FilterEngine(config)
        result = engine.filter(mock_dataset)

        # ep_002 has "jerky", ep_003 has "jerky" + "mostly_static", ep_004 has "saturated"
        assert result.episodes_excluded == 3
        assert "ep_002" in result.excluded_ids
        assert "ep_003" in result.excluded_ids
        assert "ep_004" in result.excluded_ids

    def test_include_episodes(self, mock_dataset, register_mock_format):
        """Only specified episodes are kept."""
        config = FilterConfig(include_episodes=["ep_001", "ep_003"])
        engine = FilterEngine(config)
        result = engine.filter(mock_dataset)

        assert result.episodes_kept == 2
        assert result.episodes_excluded == 3
        assert set(result.kept_ids) == {"ep_001", "ep_003"}

    def test_exclude_episodes(self, mock_dataset, register_mock_format):
        """Specified episodes are excluded."""
        config = FilterConfig(exclude_episodes=["ep_000", "ep_004"])
        engine = FilterEngine(config)
        result = engine.filter(mock_dataset)

        assert result.episodes_kept == 3
        assert result.episodes_excluded == 2
        assert "ep_000" in result.excluded_ids
        assert "ep_004" in result.excluded_ids

    def test_combined_filters(
        self, mock_dataset, register_mock_format, sample_report
    ):
        """min_quality AND exclude_flags together."""
        config = FilterConfig(
            min_quality=6.0,
            exclude_flags=["saturated"],
            from_report=sample_report,
        )
        engine = FilterEngine(config)
        result = engine.filter(mock_dataset)

        # ep_002 (5.0, jerky) → excluded by score
        # ep_003 (3.5, jerky+mostly_static) → excluded by score
        # ep_004 (8.0, saturated) → excluded by flag
        assert result.episodes_excluded == 3
        assert result.episodes_kept == 2
        assert set(result.kept_ids) == {"ep_000", "ep_001"}

    def test_write_mode(self, mock_dataset, register_mock_format, sample_report, tmp_path):
        """Writer is called for kept episodes with correct indices."""
        output = tmp_path / "filtered"
        output.mkdir()

        config = FilterConfig(min_quality=6.0, from_report=sample_report)
        engine = FilterEngine(config)
        result = engine.filter(mock_dataset, output=output)

        assert not result.dry_run
        assert result.success
        assert result.episodes_kept == 3
        assert result.output_path == output

        # Check writer was called with sequential indices
        assert len(MockFilterWriter.written_episodes) == 3
        indices = [idx for _, idx in MockFilterWriter.written_episodes]
        assert indices == [0, 1, 2]

        # Check finalize was called
        assert MockFilterWriter.finalized

    def test_from_report_missing_episodes(
        self, mock_dataset, register_mock_format, tmp_path
    ):
        """Report has fewer episodes than dataset — unmatched episodes pass quality filters."""
        report = QualityReport(dataset_path="./test")
        # Only include 2 of 5 episodes in report
        report.per_episode = [
            EpisodeQuality(episode_id="ep_000", overall_score=4.0, flags=["jerky"]),
            EpisodeQuality(episode_id="ep_001", overall_score=8.0, flags=[]),
        ]
        report.num_episodes = 2
        report_path = tmp_path / "partial_report.json"
        report.to_json(report_path)

        config = FilterConfig(min_quality=6.0, from_report=report_path)
        engine = FilterEngine(config)
        result = engine.filter(mock_dataset)

        # ep_000 (4.0) excluded by score
        # ep_001 (8.0) kept
        # ep_002, ep_003, ep_004 not in report → no quality data → kept (conservative)
        assert result.episodes_kept == 4
        assert result.episodes_excluded == 1
        assert "ep_000" in result.excluded_ids

    def test_all_excluded(self, mock_dataset, register_mock_format, tmp_path):
        """All episodes excluded — result is successful but nothing written."""
        report = QualityReport(dataset_path="./test")
        for i in range(5):
            report.per_episode.append(
                EpisodeQuality(episode_id=f"ep_{i:03d}", overall_score=2.0, flags=[])
            )
        report.num_episodes = 5
        report_path = tmp_path / "all_bad.json"
        report.to_json(report_path)

        output = tmp_path / "filtered_empty"
        output.mkdir()

        config = FilterConfig(min_quality=6.0, from_report=report_path)
        engine = FilterEngine(config)
        result = engine.filter(mock_dataset, output=output)

        assert result.success
        assert result.episodes_kept == 0
        assert result.episodes_excluded == 5
        assert len(MockFilterWriter.written_episodes) == 0

    def test_empty_dataset(self, tmp_path, register_mock_format):
        """Empty dataset returns success with zero counts."""
        # Create an empty mock dataset that reader can detect
        dataset = tmp_path / "empty_dataset"
        dataset.mkdir()
        (dataset / ".mock_filter").touch()

        # Override reader to produce 0 episodes
        original_read = MockFilterReader.read_episodes

        def empty_read(self, path):
            return iter([])

        MockFilterReader.read_episodes = empty_read

        try:
            config = FilterConfig(min_quality=6.0)
            engine = FilterEngine(config)
            result = engine.filter(dataset)

            assert result.success
            assert result.episodes_kept == 0
            assert result.episodes_excluded == 0
        finally:
            MockFilterReader.read_episodes = original_read


class TestFilterEngineEvaluate:
    """Test the _evaluate_episode method directly."""

    def test_no_filters_keeps_all(self):
        engine = FilterEngine(FilterConfig())
        keep, reasons = engine._evaluate_episode("ep_000", None)
        assert keep
        assert reasons == []

    def test_min_quality_pass(self):
        engine = FilterEngine(FilterConfig(min_quality=6.0))
        eq = EpisodeQuality(episode_id="ep_000", overall_score=8.0, flags=[])
        keep, reasons = engine._evaluate_episode("ep_000", eq)
        assert keep

    def test_min_quality_fail(self):
        engine = FilterEngine(FilterConfig(min_quality=6.0))
        eq = EpisodeQuality(episode_id="ep_000", overall_score=4.0, flags=[])
        keep, reasons = engine._evaluate_episode("ep_000", eq)
        assert not keep
        assert "score 4.0 < min 6.0" in reasons

    def test_exclude_flags_pass(self):
        engine = FilterEngine(FilterConfig(exclude_flags=["jerky"]))
        eq = EpisodeQuality(episode_id="ep_000", overall_score=8.0, flags=["saturated"])
        keep, reasons = engine._evaluate_episode("ep_000", eq)
        assert keep

    def test_exclude_flags_fail(self):
        engine = FilterEngine(FilterConfig(exclude_flags=["jerky"]))
        eq = EpisodeQuality(episode_id="ep_000", overall_score=8.0, flags=["jerky"])
        keep, reasons = engine._evaluate_episode("ep_000", eq)
        assert not keep
        assert "flag: jerky" in reasons

    def test_include_list_pass(self):
        engine = FilterEngine(FilterConfig(include_episodes=["ep_000", "ep_001"]))
        keep, reasons = engine._evaluate_episode("ep_000", None)
        assert keep

    def test_include_list_fail(self):
        engine = FilterEngine(FilterConfig(include_episodes=["ep_000", "ep_001"]))
        keep, reasons = engine._evaluate_episode("ep_999", None)
        assert not keep
        assert "not in include list" in reasons

    def test_exclude_list_pass(self):
        engine = FilterEngine(FilterConfig(exclude_episodes=["ep_999"]))
        keep, reasons = engine._evaluate_episode("ep_000", None)
        assert keep

    def test_exclude_list_fail(self):
        engine = FilterEngine(FilterConfig(exclude_episodes=["ep_000"]))
        keep, reasons = engine._evaluate_episode("ep_000", None)
        assert not keep
        assert "in exclude list" in reasons

    def test_no_quality_data_with_quality_filters(self):
        """Quality filters active but no quality data → keep (conservative)."""
        engine = FilterEngine(FilterConfig(min_quality=6.0))
        keep, reasons = engine._evaluate_episode("ep_000", None)
        assert keep

    def test_multiple_reasons(self):
        """Episode fails multiple criteria → all reasons listed."""
        engine = FilterEngine(FilterConfig(min_quality=6.0, exclude_flags=["jerky"]))
        eq = EpisodeQuality(
            episode_id="ep_000", overall_score=4.0, flags=["jerky", "mostly_static"]
        )
        keep, reasons = engine._evaluate_episode("ep_000", eq)
        assert not keep
        assert "score 4.0 < min 6.0" in reasons
        assert "flag: jerky" in reasons
