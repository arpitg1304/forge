"""Tests for forge.lint — dataset hygiene linter."""

from __future__ import annotations

from pathlib import Path

from forge.core.models import CameraInfo, DatasetInfo, Dtype, Episode, FieldSchema
from forge.lint import DatasetLinter, LintConfig, Severity


def _episode(
    episode_id: str = "episode_000000",
    task: str | None = "Pick up the red cube and place it in the bin",
    num_frames: int = 100,
    cameras: dict | None = None,
    action_dim: int = 7,
    state_dim: int = 7,
) -> Episode:
    if cameras is None:
        cameras = {
            "observation.images.top": CameraInfo("top", 480, 640),
            "observation.images.wrist": CameraInfo("wrist", 480, 640),
        }
    return Episode(
        episode_id=episode_id,
        language_instruction=task,
        cameras=cameras,
        action_dim=action_dim,
        state_dim=state_dim,
        metadata={"num_frames": num_frames},
    )


def _codes(report) -> set[str]:
    return {i.code for i in report.issues}


class TestCleanDataset:
    def test_well_formed_dataset_passes(self):
        eps = [_episode(episode_id=f"episode_{i:06d}") for i in range(5)]
        report = DatasetLinter().lint_episodes(eps)
        assert report.passed
        assert report.num_episodes == 5
        assert len(report.errors) == 0


class TestTaskChecks:
    def test_missing_task_warns(self):
        report = DatasetLinter().lint_episodes([_episode(task=None)])
        assert "task.missing" in _codes(report)
        assert report.passed  # warning, not error

    def test_empty_task_warns(self):
        report = DatasetLinter().lint_episodes([_episode(task="   ")])
        assert "task.missing" in _codes(report)

    def test_generic_task_flagged(self):
        for junk in ("task desc", "Hold", "up", "N/A"):
            report = DatasetLinter().lint_episodes([_episode(task=junk)])
            assert "task.generic" in _codes(report), junk

    def test_too_short_task_warns(self):
        report = DatasetLinter().lint_episodes([_episode(task="grab it")])
        assert "task.too_short" in _codes(report)

    def test_too_long_task_is_info(self):
        long = "Carefully pick up the small red cube from the left side of the table and " \
               "gently place it into the blue bin on the right"
        report = DatasetLinter().lint_episodes([_episode(task=long)])
        assert "task.too_long" in _codes(report)
        assert report.passed


class TestEpisodeLength:
    def test_single_frame_episode_errors(self):
        report = DatasetLinter().lint_episodes([_episode(num_frames=1)])
        assert "episode.too_short" in _codes(report)
        assert not report.passed  # error → fail

    def test_two_frame_episode_errors(self):
        report = DatasetLinter().lint_episodes([_episode(num_frames=2)])
        assert not report.passed

    def test_normal_length_ok(self):
        report = DatasetLinter().lint_episodes([_episode(num_frames=50)])
        assert "episode.too_short" not in _codes(report)


class TestCameraChecks:
    def test_ambiguous_camera_name_warns(self):
        cams = {
            "observation.images.laptop": CameraInfo("laptop", 480, 640),
            "observation.images.wrist": CameraInfo("wrist", 480, 640),
        }
        report = DatasetLinter().lint_episodes([_episode(cameras=cams)])
        assert "camera.ambiguous_name" in _codes(report)

    def test_low_resolution_is_info(self):
        cams = {
            "observation.images.top": CameraInfo("top", 96, 96),
            "observation.images.wrist": CameraInfo("wrist", 96, 96),
        }
        report = DatasetLinter().lint_episodes([_episode(cameras=cams)])
        assert "camera.low_resolution" in _codes(report)

    def test_single_view_is_info(self):
        cams = {"observation.images.top": CameraInfo("top", 480, 640)}
        report = DatasetLinter().lint_episodes([_episode(cameras=cams)])
        assert "camera.too_few_views" in _codes(report)

    def test_no_cameras_warns(self):
        report = DatasetLinter().lint_episodes([_episode(cameras={})])
        assert "camera.none" in _codes(report)


class TestDatasetWideChecks:
    def test_inconsistent_action_dim_errors(self):
        eps = [
            _episode(episode_id="episode_000000", action_dim=7),
            _episode(episode_id="episode_000001", action_dim=14),
        ]
        report = DatasetLinter().lint_episodes(eps)
        assert "action.inconsistent_dim" in _codes(report)
        assert not report.passed

    def test_inconsistent_state_dim_errors(self):
        eps = [
            _episode(episode_id="episode_000000", state_dim=7),
            _episode(episode_id="episode_000001", state_dim=8),
        ]
        report = DatasetLinter().lint_episodes(eps)
        assert "state.inconsistent_dim" in _codes(report)

    def test_inconsistent_camera_keys_warns(self):
        eps = [
            _episode(
                episode_id="episode_000000",
                cameras={
                    "observation.images.top": CameraInfo("top", 480, 640),
                    "observation.images.wrist": CameraInfo("wrist", 480, 640),
                },
            ),
            _episode(
                episode_id="episode_000001",
                cameras={
                    "observation.images.front": CameraInfo("front", 480, 640),
                    "observation.images.wrist": CameraInfo("wrist", 480, 640),
                },
            ),
        ]
        report = DatasetLinter().lint_episodes(eps)
        assert "camera.inconsistent_keys" in _codes(report)

    def test_empty_dataset_errors(self):
        report = DatasetLinter().lint_episodes([])
        assert "dataset.empty" in _codes(report)
        assert not report.passed


class TestReportSerialization:
    def test_to_dict_roundtrip(self):
        report = DatasetLinter().lint_episodes([_episode(num_frames=1)])
        d = report.to_dict()
        assert d["passed"] is False
        assert d["summary"]["errors"] >= 1
        assert "episode.too_short" in d["counts_by_code"]
        assert isinstance(d["issues"], list)

    def test_severity_partitions(self):
        report = DatasetLinter().lint_episodes(
            [_episode(task=None, num_frames=1)]
        )
        # missing task = warn, 1-frame = error
        assert any(i.severity is Severity.ERROR for i in report.issues)
        assert any(i.severity is Severity.WARN for i in report.issues)


class TestConfigOverride:
    def test_custom_min_frames(self):
        cfg = LintConfig(min_episode_frames=10)
        report = DatasetLinter(config=cfg).lint_episodes([_episode(num_frames=5)])
        assert "episode.too_short" in _codes(report)


def _info(
    num_episodes: int = 100,
    cameras: dict | None = None,
    has_language: bool = True,
    language_coverage: float = 1.0,
    sample_language: str | None = "Pick up the red cube and place it in the bin",
    action_schema: FieldSchema | None = "default",
) -> DatasetInfo:
    if cameras is None:
        cameras = {
            "observation.images.top": CameraInfo("top", 480, 640),
            "observation.images.wrist": CameraInfo("wrist", 480, 640),
        }
    if action_schema == "default":
        action_schema = FieldSchema(name="action", shape=(7,), dtype=Dtype.FLOAT32)
    return DatasetInfo(
        path=Path("/tmp/fake"),
        format="lerobot-v3",
        num_episodes=num_episodes,
        cameras=cameras,
        has_language=has_language,
        language_coverage=language_coverage,
        sample_language=sample_language,
        action_schema=action_schema,
    )


class TestMetadataPath:
    """Tests for lint_metadata — the engine the CLI actually uses."""

    def test_clean_metadata_passes(self):
        report = DatasetLinter().lint_metadata(_info())
        assert report.passed
        assert report.num_episodes == 100
        assert len(report.issues) == 0

    def test_no_language_warns(self):
        report = DatasetLinter().lint_metadata(_info(has_language=False))
        assert "task.missing" in _codes(report)
        assert report.passed  # warn, not error

    def test_partial_language_coverage_warns(self):
        report = DatasetLinter().lint_metadata(
            _info(language_coverage=0.6)
        )
        assert "task.partial_coverage" in _codes(report)

    def test_generic_sample_task_warns(self):
        report = DatasetLinter().lint_metadata(_info(sample_language="task desc"))
        assert "task.generic" in _codes(report)

    def test_ambiguous_single_lowres_camera(self):
        # Mirrors the real pusht case: one camera named 'image' at 96x96.
        cams = {"observation.image": CameraInfo("image", 96, 96)}
        report = DatasetLinter().lint_metadata(_info(cameras=cams))
        codes = _codes(report)
        assert "camera.ambiguous_name" in codes
        assert "camera.low_resolution" in codes
        assert "camera.too_few_views" in codes

    def test_no_cameras_warns(self):
        report = DatasetLinter().lint_metadata(_info(cameras={}))
        assert "camera.none" in _codes(report)

    def test_missing_action_schema_warns(self):
        report = DatasetLinter().lint_metadata(_info(action_schema=None))
        assert "action.missing" in _codes(report)

    def test_empty_dataset_errors(self):
        report = DatasetLinter().lint_metadata(_info(num_episodes=0))
        assert "dataset.empty" in _codes(report)
        assert not report.passed
