"""Tests for the CLI."""

import json
from pathlib import Path

from typer.testing import CliRunner

from forge.cli import app

runner = CliRunner()


def _make_hf_cache(root: Path, repo_id: str, sha: str, files: dict[str, str]) -> Path:
    """Build a minimal HF Hub cache layout under `root` for one repo."""
    repo_folder = "datasets--" + repo_id.replace("/", "--")
    repo_dir = root / repo_folder
    snapshot_dir = repo_dir / "snapshots" / sha
    snapshot_dir.mkdir(parents=True)
    refs_dir = repo_dir / "refs"
    refs_dir.mkdir(parents=True)
    (refs_dir / "main").write_text(sha)
    for rel_path, content in files.items():
        target = snapshot_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return snapshot_dir


class TestCLI:
    """Tests for CLI commands."""

    def test_version_command(self):
        """Test the version command."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.stdout

    def test_formats_command(self):
        """Test the formats command."""
        result = runner.invoke(app, ["formats"])
        assert result.exit_code == 0
        assert "zarr" in result.stdout
        assert "lerobot-v3" in result.stdout
        assert "rosbag" in result.stdout

    def test_inspect_zarr(self, temp_zarr_dataset: Path):
        """Test inspecting a Zarr dataset via CLI."""
        result = runner.invoke(app, ["inspect", str(temp_zarr_dataset)])
        assert result.exit_code == 0
        assert "zarr" in result.stdout.lower()
        assert "Episodes:" in result.stdout or "episodes" in result.stdout.lower()

    def test_inspect_with_format_flag(self, temp_zarr_dataset: Path):
        """Test inspect with --format flag."""
        result = runner.invoke(app, ["inspect", str(temp_zarr_dataset), "--format", "zarr"])
        assert result.exit_code == 0

    def test_inspect_nonexistent_path(self):
        """Test inspect with non-existent path."""
        result = runner.invoke(app, ["inspect", "/nonexistent/path"])
        assert result.exit_code != 0

    def test_help_command(self):
        """Test help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "inspect" in result.stdout
        assert "formats" in result.stdout
        assert "version" in result.stdout


class TestLocalCommand:
    """Tests for `forge local`."""

    def test_lists_populated_datasets_from_custom_path(self, tmp_path):
        _make_hf_cache(
            tmp_path, "lerobot/foo", "a" * 40,
            {"data/file.parquet": "x", "videos/cam/v.mp4": "y"},
        )
        _make_hf_cache(
            tmp_path, "lerobot/stub", "b" * 40, {"meta/info.json": "{}"},
        )
        result = runner.invoke(
            app, ["local", "--path", str(tmp_path), "--no-detect"]
        )
        assert result.exit_code == 0, result.stdout
        assert "lerobot/foo" in result.stdout
        # Stubs hidden by default.
        assert "lerobot/stub" not in result.stdout

    def test_all_flag_includes_stubs(self, tmp_path):
        _make_hf_cache(
            tmp_path, "lerobot/foo", "a" * 40, {"data/file.parquet": "x"},
        )
        _make_hf_cache(
            tmp_path, "lerobot/stub", "b" * 40, {"meta/info.json": "{}"},
        )
        result = runner.invoke(
            app, ["local", "--path", str(tmp_path), "--all", "--no-detect"]
        )
        assert result.exit_code == 0, result.stdout
        assert "lerobot/foo" in result.stdout
        assert "lerobot/stub" in result.stdout

    def test_json_output(self, tmp_path):
        _make_hf_cache(
            tmp_path, "lerobot/foo", "a" * 40, {"data/file.parquet": "x"},
        )
        result = runner.invoke(
            app, ["local", "--path", str(tmp_path), "--no-detect", "-o", "json"]
        )
        assert result.exit_code == 0, result.stdout
        data = json.loads(result.stdout)
        assert data["count"] == 1
        assert data["datasets"][0]["repo_id"] == "lerobot/foo"
        assert data["datasets"][0]["is_stub"] is False
        assert data["datasets"][0]["file_counts"] == {".parquet": 1}

    def test_missing_path_prints_warning(self, tmp_path):
        result = runner.invoke(
            app, ["local", "--path", str(tmp_path / "nope")]
        )
        assert result.exit_code == 0
        assert "does not exist" in result.stdout

    def test_empty_cache_message(self, tmp_path):
        result = runner.invoke(app, ["local", "--path", str(tmp_path)])
        assert result.exit_code == 0
        assert "No populated datasets" in result.stdout

    def test_resolves_bare_repo_id_via_lerobot_cache(self, tmp_path, monkeypatch):
        # Build a fake LeRobot cache with one dataset and point the env var
        # at it. `forge inspect arpit/foo` should resolve via this cache
        # instead of going to the network.
        ds = tmp_path / "arpit" / "foo"
        (ds / "meta").mkdir(parents=True)
        (ds / "data" / "chunk-000").mkdir(parents=True)
        (ds / "videos" / "chunk-000" / "cam").mkdir(parents=True)
        (ds / "meta" / "info.json").write_text(
            '{"codebase_version":"v2.1","total_episodes":1,"total_frames":1,'
            '"fps":30,"robot_type":"so101","features":{},"chunks_size":1000,'
            '"data_path":"data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",'
            '"video_path":"videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"}'
        )
        (ds / "data" / "chunk-000" / "episode_000000.parquet").write_text("x")
        (ds / "videos" / "chunk-000" / "cam" / "episode_000000.mp4").write_text("y")

        monkeypatch.setenv("LEROBOT_HOME", str(tmp_path))
        # Also force HF_HUB_CACHE to a clean tmp dir so we don't accidentally
        # hit a real cache on the developer's machine.
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "_hub"))

        result = runner.invoke(app, ["inspect", "arpit/foo"])
        # We don't fully exercise the inspector here (depends on more
        # complete metadata), just confirm cache resolution fired and we
        # never tried to download.
        assert "Using cached LeRobot dataset: arpit/foo" in result.stdout
        assert "Downloading from HuggingFace Hub" not in result.stdout

    def test_lerobot_layout(self, tmp_path):
        # Mimic ~/.cache/huggingface/lerobot/<org>/<dataset>/
        org_dir = tmp_path / "myorg"
        ds = org_dir / "stack_lego"
        (ds / "meta").mkdir(parents=True)
        (ds / "data" / "chunk-000").mkdir(parents=True)
        (ds / "meta" / "info.json").write_text('{"codebase_version":"v2.1"}')
        (ds / "data" / "chunk-000" / "episode_000000.parquet").write_text("x")

        result = runner.invoke(
            app, ["local", "--path", str(org_dir), "--no-detect"]
        )
        assert result.exit_code == 0, result.stdout
        assert "myorg/stack_lego" in result.stdout


class TestRegistryIdToHfUrl:
    """Registry ids should be rewritten to hf:// so --quick / size-guard apply."""

    def test_hf_backed_id_resolves_to_hf_url(self):
        from forge.cli import _registry_id_to_hf_url

        # metaworld's source is hf_hub: lerobot/metaworld_mt50
        assert _registry_id_to_hf_url("metaworld") == "hf://lerobot/metaworld_mt50"

    def test_prefers_hf_over_other_sources(self):
        from forge.cli import _registry_id_to_hf_url

        # droid has both a gcs and an hf_hub source; get_source prefers hf_hub.
        assert _registry_id_to_hf_url("droid") == "hf://cadene/droid"

    def test_unknown_id_returns_none(self):
        from forge.cli import _registry_id_to_hf_url

        assert _registry_id_to_hf_url("definitely_not_a_dataset") is None

    def test_repo_id_and_paths_are_ignored(self):
        from forge.cli import _registry_id_to_hf_url

        assert _registry_id_to_hf_url("lerobot/pusht") is None
        assert _registry_id_to_hf_url("./local_dir") is None

    def test_local_dir_shadows_registry_id(self, tmp_path, monkeypatch):
        from forge.cli import _registry_id_to_hf_url

        # A local directory named like a registry id must not be rewritten.
        monkeypatch.chdir(tmp_path)
        (tmp_path / "metaworld").mkdir()
        assert _registry_id_to_hf_url("metaworld") is None
