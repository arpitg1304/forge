"""Tests for HuggingFace Hub integration."""

from pathlib import Path

import pytest

from forge.hub.download import (
    find_in_hf_cache,
    find_in_lerobot_cache,
    get_hf_cache_dir,
    get_lerobot_cache_dir,
    list_hf_cache_datasets,
    list_local_datasets,
)
from forge.hub.url import HFDatasetRef, is_hf_url, parse_hf_url


class TestIsHfUrl:
    """Test is_hf_url function."""

    def test_hf_scheme(self):
        assert is_hf_url("hf://lerobot/pusht") is True
        assert is_hf_url("hf://openvla/modified_libero_rlds") is True

    def test_huggingface_scheme(self):
        assert is_hf_url("huggingface://lerobot/pusht") is True

    def test_https_url(self):
        assert is_hf_url("https://huggingface.co/datasets/lerobot/pusht") is True

    def test_local_path(self):
        assert is_hf_url("/path/to/dataset") is False
        assert is_hf_url("./dataset") is False
        assert is_hf_url("dataset") is False

    def test_non_string(self):
        assert is_hf_url(None) is False  # type: ignore
        assert is_hf_url(123) is False  # type: ignore


class TestParseHfUrl:
    """Test parse_hf_url function."""

    def test_hf_scheme_basic(self):
        ref = parse_hf_url("hf://lerobot/pusht")
        assert ref.repo_id == "lerobot/pusht"
        assert ref.revision is None
        assert ref.subset is None

    def test_hf_scheme_with_revision(self):
        ref = parse_hf_url("hf://lerobot/pusht@main")
        assert ref.repo_id == "lerobot/pusht"
        assert ref.revision == "main"
        assert ref.subset is None

    def test_hf_scheme_with_commit(self):
        ref = parse_hf_url("hf://lerobot/pusht@abc123")
        assert ref.repo_id == "lerobot/pusht"
        assert ref.revision == "abc123"

    def test_huggingface_scheme(self):
        ref = parse_hf_url("huggingface://openvla/modified_libero_rlds")
        assert ref.repo_id == "openvla/modified_libero_rlds"

    def test_https_url(self):
        ref = parse_hf_url("https://huggingface.co/datasets/lerobot/pusht")
        assert ref.repo_id == "lerobot/pusht"

    def test_https_url_with_trailing_slash(self):
        ref = parse_hf_url("https://huggingface.co/datasets/lerobot/pusht/")
        assert ref.repo_id == "lerobot/pusht"

    def test_invalid_scheme(self):
        with pytest.raises(ValueError, match="Invalid HuggingFace URL scheme"):
            parse_hf_url("http://example.com/dataset")

    def test_invalid_repo_id_format(self):
        with pytest.raises(ValueError, match="Invalid repo_id format"):
            parse_hf_url("hf://just_one_part")


class TestHFDatasetRef:
    """Test HFDatasetRef dataclass."""

    def test_basic_ref(self):
        ref = HFDatasetRef(repo_id="lerobot/pusht")
        assert ref.repo_id == "lerobot/pusht"
        assert ref.revision is None
        assert ref.subset is None

    def test_ref_with_all_fields(self):
        ref = HFDatasetRef(
            repo_id="org/dataset",
            revision="v1.0",
            subset="train",
        )
        assert ref.repo_id == "org/dataset"
        assert ref.revision == "v1.0"
        assert ref.subset == "train"


def _make_hf_cache(root: Path, repo_id: str, sha: str, files: dict[str, str]) -> Path:
    """Build a minimal HF Hub cache layout under `root` for a single repo.

    `files` maps relative paths inside the snapshot to file contents (any
    string is fine — only the file's existence/extension matters here).
    Returns the path of the synthesised snapshot directory.
    """
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


class TestFindInHfCache:
    """Test find_in_hf_cache — local HF cache auto-discovery."""

    def test_finds_populated_snapshot_via_refs_main(self, tmp_path):
        snap = _make_hf_cache(
            tmp_path,
            "lerobot/foo",
            "deadbeef" * 5,
            {
                "meta/info.json": "{}",
                "data/chunk-000/file-000.parquet": "x",
                "videos/cam/chunk-000/file-000.mp4": "x",
            },
        )
        found = find_in_hf_cache("lerobot/foo", cache_dir=tmp_path)
        assert found is not None
        assert found.resolve() == snap.resolve()

    def test_returns_none_for_metadata_only_stub(self, tmp_path):
        _make_hf_cache(
            tmp_path,
            "lerobot/stub",
            "0" * 40,
            {"meta/info.json": "{}", "README.md": "x"},
        )
        assert find_in_hf_cache("lerobot/stub", cache_dir=tmp_path) is None

    def test_returns_none_when_repo_absent(self, tmp_path):
        assert find_in_hf_cache("lerobot/nope", cache_dir=tmp_path) is None

    def test_revision_resolves_via_refs(self, tmp_path):
        snap = _make_hf_cache(
            tmp_path,
            "lerobot/foo",
            "abc123",
            {"data/file.parquet": "x"},
        )
        # Rename refs/main to refs/myrevision so the test exercises that path.
        repo_dir = tmp_path / "datasets--lerobot--foo"
        (repo_dir / "refs" / "main").rename(repo_dir / "refs" / "myrevision")
        found = find_in_hf_cache("lerobot/foo", revision="myrevision", cache_dir=tmp_path)
        assert found is not None
        assert found.resolve() == snap.resolve()

    def test_revision_as_direct_sha(self, tmp_path):
        snap = _make_hf_cache(
            tmp_path,
            "lerobot/foo",
            "deadbeef",
            {"data/file.parquet": "x"},
        )
        # Drop refs/ so we have to fall back to treating the revision as a sha.
        (tmp_path / "datasets--lerobot--foo" / "refs" / "main").unlink()
        found = find_in_hf_cache("lerobot/foo", revision="deadbeef", cache_dir=tmp_path)
        assert found is not None
        assert found.resolve() == snap.resolve()

    def test_falls_back_to_newest_populated_snapshot_without_refs(self, tmp_path):
        # No refs/ at all — find_in_hf_cache should scan snapshots and pick
        # the most recently modified populated one.
        snap_old = _make_hf_cache(
            tmp_path,
            "lerobot/foo",
            "old" + "0" * 37,
            {"data/file.parquet": "x"},
        )
        (tmp_path / "datasets--lerobot--foo" / "refs" / "main").unlink()

        snap_new = (tmp_path / "datasets--lerobot--foo" / "snapshots" / "new" / "data")
        snap_new.mkdir(parents=True)
        (snap_new / "file.parquet").write_text("y")
        snapshot_new_dir = snap_new.parent

        import os
        # Make snap_new newer than snap_old.
        os.utime(snap_old, (1, 1))
        os.utime(snapshot_new_dir, (2_000_000_000, 2_000_000_000))

        found = find_in_hf_cache("lerobot/foo", cache_dir=tmp_path)
        assert found is not None
        assert found.resolve() == snapshot_new_dir.resolve()


class TestListHfCacheDatasets:
    """Test list_hf_cache_datasets — local cache listing."""

    def test_lists_populated_datasets_only_by_default(self, tmp_path):
        _make_hf_cache(
            tmp_path,
            "lerobot/foo",
            "a" * 40,
            {"data/file.parquet": "xxxxx", "videos/cam/file.mp4": "yyyyy"},
        )
        _make_hf_cache(
            tmp_path,
            "lerobot/stub",
            "b" * 40,
            {"meta/info.json": "{}"},
        )
        results = list_hf_cache_datasets(cache_dir=tmp_path)
        assert [d.repo_id for d in results] == ["lerobot/foo"]
        ds = results[0]
        assert ds.is_stub is False
        assert ds.file_counts == {".parquet": 1, ".mp4": 1}
        assert ds.size_bytes >= 10  # both files together

    def test_includes_stubs_when_requested(self, tmp_path):
        _make_hf_cache(
            tmp_path, "lerobot/foo", "a" * 40, {"data/file.parquet": "x"}
        )
        _make_hf_cache(
            tmp_path, "lerobot/stub", "b" * 40, {"meta/info.json": "{}"}
        )
        results = list_hf_cache_datasets(cache_dir=tmp_path, include_stubs=True)
        assert [d.repo_id for d in results] == ["lerobot/foo", "lerobot/stub"]
        stub = next(d for d in results if d.repo_id == "lerobot/stub")
        assert stub.is_stub is True
        assert stub.snapshot_path is None

    def test_returns_empty_for_missing_dir(self, tmp_path):
        assert list_hf_cache_datasets(cache_dir=tmp_path / "missing") == []

    def test_ignores_non_dataset_folders(self, tmp_path):
        # Model cache folders or unrelated junk should be skipped.
        (tmp_path / "models--meta-llama--Llama").mkdir()
        (tmp_path / "random_folder").mkdir()
        _make_hf_cache(
            tmp_path, "lerobot/foo", "a" * 40, {"data/file.parquet": "x"}
        )
        results = list_hf_cache_datasets(cache_dir=tmp_path)
        assert [d.repo_id for d in results] == ["lerobot/foo"]

    def test_sorted_by_repo_id(self, tmp_path):
        for name in ["zeta/last", "alpha/first", "mid/middle"]:
            _make_hf_cache(
                tmp_path, name, name.replace("/", "") + "0" * 30,
                {"data/file.parquet": "x"},
            )
        results = list_hf_cache_datasets(cache_dir=tmp_path)
        assert [d.repo_id for d in results] == [
            "alpha/first", "mid/middle", "zeta/last"
        ]


def _make_lerobot_dataset(root: Path, rel_path: str, num_episodes: int = 2) -> Path:
    """Build a minimal lerobot-v2.1-shaped dataset directory under `root`.

    The format detector recognises lerobot-v2 by presence of meta/info.json
    plus data/chunk-*/episode_*.parquet files.
    """
    dataset_dir = root / rel_path
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "data" / "chunk-000").mkdir(parents=True)
    (dataset_dir / "videos" / "chunk-000" / "cam").mkdir(parents=True)
    (dataset_dir / "meta" / "info.json").write_text(
        '{"codebase_version":"v2.1","total_episodes":' + str(num_episodes) + "}"
    )
    for i in range(num_episodes):
        (dataset_dir / "data" / "chunk-000" / f"episode_{i:06d}.parquet").write_text(
            "x" * 100
        )
        (dataset_dir / "videos" / "chunk-000" / "cam" / f"episode_{i:06d}.mp4").write_text(
            "y" * 50
        )
    return dataset_dir


class TestListLocalDatasetsGeneric:
    """Tests for the generic-layout (LeRobot-style) branch."""

    def test_finds_lerobot_layout_one_level_deep(self, tmp_path):
        # Mirror `~/.cache/huggingface/lerobot/<org>/<dataset>` — user
        # points at the org directory.
        org_dir = tmp_path / "myorg"
        org_dir.mkdir()
        _make_lerobot_dataset(org_dir, "stack_lego")
        _make_lerobot_dataset(org_dir, "place_cube")

        results = list_local_datasets(cache_dir=org_dir)
        ids = [d.repo_id for d in results]
        assert ids == ["myorg/place_cube", "myorg/stack_lego"]
        # Generic-layout entries are never marked as stubs.
        assert all(not d.is_stub for d in results)
        # File counts should be picked up from the dataset dirs themselves.
        for d in results:
            assert d.file_counts.get(".parquet", 0) >= 1
            assert d.file_counts.get(".mp4", 0) >= 1
            assert d.snapshot_path is not None and d.snapshot_path.is_dir()

    def test_finds_lerobot_layout_two_levels_deep(self, tmp_path):
        # User points at `~/.cache/huggingface/lerobot/`; datasets nest as
        # org/repo/.
        _make_lerobot_dataset(tmp_path, "orgA/dataset_one")
        _make_lerobot_dataset(tmp_path, "orgB/dataset_two")

        results = list_local_datasets(cache_dir=tmp_path)
        ids = [d.repo_id for d in results]
        assert ids == ["orgA/dataset_one", "orgB/dataset_two"]

    def test_skips_empty_subdirectories(self, tmp_path):
        # An empty subdir shouldn't be reported as a dataset.
        (tmp_path / "empty_repo").mkdir()
        _make_lerobot_dataset(tmp_path, "real_repo")

        results = list_local_datasets(cache_dir=tmp_path)
        assert [d.repo_id for d in results] == [f"{tmp_path.name}/real_repo"]

    def test_does_not_descend_into_dataset_subdirs(self, tmp_path):
        # Once a dataset is found at a given level, the walker shouldn't
        # treat its data/ or meta/ subdirs as separate candidates.
        _make_lerobot_dataset(tmp_path, "myset")
        results = list_local_datasets(cache_dir=tmp_path)
        assert len(results) == 1
        assert results[0].repo_id == f"{tmp_path.name}/myset"

    def test_hf_hub_layout_still_works(self, tmp_path):
        # Sanity check: presence of any `datasets--*` folder forces the
        # HF Hub branch even if other directories exist alongside.
        _make_hf_cache(
            tmp_path, "lerobot/foo", "a" * 40, {"data/file.parquet": "x"}
        )
        (tmp_path / "myorg" / "myset").mkdir(parents=True)
        results = list_local_datasets(cache_dir=tmp_path)
        assert [d.repo_id for d in results] == ["lerobot/foo"]


def test_list_hf_cache_datasets_alias_matches_list_local_datasets():
    # Back-compat alias should be the same callable.
    assert list_hf_cache_datasets is list_local_datasets


class TestGetHfCacheDir:
    """Test get_hf_cache_dir's env-var overrides."""

    def test_default(self, monkeypatch):
        for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
            monkeypatch.delenv(var, raising=False)
        assert get_hf_cache_dir() == Path.home() / ".cache" / "huggingface" / "hub"

    def test_hf_hub_cache_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "custom"))
        assert get_hf_cache_dir() == tmp_path / "custom"

    def test_hf_home_override(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))
        assert get_hf_cache_dir() == tmp_path / "hfhome" / "hub"


class TestGetLerobotCacheDir:
    """Test get_lerobot_cache_dir's env-var overrides."""

    def test_default(self, monkeypatch):
        for var in ("LEROBOT_HOME", "HF_HOME"):
            monkeypatch.delenv(var, raising=False)
        expected = Path.home() / ".cache" / "huggingface" / "lerobot"
        assert get_lerobot_cache_dir() == expected

    def test_lerobot_home_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LEROBOT_HOME", str(tmp_path / "lr"))
        assert get_lerobot_cache_dir() == tmp_path / "lr"

    def test_hf_home_override(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LEROBOT_HOME", raising=False)
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))
        assert get_lerobot_cache_dir() == tmp_path / "hfhome" / "lerobot"


class TestFindInLerobotCache:
    """Tests for the LeRobot-cache lookup used by `_resolve_via_hub`."""

    def test_returns_path_for_populated_repo(self, tmp_path):
        ds_dir = _make_lerobot_dataset(tmp_path, "arpit/foo")
        found = find_in_lerobot_cache("arpit/foo", cache_dir=tmp_path)
        assert found is not None
        assert found.resolve() == ds_dir.resolve()

    def test_returns_none_for_missing_repo(self, tmp_path):
        assert find_in_lerobot_cache("nobody/nothing", cache_dir=tmp_path) is None

    def test_returns_none_for_empty_dir(self, tmp_path):
        # An existing-but-empty repo dir should not be considered a hit
        # (matches the `eval_so101_place_cylinder` empty-stub scenario).
        (tmp_path / "arpit" / "empty").mkdir(parents=True)
        assert find_in_lerobot_cache("arpit/empty", cache_dir=tmp_path) is None
