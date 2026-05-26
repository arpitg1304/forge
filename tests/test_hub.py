"""Tests for HuggingFace Hub integration."""

from pathlib import Path

import pytest

from forge.hub.download import find_in_hf_cache, get_hf_cache_dir
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
