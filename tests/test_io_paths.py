"""Tests for cloud/local path resolution (forge.io.paths).

Cloud access is exercised without touching any real cloud provider:
- fsspec's in-process ``memory://`` filesystem for the fast, always-on tests.
- a ``moto`` mock S3 server for the ``s3://`` code path (skipped if moto/s3fs
  aren't installed).
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

from forge.core.exceptions import MissingDependencyError
from forge.io.paths import (
    RemoteWriteNotSupportedError,
    cleanup_localized,
    get_filesystem,
    get_scheme,
    is_remote_uri,
    localize,
)


class TestSchemeDetection:
    def test_get_scheme(self):
        assert get_scheme("s3://bucket/key") == "s3"
        assert get_scheme("gs://bucket/key") == "gs"
        assert get_scheme("gcs://bucket/key") == "gcs"
        assert get_scheme("hf://org/repo") == "hf"
        assert get_scheme("memory://foo/bar") == "memory"
        # Local paths and Windows drive letters have no URL scheme.
        assert get_scheme("/local/abs/path") == ""
        assert get_scheme("relative/path") == ""
        assert get_scheme("C:/Users/data") == ""

    def test_is_remote_uri(self):
        assert is_remote_uri("s3://bucket/key")
        assert is_remote_uri("gs://bucket/key")
        assert is_remote_uri("gcs://bucket/key")
        # memory:// is a generic fsspec source (used by these tests).
        assert is_remote_uri("memory://foo")
        # Local paths and HF URLs are NOT remote (HF is handled by forge.hub).
        assert not is_remote_uri("/local/path")
        assert not is_remote_uri("relative/path")
        assert not is_remote_uri("hf://org/repo")
        assert not is_remote_uri("huggingface://org/repo")

    def test_is_remote_uri_accepts_pathlike(self):
        assert not is_remote_uri(Path("/local/path"))


class TestLocalPassthrough:
    def test_local_path_returned_unchanged(self):
        # Local behaviour must be identical to before this module existed.
        assert localize("/tmp/some/dataset") == Path("/tmp/some/dataset")
        assert localize(Path("relative/ds")) == Path("relative/ds")

    def test_hf_url_is_not_localized(self):
        # HF URLs are resolved elsewhere; localize must not try to download them.
        assert localize("hf://lerobot/pusht") == Path("hf://lerobot/pusht")


@pytest.fixture
def memory_fs():
    """A clean fsspec memory filesystem for each test."""
    import fsspec

    fs = fsspec.filesystem("memory")
    # Wipe any state left by other tests (MemoryFileSystem is process-global).
    try:
        for p in fs.find("/"):
            fs.rm(p)
    except FileNotFoundError:
        pass
    yield fs
    cleanup_localized()


def _write_mem(fs, path: str, data: bytes) -> None:
    with fs.open(path, "wb") as f:
        f.write(data)


class TestLocalizeMemory:
    def test_localize_directory(self, memory_fs):
        _write_mem(memory_fs, "/ds/meta/info.json", b'{"a": 1}')
        _write_mem(memory_fs, "/ds/data/f0.parquet", b"PARQ0")
        _write_mem(memory_fs, "/ds/data/f1.parquet", b"PARQ1")

        local = localize("memory:///ds")

        assert local.is_dir()
        assert (local / "meta" / "info.json").read_bytes() == b'{"a": 1}'
        assert (local / "data" / "f0.parquet").read_bytes() == b"PARQ0"
        assert (local / "data" / "f1.parquet").read_bytes() == b"PARQ1"

    def test_localize_single_file(self, memory_fs):
        _write_mem(memory_fs, "/solo/info.json", b'{"k": 2}')

        local = localize("memory:///solo/info.json")

        assert local.is_file()
        assert local.name == "info.json"
        assert local.read_bytes() == b'{"k": 2}'

    def test_localize_missing_raises(self, memory_fs):
        with pytest.raises(FileNotFoundError):
            localize("memory:///does/not/exist")

    def test_cleanup_removes_temp_dirs(self, memory_fs):
        _write_mem(memory_fs, "/ds2/a.txt", b"hi")
        local = localize("memory:///ds2")
        assert local.exists()
        cleanup_localized()
        assert not local.exists()


class TestMissingDependency:
    def test_missing_s3fs_gives_install_hint(self, monkeypatch):
        """s3:// without s3fs must fail with the exact pip command."""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "s3fs":
                raise ImportError("No module named 's3fs'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(MissingDependencyError) as excinfo:
            get_filesystem("s3://bucket/key")
        msg = str(excinfo.value)
        assert "s3fs" in msg
        assert "pip install forge-robotics[s3]" in msg

    def test_missing_gcsfs_gives_install_hint(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "gcsfs":
                raise ImportError("No module named 'gcsfs'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(MissingDependencyError) as excinfo:
            get_filesystem("gs://bucket/key")
        assert "pip install forge-robotics[gcs]" in str(excinfo.value)


class TestRemoteWriteGuard:
    def test_error_message_mentions_alternatives(self):
        err = RemoteWriteNotSupportedError("s3://bucket/out")
        msg = str(err)
        assert "s3://bucket/out" in msg
        assert "not supported" in msg.lower()


class TestCLICloud:
    """The CLI resolves cloud sources and rejects cloud outputs."""

    def _make_lerobot_v3_in_memory(self, fs, prefix: str) -> None:
        files = {
            f"{prefix}/meta/info.json": json.dumps(
                {
                    "codebase_version": "v3.0",
                    "fps": 30,
                    "robot_type": "r",
                    "features": {"action": {"dtype": "float32", "shape": [7]}},
                    "total_episodes": 2,
                    "total_frames": 100,
                }
            ),
            f"{prefix}/meta/episodes.jsonl": (
                '{"episode_index": 0, "length": 50}\n'
                '{"episode_index": 1, "length": 50}\n'
            ),
            f"{prefix}/meta/tasks.jsonl": '{"task_index": 0, "task": "t"}\n',
            f"{prefix}/data/train/.keep": "",
        }
        for path, content in files.items():
            with fs.open(path, "wb") as f:
                f.write(content.encode())

    def test_inspect_resolves_cloud_uri(self, memory_fs):
        from typer.testing import CliRunner

        from forge.cli import app

        self._make_lerobot_v3_in_memory(memory_fs, "/cli-bkt/ds")

        result = CliRunner().invoke(app, ["inspect", "memory:///cli-bkt/ds"])
        assert result.exit_code == 0, result.output
        # lerobot-v3 is streamable, so inspect reads metadata over the network
        # instead of downloading the dataset.
        assert "streamed metadata" in result.output
        assert "lerobot" in result.output.lower()

    def test_inspect_deep_downloads_cloud_uri(self, memory_fs):
        # --deep needs the full data, so it falls back to download.
        from typer.testing import CliRunner

        from forge.cli import app

        self._make_lerobot_v3_in_memory(memory_fs, "/cli-bkt-deep/ds")
        result = CliRunner().invoke(
            app, ["inspect", "memory:///cli-bkt-deep/ds", "--deep"]
        )
        assert "Fetching from cloud storage" in result.output

    def test_convert_rejects_cloud_output(self, memory_fs):
        from typer.testing import CliRunner

        from forge.cli import app

        self._make_lerobot_v3_in_memory(memory_fs, "/cli-bkt2/ds")

        result = CliRunner().invoke(
            app,
            ["convert", "memory:///cli-bkt2/ds", "s3://bucket/out", "-f", "lerobot-v3"],
        )
        assert result.exit_code == 1
        assert "not supported" in result.output.lower()
        assert "s3://bucket/out" in result.output


class TestFormatsOverMemory:
    """End-to-end: prioritized formats (lerobot-v3, zarr) resolve from cloud."""

    def _upload_dir(self, fs, local_dir: Path, remote_prefix: str) -> None:
        for path in local_dir.rglob("*"):
            if path.is_file():
                rel = path.relative_to(local_dir).as_posix()
                with fs.open(f"{remote_prefix}/{rel}", "wb") as f:
                    f.write(path.read_bytes())

    def test_lerobot_v3_localize_and_detect(
        self, memory_fs, temp_lerobot_v3_dataset: Path
    ):
        from forge.formats.registry import FormatRegistry

        self._upload_dir(memory_fs, temp_lerobot_v3_dataset, "/lerobot_ds")

        local = localize("memory:///lerobot_ds")
        assert FormatRegistry.detect_format(local) == "lerobot-v3"
        # Metadata content survived the round-trip.
        info = json.loads((local / "meta" / "info.json").read_text())
        assert info["codebase_version"] == "v3.0"

    def test_zarr_localize_and_detect(self, memory_fs, tmp_path: Path):
        zarr = pytest.importorskip("zarr")

        from forge.formats.registry import FormatRegistry

        # Build a minimal zarr store locally (version-robust: create_array on
        # zarr>=3, create_dataset on zarr 2), then round-trip it through cloud.
        store_path = tmp_path / "umi.zarr"
        root = zarr.open(str(store_path), mode="w")
        root.attrs["fps"] = 30
        make = getattr(root, "create_array", None) or root.create_dataset
        make("action", shape=(10, 7), dtype="float32")

        self._upload_dir(memory_fs, store_path, "/zarr_ds.zarr")

        local = localize("memory:///zarr_ds.zarr")
        assert FormatRegistry.detect_format(local) == "zarr"


# ---------------------------------------------------------------------------
# moto-backed S3 test — exercises the real s3:// scheme via a mock S3 server.
# ---------------------------------------------------------------------------


@pytest.fixture
def moto_s3():
    """Spin up a mock S3 server and point s3fs/botocore at it via env vars."""
    pytest.importorskip("s3fs")
    moto_server = pytest.importorskip("moto.server")

    import s3fs

    server = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"

    monkey = pytest.MonkeyPatch()
    monkey.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkey.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkey.setenv("AWS_SESSION_TOKEN", "testing")
    monkey.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkey.setenv("AWS_ENDPOINT_URL", endpoint)
    monkey.setenv("AWS_ENDPOINT_URL_S3", endpoint)

    # s3fs caches filesystem instances keyed by options; clear so our endpoint
    # env vars take effect for a fresh client.
    s3fs.S3FileSystem.clear_instance_cache()

    try:
        yield endpoint
    finally:
        s3fs.S3FileSystem.clear_instance_cache()
        monkey.undo()
        server.stop()
        cleanup_localized()


class TestLocalizeS3Moto:
    def test_localize_lerobot_v3_from_s3(
        self, moto_s3, temp_lerobot_v3_dataset: Path
    ):
        import s3fs

        from forge.formats.registry import FormatRegistry

        fs = s3fs.S3FileSystem()
        fs.mkdir("forge-test-bucket")
        for path in temp_lerobot_v3_dataset.rglob("*"):
            if path.is_file():
                rel = path.relative_to(temp_lerobot_v3_dataset).as_posix()
                fs.put_file(str(path), f"forge-test-bucket/dataset/{rel}")

        local = localize("s3://forge-test-bucket/dataset")

        assert local.is_dir()
        assert (local / "meta" / "info.json").exists()
        assert FormatRegistry.detect_format(local) == "lerobot-v3"

    def test_localize_single_file_from_s3(self, moto_s3):
        import s3fs

        fs = s3fs.S3FileSystem()
        fs.mkdir("forge-test-bucket2")
        with fs.open("forge-test-bucket2/only.json", "wb") as f:
            f.write(b'{"hello": "s3"}')

        local = localize("s3://forge-test-bucket2/only.json")

        assert local.is_file()
        assert local.name == "only.json"
        assert local.read_bytes() == b'{"hello": "s3"}'
