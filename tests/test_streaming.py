"""Streaming metadata reads for cloud datasets (lerobot-v3, zarr).

`forge inspect` on an ``s3://`` / ``gs://`` lerobot-v3 or zarr dataset reads
metadata via range requests instead of downloading the whole dataset. These
tests verify (a) it produces the right DatasetInfo over fsspec without a local
copy, and (b) — against a moto S3 mock — that it actually streams, by asserting
the bytes pulled off the wire stay under a tight budget.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from forge.formats.registry import FormatRegistry
from forge.io import cleanup_localized
from forge.io.paths import _TEMP_ROOTS


def _lerobot_v3_info() -> dict:
    return {
        "codebase_version": "v3.0",
        "robot_type": "franka",
        "total_episodes": 42,
        "total_frames": 5000,
        "fps": 20,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [7]},
            "action": {"dtype": "float32", "shape": [7]},
            "observation.images.wrist": {"dtype": "video", "shape": [240, 320, 3]},
        },
    }


@pytest.fixture
def memory_fs():
    fs = pytest.importorskip("fsspec").filesystem("memory")
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


class TestLeRobotV3Streaming:
    def test_detect_and_inspect_over_memory(self, memory_fs):
        _write_mem(
            memory_fs, "/lr3/meta/info.json", json.dumps(_lerobot_v3_info()).encode()
        )
        _write_mem(memory_fs, "/lr3/meta/tasks.jsonl", b'{"task": "pick"}\n')

        uri = "memory:///lr3"
        assert FormatRegistry.detect_format(uri) == "lerobot-v3"

        before = len(_TEMP_ROOTS)
        info = FormatRegistry.get_reader("lerobot-v3").inspect(uri)

        # correct metadata, straight from info.json
        assert info.num_episodes == 42
        assert info.total_frames == 5000
        assert info.inferred_fps == 20
        assert info.inferred_robot_type == "franka"
        assert info.action_schema.shape == (7,)
        assert "wrist" in info.cameras
        cam = info.cameras["wrist"]
        assert (cam.height, cam.width) == (240, 320)
        assert info.has_language is True
        # ...and it did NOT download anything.
        assert len(_TEMP_ROOTS) == before

    def test_not_v3_without_info(self, memory_fs):
        # a bare prefix isn't a v3 dataset
        _write_mem(memory_fs, "/empty/readme.txt", b"hi")
        assert FormatRegistry.get_reader("lerobot-v3")._can_read_remote(
            "memory:///empty"
        ) is False


class TestZarrStreaming:
    def _upload_zarr(self, fs, prefix: str, episodes=2, frames=10):
        zarr = pytest.importorskip("zarr")
        import tempfile

        tmp = Path(tempfile.mkdtemp()) / "z.zarr"
        root = zarr.open(str(tmp), mode="w")
        root.attrs["fps"] = 10
        root.attrs["robot_type"] = "ur5"
        data = root.create_group("data")
        mk = getattr(data, "create_array", None) or data.create_dataset
        cam = mk("camera0_rgb", shape=(episodes * frames, 16, 16, 3), dtype="uint8")
        cam[:] = 0
        act = mk("action", shape=(episodes * frames, 7), dtype="float32")
        act[:] = 0
        meta = root.create_group("meta")
        mm = getattr(meta, "create_array", None) or meta.create_dataset
        ends = mm("episode_ends", shape=(episodes,), dtype="int64")
        ends[:] = np.array([(i + 1) * frames for i in range(episodes)])
        for p in tmp.rglob("*"):
            if p.is_file():
                with fs.open(f"{prefix}/{p.relative_to(tmp).as_posix()}", "wb") as f:
                    f.write(p.read_bytes())

    def test_detect_and_inspect_over_memory(self, memory_fs):
        pytest.importorskip("zarr")
        self._upload_zarr(memory_fs, "/z.zarr", episodes=3, frames=8)

        uri = "memory:///z.zarr"
        assert FormatRegistry.detect_format(uri) == "zarr"

        before = len(_TEMP_ROOTS)
        info = FormatRegistry.get_reader("zarr").inspect(uri)
        assert info.num_episodes == 3
        assert info.inferred_fps == 10
        assert info.inferred_robot_type == "ur5"
        assert info.action_schema is not None
        assert len(_TEMP_ROOTS) == before  # no download


class TestStreamableRegistry:
    def test_streamable_set(self):
        assert FormatRegistry.STREAMABLE_FORMATS == frozenset({"lerobot-v3", "zarr"})


# ---------------------------------------------------------------------------
# Byte-budget: prove it streams (range reads), not downloads — via moto S3.
# ---------------------------------------------------------------------------


@pytest.fixture
def moto_s3():
    pytest.importorskip("s3fs")
    moto_server = pytest.importorskip("moto.server")
    import s3fs

    server = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"

    monkey = pytest.MonkeyPatch()
    for k, v in {
        "AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_DEFAULT_REGION": "us-east-1", "AWS_ENDPOINT_URL": endpoint,
        "AWS_ENDPOINT_URL_S3": endpoint,
    }.items():
        monkey.setenv(k, v)
    s3fs.S3FileSystem.clear_instance_cache()
    try:
        yield endpoint
    finally:
        s3fs.S3FileSystem.clear_instance_cache()
        monkey.undo()
        server.stop()


class TestByteBudgetS3:
    def test_lerobot_v3_inspect_streams_not_downloads(self, moto_s3):
        import s3fs
        from s3fs.core import S3File

        fs = s3fs.S3FileSystem()
        fs.mkdir("stream-bucket")
        # a small info.json + a deliberately "large" fake payload alongside it,
        # to prove inspect never reads the bulk.
        with fs.open("stream-bucket/ds/meta/info.json", "wb") as f:
            f.write(json.dumps(_lerobot_v3_info()).encode())
        with fs.open("stream-bucket/ds/videos/big.bin", "wb") as f:
            f.write(b"\0" * (8 * 1024 * 1024))  # 8 MB of payload

        fetched = {"n": 0}
        orig = S3File._fetch_range

        def counting(self, start, end):
            fetched["n"] += end - start
            return orig(self, start, end)

        S3File._fetch_range = counting
        try:
            info = FormatRegistry.get_reader("lerobot-v3").inspect("s3://stream-bucket/ds")
        finally:
            S3File._fetch_range = orig

        assert info.num_episodes == 42
        # streamed: fetched only info.json-scale bytes, nowhere near the 8 MB payload
        assert fetched["n"] < 1_000_000, f"expected a streaming read, got {fetched['n']} bytes"


def _write_lerobot_v3(fs, root: str, *, episodes=2, frames=10) -> None:
    """Write a minimal streamable v3 dataset (info.json + one data parquet)."""
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    info = dict(_lerobot_v3_info())
    info["total_episodes"] = episodes
    info["total_frames"] = episodes * frames
    with fs.open(f"{root}/meta/info.json", "wb") as f:
        f.write(json.dumps(info).encode())

    n = episodes * frames
    rng = np.random.RandomState(0)
    tbl = pa.table(
        {
            "episode_index": [e for e in range(episodes) for _ in range(frames)],
            "index": list(range(n)),
            "timestamp": [float(i % frames) / 10 for i in range(n)],
            "observation.state": [rng.randn(7).tolist() for _ in range(n)],
            "action": [rng.randn(7).tolist() for _ in range(n)],
        }
    )
    buf = io.BytesIO()
    pq.write_table(tbl, buf)
    with fs.open(f"{root}/data/chunk-000/file-000.parquet", "wb") as f:
        f.write(buf.getvalue())


class TestStreamingIngest:
    def test_ingest_streams_over_memory(self, memory_fs, tmp_path):
        pytest.importorskip("duckdb")
        from forge.catalog import Catalog
        from forge.catalog.ingest import ingest as ingest_datasets

        _write_lerobot_v3(memory_fs, "/ds", episodes=3, frames=8)
        cat = Catalog.init(str(tmp_path / "cat"))

        before = len(_TEMP_ROOTS)
        stats = ingest_datasets(["memory:///ds"], cat)
        assert stats.ingested == 3 and stats.failed == 0
        assert len(_TEMP_ROOTS) == before  # streamed — nothing downloaded
        # registered + quality-scored, all from parquet proprio columns
        assert cat.sql("SELECT count(*) n FROM episodes").to_pylist()[0]["n"] == 3
        scored = cat.sql(
            "SELECT count(*) n FROM v_latest_quality WHERE overall_score IS NOT NULL"
        ).to_pylist()[0]["n"]
        assert scored == 3

    def test_ingest_byte_budget_ignores_video(self, moto_s3, tmp_path):
        pytest.importorskip("duckdb")
        import s3fs
        from s3fs.core import S3File

        from forge.catalog import Catalog
        from forge.catalog.ingest import ingest as ingest_datasets

        fs = s3fs.S3FileSystem()
        fs.mkdir("ingest-bucket")
        _write_lerobot_v3(fs, "ingest-bucket/ds", episodes=2, frames=10)
        # a big "video" payload the proprio ingest must not read
        with fs.open("ingest-bucket/ds/videos/cam/big.mp4", "wb") as f:
            f.write(b"\0" * (8 * 1024 * 1024))

        fetched = {"n": 0}
        orig = S3File._fetch_range

        def counting(self, start, end):
            fetched["n"] += end - start
            return orig(self, start, end)

        S3File._fetch_range = counting
        try:
            cat = Catalog.init(str(tmp_path / "cat"))
            stats = ingest_datasets(["s3://ingest-bucket/ds"], cat)
        finally:
            S3File._fetch_range = orig

        assert stats.ingested == 2
        assert fetched["n"] < 2_000_000, f"ingest should stream proprio, got {fetched['n']} bytes"
