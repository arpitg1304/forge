# Quality validation and benchmarks

## PR validation — October 17, 2026

- 264 tests passed across quality, video, viewer handoff, filters, catalog, video caching, and LeRobot round-trip/upstream compatibility.
- 7 streaming tests passed, including the local mock S3 server.
- 1 Chrome integration test passed for playback, event navigation, camera filtering, plots, and mobile layout.
- The broader CLI suite passed 16 tests. Two existing Zarr fixture setup errors require an explicit `shape` in `create_dataset`; they occur before command execution.
- Ruff passed on the quality implementation, viewer, reader, filter engine, new tests, and the benchmark script. Existing lint issues elsewhere in the CLI and older tests remain.
- `git diff --check` passed.

The quality CI job covers the core quality, viewer backend, reader, filter, and
catalog tests on Python 3.10–3.12. Browser and upstream LeRobot tests need optional
local dependencies. This is regression coverage, not measured detection accuracy.

Run the browser check with:

```bash
FORGE_TEST_CHROME=/path/to/chrome pytest tests/test_web_viewer_browser.py --no-cov
```

## Video CPU benchmark

Local benchmark measurements use an Apple M5 Pro
with 18 CPU cores and 64 GiB RAM. The fixture contains 300 DROID frames resized
to 1280×720 and encoded as H.264. Pixel checks use a 128-pixel longest side;
optical flow uses 64 pixels. Every frame is analyzed.

| Pipeline | Median milliseconds/frame | Frames/second |
|---|---:|---:|
| Forge decode + pixel checks | 1.93 | 519 |
| Forge decode + pixel and motion checks | 3.11 | 321 |

Results are medians of five warm runs. Timings include file opening, decoding,
RGB conversion, metrics, integrity checks, and event aggregation. They exclude
fixture creation, imports, downloads, dataset indexing, proprioception, and UI.

Estimated processing time for one camera, using measured throughput:

| Input FPS | Video duration | Pixel checks | Pixel + motion |
|---|---|---:|---:|
| 20 | 10 seconds | 0.39 s | 0.62 s |
| 20 | 30 seconds | 1.16 s | 1.87 s |
| 20 | 60 seconds | 2.31 s | 3.73 s |
| 30 | 10 seconds | 0.58 s | 0.93 s |
| 30 | 30 seconds | 1.73 s | 2.80 s |
| 30 | 60 seconds | 3.47 s | 5.60 s |

Longer durations are linear estimates, not separately measured runs. Codec,
scene, hardware, and camera count affect throughput. No GPU was used. Generated JSON and video files are kept locally, outside Git.

Reproduce with a local source video:

```bash
PYTHONPATH=. python scripts/benchmark_video_quality.py SOURCE.mp4 \
  --output /tmp/forge-video-benchmark/results.json
```

See the [quality guide](../../forge/quality/README.md) and
[video guide](../../forge/quality/video/README.md) for current semantics.
