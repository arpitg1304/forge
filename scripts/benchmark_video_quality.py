"""Reproducible CPU benchmark of Forge's video checks on a local 720p H.264 clip.

Run from repo root:
  PYTHONPATH=. .venv/bin/python scripts/benchmark_video_quality.py SOURCE.mp4 \
    --output /tmp/forge-video-benchmark/results.json

The source's first N frames are resized and re-encoded as a controlled fixture.
Preparation/encoding is excluded from timings. End-to-end timings include local
file opening, decoding, RGB conversion, metrics, integrity, events and rollups;
exclude process startup, network I/O, dataset indexing, proprioception and UI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import av
import cv2
import numpy as np

from forge.core.models import Frame, LazyImage
from forge.quality.metrics import log_dimensionless_jerk
from forge.quality.video import VideoQualityAnalyzer, VideoQualityConfig
from forge.quality.video.motion import (
    affine_design,
    camera_scene_split,
    dense_flow,
    flow_magnitude,
    histogram_correlation,
    to_uint8_gray,
)
from forge.quality.video.pixel import (
    clip_fractions,
    colorfulness,
    frame_mae,
    laplacian_variance,
    prepare,
)


def summary(seconds, units):
    median = statistics.median(seconds)
    return dict(
        median_seconds=median,
        min_seconds=min(seconds),
        max_seconds=max(seconds),
        median_ms_per_unit=median * 1000 / units,
        units_per_second=units / median,
        repetitions_seconds=seconds,
        units_per_repetition=units,
    )


def micro(fn, iterations, repeats):
    for i in range(10):
        fn(i)
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        for i in range(iterations):
            fn(i)
        durations.append(time.perf_counter() - start)
    return summary(durations, iterations)


def make_clip(source, output, count):
    actual = 0
    with av.open(str(source)) as src, av.open(str(output), "w") as dst:
        stream = dst.add_stream("libx264", rate=30)
        stream.width, stream.height = 1280, 720
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "fast"}
        for i, frame in enumerate(src.decode(video=0)):
            if i >= count:
                break
            rgb = cv2.resize(
                frame.to_ndarray(format="rgb24"), (1280, 720), interpolation=cv2.INTER_LINEAR
            )
            out = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            out.pts = i
            for packet in stream.encode(out):
                dst.mux(packet)
            actual += 1
        for packet in stream.encode():
            dst.mux(packet)
    if actual != count:
        raise ValueError(f"Source has only {actual} frames; need {count}")


def run_pipeline(clip, mode, arrays=None, count=300):
    analyzer = VideoQualityAnalyzer(
        VideoQualityConfig(level=mode if mode != "decode_only" else "pixel")
    )
    acc = analyzer.new_accumulators(fps=30, cameras=["camera"])
    start = time.perf_counter()
    n = 0
    if arrays is None:
        container = av.open(str(clip))
        frames = container.decode(video=0)
    else:
        container = None
        frames = (arrays[i % len(arrays)] for i in range(count))
    try:
        for i, img in enumerate(frames):
            # Lazily decode RGB on the production analyzer's consume path.
            if arrays is None:

                def loader(f=img):
                    return f.to_ndarray(format="rgb24")
            else:

                def loader(f=img):
                    return f

            if mode == "decode_only":
                loader()
            else:
                frame = Frame(
                    index=i,
                    timestamp=i / 30,
                    images={"camera": LazyImage(loader=loader, height=720, width=1280)},
                )
                analyzer.consume_frame(frame, acc, i)
            n += 1
        if mode != "decode_only":
            result = analyzer.finalize(acc)
            assert result.per_camera["camera"].num_frames == n
    finally:
        if container:
            container.close()
    return time.perf_counter() - start, n


def run_forge_reader(clip, mode, count):
    from forge.formats.lerobot_v3.reader import VideoFrameCache

    analyzer = VideoQualityAnalyzer(VideoQualityConfig(level=mode))
    acc = analyzer.new_accumulators(fps=30, cameras=["camera"])
    cache = VideoFrameCache(clip)
    start = time.perf_counter()
    try:
        for i in range(count):
            frame = Frame(
                index=i,
                timestamp=i / 30,
                images={
                    "camera": LazyImage(
                        loader=lambda idx=i: cache.get_frame(idx), height=720, width=1280
                    )
                },
            )
            analyzer.consume_frame(frame, acc, i)
        result = analyzer.finalize(acc)
        assert result.per_camera["camera"].num_frames == count
    finally:
        cache.close()
    return time.perf_counter() - start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    if args.frames < 16 or args.repeats < 1:
        p.error("--frames must be at least 16 and --repeats must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clip = args.output.parent / "benchmark_720p_h264.mp4"
    if args.source.resolve() == clip.resolve():
        p.error("The source must differ from the generated benchmark fixture path")
    make_clip(args.source, clip, args.frames)
    images = []
    with av.open(str(clip)) as src:
        for i, f in enumerate(src.decode(video=0)):
            if i >= 16:
                break
            images.append(f.to_ndarray(format="rgb24"))
    prepared = [prepare(i, 128) for i in images]
    motion = [to_uint8_gray(prepare(i, 64)[0]) for i in images]
    design = affine_design(*motion[0].shape)
    flows = [dense_flow(motion[i - 1], motion[i]) for i in range(1, len(motion))]

    def gray(i):
        return prepared[i % len(prepared)][0]

    def rgb(i):
        return prepared[i % len(prepared)][1]

    def flow(i):
        return flows[i % len(flows)]

    funcs = {
        "prepare_720p_to_128": lambda i: prepare(images[i % len(images)], 128),
        "prepare_720p_to_64": lambda i: to_uint8_gray(prepare(images[i % len(images)], 64)[0]),
        "blur_sharpness": lambda i: laplacian_variance(gray(i)),
        "exposure_brightness": lambda i: (float(gray(i).mean()), clip_fractions(gray(i), 4, 254)),
        "frozen_pixel_difference": lambda i: frame_mae(gray(i), gray(i + 1)),
        "colorfulness": lambda i: colorfulness(rgb(i)),
        "optical_flow_and_magnitude": lambda i: flow_magnitude(
            dense_flow(motion[i % 16], motion[(i + 1) % 16])
        ).mean(),
        "camera_scene_motion_split": lambda i: camera_scene_split(flow(i), design),
        "cut_histogram_comparison": lambda i: histogram_correlation(
            motion[i % 16], motion[(i + 1) % 16]
        ),
    }
    result = dict(
        hardware=dict(
            platform=platform.platform(),
            machine=platform.machine(),
            opencv_threads=cv2.getNumThreads(),
        ),
        versions={x: version(x) for x in ("numpy", "av", "opencv-python")},
        fixture=dict(
            source=str(args.source),
            path=str(clip),
            frames=args.frames,
            width=1280,
            height=720,
            fps=30,
            codec="H.264/libx264",
            crf=23,
            bytes=clip.stat().st_size,
            sha256=hashlib.sha256(clip.read_bytes()).hexdigest(),
        ),
        settings=dict(
            pixel_longest_side=128, motion_longest_side=64, sample_stride=1, repeats=args.repeats
        ),
        per_component={},
        pipelines={},
    )
    for name, fn in funcs.items():
        result["per_component"][name] = micro(fn, 200, args.repeats)
        print(name, round(result["per_component"][name]["median_ms_per_unit"], 4), "ms", flush=True)
    # Smoothness is an episode reduction, not a per-frame kernel.
    trajectory = np.cumsum(
        np.tile(np.array([f.mean(axis=(0, 1)) for f in flows]), (args.frames // len(flows) + 1, 1)),
        axis=0,
    )[: args.frames]
    result["motion_smoothness_per_episode"] = micro(
        lambda i: log_dimensionless_jerk(trajectory, 1 / 30), 100, args.repeats
    )

    # Integrity scheduling with no pixel decoding: every reference intentionally sampled out
    # except the first. Subtracting timings is avoided; label the one-frame work included.
    def integrity_run():
        analyzer = VideoQualityAnalyzer(VideoQualityConfig(sample_stride=args.frames + 1))
        acc = analyzer.new_accumulators(fps=30, cameras=["camera"])
        start = time.perf_counter()
        for i in range(args.frames):
            analyzer.consume_frame(
                Frame(
                    index=i,
                    timestamp=i / 30,
                    images={"camera": LazyImage(loader=lambda: images[0], height=720, width=1280)},
                ),
                acc,
                i,
            )
        analyzer.finalize(acc)
        return time.perf_counter() - start

    integrity_run()
    result["integrity_and_one_pixel_sample"] = summary(
        [integrity_run() for _ in range(args.repeats)], args.frames
    )
    jobs = [
        ("decode_only", None),
        ("pixel", images),
        ("motion", images),
        ("pixel", None),
        ("motion", None),
    ]
    timings = {
        (
            "decode_only"
            if mode == "decode_only"
            else ("metrics_" if arrays is not None else "end_to_end_") + mode
        ): []
        for mode, arrays in jobs
    }
    for mode, arrays in jobs:
        run_pipeline(clip, mode, arrays, args.frames)
    for repeat in range(args.repeats):
        # Alternate order to reduce systematic warmup/thermal bias.
        for mode, arrays in jobs if repeat % 2 == 0 else reversed(jobs):
            key = (
                "decode_only"
                if mode == "decode_only"
                else ("metrics_" if arrays is not None else "end_to_end_") + mode
            )
            elapsed, n = run_pipeline(clip, mode, arrays, args.frames)
            assert n == args.frames
            timings[key].append(elapsed)
    for key, values in timings.items():
        result["pipelines"][key] = summary(values, args.frames)
        print(key, result["pipelines"][key], flush=True)
    for mode in ("pixel", "motion"):
        run_forge_reader(clip, mode, args.frames)
        values = [run_forge_reader(clip, mode, args.frames) for _ in range(args.repeats)]
        result["pipelines"]["forge_reader_" + mode] = summary(values, args.frames)
        print("forge_reader_" + mode, result["pipelines"]["forge_reader_" + mode], flush=True)
    result["estimates"] = []
    for fps in (20, 30):
        for duration in (10, 30, 60):
            row = dict(input_fps=fps, duration_seconds=duration, frames=fps * duration)
            for mode in ("pixel", "motion"):
                ms = result["pipelines"]["forge_reader_" + mode]["median_ms_per_unit"]
                row[mode + "_processing_seconds"] = fps * duration * ms / 1000
            result["estimates"].append(row)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
