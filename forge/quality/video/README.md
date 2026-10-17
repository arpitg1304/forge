# Inspectable video quality

Video checks remain separate from the proprioception score. They describe image
health and observation integrity, not task success or training usefulness.
Results are versioned `video-v2-integrity`.

```bash
forge quality ./dataset --video --visualize --export quality.json
forge quality ./dataset --video --video-level motion --video-stride 4 \
  --video-max-frames 200 --visualize --export quality.json
```

## Integrity and frame-drop estimates

Every episode observation is inspected for camera references and timestamps,
including observations skipped by image sampling or beyond the decode cap.
For each selected/declared camera, or a camera discovered during iteration:

- `expected_frames`: episode observations on which camera references are expected.
- `present_frames`, `missing_frames`, `missing_fraction`: camera-reference availability.
- `decode_attempts`, `decode_failures`, `invalid_frames`: sampled decode outcomes.
- `num_frames`: valid decoded images included in pixel statistics.
- `sampled_out_frames`, `capped_frames`: intentionally unexamined images, never counted as failures.
- `decode_coverage`: valid decoded images divided by present references. This is analysis coverage, not a codec success rate; use failures/attempts for the latter.
- `timestamp_coverage`, `effective_fps`, `timestamp_jitter_ratio`: support and observed cadence of episode timestamps.
- `estimated_dropped_frames`, `estimated_drop_fraction`, `timestamp_gaps`, `max_frame_gap_s`: estimates from elapsed observation time and expected FPS.

A gap exceeds 1.5 expected frame periods. Its estimated missing count is
`round(elapsed_seconds * expected_fps) - 1`. A `video_frame_drops` warning fires
when any gap represents at least 3 missing observations, at least 0.2 seconds of
missing time beyond one expected period, or when the estimated drop fraction
exceeds 5%. The fraction is missing estimates / (valid transitions + missing
estimates), and is withheld on incomplete/nonmonotonic timelines.

Tune those alert criteria with `--video-drop-frames`, `--video-gap-seconds`, and
`--video-drop-fraction`. Episode FPS is used by default; `--video-fps` is an
explicit override. Python exposes `VideoQualityConfig.gap_factor` too.

**These are observation-gap estimates, not verified camera capture drops.**
Forge generally receives episode timestamps rather than independent camera PTS.
Incorrect expected FPS or intentional lower-rate acquisition can also explain
gaps. Missing camera references are separate from missing timestamp intervals;
do not add those counts together. Asynchronous camera schemas need their own
availability expectations before treating missing references as faults.

With no stored timestamps, FPS may supply an explicitly inferred event timebase,
but drop detection abstains. With incomplete timestamps, estimates cover only
supported pairs and are labeled partial. With no usable timebase, events retain
frame indices and unknown seconds.

## Pixel and motion evidence

Blur, exposure, near-identical images, cut candidates, missing references, decode
failures, invalid pixels, resolution changes, invalid timestamps, and time gaps
produce camera-scoped events. Frame indices are zero-based; `end_frame` is
exclusive. Time boundaries are the first and last supporting sample, so an
isolated sampled frame has a zero-length time span.

Pixel events on strided input describe sampled evidence, not verified continuous
faults in the intervening images. Freeze durations use elapsed seconds when
available (`--video-freeze-seconds`, default 0.5); the legacy sampled-run threshold
is only a fallback without a timebase. A stationary scene can still look frozen.

Gaps, camera absence, invalid data, decode errors, and resolution changes reset
temporal comparisons. Blank/low-texture observations also reset optical flow.
Motion magnitudes use actual elapsed time where supported; `mean_motion_px_s`
provides pixels/second on the downscaled analysis frame. The existing
`mean_motion` retains pixels per expected frame. Motion smoothness abstains when
flow sequences are interrupted or irregular rather than joining disjoint motion.

Invalid/nonfinite pixels are excluded, never converted to black. Supported inputs
are uint8 or floating grayscale/RGB/RGBA intensity images; depth/integer formats
outside uint8 need separate metrics. The LeRobot v3 reader now raises on missing
requested video frames rather than substituting black/nearby/last-frame images.
Other readers that synthesize images can still hide failures before analysis.

Whole-image thresholds remain heuristics. Downscaling, image texture, dark
backgrounds, noise, and camera placement affect them. Calibrate per camera and
analysis resolution; select cameras with `--video-cameras`. These changes do not
add task-region masks, learned visibility, or camera/proprioception clock alignment.

## Reports and viewer

Episode JSON retains the nested `video` result: per-camera metrics, flags,
events, configuration, timebase provenance, and availability. Flat `video_*`
rollups remain for compatibility. Older flat reports load as `video-v1` without
invented camera detail. `video_min_sharpness` now means the true minimum sampled
sharpness across cameras, rather than the minimum camera-average value; recalibrate
thresholds that depended on the old meaning.

The viewer shows camera integrity summaries, camera-filtered event cards and
lanes, and click-to-seek playback. Missing time is labeled explicitly. A failed
image request displays an unavailable-frame placeholder. Video diagnostics do
not contribute to the numeric proprioception score.

Existing worst-camera defect rollups remain, with their aggregation documented
in provenance. The `no_motion` rollup requires evaluable motion for every selected
camera, and still describes only the analyzed sample. Review camera relevance
and coverage before filtering episodes.

## Performance

See the [CPU benchmark and reproduction command](../../../docs/reports/README.md#video-cpu-benchmark).
