# Quality metrics

Analyze robot states, actions, and timestamps with NumPy. Video checks are optional
and separate from the proprioception score.

## Quick start

```bash
forge quality ./dataset --visualize --export quality.json
forge quality aloha_sim_transfer_cube --sample 5 --video --visualize
forge quality ./dataset --gripper-dims 6,13 --action-bounds=-1,1
```

```python
from forge.quality import QualityAnalyzer

analyzer = QualityAnalyzer()  # No inferred gripper channel or physical limits.
report = analyzer.analyze_dataset("./dataset")
report.to_json("quality.json")

result = analyzer.analyze_episode_arrays(
    "episode_0", actions=actions, states=states, timestamps=timestamps, fps=30.0
)
print(result.overall_score)  # A heuristic score, or None when unavailable.
print(result.events)
```

## Proprioception diagnostics and visual review (q-v2-proprio)

```bash
forge quality ./my_dataset --sample 20 --visualize --export quality.json

# Only for datasets with known absolute position actions and matching state units:
forge quality ./my_dataset --action-semantics position \
  --tracking-pairs 0:0,1:1 --tracking-error 0.05 --tracking-lag 0.1 \
  --pause-speed 0.001 --event-min-duration 0.5 --visualize

# Explicit grippers for a known bimanual layout:
forge quality ./my_dataset --gripper-dims 6,13
```

`--visualize` opens the existing local web viewer after scoring. It displays the analyzed episodes, event lanes, shaded action/state plots, episode flags, and metric availability. Click a span or choose an event to seek playback. The command serves the viewer until Ctrl+C. Green means **no event detected by enabled checks**, not task success; amber marks review candidates, red tracking-stall candidates, and gray unavailable data/gaps. Direction reversals are hidden by default; enable the checkbox to show these descriptive events in blue. Episode-wide flags without localized evidence remain listed separately.

| Diagnostic | Interpretation |
|---|---|
| `held_command_fraction` / `held_command` | Sustained adjacent repeated commands, measured over observed duration |
| `static` / `motion_pause` | Sustained low measured speed, with frame and time intervals |
| `direction_reversal_count` / `direction_reversal` | Sustained direction changes per selected state channel; descriptive, not a retry classifier |
| `path_length_scaled`, `net_to_path_ratio` | Travel and endpoint displacement divided by travel; closed-loop tasks can naturally have a low ratio |
| `tracking_error_mean`, `tracking_error_p95` | Maximum mapped command/state residual at each sample, with optional causal lag |
| `tracking_stall_fraction` / `tracking_stall_candidate` | Low measured speed while mapped absolute-position commands remain far away; contact can cause this too |
| `valid_duration_s`, `time_coverage`, `timestamp_gap` | Observed time support; events never bridge long recording gaps |

Motion metrics assume **position** state columns. Select them with `--state-dims`; `--state-scales` supplies a positive divisor for every original state column, including unselected columns. Default scales are 1, so thresholds use native units (and native units/second for speed). For mixed units, supply meaningful scales before comparing episodes. Tracking pairs require compatible action/state units and explicit `position` semantics; vector shape does not establish correspondence. Tune duration, speed, error, and lag thresholds to the robot and task.

Scores are heuristics, not validated usefulness or success grades. Invalid/nonfinite data, nonmonotonic timestamps, irregular timebases, or degenerate measured motion can produce `overall_score: null`. Constant measured states never fall back to moving commands for smoothness. Missing gripper mappings and hardware limits are omitted from their subscores. Changing available subscores changes score comparability; **q-v1 thresholds should be recalibrated**.

JSON exports preserve `scorer_version`, configuration/timebase `provenance`, `metric_status`, raw `diagnostics`, static/timestamp results, and localized `events`. Each event has an inclusive `start_frame`, exclusive `end_frame`, timestamps, and affected state channels where applicable. Event duration is the difference between boundary timestamps, not the frame count divided by FPS. Legacy reports can still be loaded and are marked q-v1. Catalog scoring uses a new version; detailed events live in report exports.

Retries, task progress, contact success, and failure grades need task context or additional evidence. These checks intentionally expose motion candidates for review. Video checks are documented separately.

## Measurements and scoring

| Measurement | Meaning | Included in score? |
|---|---|---|
| LDLJ smoothness | Position jerk normalized by duration and peak speed | Yes |
| Dead-action fraction | Zero or adjacent repeated commands; holds can be intentional | Yes |
| Gripper chatter | Rapid transitions on explicitly configured action channels | When configured |
| Static fraction | Sustained low measured speed over valid observed time | When available |
| Timestamp regularity | Jitter and gaps against expected FPS | When available |
| Action saturation | Occupancy near supplied physical command limits | When configured |
| Action entropy | Marginal action-distribution entropy | Yes |
| Path length | Travel in selected, scaled position channels | No |
| SPARC | Frequency-based smoothness | No |
| PSD bands | Action energy in configured frequency bands | No; can flag high-band activity |
| State/action consistency | Raw action variance among nearby within-episode states | No |

The composite normalizes weights over available components. Default weights are
0.25 for smoothness; 0.15 each for dead actions, gripper health, and static fraction;
and 0.10 each for timing, saturation, and entropy. Configure them with `QualityConfig`.
Missing components and robot-specific thresholds limit comparisons between scores.
Path length, entropy, and consistency do not establish task progress or success.

## Compatibility

- Gripper detection is disabled by default. Set `gripper_dim` or `gripper_dims`.
- Without physical limits, observed action-range occupancy is descriptive only.
- Motion requires position-state columns. Command smoothness is a fallback only
  when states are absent and `action_semantics="position"` is explicit.
- Invalid data retains its frame alignment and is not replaced with zeros.
- Scores can be null. Consumers must handle unavailable results.
- Legacy JSON remains readable. Recalibrate thresholds when moving from q-v1.
- Video JSON keeps per-camera details and legacy flat fields. See the
  [video compatibility notes](video/README.md#reports-and-viewer).

## Development

`config.py` defines thresholds; `metrics.py` contains scalar measurements;
`diagnostics.py` produces temporal events. `analyzer.py` coordinates them, and
`models.py` handles report serialization.

For a new check, add configuration, result fields, availability, and provenance.
Add frame/time evidence when localizable. Keep descriptive attributes outside the
composite unless their interpretation is justified. Test valid, invalid, missing,
and degenerate inputs, plus serialization and CLI integration where applicable.

See [validation and benchmarks](../../docs/reports/README.md) and the
[viewer guide](../visualize/README.md).
