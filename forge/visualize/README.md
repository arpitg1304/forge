# Web dataset explorer

Open the general viewer with `forge visualize ./dataset --backend web`.
For localized quality review, run `forge quality ./dataset --visualize`.
Both use the same interface; quality panels appear only when a report is supplied.

## Explore

- Search episode IDs and task descriptions. Quality mode can filter episodes with review events or flags.
- Camera cards show camera names, preserve aspect ratio, and support focus/restore and fullscreen. Narrow screens provide a horizontally scrollable camera strip.
- Play, adjust speed, step frames, jump to a frame, or loop the current episode. Frame entry is one-based; exported event indices remain zero-based.
- Plot all action/state channels, toggle individual channels, and inspect current values. Click a plot to seek. Per-channel normalization changes display only; metrics are unchanged.
- Task descriptions, recording timestamps, and per-episode camera availability come from the dataset. Relative timestamp labels use stored timestamps when available; playback follows episode FPS.

## Review

The inspector shows the heuristic score, pause fraction, episode flags, and event cards. Cards explain why a span is worth reviewing without declaring task failure. Filter event types or navigate between events with the arrow buttons. Click a card or highlighted interval to seek to its start.

Direction reversals are descriptive and hidden by default. Enable them to show blue intervals. Amber intervals are review candidates, red intervals are tracking-stall candidates, and gray intervals indicate timestamp gaps. Green regions mean no **visible** event; hidden event types and unavailable metrics still matter. Metric availability is expandable in the inspector.

## Keyboard

| Key | Action |
|---|---|
| Space | Play/pause |
| Left / right | Step one frame |
| Up / down | Switch episode |
| N / P | Next / previous visible event |
| [ / ] | Change playback speed |

Keyboard shortcuts are inactive while editing inputs or using native controls. The `?` button opens the shortcut guide.

## Verification

Backend and CLI tests: `pytest tests/test_quality_visualize.py`.

Optional browser integration (requires Chrome and `websocket-client` in the test environment):

```bash
FORGE_TEST_CHROME=/path/to/chrome pytest tests/test_web_viewer_browser.py
```

The browser test uses a temporary profile and a local synthetic fixture. It covers quality and normal viewing, episode search, event seeking/filtering, camera focus, all-channel plots, normalization, playback, rapid episode changes, and mobile overflow.

## Video integrity review

Add `--video` to `forge quality ... --visualize` to include camera summaries and
video events. The inspector separates absent camera references, decode failures,
invalid images, intentional sampling, and analysis caps. Camera/source filtering
also applies to timeline lanes, event navigation, and plot overlays. Video gap
estimates use episode timestamps and declared FPS, not independent camera PTS;
unknown timing remains explicit. Camera decode errors display an unavailable-frame
placeholder. See [the video quality guide](../quality/video/README.md).
