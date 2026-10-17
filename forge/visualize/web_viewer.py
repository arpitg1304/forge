"""Web-based dataset visualizer.

Zero-dependency viewer (stdlib only) that serves frames over HTTP
and renders everything in the browser. Supports segment overlay
with phase labels.

Usage:
    forge visualize ./dataset --backend web
    forge visualize pusht --backend web --segment
"""

from __future__ import annotations

import io
import json
import logging
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.models import Episode, Frame
from forge.formats.registry import FormatRegistry

logger = logging.getLogger(__name__)


def _encode_jpeg(img_rgb: np.ndarray) -> bytes:
    """Encode RGB uint8 array to JPEG bytes.

    Tries cv2 first, then PIL, then falls back to BMP.
    """
    # Try OpenCV
    try:
        import cv2

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            return buf.tobytes()
    except ImportError:
        pass

    # Try PIL
    try:
        from PIL import Image

        pil_img = Image.fromarray(img_rgb)
        bio = io.BytesIO()
        pil_img.save(bio, format="JPEG", quality=85)
        return bio.getvalue()
    except ImportError:
        pass

    # Fallback: BMP (no dependencies, larger)
    h, w = img_rgb.shape[:2]
    row_size = (w * 3 + 3) & ~3
    padding = row_size - w * 3
    pixel_size = row_size * h
    file_size = 54 + pixel_size

    header = bytearray(54)
    header[0:2] = b"BM"
    header[2:6] = file_size.to_bytes(4, "little")
    header[10:14] = (54).to_bytes(4, "little")
    header[14:18] = (40).to_bytes(4, "little")
    header[18:22] = w.to_bytes(4, "little")
    header[22:26] = h.to_bytes(4, "little")
    header[26:28] = (1).to_bytes(2, "little")
    header[28:30] = (24).to_bytes(2, "little")
    header[34:38] = pixel_size.to_bytes(4, "little")

    pixels = bytearray()
    pad_bytes = b"\x00" * padding
    for y in range(h - 1, -1, -1):
        row = img_rgb[y]
        for x in range(w):
            r, g, b = row[x]
            pixels.extend([b, g, r])
        pixels.extend(pad_bytes)

    return bytes(header) + bytes(pixels)


class WebBackend:
    """Loads dataset data and serves it to the web frontend.

    Only extracts lightweight metadata (actions, states, camera keys)
    on init. Frame images are loaded lazily on demand.
    """

    def __init__(self, dataset_path: Path, max_episodes: int = 50, quality_report=None):
        self._quality = (
            {eq.episode_id: eq for eq in quality_report.per_episode}
            if quality_report is not None
            else {}
        )
        self._quality_mode = quality_report is not None
        self.dataset_path = Path(dataset_path)
        self.format_name = FormatRegistry.detect_format(dataset_path)
        self.reader = FormatRegistry.get_reader(self.format_name)
        self.info = self.reader.inspect(dataset_path)

        self._episode_objects: list[Episode] = []
        self._episode_lengths: list[int] = []
        self._camera_keys: list[str] = []
        self._episode_actions: list[np.ndarray | None] = []
        self._episode_states: list[np.ndarray | None] = []
        self._episode_timestamps: list[list[float | None]] = []
        self._episode_cameras: list[list[str]] = []
        self._segmentations: list[dict | None] = []

        # Cache for materialized frames (populated lazily)
        self._frame_cache: dict[int, list[Frame]] = {}

        print(f"Loading episodes from {self.format_name} dataset...")

        for i, episode in enumerate(self.reader.read_episodes(dataset_path)):
            if len(self._episode_objects) >= max_episodes:
                break
            if self._quality_mode and episode.episode_id not in self._quality:
                continue
            if (i + 1) % 10 == 0:
                print(f"  Loaded {i + 1} episodes...")

            self._episode_objects.append(episode)

            # Extract only actions/states — iterate frames without storing them
            actions = []
            states = []
            timestamps = []
            cameras = set()
            frame_count = 0
            for frame in episode.frames():
                frame_count += 1
                actions.append(frame.action)
                states.append(frame.state)
                timestamps.append(
                    float(frame.timestamp)
                    if frame.timestamp is not None and np.isfinite(frame.timestamp)
                    else None
                )
                cameras.update(frame.images)

            self._episode_timestamps.append(timestamps)
            self._episode_cameras.append(sorted(cameras))
            self._camera_keys = sorted(set(self._camera_keys) | cameras)

            def aligned(values):
                template = next((v for v in values if v is not None), None)
                if template is None:
                    return None
                return np.array(
                    [
                        v if v is not None else np.full_like(template, np.nan, dtype=float)
                        for v in values
                    ]
                )

            self._episode_lengths.append(frame_count)
            self._episode_actions.append(aligned(actions))
            self._episode_states.append(aligned(states))
            self._segmentations.append(None)

        print(f"Loaded {len(self._episode_objects)} episodes")

    def set_segmentation(self, episode_idx: int, segments_data: list[dict]) -> None:
        """Set segmentation data for an episode."""
        if episode_idx < len(self._segmentations):
            self._segmentations[episode_idx] = segments_data

    def _get_frames(self, episode_idx: int) -> list[Frame]:
        """Lazily materialize and cache frames for an episode."""
        if episode_idx not in self._frame_cache:
            episode = self._episode_objects[episode_idx]
            self._frame_cache[episode_idx] = list(episode.frames())
        return self._frame_cache[episode_idx]

    def get_num_episodes(self) -> int:
        return len(self._episode_objects)

    def get_episode_length(self, episode_idx: int) -> int:
        if episode_idx >= len(self._episode_lengths):
            return 0
        return self._episode_lengths[episode_idx]

    def get_frame_image(
        self, episode_idx: int, frame_idx: int, camera_key: str
    ) -> np.ndarray | None:
        if episode_idx >= len(self._episode_objects):
            return None
        frames = self._get_frames(episode_idx)
        if frame_idx >= len(frames):
            return None
        frame = frames[frame_idx]
        if camera_key not in frame.images:
            return None
        try:
            img = frame.images[camera_key].load()
        except Exception:
            return None
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def get_episode_actions(self, episode_idx: int) -> np.ndarray | None:
        if episode_idx >= len(self._episode_actions):
            return None
        return self._episode_actions[episode_idx]

    def get_episode_states(self, episode_idx: int) -> np.ndarray | None:
        if episode_idx >= len(self._episode_states):
            return None
        return self._episode_states[episode_idx]

    def get_segmentation(self, episode_idx: int) -> list[dict] | None:
        if episode_idx >= len(self._segmentations):
            return None
        return self._segmentations[episode_idx]

    def get_camera_keys(self) -> list[str]:
        return self._camera_keys

    def get_fps(self) -> float:
        return self.info.inferred_fps or 30.0

    def get_name(self) -> str:
        if self.dataset_path.parent.name == "snapshots":
            cache_name = self.dataset_path.parent.parent.name
            if cache_name.startswith("datasets--"):
                return cache_name.removeprefix("datasets--").replace("--", "/")
        return self.dataset_path.name

    def get_info_dict(self) -> dict:
        return {
            "name": self.get_name(),
            "format": self.format_name,
            "num_episodes": self.get_num_episodes(),
            "cameras": self.get_camera_keys(),
            "fps": self.get_fps(),
            "quality_mode": self._quality_mode,
            "episode_ids": [ep.episode_id for ep in self._episode_objects],
            "episodes": [
                {
                    "id": ep.episode_id,
                    "num_frames": self._episode_lengths[i],
                    "fps": ep.fps or self.get_fps(),
                    "task": ep.language_instruction,
                    "score": self._quality[ep.episode_id].overall_score
                    if ep.episode_id in self._quality
                    else None,
                    "flags": self._quality[ep.episode_id].flags
                    if ep.episode_id in self._quality
                    else [],
                    "event_count": sum(
                        e.kind != "direction_reversal" for e in self._quality[ep.episode_id].events
                    )
                    + (
                        sum(
                            len(c.events)
                            for c in self._quality[ep.episode_id].video.per_camera.values()
                        )
                        if self._quality[ep.episode_id].video
                        else 0
                    )
                    if ep.episode_id in self._quality
                    else 0,
                }
                for i, ep in enumerate(self._episode_objects)
            ],
            "total_frames": sum(self.get_episode_length(i) for i in range(self.get_num_episodes())),
        }

    def get_episode_dict(self, episode_idx: int) -> dict:
        num_frames = self.get_episode_length(episode_idx)

        actions = self.get_episode_actions(episode_idx)
        states = self.get_episode_states(episode_idx)

        result: dict[str, Any] = {
            "num_frames": num_frames,
            "episode_id": self._episode_objects[episode_idx].episode_id,
            "task": self._episode_objects[episode_idx].language_instruction,
            "timestamps": self._episode_timestamps[episode_idx],
            "cameras": self._episode_cameras[episode_idx],
            "actions": None,
            "states": None,
            "segments": self.get_segmentation(episode_idx),
            "quality": self._quality[self._episode_objects[episode_idx].episode_id].to_dict()
            if self._episode_objects[episode_idx].episode_id in self._quality
            else None,
            "fps": self._episode_objects[episode_idx].fps or self.get_fps(),
        }

        if actions is not None and actions.size > 0:
            a = actions
            result["actions"] = np.where(
                np.isfinite(a), np.round(a, 4).astype(object), None
            ).tolist()

        if states is not None and states.size > 0:
            s = states
            result["states"] = np.where(
                np.isfinite(s), np.round(s, 4).astype(object), None
            ).tolist()

        return result


def _make_handler(backend: WebBackend):
    """Create an HTTP request handler class with access to the backend."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Suppress request logs

        def _send_json(self, data: Any) -> None:
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_image(self, img_bytes: bytes, content_type: str = "image/jpeg") -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(img_bytes)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(img_bytes)

        def _send_404(self) -> None:
            self.send_response(404)
            self.end_headers()

        def do_GET(self) -> None:
            try:
                self._route()
            except BrokenPipeError:
                pass
            except ConnectionResetError:
                pass

        def _route(self) -> None:
            path = self.path

            if path == "/":
                self._send_html(HTML_TEMPLATE)

            elif path == "/api/info":
                self._send_json(backend.get_info_dict())

            elif path.startswith("/api/episode/"):
                try:
                    idx = int(path.split("/")[3])
                    self._send_json(backend.get_episode_dict(idx))
                except (IndexError, ValueError):
                    self._send_404()

            elif path.startswith("/frame/"):
                parts = path.split("/")
                # /frame/<episode>/<frame>/<camera...>
                if len(parts) >= 5:
                    try:
                        ep = int(parts[2])
                        fr = int(parts[3])
                        cam = "/".join(parts[4:])  # Camera key may contain /
                        img = backend.get_frame_image(ep, fr, cam)
                        if img is not None:
                            img_bytes = _encode_jpeg(img)
                            ct = "image/jpeg"
                            if img_bytes[:2] == b"BM":
                                ct = "image/bmp"
                            self._send_image(img_bytes, ct)
                        else:
                            self._send_404()
                    except (ValueError, IndexError):
                        self._send_404()
                else:
                    self._send_404()
            else:
                self._send_404()

    return Handler


class WebViewer:
    """Web-based dataset viewer.

    Starts a local HTTP server and opens the browser.

    Example:
        >>> viewer = WebViewer("path/to/dataset")
        >>> viewer.show()
    """

    def __init__(
        self,
        dataset_path: str | Path,
        max_episodes: int = 50,
        port: int = 0,
        segment: bool = False,
        quality_report=None,
    ):
        self.backend = WebBackend(Path(dataset_path), max_episodes, quality_report=quality_report)
        self.port = port

        if segment:
            self._run_segmentation()

    def _run_segmentation(self) -> None:
        """Run PELT segmentation with labels using already-extracted data."""
        try:
            from forge.segment import SegmentAnalyzer, SegmentConfig
            from forge.segment.labeler import PhaseLabeler
            from forge.segment.plot import PHASE_COLORS
        except ImportError:
            print("Segmentation requires ruptures: pip install forge-robotics[segment]")
            return

        print("Running segmentation with phase labels...")
        # Use l2 cost model for speed (rbf is ~20x slower on long episodes)
        config = SegmentConfig(label_phases=True, cost_model="l2")
        analyzer = SegmentAnalyzer(config=config)

        for i in range(self.backend.get_num_episodes()):
            try:
                # Use already-extracted states (avoids re-iterating frames)
                states = self.backend.get_episode_states(i)
                if states is None or states.size == 0:
                    continue

                es = analyzer.segment_episode_arrays(
                    episode_id=f"episode_{i}",
                    signal=states if states.ndim == 2 else states.reshape(-1, 1),
                    fps=self.backend.get_fps(),
                )

                # Label phases if configured
                if config.label_phases and es.segments:
                    labeler = PhaseLabeler()
                    labeler.label_segments(
                        es.segments, states if states.ndim == 2 else states.reshape(-1, 1)
                    )

                segments_data = []
                for seg in es.segments:
                    color = PHASE_COLORS.get(seg.label, "#bdbdbd")
                    segments_data.append(
                        {
                            "start": seg.start,
                            "end": seg.end,
                            "label": seg.label or "unknown",
                            "color": color,
                            "duration_frames": seg.duration_frames,
                        }
                    )
                self.backend.set_segmentation(i, segments_data)
            except Exception as e:
                logger.warning("Segmentation failed for episode %d: %s", i, e)

        print("Segmentation complete.")

    def show(self) -> None:
        """Start server and open browser."""
        handler = _make_handler(self.backend)

        # Find a free port
        if self.port == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                self.port = s.getsockname()[1]

        server = HTTPServer(("127.0.0.1", self.port), handler)
        url = f"http://127.0.0.1:{self.port}"

        print(f"\nForge Viewer running at: {url}")
        print("Press Ctrl+C to stop.\n")

        # Open browser after a short delay
        def open_browser():
            import time

            time.sleep(0.3)
            webbrowser.open(url)

        threading.Thread(target=open_browser, daemon=True).start()

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down viewer...")
            server.shutdown()


def web_visualize(
    dataset_path: str | Path,
    max_episodes: int = 50,
    segment: bool = False,
    port: int = 0,
    quality_report=None,
) -> None:
    """Visualize a dataset in the browser.

    Args:
        dataset_path: Path to dataset (any supported format).
        max_episodes: Maximum episodes to load.
        segment: Run segmentation and show phase overlay.
        port: Server port (0 = auto).
    """
    viewer = WebViewer(
        dataset_path,
        max_episodes=max_episodes,
        segment=segment,
        port=port,
        quality_report=quality_report,
    )
    viewer.show()


# ============================================================
# HTML Template
# ============================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forge · Dataset explorer</title>
<style>
:root{color-scheme:dark;--bg:#0d1117;--panel:#131a23;--raised:#1b2531;--border:#283342;--text:#e7edf5;--muted:#95a5b9;--accent:#8cb9ff;--green:#69c5a2;--amber:#ecb968;--red:#f08484}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:13px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}button,input,select{font:inherit}button,select,input[type=search],input[type=number]{color:var(--text);background:var(--raised);border:1px solid var(--border);border-radius:7px;padding:7px 10px}button{cursor:pointer}button:hover{border-color:var(--accent);background:#223248}button:disabled{opacity:.4;cursor:default}button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:3px}input[type=checkbox],input[type=range]{accent-color:var(--accent)}[hidden]{display:none!important}h1,h2,h3,p{margin:0}h1{font-size:15px;font-weight:600}h2{font-size:14px}h3{font-size:12px;font-weight:600}small,.muted{color:var(--muted)}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}.eyebrow{font-size:10px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted);font-weight:600}.badge{padding:3px 7px;border:1px solid var(--border);border-radius:5px;font-size:10px;color:var(--muted);white-space:nowrap}.row{display:flex;align-items:center;gap:10px}.spread{justify-content:space-between}.stack{display:flex;flex-direction:column;gap:12px}.empty{padding:28px 12px;text-align:center;color:var(--muted)}
header{height:66px;display:flex;align-items:center;justify-content:space-between;padding:0 22px;border-bottom:1px solid var(--border);gap:16px}.brand{font-size:19px;font-weight:800;letter-spacing:-.06em;margin-right:18px}.brand i{font-style:normal;color:var(--accent)}.dataset-heading{min-width:0}#dataset-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}#dataset-meta{font-size:11px;color:var(--muted)}
#app{display:grid;grid-template-columns:215px minmax(0,1fr) 285px;height:calc(100vh - 66px)}#app.no-quality{grid-template-columns:215px minmax(0,1fr)}#episodes-panel{border-right:1px solid var(--border);display:flex;flex-direction:column;min-height:0;padding:20px 12px 12px;gap:12px}#episode-search{width:100%}#episode-filter{width:100%;font-size:11px}#episode-list{overflow:auto;min-height:0;flex:1;display:flex;flex-direction:column;gap:5px}.episode-item{text-align:left;padding:10px;background:transparent;border:1px solid transparent;width:100%;flex-shrink:0}.episode-item[aria-current=true]{background:#1c2b40;border-color:#3c5d85}.episode-item .row{justify-content:space-between}.episode-item small{font-size:10px}.event-dot{color:var(--amber);font-size:10px}.sidebar-footer{font-size:10px;color:var(--muted);padding:10px 3px;border-top:1px solid var(--border)}
main{overflow:auto;padding:20px;min-width:0;display:flex;flex-direction:column;gap:16px}.workspace-heading{align-items:flex-start}#episode-title{font-size:19px;letter-spacing:-.025em}#task-label{color:var(--muted);font-size:12px;max-width:70ch;margin-top:3px}.panel{border:1px solid var(--border);border-radius:10px;background:var(--panel);overflow:hidden;flex-shrink:0}.panel-heading{padding:12px 14px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;gap:10px}.panel-heading small{font-size:10px}.camera-tools{display:flex;gap:6px}.camera-tools button{font-size:10px;padding:4px 8px}#cameras{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;padding:10px;background:#090d12}.camera-card{position:relative;min-width:0;border:1px solid var(--border);border-radius:7px;overflow:hidden;background:#06090d}.camera-card img{display:block;width:100%;height:auto;aspect-ratio:16/10;object-fit:contain}.camera-caption{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:7px 9px;background:#141c26;font-size:10px;color:#b2c1d2}.camera-caption span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.camera-caption button{font-size:10px;padding:2px 6px}.camera-card.focused{grid-column:1/-1}.camera-card.focused img{height:clamp(250px,43vh,550px)}#cameras:has(.focused) .camera-card:not(.focused){display:none}.camera-error{position:absolute;top:35%;left:0;right:0;text-align:center;color:var(--muted);font-size:11px;background:#0d1117cc;padding:8px}
#controls-bar{padding:10px 12px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}#play-btn{background:#aacdff;color:#10223c;border-color:#aacdff;font-weight:650;min-width:66px}#speed-select{font-size:11px;padding:7px;max-width:74px}#timeline-wrap{flex:1;min-width:150px;position:relative;display:flex;align-items:center;height:26px}#scrubber{width:100%;margin:0;position:relative;z-index:2;background:transparent}#segment-overlay{position:absolute;left:0;right:0;height:6px;overflow:hidden;opacity:.6}.seg{position:absolute;height:100%}.time-code{font-size:11px;white-space:nowrap}#frame-counter{width:100%;font-size:10px;text-align:right;color:var(--muted)}#frame-jump{width:72px;padding:4px;font-size:11px}.playback-footer{border-top:1px solid var(--border);padding:7px 12px;display:flex;align-items:center;justify-content:space-between;gap:10px;color:var(--muted);font-size:10px}.playback-footer label{display:flex;align-items:center;gap:6px}
#quality-panel{padding:0}#quality-timelines{padding:12px 14px 4px}.quality-row{display:flex;gap:10px;align-items:center;margin-bottom:10px}.quality-row-label{width:115px;font-size:10px;color:var(--muted);flex-shrink:0}.quality-track{height:20px;flex:1;position:relative;background:#1c302d;border-radius:4px;cursor:crosshair}.quality-span{position:absolute;height:100%;min-width:3px;border:0;border-radius:3px;padding:0;opacity:.85}.quality-span:hover{opacity:1;outline:1px solid white}.quality-cursor{position:absolute;width:1px;height:100%;background:#fff;pointer-events:none}.timeline-ruler{margin:0 14px 10px 139px;display:flex;justify-content:space-between;font-size:9px;color:var(--muted)}#quality-current{padding:9px 14px;background:#101720;font-size:11px;border-top:1px solid var(--border);min-height:36px}.legend{font-size:10px;display:flex;gap:12px;color:var(--muted);padding:0 14px 12px;flex-wrap:wrap}.legend i{width:6px;height:6px;border-radius:50%;display:inline-block;margin-right:4px}#quality-show-reversals{vertical-align:middle}.compact-label{font-size:10px;color:var(--muted)}
#signals-heading{display:flex;align-items:center;justify-content:space-between}#charts{display:grid;grid-template-columns:1fr;gap:12px}.signal-panel{border:1px solid var(--border);border-radius:9px;background:var(--panel);overflow:hidden}.signal-heading{padding:9px 12px;display:flex;align-items:center;justify-content:space-between}.signal-heading select{font-size:10px;padding:3px 6px}.chart-container{height:155px;position:relative;margin:0 8px}.chart-container canvas{width:100%;height:100%;display:block;cursor:crosshair}.marker-canvas{position:absolute;inset:0;pointer-events:none}.signal-legend{display:flex;flex-wrap:wrap;gap:5px;padding:5px 12px 10px}.channel-chip{font-size:10px;padding:3px 6px;background:transparent;border-color:var(--border)}.channel-chip[aria-pressed=false]{opacity:.4}.channel-chip i{display:inline-block;width:5px;height:5px;border-radius:50%;margin-right:4px}.channel-chip span{font-family:ui-monospace,monospace;color:var(--muted);margin-left:5px}.signal-hint{font-size:10px;color:var(--muted)}#segment-legend{display:flex;gap:10px;flex-wrap:wrap;font-size:10px;color:var(--muted)}
#inspector{border-left:1px solid var(--border);overflow:auto;padding:20px 16px;display:flex;flex-direction:column;gap:16px}#quality-summary{font-size:11px;color:var(--muted)}.score-card{padding:16px;background:linear-gradient(130deg,#1d2b3d,#151e2a);border:1px solid #32445b;border-radius:10px}.score-number{font-size:34px;letter-spacing:-.045em;line-height:1.25}.score-number small{font-size:12px;letter-spacing:0}.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:14px}.metric-grid strong{font-size:17px;display:block;font-weight:500}.metric-grid small{font-size:10px}.flags{display:flex;gap:5px;flex-wrap:wrap}.flag{font-size:10px;border:1px solid #614f34;background:#342b1e;color:#edc886;border-radius:4px;padding:3px 6px}.event-tools{display:flex;gap:5px}.event-tools button{padding:4px 8px;font-size:11px}#event-filter{width:100%;font-size:11px}.event-card{width:100%;text-align:left;background:#171f2a;border:1px solid var(--border);border-left:2px solid var(--amber);border-radius:6px;padding:10px;margin-bottom:7px}.event-card.active{background:#2b2a23;border-color:#ad854a}.event-card strong{font-size:11px;font-weight:550;display:block}.event-card small{font-size:10px}.event-card p{font-size:10px;color:var(--muted);margin-top:5px}#event-list{max-height:390px;overflow:auto}.inspector-note{font-size:10px;color:var(--muted);line-height:1.6}.detail-block{border-top:1px solid var(--border);padding-top:12px}summary{cursor:pointer;font-size:11px;color:var(--muted)}#quality-status{font:10px/1.7 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;margin-top:10px}#loading{position:fixed;inset:0;z-index:100;background:var(--bg);display:flex;align-items:center;justify-content:center;flex-direction:column;gap:12px}.spinner{width:24px;height:24px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}#error-banner{background:#402329;border:1px solid #76404a;padding:12px;border-radius:8px}#help-dialog{max-width:440px;background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:12px;padding:24px}#help-dialog::backdrop{background:#0009}#help-dialog p{margin:12px 0;color:var(--muted)}kbd{font:11px ui-monospace,monospace;padding:2px 5px;border:1px solid var(--border);border-radius:4px}#episode-select{max-width:190px;font-size:11px}.mobile-only{display:none}
@media(min-width:1400px){#charts{grid-template-columns:1fr 1fr}.chart-container{height:190px}}
@media(max-width:1180px){#app{grid-template-columns:175px minmax(0,1fr) 250px}#app.no-quality{grid-template-columns:175px minmax(0,1fr)}main{padding:14px}#inspector{padding:16px 12px}.quality-row-label{width:95px}.timeline-ruler{margin-left:119px}.camera-card img{height:160px}#dataset-meta{display:none}}
@media(max-width:900px){#app,#app.no-quality{grid-template-columns:minmax(0,1fr);height:auto}#episodes-panel{display:none}main{overflow:visible}#inspector{border:0;border-top:1px solid var(--border);overflow:visible;padding:20px}#event-list{max-height:300px}header{height:60px;padding:0 14px}.brand{margin-right:4px}.workspace-heading{flex-wrap:wrap}.camera-card img{height:180px}#charts{grid-template-columns:1fr}.mobile-only{display:block}#quality-panel .panel-heading{flex-wrap:wrap}}
@media(max-width:600px){#cameras{display:flex;overflow-x:auto;scroll-snap-type:x mandatory}.camera-card{flex:0 0 88%;scroll-snap-align:start}.camera-card img{height:180px}.camera-card.focused{flex-basis:100%}.camera-card.focused img{height:250px}.signal-hint{max-width:155px;text-align:right}}
@media(prefers-reduced-motion:reduce){.spinner{animation:none}}
</style>
</head>
<body>
<div id="loading"><div class="spinner"></div><span id="loading-text">Opening dataset…</span></div>
<header><div class="row" style="min-width:0"><span class="brand">forge<i>.</i></span><div class="dataset-heading"><div class="eyebrow">Dataset explorer</div><h1 id="dataset-name">Dataset</h1></div><span class="badge" id="format-badge"></span></div><div class="row"><span id="dataset-meta"></span><button id="help-btn" aria-label="Keyboard shortcuts">?</button></div></header>
<div id="app">
<aside id="episodes-panel" aria-label="Episodes"><div class="row spread"><h2>Episodes</h2><span class="badge" id="episode-count"></span></div><input type="search" id="episode-search" placeholder="Find episode or task…" aria-label="Search episodes"><select id="episode-filter" aria-label="Filter episodes"><option value="all">All episodes</option><option value="events">With review events</option><option value="flags">With quality flags</option></select><div id="episode-list"></div><div class="sidebar-footer"><kbd>↑</kbd> <kbd>↓</kbd> Switch episodes<br>Review events are candidates, not verdicts.</div></aside>
<main>
<div id="error-banner" role="alert" hidden></div>
<div class="workspace-heading row spread"><div><div class="eyebrow">Episode workspace</div><h2 id="episode-title">Loading…</h2><p id="task-label"></p></div><div class="row"><button id="prev-episode" aria-label="Previous episode">←</button><select id="episode-select" aria-label="Episode"></select><button id="next-episode" aria-label="Next episode">→</button></div></div>
<section class="panel" aria-label="Camera playback"><div class="panel-heading"><div class="row"><h2>Camera views</h2><span class="badge" id="camera-count"></span></div><div class="camera-tools"><button id="camera-reset">Show all views</button><button id="fullscreen-btn">Fullscreen</button></div></div><div id="cameras"></div><div id="controls-bar"><button id="step-back" aria-label="Previous frame">‹</button><button id="play-btn">Play</button><button id="step-forward" aria-label="Next frame">›</button><select id="speed-select" aria-label="Playback speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select><div id="timeline-wrap"><div id="segment-overlay"></div><input type="range" id="scrubber" min="0" max="1" value="0" aria-label="Playback position"></div><span class="time-code mono" id="time-code"></span></div><div class="playback-footer"><label><input type="checkbox" id="loop-playback"> Loop episode</label><span id="current-phase"></span><label>Frame <input id="frame-jump" type="number" min="1" value="1" aria-label="Jump to frame"><span id="frame-counter"></span></label></div></section>
<div id="segment-legend"></div>
<section id="quality-panel" class="panel" hidden aria-label="Quality timeline"><div class="panel-heading"><h2>Event timeline</h2><label class="compact-label"><input type="checkbox" id="quality-show-reversals"> Include direction reversals</label></div><div id="quality-timelines"></div><div class="timeline-ruler" id="timeline-ruler"></div><div class="legend"><span><i style="background:#69c5a2"></i>No visible event</span><span><i style="background:#ecb968"></i>Review</span><span><i style="background:#f08484"></i>Tracking</span><span><i style="background:#8cb9ff"></i>Descriptive</span></div><p id="quality-current" aria-live="polite"></p></section>
<div id="signals-heading"><h2>Signals</h2><span class="signal-hint">Click a plot to seek · toggle channels below</span></div>
<section id="charts" aria-label="Signal plots"><div class="signal-panel"><div class="signal-heading"><h3>Actions</h3><select id="action-scale" aria-label="Action plot scale"><option value="raw">Native units</option><option value="normalized">Normalize each channel</option></select></div><div class="chart-container"><canvas id="action-chart" aria-label="Action signals; click to seek"></canvas><canvas id="action-marker" class="marker-canvas"></canvas></div><div id="action-legend" class="signal-legend"></div></div><div class="signal-panel"><div class="signal-heading"><h3>States</h3><select id="state-scale" aria-label="State plot scale"><option value="raw">Native units</option><option value="normalized">Normalize each channel</option></select></div><div class="chart-container"><canvas id="state-chart" aria-label="State signals; click to seek"></canvas><canvas id="state-marker" class="marker-canvas"></canvas></div><div id="state-legend" class="signal-legend"></div></div></section>
</main>
<aside id="inspector" aria-label="Quality inspector"><div class="row spread"><h2>Quality review</h2><span class="badge" id="quality-modalities">Proprioception</span></div><div class="score-card"><div class="eyebrow">Heuristic score</div><div class="score-number"><span id="score-value">—</span> <small>/ 10</small></div><p id="quality-summary"></p><div class="metric-grid"><div><strong id="review-count">—</strong><small>Review events</small></div><div><strong id="pause-fraction">—</strong><small>Time paused</small></div></div></div><div><div class="eyebrow" style="margin-bottom:7px">Episode flags</div><div id="quality-flags" class="flags"></div></div><div id="video-camera-summary" class="stack" hidden></div><div class="stack"><div class="row spread"><h2>Events</h2><div class="event-tools"><button id="prev-event" aria-label="Previous event">↑</button><button id="next-event" aria-label="Next event">↓</button></div></div><select id="event-camera-filter" aria-label="Filter event camera" hidden><option value="all">All sources</option></select><select id="event-filter" aria-label="Filter event type"><option value="all">All event types</option></select><div id="event-list"></div></div><p class="inspector-note">Unmarked regions mean no event detected by the enabled checks. Scores and motion events do not establish task success.</p><details class="detail-block"><summary>Metric availability & assumptions</summary><pre id="quality-status"></pre></details></aside>
</div>
<dialog id="help-dialog"><h2>Move through your data</h2><p><kbd>Space</kbd> Play / pause<br><kbd>←</kbd> <kbd>→</kbd> Step one frame<br><kbd>↑</kbd> <kbd>↓</kbd> Previous / next episode<br><kbd>N</kbd> <kbd>P</kbd> Next / previous visible event<br><kbd>[</kbd> <kbd>]</kbd> Playback speed</p><p>Click a camera’s Focus button to inspect one view. Signal values use native dataset units; normalization changes the plot only.</p><form method="dialog"><button>Close</button></form></dialog>
<script>
const $=id=>document.getElementById(id);
const colors=['#8cb9ff','#69c5a2','#f08484','#ecb968','#c49cf2','#79c8dc','#e1a3ca','#cad18a','#91a2ef','#dd936b','#78b78c','#b8a7da','#c5cbd5','#5dc3ba'];
const state={ep:0,frame:0,numFrames:0,fps:30,playing:false,speed:1,info:null,epData:null,cameras:[],hidden:{action:new Set(),state:new Set()},loadToken:0,playToken:0,eventType:'all',focusCamera:null};
const friendly=s=>String(s).replaceAll('_',' ');
const seconds=n=>Number.isFinite(n)?`${n.toFixed(2)}s`:'—';
const element=(tag,cls,text)=>{const el=document.createElement(tag);if(cls)el.className=cls;if(text!==undefined)el.textContent=text;return el;};
function timeAt(i){const t=state.epData?.timestamps;return Number.isFinite(t?.[i])&&Number.isFinite(t?.[0])?t[i]-t[0]:i/state.fps;}
function eventTime(e){if(!Number.isFinite(e.start_time_s))return NaN;return Number.isFinite(state.epData?.timestamps?.[0])?e.start_time_s-state.epData.timestamps[0]:e.start_frame/state.fps;}
function allQualityEvents(){const q=state.epData?.quality;return [...(q?.events||[]),...Object.values(q?.video?.per_camera||{}).flatMap(c=>c.events||[])].sort((a,b)=>a.start_frame-b.start_frame);}
function qualityEvents(){return allQualityEvents().filter(e=>(e.kind!=='direction_reversal'||$('quality-show-reversals').checked)&&(state.eventType==='all'||state.eventType===e.kind)&&($('event-camera-filter').value==='all'||($('event-camera-filter').value==='proprio'?!e.camera:e.camera===$('event-camera-filter').value)));}
function qualityColor(kind){return kind==='tracking_stall_candidate'?'#f08484':(kind==='timestamp_gap'||kind==='video_frame_gap'||kind==='video_camera_missing'||kind==='video_decode_failure'||kind==='video_invalid_pixels'||kind==='video_invalid_timestamps')?'#95a5b9':kind==='direction_reversal'?'#8cb9ff':'#ecb968';}
function eventLabel(e){return `${e.camera?friendly(e.camera)+' · ':''}${friendly(e.kind)} · ${Number.isFinite(e.start_time_s)&&Number.isFinite(e.end_time_s)?seconds(eventTime(e))+'–'+seconds(eventTime(e)+e.end_time_s-e.start_time_s):'frames '+e.start_frame+'–'+(e.end_frame-1)+' (time unavailable)'}`;}
function stopPlayback(){state.playing=false;state.playToken++;$('play-btn').textContent='Play';}
function seek(frame){stopPlayback();state.frame=Math.max(0,Math.min(state.numFrames-1,Math.round(frame)));updateFrame();}
function seekQualityEvent(e){if(e){seek(e.start_frame);document.querySelector('main').scrollTop=0;}}
async function getJSON(url){const r=await fetch(url);if(!r.ok)throw Error(`Request failed (${r.status})`);return r.json();}
function showError(e){$('error-banner').hidden=false;$('error-banner').textContent=e.message||String(e);}
async function init(){
 try{
 state.info=await getJSON('/api/info');$('dataset-name').textContent=state.info.name;$('format-badge').textContent=state.info.format;
 $('dataset-meta').textContent=`${state.info.num_episodes} episodes · ${state.info.total_frames.toLocaleString()} frames`;
 $('app').classList.toggle('no-quality',!state.info.quality_mode);$('inspector').hidden=!state.info.quality_mode;$('episode-filter').hidden=!state.info.quality_mode;
 state.info.episodes ||= state.info.episode_ids.map(id=>({id}));$('episode-count').textContent=state.info.num_episodes;
 state.info.episodes.forEach((ep,i)=>{const o=element('option','',ep.id);o.value=i;$('episode-select').append(o);});
 $('episode-search').oninput=renderEpisodeList;$('episode-filter').onchange=renderEpisodeList;
 $('episode-select').onchange=e=>loadEpisode(Number(e.target.value));$('prev-episode').onclick=()=>loadEpisode(state.ep-1);$('next-episode').onclick=()=>loadEpisode(state.ep+1);
 $('play-btn').onclick=togglePlay;$('step-back').onclick=()=>seek(state.frame-1);$('step-forward').onclick=()=>seek(state.frame+1);
 $('speed-select').onchange=e=>{state.speed=Number(e.target.value);};$('scrubber').oninput=e=>seek(Number(e.target.value));$('frame-jump').onchange=e=>seek(Number(e.target.value)-1);
 $('camera-reset').onclick=()=>{state.focusCamera=null;renderCameras();updateFrame();};
 $('fullscreen-btn').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await $('cameras').requestFullscreen();}catch(e){showError(e);}};
 $('quality-show-reversals').onchange=()=>{state.eventType='all';renderQuality();drawCharts();updateQualityPosition();};
 $('event-camera-filter').onchange=()=>{renderQuality(false);drawCharts();updateQualityPosition();};
 $('event-filter').onchange=e=>{state.eventType=e.target.value;renderQuality(false);drawCharts();updateQualityPosition();};
 $('next-event').onclick=()=>jumpEvent(1);$('prev-event').onclick=()=>jumpEvent(-1);
 $('help-btn').onclick=()=>$('help-dialog').showModal();document.addEventListener('keydown',onKey);
 for(const type of ['action','state']){$(`${type}-scale`).onchange=()=>{drawCharts();drawChartMarkers();};$(`${type}-chart`).onclick=e=>{const p=e.target._chartParams;if(p)seek((e.offsetX-p.left)/p.width*(state.numFrames-1));};}
 renderEpisodeList();
 if(state.info.num_episodes)await loadEpisode(0);else{$('episode-title').textContent='No episodes';$('cameras').append(element('div','empty','This dataset contains no displayable episodes.'));$('play-btn').disabled=true;}
 }catch(e){showError(e);}finally{$('loading').hidden=true;}
}
function renderEpisodeList(){
 const list=$('episode-list');list.replaceChildren();const search=$('episode-search').value.toLowerCase(),filter=$('episode-filter').value;
 state.info.episodes.forEach((ep,i)=>{if(!`${ep.id} ${ep.task||''}`.toLowerCase().includes(search)||(filter==='events'&&!ep.event_count)||(filter==='flags'&&!ep.flags?.length))return;
 const btn=element('button','episode-item');btn.setAttribute('aria-current',String(i===state.ep));btn.onclick=()=>loadEpisode(i);
 const row=element('div','row');row.append(element('span','mono',(/^episode_\d+$/.test(ep.id)?`Ep. ${Number(ep.id.slice(8))}`:ep.id)));if(state.info.quality_mode)row.append(element('span','event-dot',ep.event_count?`${ep.event_count} ${ep.event_count===1?'event':'events'}`:'—'));btn.append(row);
 btn.append(element('small','',`${ep.num_frames??'—'} frames${Number.isFinite(ep.score)?' · '+ep.score.toFixed(1)+'/10':''}`));btn.title=ep.task||ep.id;list.append(btn);});
 if(!list.children.length)list.append(element('div','empty','No matching episodes.'));
}
async function loadEpisode(idx){
 if(idx<0||idx>=state.info.num_episodes)return;stopPlayback();const token=++state.loadToken;$('error-banner').hidden=true;
 try{const data=await getJSON(`/api/episode/${idx}`);if(token!==state.loadToken)return;
 state.ep=idx;state.epData=data;state.numFrames=data.num_frames;state.frame=0;state.fps=data.fps||state.info.fps||30;state.eventType='all';$('event-camera-filter').value='all';state.hidden={action:new Set(),state:new Set()};
 $('episode-select').value=idx;$('episode-title').textContent=data.episode_id||state.info.episode_ids[idx];$('task-label').textContent=data.task||'No task description provided';
 $('prev-episode').disabled=idx===0;$('next-episode').disabled=idx===state.info.num_episodes-1;$('play-btn').disabled=state.numFrames===0;
 $('scrubber').max=Math.max(0,state.numFrames-1);$('frame-jump').max=state.numFrames;
 state.cameras=data.cameras||state.info.cameras||[];renderEpisodeList();renderCameras();renderSegments();renderQuality();renderSignalLegends();drawCharts();updateFrame();
 }catch(e){if(token===state.loadToken)showError(e);}
}
function renderCameras(){
 const root=$('cameras');root.replaceChildren();$('camera-count').textContent=`${state.cameras.length} views`;
 if(!state.cameras.includes(state.focusCamera))state.focusCamera=null;
 if(!state.cameras.length){root.append(element('div','empty','No camera data · explore the recorded signals below'));return;}
 state.cameras.forEach((cam,i)=>{const card=element('div','camera-card');card.classList.toggle('focused',state.focusCamera===cam);
 const img=element('img');img.id=`camera-${i}`;img.alt=friendly(cam);img.draggable=false;const err=element('div','camera-error','Frame unavailable');err.hidden=true;
 img.onerror=()=>{err.hidden=false;};img.onload=()=>{err.hidden=true;};const caption=element('div','camera-caption');caption.append(element('span','',friendly(cam)));
 const focus=element('button','',state.focusCamera===cam?'Restore':'Focus');focus.onclick=()=>{state.focusCamera=state.focusCamera===cam?null:cam;renderCameras();updateFrame();};caption.append(focus);card.append(img,err,caption);root.append(card);});
}
function updateFrame(){
 $('scrubber').value=state.frame;$('frame-jump').value=state.frame+1;$('frame-counter').textContent=`/ ${state.numFrames}`;
 $('time-code').textContent=`${seconds(timeAt(state.frame))} / ${seconds(timeAt(Math.max(0,state.numFrames-1)))}`;
 state.cameras.forEach((cam,i)=>{const img=$(`camera-${i}`);if(!img)return;const src=`/frame/${state.ep}/${state.frame}/${cam.split('/').map(encodeURIComponent).join('/')}`;if(img.getAttribute('src')!==src)img.src=src;});
 const seg=state.epData?.segments?.find(s=>state.frame>=s.start&&state.frame<s.end);$('current-phase').textContent=seg?.label||'';
 drawChartMarkers();updateQualityPosition();updateSignalValues();
}
function renderSegments(){const overlay=$('segment-overlay');overlay.replaceChildren();$('segment-legend').replaceChildren();const seen=new Set();for(const s of state.epData?.segments||[]){const el=element('div','seg');el.style.left=`${100*s.start/state.numFrames}%`;el.style.width=`${100*(s.end-s.start)/state.numFrames}%`;el.style.background=s.color;overlay.append(el);if(!seen.has(s.label)){const label=element('span','',s.label);label.style.color=s.color;$('segment-legend').append(label);seen.add(s.label);}}}
function renderQuality(rebuildFilter=true){
 const enabled=!!state.info.quality_mode;$('quality-panel').hidden=!enabled;if(!enabled)return;
 const q=state.epData?.quality;state.qualityUnavailable=!q||q.num_frames!==state.numFrames||q.flags.some(f=>f.startsWith('nan_in_')||f==='invalid_timestamps')||q.metric_status?.score?.includes('insufficient frames');state.qualityUnavailable=state.qualityUnavailable&&!q?.video;
 $('score-value').textContent=q?.overall_score==null?'N/A':q.overall_score.toFixed(2);$('quality-summary').textContent=q?.overall_score==null?'Score unavailable. Check metric availability.':'A diagnostic summary, not a task-success grade.';
 $('review-count').textContent=allQualityEvents().filter(e=>e.kind!=='direction_reversal').length;$('pause-fraction').textContent=q?.static?`${(100*q.static.static_fraction).toFixed(1)}%`:'—';
 const flags=$('quality-flags');flags.replaceChildren();for(const f of q?.flags||[])flags.append(element('span','flag',friendly(f)));if(!flags.children.length)flags.append(element('small','',state.qualityUnavailable?'Analysis unavailable':'No episode flags'));
 $('quality-status').textContent=Object.entries(q?.metric_status||{}).map(([k,v])=>`${friendly(k)}: ${v}`).join('\n\n');
 if(rebuildFilter){const filter=$('event-filter');filter.replaceChildren();const all=element('option','','All event types');all.value='all';filter.append(all);for(const kind of new Set(allQualityEvents().filter(e=>e.kind!=='direction_reversal'||$('quality-show-reversals').checked).map(e=>e.kind))){const o=element('option','',friendly(kind));o.value=kind;filter.append(o);}filter.value=state.eventType;}
 renderVideoSummary(q,rebuildFilter);
 const events=qualityEvents(),tracks=$('quality-timelines');tracks.replaceChildren();const laneKey=e=>e.camera?e.camera+' / '+e.kind:e.kind;const kinds=[...new Set(events.map(laneKey))];if(!kinds.length)kinds.push(state.qualityUnavailable?'Analysis unavailable':'No visible events');
 for(const kind of kinds){const row=element('div','quality-row'),label=element('span','quality-row-label',friendly(kind)),track=element('div','quality-track');if(state.qualityUnavailable)track.style.background='#333d4b';track.onclick=e=>{const r=track.getBoundingClientRect();seek((e.clientX-r.left)/r.width*(state.numFrames-1));};
 for(const event of events.filter(e=>laneKey(e)===kind)){const btn=element('button','quality-span');btn.style.left=`${100*event.start_frame/Math.max(1,state.numFrames)}%`;btn.style.width=`${100*(event.end_frame-event.start_frame)/Math.max(1,state.numFrames)}%`;btn.style.background=qualityColor(event.kind);btn.title=eventLabel(event);btn.setAttribute('aria-label',eventLabel(event));btn.onclick=e=>{e.stopPropagation();seekQualityEvent(event);};track.append(btn);}
 track.append(element('div','quality-cursor'));row.append(label,track);tracks.append(row);}
 const ruler=$('timeline-ruler');ruler.replaceChildren();for(const fraction of [0,.25,.5,.75,1])ruler.append(element('span','mono',seconds(timeAt(Math.round(fraction*Math.max(0,state.numFrames-1))))));
 const list=$('event-list');list.replaceChildren();events.forEach((e,i)=>{const btn=element('button','event-card');btn.dataset.eventIndex=i;btn.style.borderLeftColor=qualityColor(e.kind);btn.append(element('strong','',`${e.camera?friendly(e.camera)+' · ':''}${friendly(e.kind)}`),element('small','mono',Number.isFinite(e.start_time_s)&&Number.isFinite(e.end_time_s)?`${seconds(eventTime(e))} · ${(e.end_time_s-e.start_time_s).toFixed(2)}s span`:`Frames ${e.start_frame}–${e.end_frame-1} · time unavailable`));btn.append(element('p','',eventDescription(e)));btn.onclick=()=>seekQualityEvent(e);list.append(btn);});
 if(!events.length)list.append(element('div','empty',state.qualityUnavailable?'Analysis unavailable for this episode.':state.eventType!=='all'?'No events match this filter.':'No review events detected. Descriptive reversals are hidden unless enabled.'));
 $('prev-event').disabled=$('next-event').disabled=events.length===0;updateQualityPosition();
}
function renderVideoSummary(q,rebuild){
 const cameras=q?.video?.per_camera||{},names=Object.keys(cameras),root=$('video-camera-summary'),filter=$('event-camera-filter');
 root.hidden=filter.hidden=!names.length;$('quality-modalities').textContent=names.length?'Proprio + video':'Proprioception';root.replaceChildren();
 if(rebuild){const value=filter.value;filter.replaceChildren();for(const [id,label] of [['all','All sources'],['proprio','Proprioception'],...names.map(n=>[n,friendly(n)])]){const o=element('option','',label);o.value=id;filter.append(o);}filter.value=[...filter.options].some(o=>o.value===value)?value:'all';}
 if(!names.length)return;root.append(element('h2','','Camera integrity'));
 for(const name of names){const c=cameras[name],card=element('div','detail-block');card.append(element('h3','',friendly(name)));
 card.append(element('p','inspector-note',`${c.num_frames} decoded / ${c.present_frames} present references · ${c.expected_frames} observations`));
 card.append(element('p','inspector-note',`${c.missing_frames} missing · ${c.decode_failures} decode failures · ${c.invalid_frames} invalid`));
 card.append(element('p','inspector-note',`Estimated dropped observations: ${c.estimated_dropped_frames??'unknown'} · timing coverage: ${c.timestamp_coverage==null?'unknown':(c.timestamp_coverage*100).toFixed(0)+'%'}`));
 card.append(element('p','inspector-note',`${c.sampled_out_frames} skipped by stride · ${c.capped_frames} outside analysis cap`));root.append(card);
 $('quality-status').textContent+='\n\n'+name+'\n'+Object.entries(c.metric_status||{}).map(([k,v])=>`${friendly(k)}: ${v}`).join('\n')+'\nTimebase: '+(c.provenance?.timebase||'unknown');}
}
function eventDescription(e){const notes={video_blur:'Low whole-image detail; low-texture scenes may qualify.',video_overexposure:'Bright clipping in sampled pixels; check task visibility.',video_underexposure:'Dark or black pixels; dark backgrounds may qualify.',video_frozen_candidate:'Near-identical samples; a stationary scene may qualify.',video_frame_gap:'Estimated missing observation intervals from episode timestamps, not camera capture PTS.',video_camera_missing:'Camera references absent on these episode observations.',video_decode_failure:'Image references could not be decoded.',video_invalid_pixels:'Unsupported or invalid image data; excluded from pixel metrics.',video_invalid_timestamps:'Timing is invalid or incomplete; frame positions remain available.',video_resolution_change:'Image dimensions changed; temporal comparisons reset.',video_cut_candidate:'Histogram discontinuity and motion spike; review for a cut.',held_command:'Repeated commands. May be an intentional hold.',motion_pause:'Low measured motion. Review the task context.',direction_reversal:'A direction change, not evidence of a retry.',tracking_stall_candidate:'Low motion with command/state mismatch. Contact can also cause this.',timestamp_gap:'Missing time support between recorded samples.'};return (notes[e.kind]||'Review this interval in context.')+(e.evidence?.estimated_missing_observations!=null?` Estimated missing: ${e.evidence.estimated_missing_observations}.`:'')+(e.evidence?.sample_stride>1?` Sample stride ${e.evidence.sample_stride}; intervening pixels were not analyzed.`:'')+(e.channels?.length?` Channels ${e.channels.join(', ')}.`:'');}
function updateQualityPosition(){if(!state.info?.quality_mode)return;document.querySelectorAll('.quality-cursor').forEach(el=>el.style.left=`${100*state.frame/Math.max(1,state.numFrames)}%`);const events=qualityEvents(),active=events.filter(e=>state.frame>=e.start_frame&&state.frame<e.end_frame);$('quality-current').textContent=state.qualityUnavailable?'Analysis unavailable for this timeline.':active.length?active.map(eventLabel).join(' · '):'No visible event at this frame. Check metric availability before interpreting this region.';document.querySelectorAll('.event-card').forEach(el=>{const e=events[Number(el.dataset.eventIndex)];el.classList.toggle('active',state.frame>=e.start_frame&&state.frame<e.end_frame);});}
function jumpEvent(direction){const events=qualityEvents();const next=direction>0?events.find(e=>e.start_frame>state.frame):[...events].reverse().find(e=>e.start_frame<state.frame);if(next)seekQualityEvent(next);}
function renderSignalLegends(){for(const type of ['action','state']){const root=$(`${type}-legend`);root.replaceChildren();const data=state.epData?.[type==='action'?'actions':'states'];const dims=data?.[0]?.length||0;for(let d=0;d<dims;d++){const btn=element('button','channel-chip');btn.setAttribute('aria-pressed','true');btn.title=`Toggle ${type} channel ${d}`;const dot=element('i');dot.style.background=colors[d%colors.length];btn.append(dot,document.createTextNode(`ch ${d}`),element('span'));btn.onclick=()=>{const hidden=state.hidden[type];hidden.has(d)?hidden.delete(d):hidden.add(d);btn.setAttribute('aria-pressed',String(!hidden.has(d)));drawCharts();drawChartMarkers();};root.append(btn);}}}
function updateSignalValues(){for(const type of ['action','state']){const row=state.epData?.[type==='action'?'actions':'states']?.[state.frame];$(`${type}-legend`).querySelectorAll('.channel-chip span').forEach((el,d)=>{el.textContent=Number.isFinite(row?.[d])?row[d].toFixed(3):'—';});}}
function drawCharts(){for(const type of ['action','state'])drawChart(type);}
function drawChart(type){
 const canvas=$(`${type}-chart`),ctx=canvas.getContext('2d'),rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;ctx.scale(dpr,dpr);ctx.clearRect(0,0,rect.width,rect.height);canvas._chartParams=null;
 const data=state.epData?.[type==='action'?'actions':'states'];ctx.font='10px ui-monospace,monospace';ctx.fillStyle='#95a5b9';if(!data?.length){ctx.fillText('No recorded signals',16,35);return;}
 const dims=data[0].length,shown=Array.from({length:dims},(_,i)=>i).filter(i=>!state.hidden[type].has(i)),norm=$(`${type}-scale`).value==='normalized';
 const ranges=Array.from({length:dims},()=>[Infinity,-Infinity]);for(const row of data)for(const d of shown)if(Number.isFinite(row[d])){ranges[d][0]=Math.min(ranges[d][0],row[d]);ranges[d][1]=Math.max(ranges[d][1],row[d]);}
 let min=norm?0:Math.min(...shown.map(d=>ranges[d][0])),max=norm?1:Math.max(...shown.map(d=>ranges[d][1]));if(!shown.length||!Number.isFinite(min)||!Number.isFinite(max)){ctx.fillText('No visible channel data',16,35);return;}if(min===max){min-=.5;max+=.5;}
 const left=48,top=12,width=Math.max(1,rect.width-left-12),height=Math.max(1,rect.height-39);canvas._chartParams={left,top,width,height};
 ctx.strokeStyle='#25313e';ctx.lineWidth=1;for(let i=0;i<3;i++){const y=top+i*height/2;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(left+width,y);ctx.stroke();ctx.fillStyle='#95a5b9';ctx.fillText((max-i*(max-min)/2).toFixed(2),2,y+3);}
 for(const f of [0,.5,1]){ctx.textAlign=f===1?'right':f===0?'left':'center';ctx.fillText(seconds(timeAt(Math.round(f*(data.length-1)))),left+f*width,rect.height-7);}ctx.textAlign='left';
 for(const e of qualityEvents()){ctx.fillStyle=qualityColor(e.kind)+'1c';ctx.fillRect(left+e.start_frame/data.length*width,top,(e.end_frame-e.start_frame)/data.length*width,height);}
 for(const d of shown){ctx.strokeStyle=colors[d%colors.length];ctx.lineWidth=1.3;ctx.beginPath();let connected=false;for(let i=0;i<data.length;i++){let v=data[i][d];if(!Number.isFinite(v)){connected=false;continue;}if(norm)v=(v-ranges[d][0])/(ranges[d][1]-ranges[d][0]||1);const x=left+i/Math.max(1,data.length-1)*width,y=top+height-(v-min)/(max-min)*height;if(connected)ctx.lineTo(x,y);else ctx.moveTo(x,y);connected=true;}ctx.stroke();}
}
function drawChartMarkers(){for(const type of ['action','state']){const chart=$(`${type}-chart`),canvas=$(`${type}-marker`),r=chart.getBoundingClientRect(),dpr=window.devicePixelRatio||1;canvas.width=r.width*dpr;canvas.height=r.height*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);const p=chart._chartParams;if(!p)continue;const x=p.left+state.frame/Math.max(1,state.numFrames-1)*p.width;ctx.strokeStyle='#e7edf5';ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(x,p.top);ctx.lineTo(x,p.top+p.height);ctx.stroke();}}
function togglePlay(){if(!state.numFrames)return;if(state.playing){stopPlayback();return;}if(state.frame>=state.numFrames-1)state.frame=0;state.playing=true;$('play-btn').textContent='Pause';const token=++state.playToken;let last=performance.now();function tick(now){if(!state.playing||token!==state.playToken)return;const interval=1000/(state.fps*state.speed),steps=Math.floor((now-last)/interval);if(steps){last+=steps*interval;state.frame+=steps;if(state.frame>=state.numFrames){if($('loop-playback').checked)state.frame%=state.numFrames;else{state.frame=state.numFrames-1;stopPlayback();}}updateFrame();}if(state.playing)requestAnimationFrame(tick);}requestAnimationFrame(tick);}
function onKey(e){if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(e.target.tagName)||$('help-dialog').open)return;switch(e.key){case ' ':e.preventDefault();togglePlay();break;case 'ArrowLeft':e.preventDefault();seek(state.frame-1);break;case 'ArrowRight':e.preventDefault();seek(state.frame+1);break;case 'ArrowUp':e.preventDefault();loadEpisode(state.ep-1);break;case 'ArrowDown':e.preventDefault();loadEpisode(state.ep+1);break;case 'n':jumpEvent(1);break;case 'p':jumpEvent(-1);break;case '[':case ']':{const el=$('speed-select');el.selectedIndex=Math.max(0,Math.min(el.options.length-1,el.selectedIndex+(e.key===']'?1:-1)));state.speed=Number(el.value);break;}}}
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{drawCharts();drawChartMarkers();},80);});
init();
</script>
</body>
</html>
"""
