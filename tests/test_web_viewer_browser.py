"""Opt-in real-browser regression: FORGE_TEST_CHROME=/path/to/chrome pytest this_file.

Requires websocket-client in the test environment. No browser package is needed
by Forge itself. The test uses a temporary Chrome profile and a local fixture.
"""

import base64
import json
import os
import subprocess
import threading
import time
import urllib.request
from http.server import HTTPServer

import numpy as np
import pytest

from forge.visualize.web_viewer import _make_handler


@pytest.mark.skipif(not os.environ.get("FORGE_TEST_CHROME"), reason="Opt-in Chrome integration")
def test_explorer_in_browser(tmp_path):
    websocket = pytest.importorskip("websocket")

    class Backend:
        quality_mode = True

        def get_info_dict(self):
            return dict(
                name="Fixture",
                format="test",
                num_episodes=2,
                total_frames=200,
                fps=15,
                cameras=["wrist"],
                quality_mode=self.quality_mode,
                episode_ids=["episode_000000", "episode_000001"],
                episodes=[
                    dict(
                        id=f"episode_{i:06d}",
                        num_frames=100,
                        task="Pick a cup",
                        score=8.0,
                        event_count=1,
                        flags=[],
                    )
                    for i in range(2)
                ],
            )

        def get_episode_dict(self, idx):
            events = [
                dict(
                    kind="held_command",
                    start_frame=50,
                    end_frame=100,
                    start_time_s=50 / 15,
                    end_time_s=99 / 15,
                    channels=[],
                ),
                dict(
                    kind="direction_reversal",
                    start_frame=20,
                    end_frame=30,
                    start_time_s=20 / 15,
                    end_time_s=29 / 15,
                    channels=[0],
                ),
            ]
            quality = dict(
                num_frames=100,
                overall_score=8.0,
                flags=[],
                events=events,
                static=None,
                metric_status={"score": "heuristic"},
                video={
                    "per_camera": {
                        "wrist": {
                            "num_frames": 95,
                            "present_frames": 100,
                            "expected_frames": 100,
                            "missing_frames": 0,
                            "decode_failures": 5,
                            "invalid_frames": 0,
                            "sampled_out_frames": 0,
                            "capped_frames": 0,
                            "estimated_dropped_frames": 3,
                            "timestamp_coverage": 1.0,
                            "metric_status": {"pixels": "available"},
                            "events": [
                                dict(
                                    kind="video_frame_gap",
                                    camera="wrist",
                                    start_frame=10,
                                    end_frame=12,
                                    start_time_s=10 / 15,
                                    end_time_s=14 / 15,
                                    evidence={"estimated_missing_observations": 3},
                                )
                            ],
                        }
                    }
                },
            )
            data = [[i / 100 + d for d in range(12)] for i in range(100)]
            return dict(
                episode_id=f"episode_{idx:06d}",
                num_frames=100,
                fps=15,
                cameras=["wrist"],
                timestamps=[i / 15 for i in range(100)],
                task="Pick a cup",
                actions=data,
                states=data,
                segments=None,
                quality=quality if self.quality_mode else None,
            )

        def get_frame_image(self, *args):
            return np.full((40, 60, 3), 90, dtype=np.uint8)

    backend = Backend()
    server = HTTPServer(("127.0.0.1", 0), _make_handler(backend))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    chrome = subprocess.Popen(
        [
            os.environ["FORGE_TEST_CHROME"],
            "--headless",
            "--disable-gpu",
            "--no-first-run",
            "--remote-debugging-port=0",
            "--remote-allow-origins=*",
            f"--user-data-dir={tmp_path / 'profile'}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    ws = None
    try:
        port_file = tmp_path / "profile" / "DevToolsActivePort"
        for _ in range(100):
            if port_file.exists():
                break
            time.sleep(0.1)
        port = port_file.read_text().splitlines()[0]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json") as response:
            targets = json.load(response)
        ws = websocket.create_connection(
            next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page"), timeout=15
        )
        seq = 0

        def call(method, **params):
            nonlocal seq
            seq += 1
            ws.send(json.dumps(dict(id=seq, method=method, params=params)))
            while True:
                result = json.loads(ws.recv())
                if result.get("id") == seq:
                    assert "error" not in result, result
                    return result.get("result", {})

        def js(expression):
            result = call(
                "Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True
            )
            assert "exceptionDetails" not in result, result
            return result.get("result", {}).get("value")

        def ready():
            for _ in range(100):
                if js("!!document.getElementById('loading')?.hidden && !!state.epData"):
                    return
                time.sleep(0.1)
            pytest.fail("Viewer did not initialize")

        call("Page.enable")
        call(
            "Emulation.setDeviceMetricsOverride",
            width=1500,
            height=1000,
            deviceScaleFactor=1,
            mobile=False,
        )
        call("Page.navigate", url=f"http://127.0.0.1:{server.server_port}/")
        ready()
        assert js("""(async()=>{
          const assert=(v,m)=>{if(!v)throw Error(m);};
          assert(qualityEvents().length===2,'default reversal visibility');
          assert(!$('video-camera-summary').hidden,'camera summary');
          $('event-camera-filter').value='wrist';$('event-camera-filter').onchange();
          assert(qualityEvents().length===1&&qualityEvents()[0].camera==='wrist','camera filter');
          seekQualityEvent(qualityEvents()[0]);assert(state.frame===10,'video event seek');
          assert($('quality-current').textContent.includes('wrist'),'camera-scoped evidence');
          $('event-camera-filter').value='proprio';$('event-camera-filter').onchange();
          seekQualityEvent(qualityEvents()[0]);assert(state.frame===50,'event seeking');
          $('episode-search').value='000001';renderEpisodeList();
          assert($('episode-list').children.length===1,'episode search');
          $('quality-show-reversals').checked=true;$('quality-show-reversals').onchange();
          assert(qualityEvents().length===2,'reversal toggle');
          $('event-filter').value='direction_reversal';
          $('event-filter').onchange({target:$('event-filter')});
          assert(qualityEvents().length===1,'event filtering');
          document.querySelector('.camera-caption button').click();
          assert(document.querySelector('.camera-card.focused'),'camera focus');
          $('camera-reset').click();assert(!document.querySelector('.focused'),'restore camera');
          assert($('action-legend').children.length===12,'all channels');
          $('action-legend').firstChild.click();assert(state.hidden.action.has(0),'channel toggle');
          $('state-scale').value='normalized';$('state-scale').onchange();
          assert($('state-chart')._chartParams,'normalized plot');
          seek(2);togglePlay();await new Promise(r=>setTimeout(r,250));stopPlayback();
          assert(state.frame>2,'playback');
          await Promise.all([loadEpisode(0),loadEpisode(1)]);
          assert(state.ep===1,'episode loading order');
          return true;
        })()""")
        call(
            "Emulation.setDeviceMetricsOverride",
            width=390,
            height=844,
            deviceScaleFactor=1,
            mobile=False,
        )
        time.sleep(0.2)
        assert js("document.documentElement.scrollWidth <= innerWidth")
        # An image artifact helps diagnose CI/browser failures when retained.
        (tmp_path / "mobile.png").write_bytes(
            base64.b64decode(call("Page.captureScreenshot", format="png")["data"])
        )
        backend.quality_mode = False
        call("Page.reload")
        ready()
        assert js("$('inspector').hidden && $('quality-panel').hidden && state.cameras.length===1")
        assert js("!$('error-banner').textContent")
    finally:
        if ws:
            ws.close()
        chrome.terminate()
        try:
            chrome.wait(timeout=5)
        except subprocess.TimeoutExpired:
            chrome.kill()
            chrome.wait()
        server.shutdown()
        server.server_close()
