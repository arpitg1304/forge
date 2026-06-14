"""Tests for Tier 1 video motion metrics (optical flow). Requires opencv."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from forge.core.models import Episode, Frame, LazyImage
from forge.quality.video import VideoQualityAnalyzer, VideoQualityConfig

H = W = 96


def _lazy(img):
    return LazyImage(loader=lambda im=img: im, height=H, width=W, channels=3)


def _bg(seed=0, lo=0, hi=120):
    return np.random.default_rng(seed).integers(lo, hi, (H, W, 3), dtype=np.uint8)


def _episode(images, ep_id="ep", cam="cam"):
    frames = [Frame(index=i, images={cam: _lazy(im)}) for i, im in enumerate(images)]
    return Episode(episode_id=ep_id, _frame_loader=lambda: iter(frames))


def _analyze(images, level="motion", **cfg):
    ep = _episode(images)
    return VideoQualityAnalyzer(VideoQualityConfig(level=level, **cfg)).analyze_episode(ep)


# ── Motion magnitude / no_motion ─────────────────────────────────


def _translating(n=20, step=3):
    base = _bg(1)
    out = []
    for t in range(n):
        f = base.copy()
        x = 5 + step * t
        f[40:60, x:x + 20, :] = 240
        out.append(f)
    return out


def test_moving_episode_has_motion():
    vq = _analyze(_translating())
    assert vq.mean_motion > 0.2
    assert "no_motion" not in vq.flags


def test_static_episode_flagged_no_motion():
    base = _bg(2)
    vq = _analyze([base.copy() for _ in range(20)])
    assert vq.mean_motion < 0.05
    assert "no_motion" in vq.flags


# ── Smoothness (pixel-space LDLJ) ────────────────────────────────


def test_smooth_pan_is_smoother_than_jitter():
    base = _bg(3)

    def block(x):
        f = base.copy()
        x = int(np.clip(x, 0, W - 20))
        f[40:60, x:x + 20, :] = 240
        return f

    smooth = [block(5 + 3 * t) for t in range(20)]                 # constant velocity
    rng = np.random.default_rng(4)
    jitter = [block(38 + rng.integers(-10, 11)) for _ in range(20)]  # random jumps

    s = _analyze(smooth).motion_smoothness
    j = _analyze(jitter).motion_smoothness
    assert s is not None and j is not None
    assert s > j  # less-negative LDLJ = smoother


# ── Cut detection ────────────────────────────────────────────────


def test_cut_detected_between_distinct_scenes():
    dark = _bg(5, lo=0, hi=80)       # distinct luma distribution ...
    bright = _bg(6, lo=160, hi=255)  # ... so the histogram correlation drops
    images = [dark.copy() for _ in range(8)] + [bright.copy() for _ in range(8)]
    vq = _analyze(images)
    assert vq.cut_count and vq.cut_count >= 1
    assert "cut_detected" in vq.flags


def test_no_cut_in_continuous_motion():
    vq = _analyze(_translating(n=20, step=2))
    assert (vq.cut_count or 0) == 0
    assert "cut_detected" not in vq.flags


# ── Camera vs scene split ────────────────────────────────────────


def test_scene_motion_fraction_present():
    vq = _analyze(_translating())
    assert vq.scene_motion_fraction is not None
    assert vq.scene_motion_fraction >= 0.0


# ── Multi-camera rollup ──────────────────────────────────────────


def test_no_motion_rollup_needs_all_cameras_static():
    base_static = _bg(7)
    moving = _translating()
    frames = [
        Frame(index=t, images={"static": _lazy(base_static.copy()), "moving": _lazy(moving[t])})
        for t in range(len(moving))
    ]
    ep = Episode(episode_id="multi", _frame_loader=lambda: iter(frames))
    vq = VideoQualityAnalyzer(VideoQualityConfig(level="motion")).analyze_episode(ep)
    # One camera moves → episode is not "no_motion".
    assert "no_motion" not in vq.flags
    assert vq.mean_motion > 0.2  # rollup is the most-active camera


# ── Degenerate-frame guard ───────────────────────────────────────


def test_black_frame_does_not_create_spurious_cut():
    base = _bg(8)
    images = [base.copy() for _ in range(10)]
    images.append(np.zeros((H, W, 3), dtype=np.uint8))  # a decoder black frame
    vq = _analyze(images)
    assert (vq.cut_count or 0) == 0


# ── Level gating: motion fields absent at pixel level ────────────


def test_pixel_level_has_no_motion_fields():
    vq = _analyze(_translating(), level="pixel")
    assert vq.mean_motion is None
    assert vq.cut_count is None
