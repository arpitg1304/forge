"""Generate a synthetic LeRobot v3 dataset to exercise `forge dedup`.

Produces 10 episodes (2 cameras: top + wrist) covering interesting dedup
conditions: exact copies, re-encode noise, brightness shift, a time-shifted
take, multi-camera partial matches, and genuinely unique episodes.

Scenes are deliberately *low-frequency* (sinusoidal gratings + a moving Gaussian
blob + per-identity tint) so they survive H.264 compression — random noise would
be blurred away by the codec and make every episode look alike.

Run:  python scripts/gen_video_dedup_samples.py
Then: forge dedup sample_data/video_dedup_samples
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from forge.core.models import CameraInfo, DatasetInfo, Episode, Frame, LazyImage
from forge.formats.lerobot_v3.writer import LeRobotV3Writer, LeRobotV3WriterConfig

H = W = 96
T = 30                      # frames per episode (1s @ 30fps)
FPS = 30.0
OUT = Path(__file__).resolve().parents[1] / "sample_data" / "video_dedup_samples"

# Camera-identity offsets so top and wrist views differ within an episode.
WRIST = 100

# Distinct scene identities.
A, B, C, D = 1, 2, 3, 4
E = 5                        # a unique wrist-only identity for the partial-match case


def scene(
    identity: int,
    t: int,
    *,
    brightness: float = 0.0,
    noise: float = 0.0,
    shift: int = 0,
    noise_seed: int = 0,
) -> np.ndarray:
    """Deterministic low-frequency RGB frame for a given identity and time."""
    r = np.random.default_rng(1000 + identity)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    yy /= H
    xx /= W

    img = np.zeros((H, W), dtype=np.float64)
    for _ in range(3):  # a few gratings → identity-specific low-freq structure
        fx, fy = r.uniform(1, 5), r.uniform(1, 5)
        ph = r.uniform(0, 2 * np.pi)
        img += np.sin(2 * np.pi * (fx * xx + fy * yy) + ph)
    img = (img - img.min()) / (np.ptp(img) + 1e-9)

    tt = (t + shift) / max(T - 1, 1)          # moving Gaussian blob (temporal signal)
    cx = 0.2 + 0.6 * tt
    cy = 0.5 + 0.3 * np.sin(2 * np.pi * tt + identity)
    blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 0.02)))
    img = np.clip(0.7 * img + 0.6 * blob, 0, 1)

    tint = r.uniform(0.6, 1.0, size=3)
    rgb = np.stack([img * tint[0], img * tint[1], img * tint[2]], axis=-1) * 255.0
    rgb += brightness
    if noise > 0:
        rgb += np.random.default_rng(noise_seed).normal(0, noise, rgb.shape)
    return np.clip(rgb, 0, 255).astype(np.uint8)


# (episode_id, top_identity, wrist_identity, transform)
SPECS = [
    ("episode_000000", A, A + WRIST, "base"),    # representative of the A cluster
    ("episode_000001", A, A + WRIST, "base"),    # EXACT copy of 000000
    ("episode_000002", A, A + WRIST, "noise"),   # re-encode (mild pixel noise)
    ("episode_000003", A, A + WRIST, "bright"),  # brightness +50 (pHash robust)
    ("episode_000004", B, B + WRIST, "base"),    # unique B
    ("episode_000005", B, B + WRIST, "shift"),   # B time-shifted 6 frames
    ("episode_000006", C, C + WRIST, "base"),    # unique C
    ("episode_000007", C, E + WRIST, "base"),    # top == 000006, wrist differs
    ("episode_000008", D, D + WRIST, "base"),    # unique D
    ("episode_000009", D, D + WRIST, "base"),    # EXACT copy of 000008
]

TRANSFORMS = {
    "base": dict(),
    "noise": dict(noise=8.0),
    "bright": dict(brightness=50.0),
    "shift": dict(shift=6),
}

CAMERAS = {
    "top": CameraInfo(name="top", height=H, width=W),
    "wrist": CameraInfo(name="wrist", height=H, width=W),
}


def make_episode(idx: int, ep_id: str, top_id: int, wrist_id: int, transform: str) -> Episode:
    kw = TRANSFORMS[transform]

    def frame_loader():
        for t in range(T):
            def top_loader(t=t):
                return scene(top_id, t, noise_seed=1000 + idx, **kw)

            def wrist_loader(t=t):
                return scene(wrist_id, t, noise_seed=2000 + idx, **kw)

            # State/action vary by identity so the parquet has real content too.
            state = np.full(7, 0.01 * top_id + 0.001 * t, dtype=np.float32)
            action = np.full(7, 0.02 * top_id, dtype=np.float32)
            yield Frame(
                index=t,
                timestamp=t / FPS,
                images={
                    "top": LazyImage(loader=top_loader, height=H, width=W, channels=3),
                    "wrist": LazyImage(loader=wrist_loader, height=H, width=W, channels=3),
                },
                state=state,
                action=action,
            )

    return Episode(
        episode_id=ep_id,
        language_instruction=f"synthetic dedup scene ({transform})",
        cameras=CAMERAS,
        fps=FPS,
        _frame_loader=frame_loader,
    )


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    episodes = [
        make_episode(i, ep_id, top_id, wrist_id, transform)
        for i, (ep_id, top_id, wrist_id, transform) in enumerate(SPECS)
    ]

    info = DatasetInfo(
        path=OUT,
        format="lerobot-v3",
        num_episodes=len(episodes),
        total_frames=len(episodes) * T,
        inferred_fps=FPS,
        inferred_robot_type="synthetic",
        cameras=CAMERAS,
    )

    writer = LeRobotV3Writer(LeRobotV3WriterConfig(fps=FPS, robot_type="synthetic"))
    writer.write_dataset(iter(episodes), OUT, dataset_info=info)

    _fix_video_path_template(OUT)
    print(f"Wrote {len(episodes)} episodes to {OUT}")


def _fix_video_path_template(out: Path) -> None:
    """Reconcile info.json's video_path with the actual one-file-per-episode layout.

    forge's LeRobot v3 writer flushes a new chunk per episode (writer.py:455),
    so every episode's video is its own ``chunk-XXX/file-000.mp4`` starting at
    local frame 0 — yet it stamps info.json with the multi-episode template
    ``...file-{file_index}.mp4``. The reader keys off ``{file_index}`` and then
    seeks by GLOBAL frame index, walking off the end of each single-episode file
    and returning black frames for every episode after the first.

    Rewriting the template to the single-episode form makes the reader use the
    local ``frame_index``, so each episode decodes from its own file correctly.
    (This is a workaround for a forge writer/reader round-trip bug, not a dedup
    issue — see the dataset README.)
    """
    import json

    info_path = out / "meta" / "info.json"
    data = json.loads(info_path.read_text())
    data["video_path"] = (
        "videos/{video_key}/chunk-{chunk_index:03d}/episode_{episode_index:06d}.mp4"
    )
    info_path.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
