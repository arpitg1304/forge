"""Thresholds for the dataset linter.

Defaults encode Hugging Face's published LeRobot recording guidelines
(https://huggingface.co/blog/lerobot-datasets): concise task descriptions,
720p-class cameras, at least two views, and ``<modality>.<location>`` naming.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LintConfig:
    # ── Task / language instruction ──
    task_min_chars: int = 25            # HF: keep descriptions 25–50 chars
    task_max_chars: int = 60            # soft upper bound (warn past this)
    # Placeholder / meaningless task strings seen across Hub datasets.
    task_generic_phrases: tuple[str, ...] = (
        "task desc",
        "task description",
        "task",
        "hold",
        "up",
        "down",
        "do something",
        "n/a",
        "none",
        "test",
        "demo",
    )

    # ── Cameras ──
    min_camera_views: int = 2           # HF recommends >= 2 views
    min_cam_height: int = 480           # >= 480x640
    min_cam_width: int = 640
    # Names that don't say where the camera is (third-person vs wrist, etc.).
    ambiguous_cam_names: tuple[str, ...] = (
        "laptop",
        "phone",
        "webcam",
        "cam",
        "camera",
        "image",
        "rgb",
        "default",
    )
    # Recommended location tokens for the <modality>.<location> convention.
    known_cam_locations: tuple[str, ...] = (
        "wrist",
        "top",
        "front",
        "side",
        "left",
        "right",
        "overhead",
        "ego",
        "exterior",
        "base",
        "high",
        "low",
    )

    # ── Episodes ──
    min_episode_frames: int = 3         # 1–2 frame episodes are broken
