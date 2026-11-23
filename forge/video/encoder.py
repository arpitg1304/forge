"""Video encoder using PyAV.

Provides H.264 video encoding for converting robotics image sequences
to MP4 format, compatible with LeRobot v3 and other formats.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

from forge.core.exceptions import MissingDependencyError

if TYPE_CHECKING:
    from forge.core.models import LazyImage

# Lazy import av to handle optional dependency
_av = None


def _get_av():
    """Lazy import PyAV."""
    global _av
    if _av is None:
        try:
            import av

            _av = av
        except ImportError:
            raise MissingDependencyError(
                dependency="av",
                feature="video encoding",
                install_hint="pip install forge-robotics[video]",
            )
    return _av


def _fps_to_fraction(fps: float) -> Fraction:
    """Convert float fps to Fraction for PyAV compatibility.

    Args:
        fps: Frames per second as float.

    Returns:
        Fraction representation of fps.
    """
    # Limit denominator to avoid precision issues
    return Fraction(fps).limit_denominator(1000)


@dataclass
class VideoEncoderConfig:
    """Configuration for video encoding.

    Attributes:
        codec: Video codec to use (default: "libx264").
        crf: Constant Rate Factor for quality (0-51, lower=better, default: 23).
        preset: Encoding preset (default: "medium"). Options: ultrafast, superfast,
            veryfast, faster, fast, medium, slow, slower, veryslow.
        pixel_format: Output pixel format (default: "yuv420p" for compatibility).
        gop_size: Group of Pictures size (keyframe interval). Default: 12.
    """

    codec: str = "libx264"
    crf: int = 23
    preset: str | None = "medium"
    pixel_format: str = "yuv420p"
    gop_size: int = 12


class VideoEncoder:
    """Encodes image frames to video using PyAV (FFmpeg).

    Supports streaming encoding - frames are written as they come in,
    not buffered in memory.

    Example:
        >>> encoder = VideoEncoder()
        >>> encoder.encode_frames(frames, "output.mp4", fps=30, width=640, height=480)
    """

    def __init__(self, config: VideoEncoderConfig | None = None):
        """Initialize encoder with configuration.

        Args:
            config: Encoding configuration. Uses defaults if None.
        """
        self.config = config or VideoEncoderConfig()

    def encode_frames(
        self,
        frames: Iterator[LazyImage],
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
    ) -> int:
        """Encode an iterator of frames to a video file.

        Args:
            frames: Iterator of LazyImage objects.
            output_path: Path to output video file.
            fps: Frames per second.
            width: Video width in pixels.
            height: Video height in pixels.

        Returns:
            Number of frames encoded.

        Raises:
            MissingDependencyError: If PyAV is not installed.
            ValueError: If no frames provided or invalid dimensions.
        """
        av = _get_av()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open output container
        container = av.open(str(output_path), mode="w")

        try:
            # Add video stream (PyAV needs Fraction for rate)
            fps_frac = _fps_to_fraction(fps)
            stream = container.add_stream(self.config.codec, rate=fps_frac)
            stream.width = width
            stream.height = height
            stream.pix_fmt = self.config.pixel_format
            options = {"crf": str(self.config.crf)}
            if self.config.preset is not None:
                options["preset"] = str(self.config.preset)
            stream.options = options
            # Set GOP size
            stream.gop_size = self.config.gop_size

            frame_count = 0

            for lazy_image in frames:
                # Load the image data
                img_array = lazy_image.load()

                # Ensure uint8 format for video encoding
                if img_array.dtype != "uint8":
                    import numpy as np

                    if img_array.dtype in (np.float32, np.float64):
                        # Check if values are in 0-1 range (normalized) or 0-255 range
                        if img_array.max() <= 1.0:
                            img_array = (img_array * 255).astype(np.uint8)
                        else:
                            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                    else:
                        img_array = img_array.astype(np.uint8)

                # Create video frame from numpy array
                # PyAV expects RGB format, shape (height, width, 3)
                frame = av.VideoFrame.from_ndarray(img_array, format="rgb24")

                # Encode and write
                for packet in stream.encode(frame):
                    container.mux(packet)

                frame_count += 1

                # Clear cache to free memory
                lazy_image.clear_cache()

            # Flush encoder
            for packet in stream.encode():
                container.mux(packet)

            if frame_count == 0:
                raise ValueError("No frames provided to encoder")

            return frame_count

        finally:
            container.close()

    def encode_from_arrays(
        self,
        frames: Iterator,
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
    ) -> int:
        """Encode raw numpy arrays to video.

        Convenience method when you have numpy arrays instead of LazyImage.

        Args:
            frames: Iterator of numpy arrays (H, W, C) in RGB format.
            output_path: Path to output video file.
            fps: Frames per second.
            width: Video width in pixels.
            height: Video height in pixels.

        Returns:
            Number of frames encoded.
        """
        av = _get_av()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        container = av.open(str(output_path), mode="w")

        try:
            fps_frac = _fps_to_fraction(fps)
            stream = container.add_stream(self.config.codec, rate=fps_frac)
            stream.width = width
            stream.height = height
            stream.pix_fmt = self.config.pixel_format
            options = {"crf": str(self.config.crf)}
            if self.config.preset is not None:
                options["preset"] = str(self.config.preset)
            stream.options = options
            stream.gop_size = self.config.gop_size

            frame_count = 0

            for img_array in frames:
                # Ensure uint8 format for video encoding
                if img_array.dtype != "uint8":
                    import numpy as np

                    if img_array.dtype in (np.float32, np.float64):
                        if img_array.max() <= 1.0:
                            img_array = (img_array * 255).astype(np.uint8)
                        else:
                            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                    else:
                        img_array = img_array.astype(np.uint8)

                frame = av.VideoFrame.from_ndarray(img_array, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
                frame_count += 1

            # Flush
            for packet in stream.encode():
                container.mux(packet)

            if frame_count == 0:
                raise ValueError("No frames provided to encoder")

            return frame_count

        finally:
            container.close()


def encode_video(
    frames: Iterator[LazyImage],
    output_path: str | Path,
    fps: float,
    width: int,
    height: int,
    config: VideoEncoderConfig | None = None,
) -> int:
    """Convenience function to encode frames to video.

    Args:
        frames: Iterator of LazyImage objects.
        output_path: Path to output video file.
        fps: Frames per second.
        width: Video width in pixels.
        height: Video height in pixels.
        config: Optional encoder configuration.

    Returns:
        Number of frames encoded.
    """
    encoder = VideoEncoder(config)
    return encoder.encode_frames(frames, output_path, fps, width, height)
