"""Format registry for Forge.

The registry provides a plugin pattern for format readers and writers.
Formats register themselves using decorators, and the registry handles
format detection and instantiation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from forge.core.exceptions import FormatDetectionError, UnsupportedFormatError

if TYPE_CHECKING:
    from forge.core.protocols import FormatReader, FormatWriter


class FormatRegistry:
    """Central registry for format readers and writers.

    Supports auto-discovery and format detection. Formats register themselves
    using class decorators:

        @FormatRegistry.register_reader("rlds")
        class RLDSReader:
            ...

    Usage:
        reader = FormatRegistry.get_reader("rlds")
        writer = FormatRegistry.get_writer("lerobot-v3")
        detected = FormatRegistry.detect_format(some_path)
    """

    _readers: dict[str, type[FormatReader]] = {}
    _writers: dict[str, type[FormatWriter]] = {}

    @classmethod
    def register_reader(cls, format_name: str):
        """Decorator to register a reader class.

        Args:
            format_name: Format identifier (e.g., "rlds", "lerobot-v3").

        Returns:
            Decorator function.

        Example:
            @FormatRegistry.register_reader("rlds")
            class RLDSReader:
                ...
        """

        def decorator(reader_cls: type[FormatReader]) -> type[FormatReader]:
            cls._readers[format_name] = reader_cls
            return reader_cls

        return decorator

    @classmethod
    def register_writer(cls, format_name: str):
        """Decorator to register a writer class.

        Args:
            format_name: Format identifier.

        Returns:
            Decorator function.
        """

        def decorator(writer_cls: type[FormatWriter]) -> type[FormatWriter]:
            cls._writers[format_name] = writer_cls
            return writer_cls

        return decorator

    @classmethod
    def get_reader(cls, format_name: str) -> FormatReader:
        """Get an instantiated reader for the specified format.

        Args:
            format_name: Format identifier.

        Returns:
            Instantiated FormatReader.

        Raises:
            UnsupportedFormatError: If no reader registered for format.
        """
        if format_name not in cls._readers:
            raise UnsupportedFormatError(
                format_name,
                available_formats=list(cls._readers.keys()),
            )
        return cls._readers[format_name]()

    @classmethod
    def get_writer(cls, format_name: str) -> FormatWriter:
        """Get an instantiated writer for the specified format.

        Args:
            format_name: Format identifier.

        Returns:
            Instantiated FormatWriter.

        Raises:
            UnsupportedFormatError: If no writer registered for format.
        """
        if format_name not in cls._writers:
            raise UnsupportedFormatError(
                format_name,
                available_formats=list(cls._writers.keys()),
            )
        return cls._writers[format_name]()

    # Formats whose metadata can be inspected over the network without a full
    # download (range reads). Used by `forge inspect` to stream cloud datasets.
    STREAMABLE_FORMATS: frozenset[str] = frozenset({"lerobot-v3", "zarr"})

    # Priority order for format detection (more specific formats first)
    _detection_priority: list[str] = [
        "groot",  # Check GR00T before LeRobot (more specific)
        "lerobot-v3",  # Check v3 before v2 (more specific)
        "lerobot-v2",
        "robodm",  # .vla files are distinctive
        "rlds",
        "zarr",
        "hdf5",
        "mcap",  # Check before rosbag — MCAP is its own format, not ROS-only.
        "rosbag",
    ]

    @classmethod
    def detect_format(cls, path: Path | str) -> str:
        """Try each reader's can_read() to detect format.

        Formats are checked in priority order (more specific first).

        Args:
            path: Path to dataset.

        Returns:
            Detected format identifier.

        Raises:
            FormatDetectionError: If no reader can handle the path.
        """
        from forge.io import is_remote_uri

        # Keep remote URIs as strings (Path() would mangle "s3://"); readers'
        # can_read handle both. Non-streaming readers safely return False for
        # a remote URI, so only streamable formats match without a download.
        path = str(path) if is_remote_uri(path) else Path(path)

        # Check in priority order first
        for format_name in cls._detection_priority:
            if format_name in cls._readers:
                try:
                    if cls._readers[format_name].can_read(path):
                        return format_name
                except Exception:
                    continue

        # Then check any remaining formats not in priority list
        for format_name, reader_cls in cls._readers.items():
            if format_name in cls._detection_priority:
                continue  # Already checked
            try:
                if reader_cls.can_read(path):
                    return format_name
            except Exception:
                continue

        raise FormatDetectionError(path)

    @classmethod
    def list_formats(cls) -> dict[str, dict[str, bool]]:
        """List available formats and their capabilities.

        Returns:
            Dictionary mapping format names to capability dicts.

        Example:
            {
                "rlds": {"can_read": True, "can_write": False},
                "lerobot-v3": {"can_read": True, "can_write": True},
            }
        """
        all_formats = set(cls._readers.keys()) | set(cls._writers.keys())
        return {
            name: {
                "can_read": name in cls._readers,
                "can_write": name in cls._writers,
            }
            for name in sorted(all_formats)
        }

    @classmethod
    def has_reader(cls, format_name: str) -> bool:
        """Check if a reader is registered for the format.

        Args:
            format_name: Format identifier.

        Returns:
            True if reader is registered.
        """
        return format_name in cls._readers

    @classmethod
    def has_writer(cls, format_name: str) -> bool:
        """Check if a writer is registered for the format.

        Args:
            format_name: Format identifier.

        Returns:
            True if writer is registered.
        """
        return format_name in cls._writers

    @classmethod
    def clear(cls) -> None:
        """Clear all registered formats. Primarily for testing."""
        cls._readers.clear()
        cls._writers.clear()
