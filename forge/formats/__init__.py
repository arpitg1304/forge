"""Format readers and writers for Forge.

This module provides the format registry and auto-import of format implementations.
"""

from forge.formats.registry import FormatRegistry

__all__ = ["FormatRegistry"]


def _register_formats() -> None:
    """Import format modules to trigger registration decorators."""
    # Import format modules - this triggers @FormatRegistry.register_* decorators
    try:
        from forge.formats import rlds  # noqa: F401
    except ImportError:
        pass  # RLDS dependencies not installed

    try:
        from forge.formats import lerobot_v2  # noqa: F401
    except ImportError:
        pass

    try:
        from forge.formats import lerobot_v3  # noqa: F401
    except ImportError:
        pass

    try:
        from forge.formats import zarr  # noqa: F401
    except ImportError:
        pass

    try:
        from forge.formats import mcap  # noqa: F401
    except ImportError:
        pass

    try:
        from forge.formats import rosbag  # noqa: F401
    except ImportError:
        pass

    try:
        from forge.formats import hdf5  # noqa: F401
    except ImportError:
        pass

    try:
        from forge.formats import robodm  # noqa: F401
    except ImportError:
        pass

    try:
        from forge.formats import groot  # noqa: F401
    except ImportError:
        pass

    try:
        from forge.formats import tlabel  # noqa: F401
    except ImportError:
        pass  # TLabel dependencies not installed


# Register formats on module import
_register_formats()

