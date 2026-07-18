"""IO utilities for Forge.

Centralizes filesystem access so that every command can accept local paths,
HuggingFace URLs (handled in :mod:`forge.hub`), and cloud object-store URIs
(``s3://``, ``gs://``) through a single abstraction built on fsspec.

The public surface is intentionally small — see :mod:`forge.io.paths`.
"""

from __future__ import annotations

from forge.io.paths import (
    RemoteWriteNotSupportedError,
    cleanup_localized,
    get_filesystem,
    get_scheme,
    is_remote_uri,
    localize,
)

__all__ = [
    "RemoteWriteNotSupportedError",
    "cleanup_localized",
    "get_filesystem",
    "get_scheme",
    "is_remote_uri",
    "localize",
]
