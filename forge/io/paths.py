"""Path resolution for local, HuggingFace, and cloud object-store sources.

This is the single place Forge turns a user-supplied source string into
something the format readers can consume. Readers throughout the codebase are
built around :class:`pathlib.Path` and local file access (``h5py.File``,
``av.open``, ``pq.read_table``, ``Path.glob``, ``open()`` …), so the strategy
here is deliberately simple and uniform:

* **Local paths** and **HuggingFace URLs** are untouched — ``localize`` returns
  them as-is (HF URLs are resolved separately by :mod:`forge.hub`). Local
  behaviour is therefore identical to before this module existed.
* **Cloud URIs** (``s3://``, ``gs://``) are transparently downloaded to a
  temporary directory via fsspec, and that local path is handed back. The temp
  directory is cleaned up automatically at process exit.

Why download instead of streaming? Every reader either seeks randomly inside
files (video keyframe seeks, HDF5 chunk reads, rosbag indexes) or probes the
directory tree with local ``Path`` operations for format detection. Downloading
once keeps those readers — and their tests — completely unchanged while making
cloud sources work everywhere. The fsspec primitives exposed here
(:func:`get_filesystem`) are the foundation for adding true range-read
streaming to individual formats later.

Authentication is delegated entirely to the underlying libraries' default
credential chains (AWS: env vars / shared config / IAM roles; GCP: Application
Default Credentials). Forge never handles credentials itself.
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from forge.core.exceptions import ForgeError, MissingDependencyError

if TYPE_CHECKING:
    from rich.console import Console

# Cloud schemes that Forge advertises + the optional dependency each needs.
# Maps scheme -> (import module, extras name, pip install hint).
_CLOUD_BACKENDS: dict[str, tuple[str, str, str]] = {
    "s3": ("s3fs", "s3", "pip install forge-robotics[s3]"),
    "gs": ("gcsfs", "gcs", "pip install forge-robotics[gcs]"),
    "gcs": ("gcsfs", "gcs", "pip install forge-robotics[gcs]"),
}

# Schemes we do NOT route through fsspec here: local paths (no scheme / file://)
# and HuggingFace URLs (resolved by forge.hub). Everything else with a URL
# scheme (s3, gs, gcs, memory, …) is treated as a remote fsspec source.
_NON_REMOTE_SCHEMES = {"", "file", "hf", "huggingface"}

# Matches a leading "<scheme>://". Kept strict so Windows drive letters
# ("C:\\...", which have no "//") are never mistaken for a URI scheme.
_SCHEME_RE = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*)://")

# Temp directories created for downloaded remote datasets, removed at exit.
_TEMP_ROOTS: list[str] = []
_ATEXIT_REGISTERED = False


class RemoteWriteNotSupportedError(ForgeError):
    """Raised when a cloud URI is given as an output destination.

    Writing outputs directly to object stores is not yet supported — the
    format writers use local filesystem operations. Write locally, then upload.
    """

    def __init__(self, uri: str):
        self.uri = uri
        super().__init__(
            f"Writing outputs to a cloud URI is not supported yet: {uri}\n"
            "Write to a local directory and upload it afterwards, e.g.:\n"
            "  aws s3 cp --recursive ./out s3://bucket/out   (S3)\n"
            "  gcloud storage cp --recursive ./out gs://bucket/out   (GCS)"
        )


def get_scheme(uri: str | os.PathLike[str]) -> str:
    """Return the lowercased URL scheme of ``uri`` (``""`` for local paths)."""
    match = _SCHEME_RE.match(str(uri))
    return match.group("scheme").lower() if match else ""


def is_remote_uri(uri: str | os.PathLike[str]) -> bool:
    """True if ``uri`` is a remote fsspec source (``s3://``, ``gs://``, …).

    Local paths and HuggingFace URLs return False — the former need no
    resolution, the latter are handled by :mod:`forge.hub`.
    """
    return get_scheme(uri) not in _NON_REMOTE_SCHEMES


def _ensure_backend(scheme: str) -> None:
    """Import the fsspec backend for ``scheme`` or raise a clear install error."""
    backend = _CLOUD_BACKENDS.get(scheme)
    if backend is None:
        return  # e.g. "memory" — built into fsspec, nothing to install.
    module, _extra, hint = backend
    try:
        __import__(module)
    except ImportError as exc:
        raise MissingDependencyError(
            dependency=module,
            feature=f"{scheme}:// cloud storage access",
            install_hint=hint,
        ) from exc


def get_filesystem(uri: str | os.PathLike[str]):
    """Return ``(filesystem, path)`` for ``uri`` using fsspec.

    Raises :class:`MissingDependencyError` with the exact pip command if the
    backend for the scheme (e.g. ``s3fs`` for ``s3://``) isn't installed.
    """
    scheme = get_scheme(uri)
    _ensure_backend(scheme)
    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover - fsspec is a core dependency
        raise MissingDependencyError(
            dependency="fsspec",
            feature="cloud storage access",
            install_hint="pip install forge-robotics[s3]  # or [gcs]",
        ) from exc
    return fsspec.core.url_to_fs(str(uri))


def _register_atexit_cleanup() -> None:
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(cleanup_localized)
        _ATEXIT_REGISTERED = True


def cleanup_localized() -> None:
    """Remove all temp directories created for downloaded remote datasets."""
    while _TEMP_ROOTS:
        root = _TEMP_ROOTS.pop()
        shutil.rmtree(root, ignore_errors=True)


def _new_temp_root() -> Path:
    root = tempfile.mkdtemp(prefix="forge-cloud-")
    _TEMP_ROOTS.append(root)
    _register_atexit_cleanup()
    return Path(root)


def _download(
    fs,
    remote_path: str,
    scheme: str,
    original: str,
    console: Console | None,
) -> Path:
    """Download a remote file or directory to a fresh temp dir; return local path."""
    dest_root = _new_temp_root()

    is_dir = fs.isdir(remote_path)
    is_file = False if is_dir else fs.isfile(remote_path)
    if not is_dir and not is_file:
        raise FileNotFoundError(
            f"Remote source not found (or empty): {original}"
        )

    if is_file:
        local = dest_root / os.path.basename(remote_path.rstrip("/"))
        _copy_files(fs, [(remote_path, local)], console, original)
        return local

    # Directory: mirror the tree under dest_root, preserving relative layout.
    prefix = remote_path.rstrip("/")
    detail = fs.find(prefix, detail=True)
    pairs: list[tuple[str, Path]] = []
    for key, meta in detail.items():
        if isinstance(meta, dict) and meta.get("type") == "directory":
            continue
        rel = key[len(prefix):].lstrip("/")
        if not rel:  # a single object living exactly at the prefix
            rel = os.path.basename(key)
        pairs.append((key, dest_root / rel))

    if not pairs:
        raise FileNotFoundError(
            f"Remote source is empty: {original}"
        )

    _copy_files(fs, pairs, console, original)
    return dest_root


def _copy_files(
    fs,
    pairs: list[tuple[str, Path]],
    console: Console | None,
    original: str,
) -> None:
    """Copy ``(remote_key, local_path)`` pairs, with a progress bar if possible."""
    sizes: list[int] = []
    total_bytes = 0
    for key, _ in pairs:
        try:
            size = int(fs.info(key).get("size") or 0)
        except Exception:
            size = 0
        sizes.append(size)
        total_bytes += size

    progress = None
    task = None
    if console is not None:
        try:
            from rich.progress import (
                BarColumn,
                DownloadColumn,
                Progress,
                TextColumn,
                TimeRemainingColumn,
                TransferSpeedColumn,
            )

            progress = Progress(
                TextColumn("[cyan]Downloading[/cyan] {task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=True,
            )
            label = original if len(original) <= 48 else "…" + original[-47:]
            task = progress.add_task(label, total=total_bytes or None)
            progress.start()
        except Exception:
            progress = None

    try:
        for (key, local), size in zip(pairs, sizes):
            local.parent.mkdir(parents=True, exist_ok=True)
            fs.get_file(key, str(local))
            if progress is not None and task is not None:
                progress.advance(task, size)
    finally:
        if progress is not None:
            progress.stop()


def localize(
    uri: str | os.PathLike[str],
    *,
    console: Console | None = None,
) -> Path:
    """Return a local :class:`Path` for ``uri``, downloading it if remote.

    * Local paths and HuggingFace URLs are returned unchanged (as ``Path`` —
      HF URLs are resolved elsewhere, so callers should handle those first).
    * ``s3://`` / ``gs://`` sources are downloaded to a temp directory that is
      cleaned up at process exit, and the local path is returned.

    Args:
        uri: Local path or remote object-store URI.
        console: Optional rich console for a download progress bar.

    Raises:
        MissingDependencyError: If the backend (s3fs/gcsfs) isn't installed.
        FileNotFoundError: If the remote source doesn't exist or is empty.
    """
    if not is_remote_uri(uri):
        return Path(uri)

    original = str(uri)
    scheme = get_scheme(original)
    fs, remote_path = get_filesystem(original)
    return _download(fs, remote_path, scheme, original, console)
