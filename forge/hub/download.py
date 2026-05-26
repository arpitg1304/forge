"""Download datasets from HuggingFace Hub.

Uses huggingface_hub for efficient downloading with:
- Automatic caching
- Resume support for interrupted downloads
- Progress tracking
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from forge.core.exceptions import MissingDependencyError
from forge.hub.url import HFDatasetRef, is_hf_url, parse_hf_url

if TYPE_CHECKING:
    from collections.abc import Callable

# File extensions that indicate a snapshot has actual dataset content,
# not just metadata stubs (info.json/README/.gitattributes).
_DATASET_FILE_EXTENSIONS = {
    ".parquet",
    ".mp4",
    ".webm",
    ".avi",
    ".h5",
    ".hdf5",
    ".tfrecord",
    ".npy",
    ".npz",
    ".zarr",
    ".arrow",
    ".bag",
    ".mcap",
    ".db3",
}


def _check_huggingface_hub() -> None:
    """Check if huggingface_hub is available."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        raise MissingDependencyError(
            dependency="huggingface_hub",
            feature="HuggingFace Hub integration",
            install_hint="pip install huggingface_hub",
        )


def get_cache_dir() -> Path | None:
    """Get the cache directory to use for downloads.

    If FORGE_CACHE_DIR is set, returns that (with a /datasets subdir for
    backwards compatibility with older forge cache layouts). Otherwise
    returns None, signalling "use the standard HuggingFace Hub cache"
    (`~/.cache/huggingface/hub/`) so downloads dedupe with other HF tools.
    """
    cache_dir = os.environ.get("FORGE_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir) / "datasets"
    return None


def get_hf_cache_dir() -> Path:
    """Get the path to the standard HuggingFace Hub cache directory.

    Honours HF_HOME / HUGGINGFACE_HUB_CACHE / HF_HUB_CACHE if set, falling
    back to ~/.cache/huggingface/hub.
    """
    hf_hub_cache = os.environ.get("HF_HUB_CACHE") or os.environ.get(
        "HUGGINGFACE_HUB_CACHE"
    )
    if hf_hub_cache:
        return Path(hf_hub_cache)

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_folder_name(repo_id: str) -> str:
    """Translate a repo_id into the HF cache's folder name.

    HF Hub names dataset cache folders as `datasets--{org}--{repo}`, with
    any `/` in the repo_id replaced by `--`.
    """
    return "datasets--" + repo_id.replace("/", "--")


def _snapshot_is_populated(snapshot_dir: Path) -> bool:
    """Return True if the snapshot contains real dataset files.

    HF caches will sometimes contain only a `meta/info.json` (or just a
    README) for repos the user browsed but never fully pulled. Those stubs
    are useless to forge — we want a snapshot with at least one parquet,
    mp4, hdf5, tfrecord, etc.
    """
    if not snapshot_dir.is_dir():
        return False
    for entry in snapshot_dir.rglob("*"):
        if entry.is_file() and entry.suffix.lower() in _DATASET_FILE_EXTENSIONS:
            return True
    return False


def find_in_hf_cache(
    repo_id: str,
    revision: str | None = None,
    cache_dir: Path | str | None = None,
) -> Path | None:
    """Look up a dataset snapshot in the local HuggingFace Hub cache.

    Returns the path to a populated snapshot if found, else None. A
    "populated" snapshot is one that contains at least one real dataset
    file (parquet/mp4/hdf5/tfrecord/etc.) — metadata-only stubs are
    ignored because forge can't actually work with them.

    Args:
        repo_id: HF repo identifier, e.g. "lerobot/aloha_static_coffee".
        revision: Optional branch name or commit sha. If None, tries
            common default branches (main, master) before falling back to
            the most recently populated snapshot.
        cache_dir: Override the HF cache root (defaults to the standard
            ~/.cache/huggingface/hub location).

    Returns:
        Path to the populated snapshot directory, or None.
    """
    root = Path(cache_dir) if cache_dir is not None else get_hf_cache_dir()
    repo_dir = root / _repo_folder_name(repo_id)
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    candidates: list[Path] = []

    if revision:
        ref_file = repo_dir / "refs" / revision
        if ref_file.is_file():
            try:
                sha = ref_file.read_text().strip()
            except OSError:
                sha = ""
            if sha:
                candidates.append(snapshots_dir / sha)
        candidates.append(snapshots_dir / revision)
    else:
        for branch in ("main", "master"):
            ref_file = repo_dir / "refs" / branch
            if ref_file.is_file():
                try:
                    sha = ref_file.read_text().strip()
                except OSError:
                    continue
                if sha:
                    candidates.append(snapshots_dir / sha)

    for candidate in candidates:
        if _snapshot_is_populated(candidate):
            return candidate

    # Fall back: scan all snapshots, pick the most recently modified
    # populated one. Useful when refs/ is missing or when the user pulled
    # via a tool that didn't write refs.
    fallback: tuple[float, Path] | None = None
    for snap in snapshots_dir.iterdir():
        if not snap.is_dir() or not _snapshot_is_populated(snap):
            continue
        mtime = snap.stat().st_mtime
        if fallback is None or mtime > fallback[0]:
            fallback = (mtime, snap)

    return fallback[1] if fallback else None


def download_dataset(
    source: str | HFDatasetRef,
    *,
    revision: str | None = None,
    cache_dir: Path | str | None = None,
    force_download: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Download a dataset from HuggingFace Hub.

    Args:
        source: Dataset repo_id (e.g., "lerobot/pusht") or HFDatasetRef or hf:// URL.
        revision: Git revision (branch, tag, or commit hash).
        cache_dir: Where to cache downloaded files. Defaults to ~/.cache/forge/datasets.
        force_download: Re-download even if cached.
        progress_callback: Optional callback(downloaded_bytes, total_bytes).

    Returns:
        Path to the downloaded dataset directory.

    Raises:
        MissingDependencyError: If huggingface_hub is not installed.
        ValueError: If the source format is invalid.
    """
    _check_huggingface_hub()
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    # Parse the source
    if isinstance(source, HFDatasetRef):
        ref = source
    elif is_hf_url(source):
        ref = parse_hf_url(source)
    else:
        # Assume it's a repo_id
        ref = HFDatasetRef(repo_id=source)

    # Override revision if provided
    if revision:
        ref = HFDatasetRef(
            repo_id=ref.repo_id,
            revision=revision,
            subset=ref.subset,
        )

    # Resolve cache directory. None means "use the HF default cache",
    # which gives us free dedup with anything pulled via huggingface-cli
    # or the `datasets` library.
    if cache_dir is None:
        cache_dir = get_cache_dir()
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Fast path: a populated snapshot already on disk → skip the download.
    if not force_download:
        cached = find_in_hf_cache(
            ref.repo_id,
            revision=ref.revision,
            cache_dir=cache_dir,
        )
        if cached is not None:
            return cached

    # Download the dataset
    try:
        local_dir = snapshot_download(
            repo_id=ref.repo_id,
            repo_type="dataset",
            revision=ref.revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            force_download=force_download,
            # Use symlinks to save disk space
            local_dir_use_symlinks=True,
        )
        return Path(local_dir)

    except RepositoryNotFoundError:
        raise ValueError(
            f"Dataset not found: {ref.repo_id}\n"
            f"Check that the dataset exists at: https://huggingface.co/datasets/{ref.repo_id}"
        )
    except GatedRepoError:
        raise ValueError(
            f"Dataset '{ref.repo_id}' requires authentication.\n"
            f"Please run: huggingface-cli login\n"
            f"And ensure you have access to: https://huggingface.co/datasets/{ref.repo_id}"
        )


def resolve_path(path: str | Path) -> Path:
    """Resolve a path that may be a local path or HuggingFace URL.

    If the path is an hf:// URL, downloads the dataset and returns the local path.
    Otherwise, returns the path as-is.

    Args:
        path: Local path or HuggingFace URL.

    Returns:
        Resolved local path.
    """
    path_str = str(path)

    if is_hf_url(path_str):
        return download_dataset(path_str)

    return Path(path)
