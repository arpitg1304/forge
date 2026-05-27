"""Download datasets from HuggingFace Hub.

Uses huggingface_hub for efficient downloading with:
- Automatic caching
- Resume support for interrupted downloads
- Progress tracking
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from forge.core.exceptions import MissingDependencyError
from forge.hub.url import HFDatasetRef, is_hf_url, parse_hf_url

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class CachedDataset:
    """A dataset found in a local HuggingFace-style cache.

    Attributes:
        repo_id: The HF repo identifier, e.g. "lerobot/aloha_static_coffee".
        snapshot_path: Path to the snapshot directory on disk. None if the
            repo folder exists but no populated snapshot was found (stub).
        size_bytes: Total disk usage of this repo's cache (blobs + small
            metadata). 0 if the cache layout doesn't track blob size.
        file_counts: Counts of dataset-like files in the snapshot, keyed by
            extension (".parquet", ".mp4", etc.). Empty for stubs.
        is_stub: True if no populated snapshot exists (only metadata).
    """

    repo_id: str
    snapshot_path: Path | None
    size_bytes: int = 0
    file_counts: dict[str, int] = field(default_factory=dict)
    is_stub: bool = False

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


def get_lerobot_cache_dir() -> Path:
    """Get the path to the LeRobot library's local cache directory.

    LeRobot stores datasets under `<root>/<org>/<repo>/` rather than the
    HF Hub `datasets--<org>--<repo>/snapshots/<sha>/` layout.

    Honours LEROBOT_HOME if set, otherwise falls back to
    `$HF_HOME/lerobot` (if HF_HOME is set) or `~/.cache/huggingface/lerobot`.
    """
    lerobot_home = os.environ.get("LEROBOT_HOME")
    if lerobot_home:
        return Path(lerobot_home)

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "lerobot"

    return Path.home() / ".cache" / "huggingface" / "lerobot"


def find_in_lerobot_cache(
    repo_id: str,
    cache_dir: Path | str | None = None,
) -> Path | None:
    """Look up a dataset in the local LeRobot cache.

    LeRobot stores datasets as plain `<cache>/<org>/<repo>/` directories
    with the dataset's content (data/, meta/, videos/) directly inside.
    Returns the directory if it exists and contains at least one real
    dataset file (mp4/parquet/etc.), else None.
    """
    root = Path(cache_dir) if cache_dir is not None else get_lerobot_cache_dir()
    candidate = root / repo_id
    if not candidate.is_dir():
        return None
    if not _snapshot_is_populated(candidate):
        return None
    return candidate


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


def _repo_id_from_folder(folder_name: str) -> str | None:
    """Reverse of `_repo_folder_name`: "datasets--org--repo" → "org/repo".

    Returns None for folder names that don't fit the dataset cache pattern
    (e.g. `models--*` entries that share the same cache root).
    """
    if not folder_name.startswith("datasets--"):
        return None
    body = folder_name[len("datasets--"):]
    if "--" not in body:
        return None
    org, _, repo = body.partition("--")
    if not org or not repo:
        return None
    return f"{org}/{repo}"


def _scan_snapshot_files(snapshot_dir: Path) -> dict[str, int]:
    """Count dataset-like files per extension in a snapshot directory."""
    counts: dict[str, int] = {}
    for entry in snapshot_dir.rglob("*"):
        if not entry.is_file():
            continue
        ext = entry.suffix.lower()
        if ext in _DATASET_FILE_EXTENSIONS:
            counts[ext] = counts.get(ext, 0) + 1
    return counts


def _repo_size_bytes(repo_dir: Path) -> int:
    """Sum the size of all files (resolving symlinks once) under a repo dir.

    For the HF cache layout we mostly care about `blobs/` — symlinks in
    `snapshots/` point back into blobs, so following symlinks here would
    double-count. We walk the tree and only stat regular files, which
    naturally skips the symlinked snapshot entries.
    """
    total = 0
    for entry in repo_dir.rglob("*"):
        try:
            if entry.is_symlink() or not entry.is_file():
                continue
            total += entry.stat().st_size
        except OSError:
            continue
    return total


def _is_hf_hub_cache_root(root: Path) -> bool:
    """True if `root` looks like an HF Hub cache (contains datasets--*)."""
    try:
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith("datasets--"):
                return True
    except OSError:
        pass
    return False


def _detect_format(path: Path) -> str | None:
    """Best-effort format detection. Returns None instead of raising."""
    try:
        from forge.formats.registry import FormatRegistry

        return FormatRegistry.detect_format(path)
    except Exception:
        return None


def _walk_for_datasets(
    root: Path, max_depth: int = 3
) -> "list[tuple[Path, str]]":
    """Yield (dataset_dir, format) pairs found under `root`.

    Stops descending once a directory is identified as a dataset — we
    don't expect datasets to nest within datasets in any supported layout.
    """
    found: list[tuple[Path, str]] = []

    def _walk(d: Path, depth: int) -> None:
        fmt = _detect_format(d)
        if fmt is not None:
            found.append((d, fmt))
            return
        if depth >= max_depth:
            return
        try:
            children = sorted(d.iterdir())
        except OSError:
            return
        for child in children:
            if child.is_dir() and not child.name.startswith("."):
                _walk(child, depth + 1)

    _walk(root, 0)
    return found


def _generic_repo_id(dataset_dir: Path, root: Path) -> str:
    """Compose a readable id for a dataset found via generic-layout scan.

    If the dataset is exactly one level under `root`, prefix the root's
    basename so the id looks HF-ish (e.g. `arpitg1304/stack_lego` when
    root is `~/.cache/huggingface/lerobot/arpitg1304`). Otherwise use the
    relative path under `root`.
    """
    rel = dataset_dir.relative_to(root)
    parts = rel.parts
    if len(parts) == 1:
        return f"{root.name}/{parts[0]}"
    return str(rel)


def _list_hf_hub_cache(
    root: Path, *, include_stubs: bool
) -> list[CachedDataset]:
    """Scan an HF Hub-style cache root."""
    results: list[CachedDataset] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        repo_id = _repo_id_from_folder(entry.name)
        if repo_id is None:
            continue

        snapshot = find_in_hf_cache(repo_id, cache_dir=root)
        is_stub = snapshot is None
        if is_stub and not include_stubs:
            continue

        size = _repo_size_bytes(entry)
        file_counts = _scan_snapshot_files(snapshot) if snapshot else {}

        results.append(
            CachedDataset(
                repo_id=repo_id,
                snapshot_path=snapshot,
                size_bytes=size,
                file_counts=file_counts,
                is_stub=is_stub,
            )
        )

    results.sort(key=lambda d: d.repo_id)
    return results


def _list_generic_cache(root: Path) -> list[CachedDataset]:
    """Walk a generic / LeRobot-style cache root for dataset directories."""
    results: list[CachedDataset] = []
    for dataset_dir, _fmt in _walk_for_datasets(root):
        repo_id = _generic_repo_id(dataset_dir, root)
        results.append(
            CachedDataset(
                repo_id=repo_id,
                snapshot_path=dataset_dir,
                size_bytes=_repo_size_bytes(dataset_dir),
                file_counts=_scan_snapshot_files(dataset_dir),
                is_stub=False,
            )
        )
    results.sort(key=lambda d: d.repo_id)
    return results


def list_local_datasets(
    cache_dir: Path | str | None = None,
    *,
    include_stubs: bool = False,
) -> list[CachedDataset]:
    """List datasets in a local cache directory.

    Auto-detects two layouts:

    - **HuggingFace Hub cache** (`datasets--{org}--{repo}/snapshots/{sha}/...`).
      Used by `huggingface_hub` and the `datasets` library.
    - **Generic / LeRobot cache** (`{org}/{repo}/data/...`, `{org}/{repo}/meta/...`,
      or any nested directory layout up to 3 levels deep). Used by the
      LeRobot library's `~/.cache/huggingface/lerobot/...` tree and by
      ad-hoc local directories of datasets.

    Args:
        cache_dir: Cache root to scan. Defaults to `get_hf_cache_dir()`.
        include_stubs: For HF Hub layout only — include metadata-only
            stubs that lack real data files. Ignored for generic layout
            (where the walker already requires a detectable format).

    Returns:
        List of CachedDataset entries, sorted by repo_id.
    """
    root = Path(cache_dir) if cache_dir is not None else get_hf_cache_dir()
    if not root.is_dir():
        return []

    if _is_hf_hub_cache_root(root):
        return _list_hf_hub_cache(root, include_stubs=include_stubs)
    return _list_generic_cache(root)


# Back-compat alias for the original function name. Same semantics now
# that dispatch is layout-aware.
list_hf_cache_datasets = list_local_datasets


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
