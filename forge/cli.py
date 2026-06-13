"""Command-line interface for Forge.

Provides a thin wrapper around the Forge API for command-line usage.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="forge",
    help="Forge - Robotics dataset format converter",
    add_completion=False,
)
registry_app = typer.Typer(
    name="registry",
    help="Dataset registry - browse and search known robotics datasets.",
)
app.add_typer(registry_app, name="registry")
tokenize_app = typer.Typer(
    name="tokenize",
    help="Action tokenizers - discretize action streams into tokens.",
)
app.add_typer(tokenize_app, name="tokenize")
console = Console()


def _looks_like_hf_repo_id(path_str: str) -> bool:
    """Return True if `path_str` looks like a bare HF repo_id (`org/repo`).

    Used to give `forge inspect lerobot/aloha_static_coffee` the same
    treatment as `hf://lerobot/aloha_static_coffee` when there's no local
    directory with that name.
    """
    if "/" not in path_str or path_str.startswith((".", "/")):
        return False
    parts = path_str.split("/")
    if len(parts) != 2:
        return False
    org, repo = parts
    if not org or not repo:
        return False
    # Excludes obvious file-like names (e.g. "foo/data.parquet").
    if "." in repo and repo.rsplit(".", 1)[-1].isalpha():
        return False
    return True


def _resolve_via_hub(repo_id_or_url: str) -> Path:
    """Resolve a HuggingFace dataset reference to a local path.

    Resolution order:
    1. Local HF Hub cache (`~/.cache/huggingface/hub/datasets--*`)
    2. Local LeRobot cache (`~/.cache/huggingface/lerobot/<org>/<repo>/`)
    3. Download from HuggingFace Hub
    """
    from forge.hub import (
        download_dataset,
        find_in_hf_cache,
        find_in_lerobot_cache,
        is_hf_url,
        parse_hf_url,
    )
    from forge.hub.url import HFDatasetRef

    if is_hf_url(repo_id_or_url):
        ref = parse_hf_url(repo_id_or_url)
    else:
        ref = HFDatasetRef(repo_id=repo_id_or_url)

    cached = find_in_hf_cache(ref.repo_id, revision=ref.revision)
    if cached is not None:
        console.print(
            f"[cyan]Using cached HuggingFace dataset:[/cyan] {ref.repo_id}"
        )
        console.print(f"[dim]{cached}[/dim]")
        return cached

    lerobot_cached = find_in_lerobot_cache(ref.repo_id)
    if lerobot_cached is not None:
        console.print(
            f"[cyan]Using cached LeRobot dataset:[/cyan] {ref.repo_id}"
        )
        console.print(f"[dim]{lerobot_cached}[/dim]")
        return lerobot_cached

    console.print(f"[cyan]Downloading from HuggingFace Hub:[/cyan] {ref.repo_id}")
    with console.status("[bold green]Downloading dataset..."):
        local_path = download_dataset(ref)
    console.print(f"[green]Downloaded to:[/green] {local_path}")
    return local_path


def _resolve_dataset_path(path_str: str, demo: bool = False) -> Path:
    """Resolve a dataset path, downloading from HuggingFace Hub if needed.

    Resolution order:
    1. HuggingFace URL (hf://org/repo) — reuse local HF cache or download
    2. Existing filesystem path — return as-is
    3. Registry dataset ID — lookup and resolve source URI
    4. Bare HF repo_id (`org/repo`) — reuse local HF cache or download
    5. Fall through as filesystem path (for error reporting by caller)

    Args:
        path_str: Local path, HuggingFace URL, or registry dataset ID.
        demo: If True and resolving via registry, use demo-suitable source.

    Returns:
        Resolved local path.
    """
    from forge.hub import is_hf_url

    if is_hf_url(path_str):
        return _resolve_via_hub(path_str)

    # Check filesystem first
    path = Path(path_str)
    if path.exists():
        return path

    # Try registry lookup for bare identifiers (no /, no . prefix)
    if "/" not in path_str and not path_str.startswith("."):
        try:
            from forge.registry import DatasetRegistry

            entry = DatasetRegistry.get(path_str)
            source = DatasetRegistry.get_source(path_str, demo=demo)
            console.print(
                f"[cyan]Resolved from registry:[/cyan] {entry.name} "
                f"({entry.format})"
            )

            if source.type == "hf_hub":
                return _resolve_dataset_path(f"hf://{source.uri}")
            elif source.type == "gcs":
                console.print(f"[yellow]GCS source:[/yellow] {source.uri}")
                if source.notes:
                    console.print(f"[dim]{source.notes}[/dim]")
                console.print("Download manually with: gsutil cp -r ...")
                raise typer.Exit(1)
            else:
                console.print(f"[yellow]Source:[/yellow] {source.uri}")
                if source.notes:
                    console.print(f"[dim]{source.notes}[/dim]")
                raise typer.Exit(1)
        except ImportError:
            pass
        except Exception as e:
            from forge.core.exceptions import DatasetNotFoundError

            if isinstance(e, DatasetNotFoundError):
                # Don't swallow — let it fall through to filesystem path
                pass
            elif isinstance(e, (typer.Exit, SystemExit)):
                raise
            # Other errors: fall through silently

    # Bare `org/repo` that didn't match a local path or registry entry —
    # try treating it as a HuggingFace repo_id.
    if _looks_like_hf_repo_id(path_str):
        return _resolve_via_hub(path_str)

    return path


def _quick_inspect_hub(path: str, output: str = "text") -> None:
    """Quick inspect a HuggingFace Hub dataset without downloading.

    Fetches metadata from Hub API and analyzes file structure.
    """
    from forge.hub import parse_hf_url

    try:
        from huggingface_hub import HfApi, hf_hub_url
    except ImportError:
        console.print("[red]Error:[/red] huggingface_hub is required.")
        console.print("Install with: pip install huggingface_hub")
        raise typer.Exit(1)

    ref = parse_hf_url(path)
    api = HfApi()

    with console.status(f"[bold green]Fetching metadata for {ref.repo_id}..."):
        try:
            # Get dataset info from Hub API
            info = api.dataset_info(ref.repo_id, revision=ref.revision)
            files = list(api.list_repo_files(ref.repo_id, repo_type="dataset", revision=ref.revision))
        except Exception as e:
            console.print(f"[red]Error fetching dataset info:[/red] {e}")
            raise typer.Exit(1)

    # Analyze file structure to detect format
    format_detected = "unknown"
    file_stats: dict[str, int] = {}

    for f in files:
        ext = Path(f).suffix.lower()
        if ext:
            file_stats[ext] = file_stats.get(ext, 0) + 1

    # Detect format from files
    if any(f.endswith(".tfrecord") for f in files):
        format_detected = "rlds"
    elif any("parquet" in f for f in files):
        if any("meta/info.json" in f for f in files):
            format_detected = "lerobot-v3"
        else:
            format_detected = "lerobot-v2"
    elif any(f.endswith(".zarr") or "/.zarray" in f for f in files):
        format_detected = "zarr"
    elif any(f.endswith((".hdf5", ".h5")) for f in files):
        format_detected = "hdf5"
    elif any(f.endswith((".bag", ".mcap", ".db3")) for f in files):
        format_detected = "rosbag"

    # Count episodes/files
    parquet_files = [f for f in files if f.endswith(".parquet")]
    hdf5_files = [f for f in files if f.endswith((".hdf5", ".h5"))]
    tfrecord_files = [f for f in files if f.endswith(".tfrecord")]
    video_files = [f for f in files if f.endswith((".mp4", ".webm", ".avi"))]

    # Infer how camera frames are physically stored.
    # For lerobot-v3, the absence of any .mp4 alongside data parquet shards
    # means frames are inline image bytes in the parquet rows (dtype=image).
    camera_storage: str | None = None
    if format_detected == "lerobot-v3" and parquet_files:
        camera_storage = "mp4" if video_files else "inline images in parquet"
    elif format_detected == "lerobot-v2" and parquet_files:
        camera_storage = "mp4" if video_files else "inline images in parquet"

    # Calculate total size
    total_size = 0
    if info.siblings:
        total_size = sum(s.size or 0 for s in info.siblings if s.size)

    if output == "json":
        import json
        data = {
            "repo_id": ref.repo_id,
            "format": format_detected,
            "total_files": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1_000_000, 2),
            "file_types": file_stats,
            "parquet_files": len(parquet_files),
            "hdf5_files": len(hdf5_files),
            "tfrecord_files": len(tfrecord_files),
            "video_files": len(video_files),
            "camera_storage": camera_storage,
            "downloads": info.downloads,
            "likes": info.likes,
            "tags": info.tags,
        }
        console.print(json.dumps(data, indent=2))
        return

    # Rich formatted output
    console.print()
    console.print(f"[bold]Dataset:[/bold] {ref.repo_id}")
    console.print(f"[bold]Format:[/bold] {format_detected} [dim](detected from files)[/dim]")

    if total_size > 0:
        if total_size > 1_000_000_000:
            size_str = f"{total_size / 1_000_000_000:.2f} GB"
        else:
            size_str = f"{total_size / 1_000_000:.2f} MB"
        console.print(f"[bold]Total size:[/bold] {size_str}")

    console.print(f"[bold]Total files:[/bold] {len(files)}")

    # File breakdown
    if file_stats:
        console.print()
        table = Table(title="File Types")
        table.add_column("Extension", style="cyan")
        table.add_column("Count", justify="right")
        for ext, count in sorted(file_stats.items(), key=lambda x: -x[1]):
            table.add_row(ext, str(count))
        console.print(table)

    # Key counts
    console.print()
    if parquet_files:
        console.print(f"[bold]Parquet files:[/bold] {len(parquet_files)}")
    if hdf5_files:
        console.print(f"[bold]HDF5 files:[/bold] {len(hdf5_files)}")
    if tfrecord_files:
        console.print(f"[bold]TFRecord files:[/bold] {len(tfrecord_files)}")
    if video_files:
        console.print(f"[bold]Video files:[/bold] {len(video_files)}")

    if camera_storage:
        console.print(f"[bold]Camera storage:[/bold] {camera_storage}")

    # Hub stats
    console.print()
    console.print(f"[dim]Downloads: {info.downloads or 0} | Likes: {info.likes or 0}[/dim]")

    if info.tags:
        console.print(f"[dim]Tags: {', '.join(info.tags[:5])}{'...' if len(info.tags) > 5 else ''}[/dim]")

    console.print()
    console.print("[yellow]Note:[/yellow] Use without --quick to download and get full schema details")


@app.command("inspect")
def inspect_cmd(
    path: str = typer.Argument(..., help="Path to dataset (local path or hf://org/repo)"),
    format: str | None = typer.Option(
        None, "--format", "-f", help="Format hint (auto-detected if not provided)"
    ),
    output: str = typer.Option("text", "--output", "-o", help="Output format: text, json"),
    deep: bool = typer.Option(
        False, "--deep", "-d", help="Deep scan all episodes (slower but more accurate)"
    ),
    samples: int = typer.Option(5, "--samples", "-s", help="Number of episodes to sample"),
    generate_config: Path | None = typer.Option(
        None, "--generate-config", "-g", help="Generate a YAML config template and save to this path"
    ),
    quick: bool = typer.Option(
        False, "--quick", "-q", help="Quick inspect for Hub datasets (metadata only, no download)"
    ),
) -> None:
    """Inspect a dataset and show its structure.

    Supports both local paths and HuggingFace Hub datasets.
    Use --generate-config to create a YAML config template based on the detected schema.
    Use --quick for Hub datasets to see metadata without downloading.

    Examples:
        forge inspect my_dataset/
        forge inspect hf://lerobot/pusht
        forge inspect hf://lerobot/pusht --quick
        forge inspect my_dataset/ --generate-config config.yaml
    """
    from forge.core.exceptions import ForgeError
    from forge.hub import is_hf_url

    # Quick inspect for Hub datasets (metadata only, no download)
    if quick and is_hf_url(path):
        _quick_inspect_hub(path, output)
        return

    # Check dataset size for HuggingFace URLs and offer --quick for large datasets
    if is_hf_url(path) and not quick:
        try:
            from huggingface_hub import HfApi
            from forge.hub import parse_hf_url

            ref = parse_hf_url(path)
            api = HfApi()

            with console.status("[dim]Checking dataset size...[/dim]"):
                # Use files_metadata=True to get file sizes
                info = api.dataset_info(ref.repo_id, revision=ref.revision, files_metadata=True)
                # Calculate total size from siblings (file list)
                total_size = 0
                num_files = 0
                if info.siblings:
                    total_size = sum(s.size or 0 for s in info.siblings if s.size)
                    num_files = len(info.siblings)

            # Warn if dataset is large (> 500MB)
            if total_size > 500_000_000:
                size_gb = total_size / 1_000_000_000
                console.print(f"[yellow]Warning:[/yellow] Dataset is large ({size_gb:.2f} GB, {num_files} files)")
                use_quick = typer.confirm("Use --quick mode (metadata only, no download)?", default=True)
                if use_quick:
                    _quick_inspect_hub(path, output)
                    return
        except Exception:
            pass  # Continue with normal flow if size check fails

    from forge.inspect import InspectionOptions, Inspector

    # Resolve HuggingFace URLs to local paths
    try:
        resolved_path = _resolve_dataset_path(path)
    except Exception as e:
        console.print(f"[red]Error downloading dataset:[/red] {e}")
        raise typer.Exit(1)

    options = InspectionOptions(
        sample_episodes=samples,
        deep_scan=deep,
    )
    inspector = Inspector(options)

    try:
        info = inspector.inspect(resolved_path, format)
    except ForgeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if output == "json":
        # Convert to JSON-serializable dict
        data = {
            "path": str(info.path),
            "format": info.format,
            "format_version": info.format_version,
            "num_episodes": info.num_episodes,
            "total_frames": info.total_frames,
            "cameras": {
                name: {
                    "height": cam.height,
                    "width": cam.width,
                    "channels": cam.channels,
                    "encoding": cam.encoding,
                    "storage": cam.storage,
                }
                for name, cam in info.cameras.items()
            },
            "observation_schema": {
                name: {
                    "shape": list(schema.shape),
                    "dtype": schema.dtype.value,
                }
                for name, schema in info.observation_schema.items()
            },
            "action_schema": (
                {
                    "shape": list(info.action_schema.shape),
                    "dtype": info.action_schema.dtype.value,
                }
                if info.action_schema
                else None
            ),
            "has_timestamps": info.has_timestamps,
            "has_language": info.has_language,
            "language_coverage": info.language_coverage,
            "has_rewards": info.has_rewards,
            "inferred_fps": info.inferred_fps,
            "inferred_gripper_index": info.inferred_gripper_index,
            "missing_required": info.missing_required,
            "sample_episode_id": info.sample_episode_id,
            "sample_num_frames": info.sample_num_frames,
            "sample_language": info.sample_language,
        }
        console.print(json.dumps(data, indent=2))
        return

    # Rich formatted output
    console.print()
    console.print(f"[bold]Dataset:[/bold] {info.path}")
    console.print(
        f"[bold]Format:[/bold] {info.format}"
        + (f" (v{info.format_version})" if info.format_version else "")
    )
    console.print(f"[bold]Episodes:[/bold] {info.num_episodes}")
    if info.total_frames:
        console.print(f"[bold]Total frames:[/bold] {info.total_frames}")

    # Schema table
    if info.observation_schema:
        console.print()
        table = Table(title="Observation Schema")
        table.add_column("Field", style="cyan")
        table.add_column("Type")
        table.add_column("Shape")

        for name, schema in info.observation_schema.items():
            table.add_row(name, schema.dtype.value, str(schema.shape))

        console.print(table)

    # Action schema
    if info.action_schema:
        console.print()
        console.print(
            f"[bold]Action:[/bold] {info.action_schema.dtype.value} {info.action_schema.shape}"
        )

    # Cameras
    if info.cameras:
        console.print()
        console.print("[bold]Cameras:[/bold]")
        storage_label = {"mp4": "mp4 video", "image": "inline image in parquet"}
        for name, cam in info.cameras.items():
            label = storage_label.get(cam.storage)
            suffix = f", {label}" if label else ""
            console.print(
                f"  {name}: {cam.width}x{cam.height} ({cam.encoding}{suffix})"
            )

    # Inferred properties
    console.print()
    console.print("[bold]Inferred Properties:[/bold]")
    console.print(f"  FPS: {info.inferred_fps or '[dim]unknown[/dim]'}")
    console.print(
        f"  Gripper index: {info.inferred_gripper_index if info.inferred_gripper_index is not None else '[dim]unknown[/dim]'}"
    )
    console.print(f"  Timestamps: {'yes' if info.has_timestamps else 'no'}")
    console.print(f"  Language: {'yes' if info.has_language else 'no'}")
    if info.has_language:
        console.print(f"  Language coverage: {info.language_coverage:.0%}")
    console.print(f"  Rewards: {'yes' if info.has_rewards else 'no'}")

    # Sample
    if info.sample_episode_id:
        console.print()
        console.print("[bold]Sample Episode:[/bold]")
        console.print(f"  ID: {info.sample_episode_id}")
        if info.sample_num_frames:
            console.print(f"  Frames: {info.sample_num_frames}")
        if info.sample_language:
            console.print(f'  Language: "{info.sample_language}"')

    # Missing requirements
    if info.missing_required:
        console.print()
        console.print(
            f"[yellow]Missing for conversion:[/yellow] {', '.join(info.missing_required)}"
        )
        console.print("\n[dim]Provide these values in a config file for conversion.[/dim]")
    else:
        console.print()
        console.print("[green]Ready for conversion[/green]")

    # Generate config template if requested
    if generate_config:
        _generate_config_template(info, generate_config)


def _generate_config_template(info: "DatasetInfo", output_path: Path) -> None:
    """Generate a YAML config template based on dataset inspection.

    For MCAP sources, emits a topic config (drives the MCAP reader) since that
    is the artifact downstream commands need. For other formats, emits a
    conversion config (target_format / fps / camera mapping).

    Args:
        info: Dataset inspection info.
        output_path: Path to save the generated config.
    """
    if info.format == "mcap":
        _generate_mcap_topic_config(info, output_path)
        return

    lines = [
        "# Forge conversion configuration",
        f"# Generated from: {info.path}",
        f"# Source format: {info.format}",
        "",
        "# Target format for conversion",
        "target_format: lerobot-v3",
        "",
    ]

    # Add FPS and robot type
    if info.inferred_fps:
        lines.append(f"fps: {info.inferred_fps}")
    else:
        lines.append("# fps: 30  # Specify FPS (required if not in source)")

    if info.inferred_robot_type:
        lines.append(f"robot_type: {info.inferred_robot_type}")
    else:
        lines.append("# robot_type: franka  # Specify robot type")

    lines.append("")

    # Camera mappings
    if info.cameras:
        lines.append("# Camera name mapping (source → target)")
        lines.append("# Modify target names as needed for your use case")
        lines.append("cameras:")
        for cam_name in info.cameras:
            # Normalize the camera name for suggestion
            normalized = cam_name
            for prefix in ["steps/observation/", "observation.images.", "observation/", "steps/"]:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
            # Suggest keeping the normalized name, user can change it
            lines.append(f"  {normalized}: {normalized}")
        lines.append("")

    # Field mappings - try to identify action and state fields
    if info.observation_schema:
        lines.append("# Field mapping (source → target)")
        lines.append("# Uncomment and modify as needed")
        lines.append("fields:")

        # Look for action field
        action_candidates = []
        state_candidates = []
        for field_name, schema in info.observation_schema.items():
            name_lower = field_name.lower()
            if "action" in name_lower:
                action_candidates.append(field_name)
            elif any(kw in name_lower for kw in ["state", "proprio", "joint", "ee_pos", "qpos"]):
                state_candidates.append(field_name)

        if action_candidates:
            lines.append(f"  action: {action_candidates[0]}")
        else:
            lines.append("  # action: steps/action  # Specify action field path")

        if state_candidates:
            lines.append(f"  state: {state_candidates[0]}")
        else:
            lines.append("  # state: observation/robot_state  # Specify state field path")

        lines.append("")

    # Video settings
    lines.append("# Video encoding settings (optional)")
    lines.append("# video:")
    lines.append("#   codec: h264")
    lines.append("#   crf: 23  # Quality (lower = better, 18-28 typical)")
    lines.append("")

    # Behavior settings
    lines.append("# Behavior settings")
    lines.append("fail_on_error: false")
    lines.append("skip_existing: true")

    # Write the file
    config_content = "\n".join(lines) + "\n"
    output_path.write_text(config_content)

    console.print()
    console.print(f"[green]Config template saved to:[/green] {output_path}")
    console.print("[dim]Edit the file to customize camera/field mappings before conversion.[/dim]")


def _generate_mcap_topic_config(info: "DatasetInfo", output_path: Path) -> None:
    """Generate an MCAP topic config from a parsed channel inventory.

    Delegates to forge.formats.mcap.generate_config — the same heuristics
    used elsewhere — and writes the result via dump_config so the YAML stays
    in sync with the dataclass schema.
    """
    from forge.formats.mcap import generate_config
    from forge.formats.mcap.topic_config import dump_config

    result = generate_config(info.path)
    if result.skipped or result.config is None:
        console.print(
            f"[yellow]Skipped:[/yellow] could not auto-generate MCAP topic config "
            f"({result.reason}). No file written."
        )
        raise typer.Exit(0)

    dump_config(result.config, output_path)
    console.print()
    console.print(f"[green]MCAP topic config saved to:[/green] {output_path}")
    if result.notes:
        console.print("[yellow]Notes:[/yellow]")
        for note in result.notes:
            console.print(f"  {note}")
    console.print(
        "[dim]Edit the YAML to refine field/topic mappings, then use it with "
        "`forge convert <file>.mcap ./out --config <yaml>`.[/dim]"
    )


@app.command("convert")
def convert_cmd(
    source: str = typer.Argument(..., help="Path to source dataset (local path or hf://org/repo)"),
    output: Path = typer.Argument(..., help="Path for output dataset"),
    config_file: Path | None = typer.Option(
        None, "--config", help="YAML config file for conversion settings"
    ),
    target_format: str | None = typer.Option(
        None, "--format", "-f", help="Target format (default: lerobot-v3)"
    ),
    source_format: str | None = typer.Option(
        None, "--source-format", "-s", help="Source format (auto-detected if not provided)"
    ),
    fps: float | None = typer.Option(None, "--fps", help="Override frames per second"),
    robot_type: str | None = typer.Option(None, "--robot-type", "-r", help="Robot type"),
    camera: list[str] | None = typer.Option(
        None, "--camera", "-c", help="Camera name mapping (format: source=target). Can be repeated."
    ),
    fail_on_error: bool = typer.Option(
        False, "--fail-on-error", help="Stop on first error instead of continuing"
    ),
    visualize: bool = typer.Option(
        False, "--visualize", "-v", help="Open visualizer after conversion to compare source and output"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Show what would be converted without writing any files"
    ),
    workers: int = typer.Option(
        1, "--workers", "-w", help="Number of parallel workers for episode processing (default: 1)"
    ),
) -> None:
    """Convert a dataset from one format to another.

    Supports both local paths and HuggingFace Hub datasets.

    Example:
        forge convert ./rlds_dataset ./output --format lerobot-v3 --fps 30
        forge convert hf://lerobot/pusht ./output --format lerobot-v3
        forge convert ./data.zarr ./output --format lerobot-v3 --visualize
        forge convert ./data.zarr ./output -c front_cam=observation.images.front -c side_cam=observation.images.side
        forge convert ./data.zarr ./output --dry-run  # Preview without writing
        forge convert ./dataset ./output --config conversion.yaml
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from forge.convert import ConversionConfig, Converter
    from forge.core.exceptions import ForgeError
    from forge.formats.registry import FormatRegistry

    # Resolve HuggingFace URLs to local paths
    try:
        resolved_source = _resolve_dataset_path(source)
    except Exception as e:
        console.print(f"[red]Error downloading dataset:[/red] {e}")
        raise typer.Exit(1)

    # Load config from YAML file if provided
    if config_file:
        if not config_file.exists():
            console.print(f"[red]Error:[/red] Config file not found: {config_file}")
            raise typer.Exit(1)
        try:
            config = ConversionConfig.from_yaml(config_file)
            console.print(f"[dim]Loaded config from {config_file}[/dim]")
        except Exception as e:
            console.print(f"[red]Error loading config:[/red] {e}")
            raise typer.Exit(1)
    else:
        config = ConversionConfig()

    # CLI arguments override config file settings
    if target_format:
        config.target_format = target_format
    elif not config.target_format:
        config.target_format = "lerobot-v3"  # Default

    if fps is not None:
        config.fps = fps
    if robot_type:
        config.robot_type = robot_type
    if fail_on_error:
        config.fail_on_error = fail_on_error
    if workers > 1:
        config.num_workers = workers

    # Parse and merge camera mappings (format: source=target)
    # CLI camera mappings override config file mappings
    if camera:
        for mapping in camera:
            if "=" not in mapping:
                console.print(f"[red]Invalid camera mapping '{mapping}'. Use format: source=target[/red]")
                raise typer.Exit(1)
            src, tgt = mapping.split("=", 1)
            config.camera_mapping[src.strip()] = tgt.strip()

    # Dry run mode - just inspect and show what would happen
    if dry_run:
        console.print("[cyan]Dry run mode - no files will be written[/cyan]")
        console.print()

        # Detect source format
        detected_format = source_format or FormatRegistry.detect_format(resolved_source)
        if not detected_format:
            console.print(f"[red]Error:[/red] Could not detect format for {resolved_source}")
            raise typer.Exit(1)

        # Get reader and inspect
        reader = FormatRegistry.get_reader(detected_format)
        info = reader.inspect(resolved_source)

        console.print(f"[bold]Source:[/bold] {source}")
        console.print(f"  Format: {detected_format}")
        console.print(f"  Episodes: {info.num_episodes}")
        console.print(f"  Total frames: {info.total_frames}")
        console.print(f"  FPS: {config.fps or info.inferred_fps or 'unknown'}")

        if info.cameras:
            console.print(f"  Cameras: {len(info.cameras)}")
            for cam_name, cam_info in info.cameras.items():
                mapped_name = config.get_camera_target(cam_name)
                if cam_name != mapped_name:
                    console.print(f"    - {cam_name} → {mapped_name} ({cam_info.width}x{cam_info.height})")
                else:
                    console.print(f"    - {cam_name} ({cam_info.width}x{cam_info.height})")

        if info.action_schema:
            console.print(f"  Action: {info.action_schema.shape}")
        if info.observation_schema:
            console.print(f"  Observations: {len(info.observation_schema)} fields")

        # Show field mappings if configured
        if config.field_mapping:
            console.print(f"  Field mappings: {len(config.field_mapping)}")
            for key, mapping in config.field_mapping.items():
                target = mapping.get_target()
                if mapping.transform:
                    console.print(f"    - {mapping.source} → {target} (transform: {mapping.transform})")
                else:
                    console.print(f"    - {mapping.source} → {target}")

        console.print()
        console.print(f"[bold]Output:[/bold] {output}")
        console.print(f"  Format: {config.target_format}")
        console.print(f"  Robot type: {config.robot_type or info.inferred_robot_type or 'unknown'}")

        console.print()
        console.print("[green]Ready to convert.[/green] Run without --dry-run to proceed.")
        return

    converter = Converter(config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting conversion...", total=None)

        def progress_callback(stage: str, current: int, total: int) -> None:
            if stage == "inspect":
                progress.update(task, description="Inspecting source dataset...")
            elif stage == "episode":
                progress.update(
                    task, description=f"Converting episode {current + 1}/{total or '?'}..."
                )
            elif stage == "finalize":
                progress.update(task, description="Writing metadata...")

        try:
            result = converter.convert(
                resolved_source,
                output,
                target_format=config.target_format,
                source_format=source_format,
                progress_callback=progress_callback,
            )
        except ForgeError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    # Show results
    console.print()
    if result.success:
        console.print("[green]Conversion successful![/green]")
    else:
        console.print("[yellow]Conversion completed with errors[/yellow]")

    console.print(f"  Source format: {result.source_format}")
    console.print(f"  Target format: {result.target_format}")
    console.print(f"  Episodes converted: {result.episodes_converted}")
    if result.episodes_failed > 0:
        console.print(f"  Episodes failed: [red]{result.episodes_failed}[/red]")
    console.print(f"  Total frames: {result.total_frames}")
    console.print(f"  Output: {result.output_path}")

    if result.errors:
        console.print()
        console.print("[yellow]Errors:[/yellow]")
        for error in result.errors[:10]:  # Show first 10 errors
            console.print(f"  - {error}")
        if len(result.errors) > 10:
            console.print(f"  ... and {len(result.errors) - 10} more errors")

    if not result.success:
        raise typer.Exit(1)

    # Open visualizer if requested
    if visualize and result.success:
        try:
            from forge.visualize import UnifiedViewer

            console.print()
            console.print("[cyan]Opening comparison viewer...[/cyan]")
            console.print(f"  Source: {resolved_source}")
            console.print(f"  Output: {output}")
            console.print("[dim]Controls: Episode/Frame sliders, Play/Pause button[/dim]")
            console.print()

            viewer = UnifiedViewer(resolved_source, output)
            viewer.show()
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Could not open visualizer: {e}")


@app.command("visualize")
def visualize_cmd(
    path: str = typer.Argument(..., help="Path to dataset (local path, hf://org/repo, or registry ID)"),
    compare: str | None = typer.Option(
        None, "--compare", "-c", help="Second dataset for side-by-side comparison"
    ),
    episode: int = typer.Option(0, "--episode", "-e", help="Starting episode index"),
    backend: str = typer.Option(
        "web", "--backend", "-b", help="Viewer backend: web (default), matplotlib, opencv, rerun"
    ),
    samples: int = typer.Option(10, "--samples", "-s", help="Max episodes to load"),
    segment: bool = typer.Option(False, "--segment", help="Run segmentation and show phase labels"),
    port: int = typer.Option(0, "--port", help="Port for web viewer (0 = auto)"),
) -> None:
    """Visualize a dataset interactively.

    Supports all formats (RLDS, LeRobot v2/v3, Zarr) through the unified viewer.
    Use --compare to show two datasets side-by-side for comparison.
    Supports HuggingFace Hub URLs (hf://org/repo) and registry IDs.

    Backends:
        web: Browser-based viewer (default). Segment overlay, keyboard controls.
        matplotlib: Interactive with sliders. Slower playback.
        opencv: Fast playback with keyboard controls. No comparison mode.
        rerun: Rerun viewer. Images, time-series, and segment labels on one timeline.

    Examples:
        forge visualize pusht
        forge visualize pusht --segment
        forge visualize hf://lerobot/pusht --backend matplotlib
        forge visualize original/ --compare converted/
        forge visualize dataset/ --backend opencv
        forge visualize dataset/ --backend rerun
        forge visualize dataset/ --backend rerun --episode 2 --segment
    """
    from forge.core.exceptions import ForgeError

    # Resolve path (supports hf:// URLs, registry IDs, and local paths)
    try:
        resolved_path = _resolve_dataset_path(path)
    except Exception as e:
        console.print(f"[red]Error resolving dataset:[/red] {e}")
        raise typer.Exit(1)

    if not resolved_path.exists():
        console.print(f"[red]Error:[/red] Dataset not found: {path}")
        raise typer.Exit(1)

    # Resolve comparison dataset if provided
    resolved_compare: Path | None = None
    if compare:
        try:
            resolved_compare = _resolve_dataset_path(compare)
        except Exception as e:
            console.print(f"[red]Error resolving comparison dataset:[/red] {e}")
            raise typer.Exit(1)
        if not resolved_compare.exists():
            console.print(f"[red]Error:[/red] Comparison dataset not found: {compare}")
            raise typer.Exit(1)

    try:
        if backend.lower() == "web":
            if resolved_compare:
                console.print("[yellow]Warning:[/yellow] Comparison mode not supported with web backend")

            console.print(f"[cyan]Opening web viewer for:[/cyan] {resolved_path}")
            if segment:
                console.print("[dim]Segmentation with phase labels enabled[/dim]")

            from forge.visualize.web_viewer import WebViewer

            viewer = WebViewer(resolved_path, max_episodes=samples, segment=segment, port=port)
            viewer.show()

        elif backend.lower() == "opencv":
            if resolved_compare:
                console.print("[yellow]Warning:[/yellow] Comparison mode not supported with opencv backend")

            console.print(f"[cyan]Opening fast viewer for:[/cyan] {resolved_path}")
            console.print("[dim]Controls: Space=Play/Pause, Arrows=Navigate, +/-=Speed, Q=Quit[/dim]")
            console.print()

            from forge.visualize.cv_viewer import CVViewer

            viewer = CVViewer(resolved_path, max_episodes=samples)
            if episode > 0:
                viewer.current_episode = min(episode, viewer.backend.get_num_episodes() - 1)
            viewer.show()
        elif backend.lower() == "rerun":
            if resolved_compare:
                console.print("[yellow]Warning:[/yellow] Comparison mode not supported with rerun backend")

            console.print(f"[cyan]Opening Rerun viewer for:[/cyan] {resolved_path}")
            if segment:
                console.print("[dim]Segmentation with phase labels enabled[/dim]")
            console.print()

            from forge.visualize.rerun_backend import visualize_rerun

            visualize_rerun(resolved_path, episode_idx=episode, segment=segment, max_episodes=samples)

        else:
            # Matplotlib backend
            if resolved_compare:
                console.print(f"[cyan]Opening comparison viewer:[/cyan]")
                console.print(f"  Left:  {resolved_path}")
                console.print(f"  Right: {resolved_compare}")
            else:
                console.print(f"[cyan]Opening viewer for:[/cyan] {resolved_path}")

            console.print("[dim]Controls: Episode/Frame sliders, Play/Pause button[/dim]")
            console.print()

            from forge.visualize import UnifiedViewer

            viewer = UnifiedViewer(resolved_path, resolved_compare)
            if episode > 0:
                viewer.current_episode = min(episode, viewer.backends[0].get_num_episodes() - 1)
            viewer.show()

    except ForgeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("stats")
def stats_cmd(
    path: str = typer.Argument(..., help="Path to dataset (local or hf://org/repo)"),
    plot: bool = typer.Option(False, "--plot", "-p", help="Show distribution plots"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Save stats to JSON file"),
    sample: int = typer.Option(0, "--sample", "-s", help="Sample N episodes (0 = all)"),
    quality: bool = typer.Option(False, "--quality", "-q", help="Include quality metrics"),
    gripper_dim: int = typer.Option(-1, "--gripper-dim", help="Gripper action dimension index"),
    fps: float = typer.Option(30.0, "--fps", help="Recording FPS (fallback if no timestamps)"),
) -> None:
    """Compute and display dataset statistics.

    Shows episode length distributions, action/state statistics, and coverage metrics.

    Examples:
        forge stats dataset/
        forge stats dataset/ --plot
        forge stats hf://lerobot/aloha_sim_cube --sample 100
        forge stats dataset/ --output stats.json
        forge stats dataset/ --quality
    """
    import numpy as np

    from forge.core.exceptions import ForgeError
    from forge.formats.registry import FormatRegistry
    from forge.inspect.stats_collector import StatsCollector

    # Resolve path (handles hf:// URLs)
    resolved_path = _resolve_dataset_path(path)

    if not resolved_path.exists():
        console.print(f"[red]Error:[/red] Dataset not found: {resolved_path}")
        raise typer.Exit(1)

    # Detect format and get reader
    try:
        format_name = FormatRegistry.detect_format(resolved_path)
        reader = FormatRegistry.get_reader(format_name)
    except ForgeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[cyan]Computing statistics for:[/cyan] {resolved_path}")
    console.print(f"[dim]Format: {format_name}[/dim]")
    console.print()

    # Collect statistics
    collector = StatsCollector()
    episode_lengths: list[int] = []
    all_actions: list[np.ndarray] = []
    all_states: list[np.ndarray] = []

    try:
        with console.status("[bold green]Analyzing episodes...") as status:
            for i, episode in enumerate(reader.read_episodes(resolved_path)):
                if sample > 0 and i >= sample:
                    break

                collector.collect_episode(episode)
                episode_lengths.append(collector._episode_stats[-1].num_frames)

                # Collect action/state distributions (sample frames)
                for j, frame in enumerate(episode.frames()):
                    if j >= 10:  # Sample first 10 frames per episode
                        break
                    if frame.action is not None:
                        all_actions.append(frame.action)
                    if frame.state is not None:
                        all_states.append(frame.state)

                status.update(f"[bold green]Analyzing episodes... ({i + 1} processed)")

    except ForgeError as e:
        console.print(f"[red]Error reading dataset:[/red] {e}")
        raise typer.Exit(1)

    # Aggregate stats
    stats = collector.aggregate()

    # Display statistics
    console.print("[bold]Dataset Statistics[/bold]")
    console.print()

    # Episode counts
    table = Table(title="Episode Statistics", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Total Episodes", str(stats.total_episodes))
    table.add_row("Total Frames", str(stats.total_frames))
    table.add_row("Min Frames/Episode", str(stats.min_frames_per_episode))
    table.add_row("Max Frames/Episode", str(stats.max_frames_per_episode))
    table.add_row("Mean Frames/Episode", f"{stats.mean_frames_per_episode:.1f}")

    console.print(table)
    console.print()

    # Coverage metrics
    coverage_table = Table(title="Coverage Metrics", show_header=False)
    coverage_table.add_column("Metric", style="cyan")
    coverage_table.add_column("Value", style="white")

    coverage_table.add_row("Language Instructions", f"{stats.language_coverage * 100:.1f}%")
    coverage_table.add_row("Success Labels", f"{stats.success_label_coverage * 100:.1f}%")
    coverage_table.add_row("Rewards", f"{stats.reward_coverage * 100:.1f}%")

    console.print(coverage_table)
    console.print()

    # Action/State statistics
    if all_actions:
        actions_array = np.stack(all_actions)
        action_table = Table(title="Action Statistics", show_header=True)
        action_table.add_column("Dim", style="cyan")
        action_table.add_column("Min", style="white")
        action_table.add_column("Max", style="white")
        action_table.add_column("Mean", style="white")
        action_table.add_column("Std", style="white")

        for dim in range(min(actions_array.shape[1], 14)):  # Show up to 14 dims
            action_table.add_row(
                str(dim),
                f"{actions_array[:, dim].min():.3f}",
                f"{actions_array[:, dim].max():.3f}",
                f"{actions_array[:, dim].mean():.3f}",
                f"{actions_array[:, dim].std():.3f}",
            )

        if actions_array.shape[1] > 14:
            action_table.add_row("...", "...", "...", "...", "...")

        console.print(action_table)
        console.print()

    if all_states:
        states_array = np.stack(all_states)
        state_table = Table(title="State Statistics", show_header=True)
        state_table.add_column("Dim", style="cyan")
        state_table.add_column("Min", style="white")
        state_table.add_column("Max", style="white")
        state_table.add_column("Mean", style="white")
        state_table.add_column("Std", style="white")

        for dim in range(min(states_array.shape[1], 14)):  # Show up to 14 dims
            state_table.add_row(
                str(dim),
                f"{states_array[:, dim].min():.3f}",
                f"{states_array[:, dim].max():.3f}",
                f"{states_array[:, dim].mean():.3f}",
                f"{states_array[:, dim].std():.3f}",
            )

        if states_array.shape[1] > 14:
            state_table.add_row("...", "...", "...", "...", "...")

        console.print(state_table)
        console.print()

    # Schema consistency
    if not stats.consistent_action_dim or not stats.consistent_state_dim or not stats.consistent_cameras:
        console.print("[yellow]Warning: Schema inconsistencies detected[/yellow]")
        if not stats.consistent_action_dim:
            console.print("  - Action dimensions vary across episodes")
        if not stats.consistent_state_dim:
            console.print("  - State dimensions vary across episodes")
        if not stats.consistent_cameras:
            console.print("  - Camera sets vary across episodes")
        console.print()

    # Save to JSON if requested
    if output:
        stats_dict = {
            "total_episodes": stats.total_episodes,
            "total_frames": stats.total_frames,
            "min_frames_per_episode": stats.min_frames_per_episode,
            "max_frames_per_episode": stats.max_frames_per_episode,
            "mean_frames_per_episode": stats.mean_frames_per_episode,
            "language_coverage": stats.language_coverage,
            "success_label_coverage": stats.success_label_coverage,
            "reward_coverage": stats.reward_coverage,
            "consistent_action_dim": stats.consistent_action_dim,
            "consistent_state_dim": stats.consistent_state_dim,
            "consistent_cameras": stats.consistent_cameras,
        }

        if all_actions:
            actions_array = np.stack(all_actions)
            stats_dict["action_stats"] = {
                "shape": list(actions_array.shape),
                "min": actions_array.min(axis=0).tolist(),
                "max": actions_array.max(axis=0).tolist(),
                "mean": actions_array.mean(axis=0).tolist(),
                "std": actions_array.std(axis=0).tolist(),
            }

        if all_states:
            states_array = np.stack(all_states)
            stats_dict["state_stats"] = {
                "shape": list(states_array.shape),
                "min": states_array.min(axis=0).tolist(),
                "max": states_array.max(axis=0).tolist(),
                "mean": states_array.mean(axis=0).tolist(),
                "std": states_array.std(axis=0).tolist(),
            }

        with open(output, "w") as f:
            json.dump(stats_dict, f, indent=2)
        console.print(f"[green]Stats saved to:[/green] {output}")

    # Quality metrics if requested
    if quality:
        from forge.quality import QualityAnalyzer, QualityConfig

        qconfig = QualityConfig(gripper_dim=gripper_dim, fps=fps)
        analyzer = QualityAnalyzer(config=qconfig)

        console.print("[bold]Quality Metrics[/bold]")
        console.print()

        report = analyzer.analyze_dataset(resolved_path, sample=sample)

        console.print(f"  Overall Quality Score: [bold]{report.overall_score:.1f}[/bold] / 10")
        console.print()

        for key, val in report.subscores.items():
            bar_len = int(val * 10)
            bar = "[green]" + "█" * bar_len + "[/green]" + "[dim]░[/dim]" * (10 - bar_len)
            label = key.replace("_", " ").title()
            console.print(f"  {label:<24s} {bar}  {val:.2f}")

        if report.flags:
            console.print()
            for flag in report.flags:
                console.print(f"  [yellow]⚠[/yellow] {flag}")

        console.print()

    # Plot if requested
    if plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Episode length histogram
            axes[0, 0].hist(episode_lengths, bins=30, edgecolor="black", alpha=0.7)
            axes[0, 0].set_xlabel("Episode Length (frames)")
            axes[0, 0].set_ylabel("Count")
            axes[0, 0].set_title("Episode Length Distribution")
            axes[0, 0].axvline(
                stats.mean_frames_per_episode, color="red", linestyle="--", label=f"Mean: {stats.mean_frames_per_episode:.1f}"
            )
            axes[0, 0].legend()

            # Action distribution (box plot)
            if all_actions:
                actions_array = np.stack(all_actions)
                num_dims = min(actions_array.shape[1], 14)
                axes[0, 1].boxplot([actions_array[:, i] for i in range(num_dims)])
                axes[0, 1].set_xlabel("Action Dimension")
                axes[0, 1].set_ylabel("Value")
                axes[0, 1].set_title("Action Distribution by Dimension")
            else:
                axes[0, 1].text(0.5, 0.5, "No action data", ha="center", va="center")
                axes[0, 1].set_title("Action Distribution")

            # State distribution (box plot)
            if all_states:
                states_array = np.stack(all_states)
                num_dims = min(states_array.shape[1], 14)
                axes[1, 0].boxplot([states_array[:, i] for i in range(num_dims)])
                axes[1, 0].set_xlabel("State Dimension")
                axes[1, 0].set_ylabel("Value")
                axes[1, 0].set_title("State Distribution by Dimension")
            else:
                axes[1, 0].text(0.5, 0.5, "No state data", ha="center", va="center")
                axes[1, 0].set_title("State Distribution")

            # Coverage bar chart
            coverages = [
                ("Language", stats.language_coverage),
                ("Success", stats.success_label_coverage),
                ("Rewards", stats.reward_coverage),
            ]
            axes[1, 1].bar(
                [c[0] for c in coverages], [c[1] * 100 for c in coverages], color=["blue", "green", "orange"], alpha=0.7
            )
            axes[1, 1].set_ylabel("Coverage (%)")
            axes[1, 1].set_title("Data Coverage")
            axes[1, 1].set_ylim(0, 105)

            plt.tight_layout()
            plt.show()

        except ImportError:
            console.print("[yellow]Warning:[/yellow] matplotlib not installed. Install with: pip install matplotlib")


@app.command("export-video")
def export_video_cmd(
    path: str = typer.Argument(..., help="Path to dataset (local or hf://org/repo)"),
    output: Path = typer.Option(None, "--output", "-o", help="Output path (file or directory)"),
    episode: int | None = typer.Option(None, "--episode", "-e", help="Episode index to export (default: 0)"),
    camera: str | None = typer.Option(None, "--camera", "-c", help="Camera name to export (default: all cameras)"),
    all_episodes: bool = typer.Option(False, "--all", "-a", help="Export all episodes"),
    fps: int | None = typer.Option(None, "--fps", "-f", help="Override FPS (default: from dataset)"),
    grid: bool = typer.Option(False, "--grid", "-g", help="Combine all cameras into a grid layout"),
) -> None:
    """Export videos from dataset cameras.

    Extract camera feeds from any supported format and save as MP4 files.

    Examples:
        forge export-video dataset/ -o demo.mp4                    # First episode, all cameras grid
        forge export-video dataset/ -e 5 -o episode5.mp4           # Specific episode
        forge export-video dataset/ -c wrist_cam -o wrist.mp4      # Specific camera
        forge export-video dataset/ --all -o ./videos/             # All episodes to directory
        forge export-video hf://lerobot/pusht -o pusht_demo.mp4    # From HuggingFace
    """
    import numpy as np

    from forge.core.exceptions import ForgeError
    from forge.formats.registry import FormatRegistry
    from forge.video.encoder import VideoEncoder, VideoEncoderConfig

    # Resolve path (handles hf:// URLs)
    resolved_path = _resolve_dataset_path(path)

    if not resolved_path.exists():
        console.print(f"[red]Error:[/red] Dataset not found: {resolved_path}")
        raise typer.Exit(1)

    # Detect format and get reader
    try:
        format_name = FormatRegistry.detect_format(resolved_path)
        reader = FormatRegistry.get_reader(format_name)
    except ForgeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Inspect to get metadata
    info = reader.inspect(resolved_path)

    # Determine FPS
    video_fps = fps or info.inferred_fps or 30
    console.print(f"[cyan]Exporting video from:[/cyan] {resolved_path}")
    console.print(f"[dim]Format: {format_name}, FPS: {video_fps}[/dim]")

    # Determine which episodes to export
    if all_episodes:
        episode_indices = list(range(info.num_episodes or 1))
    else:
        episode_indices = [episode if episode is not None else 0]

    # Determine output path
    if output is None:
        output = Path("./output.mp4") if len(episode_indices) == 1 else Path("./videos")

    # If exporting multiple episodes, output must be a directory
    if len(episode_indices) > 1:
        output.mkdir(parents=True, exist_ok=True)

    # Create encoder
    encoder = VideoEncoder(VideoEncoderConfig(codec="libx264", crf=23, preset="medium"))

    # Load all episodes into a list for indexed access
    # (streaming would be more memory-efficient for large datasets)
    episodes_list: list = []
    try:
        with console.status("[bold green]Loading episode index..."):
            for ep in reader.read_episodes(resolved_path):
                episodes_list.append(ep)
    except Exception as e:
        console.print(f"[red]Error:[/red] Could not read episodes: {e}")
        raise typer.Exit(1)

    if not episodes_list:
        console.print("[red]Error:[/red] No episodes found in dataset")
        raise typer.Exit(1)

    # Process episodes
    for ep_idx in episode_indices:
        console.print(f"\n[bold]Episode {ep_idx}[/bold]")

        if ep_idx >= len(episodes_list):
            console.print(f"[yellow]Warning:[/yellow] Episode {ep_idx} not found (dataset has {len(episodes_list)} episodes)")
            continue

        ep = episodes_list[ep_idx]

        # Collect frames
        frames_by_camera: dict[str, list[np.ndarray]] = {}
        frame_count = 0

        with console.status(f"[bold green]Loading frames...") as status:
            for frame in ep.frames():
                frame_count += 1
                for cam_name, lazy_img in frame.images.items():
                    # Filter by camera if specified
                    if camera and cam_name != camera:
                        continue
                    if cam_name not in frames_by_camera:
                        frames_by_camera[cam_name] = []
                    frames_by_camera[cam_name].append(lazy_img.load())
                status.update(f"[bold green]Loading frames... ({frame_count})")

        if not frames_by_camera:
            console.print(f"[yellow]No camera data found for episode {ep_idx}[/yellow]")
            continue

        console.print(f"  Frames: {frame_count}")
        console.print(f"  Cameras: {', '.join(frames_by_camera.keys())}")

        # Determine output file path
        if len(episode_indices) == 1:
            out_path = output if output.suffix == ".mp4" else output / "output.mp4"
        else:
            out_path = output / f"episode_{ep_idx:05d}.mp4"

        out_path.parent.mkdir(parents=True, exist_ok=True)

        if grid and len(frames_by_camera) > 1:
            # Combine cameras into grid
            _export_grid_video(frames_by_camera, out_path, video_fps, encoder, console)
        elif camera:
            # Export single camera
            if camera not in frames_by_camera:
                console.print(f"[red]Error:[/red] Camera '{camera}' not found. Available: {list(frames_by_camera.keys())}")
                raise typer.Exit(1)
            frames = frames_by_camera[camera]
            h, w = frames[0].shape[:2]
            encoder.encode_from_arrays(iter(frames), out_path, fps=video_fps, width=w, height=h)
            console.print(f"  [green]Saved:[/green] {out_path}")
        else:
            # Export each camera separately or as grid
            if len(frames_by_camera) == 1:
                cam_name = list(frames_by_camera.keys())[0]
                frames = frames_by_camera[cam_name]
                h, w = frames[0].shape[:2]
                encoder.encode_from_arrays(iter(frames), out_path, fps=video_fps, width=w, height=h)
                console.print(f"  [green]Saved:[/green] {out_path} ({cam_name})")
            else:
                # Multiple cameras - export as grid by default
                _export_grid_video(frames_by_camera, out_path, video_fps, encoder, console)

    console.print()
    console.print("[green]Export complete![/green]")


def _export_grid_video(
    frames_by_camera: dict[str, list],
    output_path: Path,
    fps: float,
    encoder: "VideoEncoder",
    console: Console,
) -> None:
    """Export multiple cameras as a grid video.

    Args:
        frames_by_camera: Dict mapping camera names to frame lists.
        output_path: Output video path.
        fps: Frames per second.
        encoder: VideoEncoder instance.
        console: Rich console for output.
    """
    import math

    import numpy as np

    camera_names = list(frames_by_camera.keys())
    num_cameras = len(camera_names)
    num_frames = min(len(frames) for frames in frames_by_camera.values())

    # Calculate grid dimensions
    cols = math.ceil(math.sqrt(num_cameras))
    rows = math.ceil(num_cameras / cols)

    # Get frame dimensions (assume all cameras have same size, resize if not)
    sample_frame = frames_by_camera[camera_names[0]][0]
    cell_h, cell_w = sample_frame.shape[:2]

    # Create grid frames
    grid_h = rows * cell_h
    grid_w = cols * cell_w

    def generate_grid_frames():
        for frame_idx in range(num_frames):
            grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

            for cam_idx, cam_name in enumerate(camera_names):
                row = cam_idx // cols
                col = cam_idx % cols
                y_start = row * cell_h
                x_start = col * cell_w

                frame = frames_by_camera[cam_name][frame_idx]

                # Resize if needed
                if frame.shape[:2] != (cell_h, cell_w):
                    import cv2
                    frame = cv2.resize(frame, (cell_w, cell_h))

                grid[y_start:y_start + cell_h, x_start:x_start + cell_w] = frame

            yield grid

    encoder.encode_from_arrays(generate_grid_frames(), output_path, fps=fps, width=grid_w, height=grid_h)
    console.print(f"  [green]Saved:[/green] {output_path} ({cols}x{rows} grid: {', '.join(camera_names)})")


@app.command("formats")
def formats_cmd() -> None:
    """List supported formats."""
    from forge.formats import FormatRegistry

    table = Table(title="Supported Formats")
    table.add_column("Format", style="cyan")
    table.add_column("Read", justify="center")
    table.add_column("Write", justify="center")
    table.add_column("Visualize", justify="center")

    for name, caps in FormatRegistry.list_formats().items():
        read = "[green]✓[/green]" if caps["can_read"] else "[dim]-[/dim]"
        write = "[green]✓[/green]" if caps["can_write"] else "[dim]-[/dim]"
        # Any format with a reader can be visualized
        viz = "[green]✓[/green]" if caps["can_read"] else "[dim]-[/dim]"
        table.add_row(name, read, write, viz)

    console.print()
    console.print(table)


@app.command("hub")
def hub_cmd(
    query: str = typer.Argument(
        None, help="Search query (e.g., 'robot manipulation', 'lerobot', 'pusht')"
    ),
    author: str | None = typer.Option(
        None, "--author", "-a", help="Filter by author/organization (e.g., 'lerobot', 'openvla')"
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of results"),
    download: str | None = typer.Option(
        None, "--download", "-d", help="Download a specific dataset by repo_id (e.g., 'lerobot/pusht')"
    ),
) -> None:
    """Search and download datasets from HuggingFace Hub.

    Examples:
        forge hub                              # List popular robotics datasets
        forge hub "robot manipulation"         # Search for datasets
        forge hub --author lerobot             # List all LeRobot datasets
        forge hub --download lerobot/pusht     # Download a specific dataset
    """
    try:
        from huggingface_hub import HfApi, list_datasets
    except ImportError:
        console.print("[red]Error:[/red] huggingface_hub is required for this command.")
        console.print("Install with: pip install huggingface_hub")
        raise typer.Exit(1)

    # Handle download mode
    if download:
        from forge.hub import download_dataset

        console.print(f"[cyan]Downloading dataset:[/cyan] {download}")
        with console.status("[bold green]Downloading..."):
            try:
                local_path = download_dataset(download)
                console.print(f"[green]Downloaded to:[/green] {local_path}")
                console.print()
                console.print("To inspect this dataset:")
                console.print(f"  forge inspect {local_path}")
                console.print()
                console.print("Or use the hf:// URL directly:")
                console.print(f"  forge inspect hf://{download}")
            except Exception as e:
                console.print(f"[red]Error downloading:[/red] {e}")
                raise typer.Exit(1)
        return

    # Search mode
    api = HfApi()

    # Build search parameters
    search_params = {
        "limit": limit,
        "sort": "downloads",
        "direction": -1,
    }

    # Default to robotics-related search if no query
    if not query and not author:
        # Show popular robotics datasets
        query = "robot"

    if query:
        search_params["search"] = query
    if author:
        search_params["author"] = author

    console.print(f"[cyan]Searching HuggingFace Hub...[/cyan]")
    if query:
        console.print(f"  Query: {query}")
    if author:
        console.print(f"  Author: {author}")
    console.print()

    try:
        datasets = list(list_datasets(**search_params))
    except Exception as e:
        console.print(f"[red]Error searching:[/red] {e}")
        raise typer.Exit(1)

    if not datasets:
        console.print("[yellow]No datasets found matching your criteria.[/yellow]")
        return

    # Display results
    table = Table(title=f"HuggingFace Datasets ({len(datasets)} results)")
    table.add_column("Repository", style="cyan", no_wrap=True)
    table.add_column("Downloads", justify="right")
    table.add_column("Updated", justify="right")
    table.add_column("Tags", max_width=40)

    for ds in datasets:
        # Format downloads
        downloads = ds.downloads if hasattr(ds, 'downloads') else 0
        if downloads >= 1_000_000:
            dl_str = f"{downloads / 1_000_000:.1f}M"
        elif downloads >= 1_000:
            dl_str = f"{downloads / 1_000:.1f}K"
        else:
            dl_str = str(downloads)

        # Format date
        updated = ds.last_modified if hasattr(ds, 'last_modified') else None
        date_str = updated.strftime("%Y-%m-%d") if updated else "-"

        # Get relevant tags
        tags = ds.tags if hasattr(ds, 'tags') else []
        # Filter to interesting tags
        interesting_tags = [t for t in tags if not t.startswith(('license:', 'size_categories:'))]
        tags_str = ", ".join(interesting_tags[:3]) if interesting_tags else "-"

        table.add_row(ds.id, dl_str, date_str, tags_str)

    console.print(table)
    console.print()
    console.print("[dim]To inspect a dataset:[/dim]")
    console.print("  forge inspect hf://<repo_id>")
    console.print()
    console.print("[dim]To download a dataset:[/dim]")
    console.print("  forge hub --download <repo_id>")


def _format_size(num_bytes: int) -> str:
    """Render a byte count as a short human-readable string."""
    if num_bytes <= 0:
        return "-"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024 or unit == "TB":
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{num_bytes} B"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


@app.command("local")
def local_cmd(
    path: Path | None = typer.Option(
        None,
        "--path",
        "-p",
        help="Cache directory to scan. Defaults to the HuggingFace Hub cache "
        "(~/.cache/huggingface/hub, or $HF_HUB_CACHE/$HF_HOME if set).",
    ),
    all_entries: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Include metadata-only stubs (repos forge cannot actually load).",
    ),
    output: str = typer.Option(
        "text", "--output", "-o", help="Output format: text, json"
    ),
    detect: bool = typer.Option(
        True,
        "--detect/--no-detect",
        help="Detect each dataset's format. Disable for a faster listing on "
        "very large caches.",
    ),
) -> None:
    """List datasets already present in a local HuggingFace-style cache.

    Scans the standard HF Hub cache by default and shows every dataset
    that has data files on disk (not just metadata stubs). Pass `--path`
    to point at a different cache root.

    Examples:
        forge local
        forge local --all
        forge local --path /data/hf_cache
        forge local --output json
    """
    from forge.hub import get_hf_cache_dir, list_local_datasets

    cache_root = path if path is not None else get_hf_cache_dir()
    if not cache_root.is_dir():
        console.print(
            f"[yellow]Cache directory does not exist:[/yellow] {cache_root}"
        )
        raise typer.Exit(0 if output == "text" else 1)

    datasets = list_local_datasets(cache_dir=cache_root, include_stubs=all_entries)

    # Resolve format per dataset (best-effort).
    formats: dict[str, str] = {}
    if detect:
        from forge.core.exceptions import ForgeError
        from forge.formats.registry import FormatRegistry

        for ds in datasets:
            if ds.snapshot_path is None:
                continue
            try:
                formats[ds.repo_id] = FormatRegistry.detect_format(ds.snapshot_path)
            except (ForgeError, Exception):
                formats[ds.repo_id] = "unknown"

    if output == "json":
        data = {
            "cache_dir": str(cache_root),
            "count": len(datasets),
            "datasets": [
                {
                    "repo_id": ds.repo_id,
                    "snapshot_path": str(ds.snapshot_path) if ds.snapshot_path else None,
                    "size_bytes": ds.size_bytes,
                    "file_counts": ds.file_counts,
                    "is_stub": ds.is_stub,
                    "format": formats.get(ds.repo_id),
                }
                for ds in datasets
            ],
        }
        # Use plain print, not console.print, so Rich doesn't wrap long
        # paths and break json parsing.
        print(json.dumps(data, indent=2))
        return

    console.print(f"[bold]Cache:[/bold] {cache_root}")

    if not datasets:
        if all_entries:
            console.print("[yellow]No dataset folders found.[/yellow]")
        else:
            console.print(
                "[yellow]No populated datasets found.[/yellow] "
                "Pass [bold]--all[/bold] to also list metadata-only stubs."
            )
        return

    table = Table(title=f"Local Datasets ({len(datasets)})")
    table.add_column("Repository", style="cyan", no_wrap=True)
    table.add_column("Size", justify="right", no_wrap=True)
    if detect:
        table.add_column("Format", no_wrap=True)
    # Compact "kinds" column — list which media types are present without
    # repeating per-extension counts (which already wrap badly on long ids).
    table.add_column("Kinds", no_wrap=True, overflow="ellipsis")

    total_bytes = 0
    for ds in datasets:
        total_bytes += ds.size_bytes
        size_str = _format_size(ds.size_bytes)
        if ds.is_stub:
            size_str = "[dim]stub[/dim]"

        if ds.file_counts:
            kinds = ", ".join(ext.lstrip(".") for ext in sorted(ds.file_counts))
        else:
            kinds = "[dim]—[/dim]"

        if detect:
            fmt = formats.get(ds.repo_id, "—")
            if ds.is_stub:
                fmt = "[dim]—[/dim]"
            table.add_row(ds.repo_id, size_str, fmt, kinds)
        else:
            table.add_row(ds.repo_id, size_str, kinds)

    console.print(table)
    console.print(
        f"[dim]Total: {_format_size(total_bytes)} across {len(datasets)} datasets[/dim]"
    )
    console.print()

    # Pick an example to make the hint copy-pasteable. For HF Hub layout we
    # can use the repo_id directly; for generic layouts we need the full
    # path because the repo_id is just a display label.
    from forge.hub.download import _is_hf_hub_cache_root

    example_ds = next((d for d in datasets if d.snapshot_path is not None), None)
    if example_ds is not None and _is_hf_hub_cache_root(cache_root):
        console.print(
            f"[dim]Inspect a dataset with:[/dim] forge inspect {example_ds.repo_id}"
        )
    elif example_ds is not None:
        console.print(
            f"[dim]Inspect a dataset with:[/dim] forge inspect {example_ds.snapshot_path}"
        )


@app.command("quality")
def quality_cmd(
    path: str = typer.Argument(..., help="Path to dataset (local or hf://org/repo)"),
    gripper_dim: int = typer.Option(-1, "--gripper-dim", "-g", help="Gripper dimension index (-1 = last)"),
    fps: float = typer.Option(30.0, "--fps", "-f", help="Fallback FPS if timestamps unavailable"),
    sample: int = typer.Option(0, "--sample", "-s", help="Analyze N episodes (0 = all)"),
    export: Path | None = typer.Option(None, "--export", "-e", help="Export full report to JSON"),
    export_flagged: Path | None = typer.Option(None, "--export-flagged", help="Export flagged episode IDs to JSON"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Quick mode (sample 50 episodes)"),
    action_bounds: str | None = typer.Option(None, "--action-bounds", help="Known action bounds as 'min,max' (e.g., '-1,1')"),
    video: bool = typer.Option(False, "--video", help="Also analyze camera streams (Tier 0 pixel metrics)"),
    video_level: str = typer.Option("pixel", "--video-level", help="Video metric tier: 'pixel' (Tier 0). motion/semantic coming soon."),
    video_downscale: int = typer.Option(128, "--video-downscale", help="Longest side (px) of the analysis frame"),
    video_stride: int = typer.Option(1, "--video-stride", help="Analyze every Nth image-bearing frame"),
    video_max_frames: int = typer.Option(0, "--video-max-frames", help="Cap analyzed frames per camera (0 = all)"),
    video_cameras: str | None = typer.Option(None, "--video-cameras", help="Comma-separated camera names (default: all)"),
) -> None:
    """Compute quality metrics on dataset episodes.

    Analyzes proprioception data (actions, states, timestamps) to score episode
    quality. Pure numpy number crunching by default — pass --video to also score
    camera streams (Tier 0 pixel metrics: blur, exposure, frozen frames, colour).

    Metrics: smoothness (LDLJ), dead actions, gripper chatter, static detection,
    timestamp regularity, action saturation, action diversity.

    Examples:
        forge quality ./dataset
        forge quality ./dataset --gripper-dim 6 --fps 30
        forge quality ./dataset --export quality_report.json
        forge quality ./dataset --quick
        forge quality ./dataset --video --video-stride 4
        forge quality hf://lerobot/aloha_sim_cube --sample 100
    """
    from rich.panel import Panel
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

    from forge.core.exceptions import ForgeError
    from forge.formats.registry import FormatRegistry
    from forge.quality import QualityAnalyzer, QualityConfig

    # Parse action bounds
    bounds = None
    if action_bounds:
        try:
            parts = action_bounds.split(",")
            bounds = (float(parts[0]), float(parts[1]))
        except (ValueError, IndexError):
            console.print("[red]Error:[/red] --action-bounds must be 'min,max' (e.g., '-1,1')")
            raise typer.Exit(1)

    config = QualityConfig(
        gripper_dim=gripper_dim,
        fps=fps,
        action_bounds=bounds,
    )

    # Video (Tier 0) config — only built when --video is passed.
    video_config = None
    if video:
        if video_level != "pixel":
            console.print(
                f"[red]Error:[/red] --video-level '{video_level}' not available yet. "
                "Tier 0 ('pixel') is the only supported level; motion/semantic are coming soon."
            )
            raise typer.Exit(1)
        from forge.quality.video import VideoQualityConfig

        cam_list = [c.strip() for c in video_cameras.split(",")] if video_cameras else None
        video_config = VideoQualityConfig(
            downscale=video_downscale,
            sample_stride=video_stride,
            max_frames=video_max_frames,
            cameras=cam_list,
        )

    if quick and sample == 0:
        sample = 50

    # Resolve path
    resolved_path = _resolve_dataset_path(path)

    if not resolved_path.exists():
        console.print(f"[red]Error:[/red] Dataset not found: {resolved_path}")
        raise typer.Exit(1)

    # Detect format
    try:
        format_name = FormatRegistry.detect_format(resolved_path)
        reader = FormatRegistry.get_reader(format_name)
    except ForgeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[cyan]Quality analysis:[/cyan] {path}")
    console.print(f"[dim]Format: {format_name}[/dim]")
    console.print()

    # Analyze with progress bar
    analyzer = QualityAnalyzer(config=config, video_config=video_config)
    from collections import defaultdict

    from forge.quality.models import QualityReport

    report = QualityReport(dataset_path=str(path))
    flagged: dict[str, list[str]] = defaultdict(list)

    try:
        with Progress(
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing episodes...", total=sample or None)

            for i, episode in enumerate(reader.read_episodes(resolved_path)):
                if sample > 0 and i >= sample:
                    break

                eq = analyzer.analyze_episode(episode)
                report.per_episode.append(eq)

                for flag in eq.flags:
                    flagged[flag].append(eq.episode_id)

                progress.advance(task)

            if sample == 0:
                progress.update(task, total=len(report.per_episode), completed=len(report.per_episode))

    except ForgeError as e:
        console.print(f"[red]Error reading dataset:[/red] {e}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — showing partial results[/yellow]")

    report.num_episodes = len(report.per_episode)
    report.flagged_episodes = dict(flagged)

    if report.num_episodes == 0:
        console.print("[yellow]No episodes analyzed.[/yellow]")
        raise typer.Exit(0)

    # Aggregate scores
    import numpy as np

    scores = [eq.overall_score for eq in report.per_episode if eq.overall_score is not None]
    report.overall_score = float(np.mean(scores)) if scores else 0.0

    all_keys: set[str] = set()
    for eq in report.per_episode:
        all_keys.update(eq.subscores.keys())

    for key in all_keys:
        vals = [eq.subscores[key] for eq in report.per_episode if key in eq.subscores]
        if vals:
            report.subscores[key] = float(np.mean(vals))

    # Generate recommendations
    from forge.quality.analyzer import _generate_recommendations

    report.flags, report.recommendations = _generate_recommendations(report, config)

    # ── Display results ──
    console.print()

    # Overall score with color
    score = report.overall_score
    if score >= 8.0:
        score_color = "green"
    elif score >= 6.0:
        score_color = "yellow"
    else:
        score_color = "red"

    # Build the quality report panel
    lines: list[str] = []
    lines.append(f"Overall Quality Score: [{score_color}]{score:.1f} / 10[/{score_color}]")
    lines.append("")

    # Subscores with bar visualization
    subscore_labels = {
        "smoothness": "Smoothness (LDLJ)",
        "dead_actions": "Dead Actions",
        "gripper_health": "Gripper Health",
        "static_detection": "Static Detection",
        "timestamp_regularity": "Timestamp Regularity",
        "action_saturation": "Action Saturation",
        "action_diversity": "Action Diversity",
    }

    for key, label in subscore_labels.items():
        if key in report.subscores:
            val = report.subscores[key]
            filled = int(val * 10)
            bar = "[green]" + "\u2588" * filled + "[/green]" + "[dim]" + "\u2591" * (10 - filled) + "[/dim]"

            # Count flagged episodes for this metric
            flag_map = {
                "smoothness": "jerky",
                "dead_actions": "dead_actions",
                "gripper_health": "gripper_chatter",
                "static_detection": "mostly_static",
                "timestamp_regularity": "timestamp_jitter",
                "action_saturation": "saturated",
                "action_diversity": "low_entropy",
            }
            flag_key = flag_map.get(key, "")
            flagged_count = len(report.flagged_episodes.get(flag_key, []))

            flag_str = f"  {flagged_count} flagged" if flagged_count > 0 else "  OK"
            lines.append(f"{label:<24} {bar}  {val:.2f}{flag_str}")

    # Additional signals \u2014 surfaced for inspection but not part of the
    # composite score (different scales: SPARC is negative, PSD a fraction,
    # state-action variance unbounded positive).
    def _collect(attr: str) -> list[float]:
        return [
            getattr(eq, attr)
            for eq in report.per_episode
            if getattr(eq, attr) is not None
        ]

    sparc_vals = _collect("sparc")
    psd_vals = _collect("psd_high_fraction")
    sa_vals = _collect("state_action_consistency")

    if sparc_vals or psd_vals or sa_vals:
        lines.append("")
        lines.append("[bold]Additional Signals[/bold] [dim](not in composite)[/dim]")
        if sparc_vals:
            lines.append(
                f"  SPARC smoothness         mean={np.mean(sparc_vals):+.2f}  "
                f"[dim](more negative = jerkier; complement to LDLJ)[/dim]"
            )
        if psd_vals:
            n_flag = len(report.flagged_episodes.get("psd_high_band_chatter", []))
            flag_str = f"  [yellow]{n_flag} flagged[/yellow]" if n_flag else "  OK"
            lines.append(
                f"  PSD high-band fraction   mean={np.mean(psd_vals):.3f}{flag_str}"
            )
        if sa_vals:
            lines.append(
                f"  State-action variance    mean={np.mean(sa_vals):.3f}  "
                f"[dim](range {np.min(sa_vals):.2f}\u2013{np.max(sa_vals):.2f})[/dim]"
            )

    # Video (Tier 0) summary
    vid_eps = [eq for eq in report.per_episode if eq.video is not None]
    if vid_eps:
        cams: set[str] = set()
        for eq in vid_eps:
            cams.update(eq.video.per_camera.keys())

        def _vid_mean(attr: str) -> float | None:
            vals = [
                getattr(eq.video, attr)
                for eq in vid_eps
                if getattr(eq.video, attr) is not None
            ]
            return float(np.mean(vals)) if vals else None

        lines.append("")
        lines.append(
            f"[bold]Video[/bold] [dim](Tier 0 — pixel, {len(vid_eps)} episodes, "
            f"{len(cams)} camera{'s' if len(cams) != 1 else ''})[/dim]"
        )

        def _vid_line(label: str, value: float | None, flag_key: str, fmt: str) -> None:
            if value is None:
                return
            n_flag = len(report.flagged_episodes.get(flag_key, []))
            flag_str = f"  [yellow]{n_flag} flagged[/yellow]" if n_flag else "  OK"
            lines.append(f"  {label:<26} {format(value, fmt)}{flag_str}")

        _vid_line("Sharpness (min, mean)", _vid_mean("min_sharpness"), "blurry", ".0f")
        _vid_line("Overexposed fraction", _vid_mean("overexposed_fraction"), "over_exposed", ".2f")
        _vid_line("Underexposed fraction", _vid_mean("underexposed_fraction"), "under_exposed", ".2f")
        _vid_line("Frozen-frame fraction", _vid_mean("frozen_fraction"), "frozen_frames", ".2f")
        color = _vid_mean("mean_colorfulness")
        if color is not None:
            lines.append(f"  {'Colorfulness':<26} {color:.1f}  [dim](scene diversity)[/dim]")

    # Top issues
    if report.flags:
        lines.append("")
        lines.append("[bold]Top Issues:[/bold]")
        for flag in report.flags[:5]:
            lines.append(f"  [yellow]\u26a0[/yellow] {flag}")

    # Recommendations
    if report.recommendations:
        lines.append("")
        lines.append("[bold]Recommendations:[/bold]")
        for rec in report.recommendations[:4]:
            lines.append(f"  [cyan]\u2192[/cyan] {rec}")

    panel_text = "\n".join(lines)
    panel = Panel(
        panel_text,
        title=f"Quality Report: {path}    ({report.num_episodes} episodes)",
        border_style="blue",
        padding=(1, 2),
    )
    console.print(panel)

    # JSON export
    if export:
        report.to_json(export)
        console.print(f"\n[green]Report saved to:[/green] {export}")

    if export_flagged:
        with open(export_flagged, "w") as f:
            json.dump(report.flagged_episodes, f, indent=2)
        console.print(f"[green]Flagged episodes saved to:[/green] {export_flagged}")


@app.command("filter")
def filter_cmd(
    source: str = typer.Argument(..., help="Path to dataset (local or hf://org/repo)"),
    output: Path | None = typer.Argument(None, help="Output path for filtered dataset (omit for dry-run)"),
    min_quality: float | None = typer.Option(None, "--min-quality", "-q", help="Keep episodes with overall_score >= this value (0-10)"),
    exclude_flags: str | None = typer.Option(None, "--exclude-flags", help="Exclude episodes with ANY of these flags (comma-separated)"),
    min_sharpness: float | None = typer.Option(None, "--min-sharpness", help="[video] Exclude episodes whose min camera sharpness is below this (var-of-Laplacian)"),
    max_frozen: float | None = typer.Option(None, "--max-frozen", help="[video] Exclude episodes whose frozen-frame fraction exceeds this (0-1)"),
    max_overexposed: float | None = typer.Option(None, "--max-overexposed", help="[video] Exclude episodes whose overexposed fraction exceeds this (0-1)"),
    max_underexposed: float | None = typer.Option(None, "--max-underexposed", help="[video] Exclude episodes whose underexposed fraction exceeds this (0-1)"),
    include_episodes: str | None = typer.Option(None, "--include-episodes", help="Only include these episode IDs (comma-separated)"),
    exclude_episodes: str | None = typer.Option(None, "--exclude-episodes", help="Exclude these episode IDs (comma-separated)"),
    from_report: Path | None = typer.Option(None, "--from-report", "-r", help="Use pre-computed quality report JSON"),
    gripper_dim: int = typer.Option(-1, "--gripper-dim", "-g", help="Gripper dimension index (-1 = last)"),
    fps: float = typer.Option(30.0, "--fps", "-f", help="Fallback FPS if timestamps unavailable"),
    action_bounds: str | None = typer.Option(None, "--action-bounds", help="Known action bounds as 'min,max'"),
    video_stride: int = typer.Option(1, "--video-stride", help="[video] Analyze every Nth image-bearing frame during live analysis"),
) -> None:
    """Filter dataset episodes based on quality scores and flags.

    Reads a dataset, evaluates each episode against quality criteria, and
    writes only passing episodes to the output (same format). If no output
    path is given, runs in dry-run mode and prints a summary.

    Video (Tier 0) criteria — --min-sharpness, --max-frozen, --max-overexposed,
    --max-underexposed, and the video flags (blurry, frozen_frames, over_exposed,
    under_exposed) via --exclude-flags — trigger live video analysis automatically,
    or read video fields straight from a --from-report JSON.

    Examples:
        forge filter ./dataset                                    # Dry-run
        forge filter ./dataset ./filtered --min-quality 6.0
        forge filter ./dataset ./filtered --exclude-flags jerky,mostly_static
        forge filter ./dataset ./filtered --min-sharpness 80 --exclude-flags frozen_frames
        forge filter ./dataset ./filtered --from-report report.json --min-quality 7.0
    """
    from rich.panel import Panel
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

    from forge.core.exceptions import ForgeError
    from forge.filter.engine import FilterConfig, FilterEngine

    # Validate mutual exclusivity
    if include_episodes and exclude_episodes:
        console.print("[red]Error:[/red] Cannot use both --include-episodes and --exclude-episodes")
        raise typer.Exit(1)

    # Parse action bounds
    bounds = None
    if action_bounds:
        try:
            parts = action_bounds.split(",")
            bounds = (float(parts[0]), float(parts[1]))
        except (ValueError, IndexError):
            console.print("[red]Error:[/red] --action-bounds must be 'min,max' (e.g., '-1,1')")
            raise typer.Exit(1)

    # Parse comma-separated lists
    flags_list = [f.strip() for f in exclude_flags.split(",")] if exclude_flags else None
    include_list = [e.strip() for e in include_episodes.split(",")] if include_episodes else None
    exclude_list = [e.strip() for e in exclude_episodes.split(",")] if exclude_episodes else None

    config = FilterConfig(
        min_quality=min_quality,
        exclude_flags=flags_list,
        min_sharpness=min_sharpness,
        max_frozen_fraction=max_frozen,
        max_overexposed_fraction=max_overexposed,
        max_underexposed_fraction=max_underexposed,
        include_episodes=include_list,
        exclude_episodes=exclude_list,
        from_report=from_report,
        gripper_dim=gripper_dim,
        fps=fps,
        action_bounds=bounds,
        video_stride=video_stride,
    )

    # Resolve source
    resolved_path = _resolve_dataset_path(source)
    if not resolved_path.exists():
        console.print(f"[red]Error:[/red] Dataset not found: {resolved_path}")
        raise typer.Exit(1)

    # Show mode
    dry_run = output is None
    if dry_run:
        console.print(f"[cyan]Filter dry-run:[/cyan] {source}")
    else:
        console.print(f"[cyan]Filtering:[/cyan] {source} [dim]→[/dim] {output}")

    # Build criteria string
    criteria: list[str] = []
    if min_quality is not None:
        criteria.append(f"min_quality={min_quality}")
    if flags_list:
        criteria.append(f"exclude_flags=[{', '.join(flags_list)}]")
    if min_sharpness is not None:
        criteria.append(f"min_sharpness={min_sharpness:g}")
    if max_frozen is not None:
        criteria.append(f"max_frozen={max_frozen:g}")
    if max_overexposed is not None:
        criteria.append(f"max_overexposed={max_overexposed:g}")
    if max_underexposed is not None:
        criteria.append(f"max_underexposed={max_underexposed:g}")
    if include_list:
        criteria.append(f"include={len(include_list)} episodes")
    if exclude_list:
        criteria.append(f"exclude={len(exclude_list)} episodes")
    if from_report:
        criteria.append(f"from_report={from_report}")

    if not criteria:
        console.print("[yellow]Warning:[/yellow] No filter criteria specified — all episodes will pass")

    console.print(f"[dim]Criteria: {', '.join(criteria) if criteria else 'none'}[/dim]")
    console.print()

    engine = FilterEngine(config)

    try:
        with Progress(
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Filtering episodes...", total=None)

            def on_progress(stage: str, current: int, total: int) -> None:
                if total > 0:
                    progress.update(task, total=total, completed=current)

            result = engine.filter(
                source=resolved_path,
                output=output,
                progress_callback=on_progress,
            )

            progress.update(
                task,
                total=result.total_episodes or (result.episodes_kept + result.episodes_excluded),
                completed=result.episodes_kept + result.episodes_excluded,
            )

    except ForgeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print()

    # Display results
    if dry_run:
        # Show table of episodes
        table = Table(show_header=True, header_style="bold")
        table.add_column("Episode", style="dim")
        table.add_column("Score", justify="right")
        table.add_column("Flags")
        table.add_column("Status")

        # We need quality data for the table — re-derive from result
        for ep_id in result.kept_ids:
            table.add_row(ep_id, "", "", "[green]KEEP[/green]")
        for ep_id in result.excluded_ids:
            reasons = result.exclusion_reasons.get(ep_id, [])
            table.add_row(ep_id, "", "", f"[red]EXCLUDE[/red] ({'; '.join(reasons)})")

        console.print(table)
        console.print()

    # Summary
    total = result.episodes_kept + result.episodes_excluded
    kept_color = "green" if result.episodes_kept > 0 else "yellow"
    excl_color = "red" if result.episodes_excluded > 0 else "dim"

    lines: list[str] = []
    lines.append(f"Episodes kept: [{kept_color}]{result.episodes_kept}[/{kept_color}] / {total}")
    lines.append(f"Episodes excluded: [{excl_color}]{result.episodes_excluded}[/{excl_color}]")

    if not dry_run and result.output_path:
        lines.append(f"Total frames: {result.total_frames_kept:,}")
        lines.append(f"Output: [bold]{result.output_path}[/bold]")

    if result.errors:
        lines.append("")
        for err in result.errors[:5]:
            lines.append(f"[red]Error:[/red] {err}")

    panel = Panel(
        "\n".join(lines),
        title=f"Filter {'Preview' if dry_run else 'Result'}: {result.format}",
        border_style="blue",
        padding=(1, 2),
    )
    console.print(panel)

    if dry_run and output is None and result.episodes_excluded > 0:
        console.print(
            f"\n[dim]Run with output path to write filtered dataset:[/dim]"
            f"\n  forge filter {source} ./filtered"
            + (f" --min-quality {min_quality}" if min_quality else "")
            + (f" --exclude-flags {exclude_flags}" if exclude_flags else "")
            + (f" --from-report {from_report}" if from_report else "")
        )


# --- Registry subcommands ---


@registry_app.command("list")
def registry_list_cmd(
    format: str | None = typer.Option(None, "--format", "-f", help="Filter by format (e.g., rlds, lerobot)"),
    embodiment: str | None = typer.Option(None, "--embodiment", "-e", help="Filter by robot (e.g., franka)"),
    tag: str | None = typer.Option(None, "--tag", "-t", help="Filter by tag (e.g., language_conditioned)"),
    demo: bool = typer.Option(False, "--demo", help="Show only demo-suitable datasets (<=100 episodes)"),
    html: bool = typer.Option(False, "--html", help="Open interactive HTML view in browser"),
) -> None:
    """List all datasets in the registry, with optional filters."""
    from forge.registry import DatasetRegistry

    try:
        entries = DatasetRegistry.list(
            format=format, embodiment=embodiment, tag=tag, demo_only=demo,
        )
    except Exception as e:
        console.print(f"[red]Error loading registry:[/red] {e}")
        raise typer.Exit(1)

    if not entries:
        console.print("[yellow]No datasets match the given filters.[/yellow]")
        raise typer.Exit(0)

    if html:
        from forge.registry.html import open_registry_html

        path = open_registry_html(entries)
        console.print(f"[green]Opened registry browser:[/green] {path}")
        return

    table = Table(title=f"Forge Dataset Registry ({len(entries)} datasets)")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Format", style="green")
    table.add_column("Embodiment")
    table.add_column("Episodes", justify="right")
    table.add_column("Demo", justify="center")

    for entry in entries:
        eps = ""
        if entry.scale and entry.scale.episodes is not None:
            eps = f"{'~' if entry.scale.approximate else ''}{entry.scale.episodes:,}"
        table.add_row(
            entry.id,
            entry.name,
            entry.format,
            ", ".join(entry.embodiment[:2]) + ("..." if len(entry.embodiment) > 2 else ""),
            eps,
            "[green]yes[/green]" if entry.demo_suitable else "",
        )

    console.print(table)


@registry_app.command("info")
def registry_info_cmd(
    dataset_id: str = typer.Argument(..., help="Dataset ID (e.g., droid, bridge_v2)"),
) -> None:
    """Show detailed metadata for a dataset."""
    from rich.panel import Panel
    from rich.text import Text

    from forge.core.exceptions import DatasetNotFoundError
    from forge.registry import DatasetRegistry

    try:
        entry = DatasetRegistry.get(dataset_id)
    except DatasetNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    lines: list[str] = []
    lines.append(f"[bold]{entry.name}[/bold]")
    lines.append(f"[dim]{entry.description}[/dim]")
    lines.append("")
    lines.append(f"  [cyan]ID:[/cyan]          {entry.id}")
    lines.append(f"  [cyan]Format:[/cyan]      {entry.format}")
    lines.append(f"  [cyan]Embodiment:[/cyan]  {', '.join(entry.embodiment)}")
    if entry.license:
        lines.append(f"  [cyan]License:[/cyan]     {entry.license}")
    if entry.paper_url:
        lines.append(f"  [cyan]Paper:[/cyan]       {entry.paper_url}")

    if entry.scale:
        scale_parts = []
        if entry.scale.episodes is not None:
            scale_parts.append(f"{entry.scale.episodes:,} episodes")
        if entry.scale.hours is not None:
            scale_parts.append(f"{entry.scale.hours:.0f} hours")
        if entry.scale.approximate:
            scale_parts.append("(approximate)")
        lines.append(f"  [cyan]Scale:[/cyan]       {' | '.join(scale_parts)}")

    if entry.task_types:
        lines.append(f"  [cyan]Tasks:[/cyan]       {', '.join(entry.task_types)}")
    if entry.tags:
        lines.append(f"  [cyan]Tags:[/cyan]        {', '.join(entry.tags)}")

    lines.append("")
    lines.append("  [bold]Sources:[/bold]")
    for i, source in enumerate(entry.sources):
        marker = " [green](demo)[/green]" if i == entry.demo_source_index else ""
        lines.append(f"    [{source.type}] {source.uri}{marker}")
        if source.notes:
            lines.append(f"      [dim]{source.notes}[/dim]")

    if entry.demo_suitable:
        lines.append("")
        eps = f" ({entry.demo_episodes} episodes)" if entry.demo_episodes else ""
        lines.append(f"  [green]Demo suitable[/green]{eps}")

    if entry.notes:
        lines.append("")
        lines.append(f"  [dim]Note: {entry.notes}[/dim]")

    console.print(Panel("\n".join(lines), title=f"Dataset: {entry.id}"))


@registry_app.command("search")
def registry_search_cmd(
    query: str = typer.Argument(..., help="Search query (e.g., 'franka manipulation')"),
) -> None:
    """Search datasets by keyword across names, tags, embodiments, and task types."""
    from forge.registry import DatasetRegistry

    try:
        results = DatasetRegistry.search(query)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if not results:
        console.print(f"[yellow]No datasets matching '{query}'.[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Search results for '{query}' ({len(results)} matches)")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Format", style="green")
    table.add_column("Description")

    for entry in results:
        desc = entry.description
        if len(desc) > 80:
            desc = desc[:77] + "..."
        table.add_row(entry.id, entry.name, entry.format, desc)

    console.print(table)


@registry_app.command("validate")
def registry_validate_cmd(
    path: str | None = typer.Option(None, "--path", "-p", help="Path to datasets.json (default: bundled)"),
    probe: bool = typer.Option(False, "--probe", help="Check that source URIs are reachable (requires network)"),
) -> None:
    """Validate the registry JSON for schema correctness and integrity."""
    from forge.registry.validation import validate_registry

    registry_path = Path(path) if path else None
    result = validate_registry(path=registry_path, probe=probe)

    if result.errors:
        console.print(f"[red bold]Errors ({len(result.errors)}):[/red bold]")
        for err in result.errors:
            console.print(f"  [red]x[/red] {err}")

    if result.warnings:
        console.print(f"[yellow bold]Warnings ({len(result.warnings)}):[/yellow bold]")
        for warn in result.warnings:
            console.print(f"  [yellow]![/yellow] {warn}")

    if result.ok:
        console.print("[green]Registry is valid.[/green]")
    else:
        raise typer.Exit(1)


@app.command("lint")
def lint_cmd(
    path: str = typer.Argument(..., help="Path to dataset (local or hf://org/repo)"),
    export: Path | None = typer.Option(
        None, "--export", "-e", help="Export the full lint report to JSON"
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Exit non-zero on warnings too, not just errors"
    ),
    max_examples: int = typer.Option(
        5, "--max-examples", help="Per-code episode examples to print (0 = all)"
    ),
) -> None:
    """Check a dataset against Hugging Face's recording guidelines.

    Flags hygiene defects that degrade training or downstream tooling: missing
    or low-coverage task instructions, placeholder/too-short task strings,
    ambiguous camera naming, low-resolution or single-view camera setups, and
    missing action fields. Pure metadata inspection (uses the reader's
    `inspect`, no video decode). Exits non-zero if any ERROR is found (add
    --strict to also fail on warnings).

    Examples:
        forge lint ./dataset
        forge lint hf://lerobot/pusht --export lint.json
        forge lint ./dataset --strict
    """
    import json as _json

    from forge.core.exceptions import ForgeError
    from forge.formats.registry import FormatRegistry
    from forge.lint import DatasetLinter

    resolved_path = _resolve_dataset_path(path)
    if not resolved_path.exists():
        console.print(f"[red]Error:[/red] Dataset not found: {resolved_path}")
        raise typer.Exit(1)

    try:
        format_name = FormatRegistry.detect_format(resolved_path)
        reader = FormatRegistry.get_reader(format_name)
    except ForgeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[cyan]Linting:[/cyan] {path}")
    console.print(f"[dim]Format: {format_name}[/dim]")
    console.print()

    try:
        info = reader.inspect(resolved_path)
    except ForgeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    linter = DatasetLinter(reader=reader)
    report = linter.lint_metadata(info, dataset_path=str(path))

    _color = {"error": "red", "warn": "yellow", "info": "dim"}
    if report.issues:
        # Group by code so a defect across 500 episodes prints once, not 500x.
        from collections import defaultdict

        by_code: dict[str, list] = defaultdict(list)
        for issue in report.issues:
            by_code[issue.code].append(issue)

        # Errors first, then warnings, then info.
        order = {"error": 0, "warn": 1, "info": 2}
        for code in sorted(
            by_code, key=lambda c: (order[by_code[c][0].severity.value], c)
        ):
            issues = by_code[code]
            sev = issues[0].severity.value
            color = _color[sev]
            scoped = [i for i in issues if i.scope != "dataset"]
            count = (
                f"  ({len(scoped)} episode{'s' if len(scoped) != 1 else ''})"
                if scoped
                else ""
            )
            console.print(
                f"[{color}]{sev.upper():<5}[/{color}] [bold]{code}[/bold]{count}"
            )
            console.print(f"        {issues[0].message}")
            if issues[0].hint:
                console.print(f"        [dim]→ {issues[0].hint}[/dim]")
            if scoped and len(scoped) > 1:
                shown = scoped if max_examples == 0 else scoped[:max_examples]
                ids = ", ".join(i.scope for i in shown)
                more = len(scoped) - len(shown)
                suffix = f" … (+{more} more)" if more > 0 else ""
                console.print(f"        [dim]e.g. {ids}{suffix}[/dim]")
            console.print()

    n_err, n_warn, n_info = (
        len(report.errors),
        len(report.warnings),
        len(report.infos),
    )
    status = "[green]PASS[/green]" if report.passed else "[red]FAIL[/red]"
    console.print(
        f"{status}  {report.num_episodes} episodes  "
        f"[red]{n_err} errors[/red], [yellow]{n_warn} warnings[/yellow], "
        f"[dim]{n_info} info[/dim]"
    )

    if export:
        export.write_text(_json.dumps(report.to_dict(), indent=2))
        console.print(f"[dim]Report written to {export}[/dim]")

    if n_err > 0 or (strict and n_warn > 0):
        raise typer.Exit(1)


@app.command("demo")
def demo_cmd(
    dataset_id: str | None = typer.Argument(
        None, help="Registry dataset ID (default: picks a random demo dataset)"
    ),
    skip_quality: bool = typer.Option(
        False, "--skip-quality", help="Skip quality analysis step"
    ),
) -> None:
    """Download a demo dataset and run inspect + quality. Great for getting started.

    Examples:
        forge demo                    # random demo dataset
        forge demo pusht              # specific dataset
        forge demo droid_100          # DROID 100-episode subset
    """
    import random

    from rich.panel import Panel

    from forge.core.exceptions import DatasetNotFoundError
    from forge.registry import DatasetRegistry

    # Pick a demo dataset
    if dataset_id:
        try:
            entry = DatasetRegistry.get(dataset_id)
        except DatasetNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
        if not entry.demo_suitable:
            console.print(
                f"[yellow]Warning:[/yellow] '{dataset_id}' is not a demo-suitable dataset "
                f"({entry.scale.episodes:,} episodes). It may take a while to download."
            )
            console.print(
                "[dim]Demo-suitable datasets: "
                + ", ".join(e.id for e in DatasetRegistry.demo_datasets())
                + "[/dim]"
            )
    else:
        demos = DatasetRegistry.demo_datasets()
        if not demos:
            console.print("[red]No demo-suitable datasets found in registry.[/red]")
            raise typer.Exit(1)
        entry = random.choice(demos)
        console.print(f"[cyan]Selected demo dataset:[/cyan] {entry.name} ({entry.id})")

    console.print()

    # Step 1: Download
    console.print("[bold]Step 1/3:[/bold] Downloading dataset...")
    console.print()
    source = DatasetRegistry.get_source(entry.id, demo=entry.demo_suitable)
    if source.type == "hf_hub":
        local_path = _resolve_dataset_path(f"hf://{source.uri}")
    else:
        console.print(f"[yellow]Source type '{source.type}' requires manual download:[/yellow]")
        console.print(f"  {source.uri}")
        raise typer.Exit(1)

    console.print()

    # Step 2: Inspect
    console.print("[bold]Step 2/3:[/bold] Inspecting dataset...")
    console.print()
    try:
        from forge.inspect import Inspector

        inspector = Inspector()
        info = inspector.inspect(local_path)

        lines = [
            f"  Format:     [green]{info.format}[/green]",
            f"  Episodes:   {info.num_episodes:,}",
            f"  Frames:     {info.total_frames:,}",
        ]
        if info.cameras:
            cam_names = ", ".join(info.cameras.keys())
            lines.append(f"  Cameras:    {cam_names}")
        if info.inferred_fps:
            lines.append(f"  FPS:        {info.inferred_fps}")

        console.print(Panel("\n".join(lines), title="Inspection Results", border_style="blue"))
        console.print()
    except Exception as e:
        console.print(f"[yellow]Inspection failed:[/yellow] {e}")
        console.print()

    # Step 3: Quality
    if not skip_quality:
        console.print("[bold]Step 3/3:[/bold] Running quality analysis...")
        console.print()
        try:
            from forge.quality import QualityAnalyzer, QualityConfig

            config = QualityConfig()
            analyzer = QualityAnalyzer(config)
            report = analyzer.analyze(local_path)

            lines = [
                f"  Overall score:  [bold]{report.mean_score:.1f}[/bold] / 10",
                f"  Episodes:       {report.num_episodes}",
            ]
            if report.flagged_episodes:
                lines.append(
                    f"  Flagged:        [yellow]{len(report.flagged_episodes)}[/yellow] episodes"
                )

            console.print(
                Panel("\n".join(lines), title="Quality Report", border_style="green")
            )
            console.print()
        except Exception as e:
            console.print(f"[yellow]Quality analysis failed:[/yellow] {e}")
            console.print()
    else:
        console.print("[dim]Step 3/3: Skipped (--skip-quality)[/dim]")
        console.print()

    # Summary
    console.print("[bold green]Done![/bold green] Next steps:")
    console.print(f"  [cyan]forge visualize[/cyan] {local_path}")
    console.print(f"  [cyan]forge convert[/cyan] {local_path} ./output --format lerobot-v3")
    console.print(f"  [cyan]forge quality[/cyan] {local_path} --export report.json")
    if entry.paper_url:
        console.print(f"  Paper: {entry.paper_url}")


@app.command("segment")
def segment_cmd(
    path: str = typer.Argument(..., help="Path to dataset (local or hf://org/repo)"),
    signal: str = typer.Option(
        "observation.state",
        "--signal",
        help="Signal to segment: observation.state, qpos, action, joint_positions, joint_velocities",
    ),
    penalty: str = typer.Option("bic", "--penalty", "-p", help="PELT penalty: 'bic', 'aic', or numeric"),
    cost_model: str = typer.Option("rbf", "--cost-model", help="PELT cost model: rbf, l2, l1, normal, ar"),
    min_segment_length: int = typer.Option(10, "--min-segment-length", help="Minimum segment length in frames"),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize", help="Z-score normalize per dimension"),
    sample: int = typer.Option(0, "--sample", "-s", help="Segment N episodes (0 = all)"),
    export: Path | None = typer.Option(None, "--export", "-e", help="Export segmentation report to JSON"),
    plot: Path | None = typer.Option(None, "--plot", help="Generate timeline visualization PNG"),
    label: bool = typer.Option(False, "--label/--no-label", help="Apply semantic phase labels to segments"),
    format: str | None = typer.Option(None, "--format", "-f", help="Format hint (auto-detected if omitted)"),
) -> None:
    """Detect phase transitions in episode signals via PELT changepoint detection.

    Segments episodes into contiguous phases by running PELT on a proprioception
    signal (observation.state by default). Outputs per-episode changepoints.

    Examples:
        forge segment ./dataset
        forge segment ./dataset --signal action --penalty 5.0
        forge segment ./dataset --export segments.json --plot timeline.png
        forge segment hf://lerobot/aloha_sim_cube --sample 20
    """
    from rich.panel import Panel
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

    from forge.core.exceptions import ForgeError, MissingDependencyError
    from forge.formats.registry import FormatRegistry
    from forge.segment import SegmentAnalyzer, SegmentConfig
    from forge.segment.models import SegmentationReport

    config = SegmentConfig(
        signal=signal,
        penalty=penalty,
        cost_model=cost_model,
        min_segment_length=min_segment_length,
        normalize=normalize,
        label_phases=label,
    )

    # Resolve path
    resolved_path = _resolve_dataset_path(path)

    if not resolved_path.exists():
        console.print(f"[red]Error:[/red] Dataset not found: {resolved_path}")
        raise typer.Exit(1)

    # Detect format
    try:
        format_name = FormatRegistry.detect_format(resolved_path)
        reader = FormatRegistry.get_reader(format_name)
    except ForgeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[cyan]Segmentation:[/cyan] {path}")
    console.print(f"[dim]Format: {format_name} | Signal: {signal} | Penalty: {penalty} | Cost: {cost_model}[/dim]")
    console.print()

    analyzer = SegmentAnalyzer(config=config)
    report = SegmentationReport(dataset_path=str(path))

    try:
        with Progress(
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Segmenting episodes...", total=sample or None)

            for i, episode in enumerate(reader.read_episodes(resolved_path)):
                if sample > 0 and i >= sample:
                    break

                es = analyzer.segment_episode(episode)
                report.per_episode.append(es)
                progress.advance(task)

            if sample == 0:
                progress.update(
                    task, total=len(report.per_episode), completed=len(report.per_episode)
                )

    except MissingDependencyError:
        raise
    except ForgeError as e:
        console.print(f"[red]Error reading dataset:[/red] {e}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — showing partial results[/yellow]")

    report.num_episodes = len(report.per_episode)
    report.config = {
        "signal": signal,
        "penalty": penalty,
        "cost_model": cost_model,
        "min_segment_length": min_segment_length,
        "normalize": normalize,
        "label_phases": label,
    }
    report.compute_summary()

    if report.num_episodes == 0:
        console.print("[yellow]No episodes segmented.[/yellow]")
        raise typer.Exit(0)

    # ── Display results ──
    console.print()

    # Summary table
    table = Table(title="Segmentation Results", show_lines=False)
    table.add_column("Episode", style="cyan")
    table.add_column("Frames", justify="right")
    table.add_column("Segments", justify="right", style="green")
    table.add_column("Changepoints", style="dim")
    if label:
        table.add_column("Labels", style="yellow")

    for ep in report.per_episode:
        cp_str = ", ".join(str(c) for c in ep.changepoints) if ep.changepoints else "-"
        # Truncate long changepoint lists
        if len(cp_str) > 50:
            cp_str = cp_str[:47] + "..."
        row = [
            ep.episode_id,
            str(ep.num_frames),
            str(ep.num_segments),
            cp_str,
        ]
        if label:
            labels_str = " -> ".join(s.label or "?" for s in ep.segments) if ep.segments else "-"
            if len(labels_str) > 60:
                labels_str = labels_str[:57] + "..."
            row.append(labels_str)
        table.add_row(*row)

    console.print(table)

    # Summary panel
    summary = report.summary
    if summary:
        lines = [
            f"Episodes: {report.num_episodes}",
            f"Mean segments/episode: [green]{summary.get('mean_segments', '?')}[/green]",
            f"Range: {summary.get('min_segments', '?')} — {summary.get('max_segments', '?')}",
            f"Total changepoints: {summary.get('total_changepoints', '?')}",
        ]
        panel = Panel(
            "\n".join(lines),
            title="Summary",
            border_style="blue",
            padding=(0, 2),
        )
        console.print(panel)

    # JSON export
    if export:
        report.to_json(export)
        console.print(f"\n[green]Report saved to:[/green] {export}")

    # Plot
    if plot:
        try:
            from forge.segment.plot import plot_segmentation

            plot_segmentation(report, plot)
            console.print(f"[green]Timeline saved to:[/green] {plot}")
        except MissingDependencyError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)


@tokenize_app.command("list")
def tokenize_list_cmd() -> None:
    """List registered action tokenizer strategies."""
    from forge.tokenize import TokenizerRegistry

    table = Table(title="Action Tokenizer Strategies")
    table.add_column("Strategy", style="cyan", no_wrap=True)
    table.add_column("Granularity", style="green")
    table.add_column("Default vocab", justify="right")

    for name in TokenizerRegistry.list_strategies():
        tok = TokenizerRegistry.create(name)
        table.add_row(name, tok.granularity, str(tok.vocab_size))

    console.print(table)


@tokenize_app.command("compare")
def tokenize_compare_cmd(
    path: str = typer.Argument(..., help="Path to dataset (local or hf://org/repo)"),
    strategies: str | None = typer.Option(
        None, "--strategies", help="Comma-separated subset of strategies (default: all)"
    ),
    sample: int = typer.Option(
        0, "--sample", "-s", help="Score on N evenly-spaced frames (0 = all)"
    ),
    num_bins: int = typer.Option(256, "--num-bins", "-b", help="Vocabulary size"),
    export: Path | None = typer.Option(
        None, "--export", "-e", help="Export comparison report to JSON"
    ),
) -> None:
    """Benchmark every tokenizer strategy on your dataset.

    Reports reconstruction error (MSE/MAE/max-abs), tokens-per-step and vocab
    utilization so you can pick the discretization that fits your actions.

    Examples:
        forge tokenize compare ./dataset
        forge tokenize compare pusht --sample 20 --export report.json
    """
    from forge.core.exceptions import ForgeError
    from forge.tokenize import TokenizerComparator

    names = [s.strip() for s in strategies.split(",")] if strategies else None
    resolved = _resolve_dataset_path(path)
    if not resolved.exists():
        console.print(f"[red]Error:[/red] Dataset not found: {resolved}")
        raise typer.Exit(1)

    console.print(f"[cyan]Tokenizer comparison:[/cyan] {path}")
    try:
        with console.status("[bold green]Fitting and scoring strategies..."):
            report = TokenizerComparator().compare_dataset(
                resolved, strategies=names, sample=sample
            )
    except ForgeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(
        f"[dim]{report.num_frames} frames, action_dim={report.action_dim}, "
        f"scored on {report.sample_size}[/dim]\n"
    )

    best = report.best_by("mae")
    table = Table(title="Tokenizer Comparison")
    table.add_column("Strategy", style="cyan", no_wrap=True)
    table.add_column("Vocab", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("MSE", justify="right")
    table.add_column("Max-abs", justify="right")
    table.add_column("Tokens/step", justify="right")
    table.add_column("Vocab util", justify="right")

    for name, st in sorted(report.stats_by_strategy.items(), key=lambda kv: kv[1].mae):
        label = f"{name} [green]✓[/green]" if name == best else name
        table.add_row(
            label,
            str(st.vocab_size),
            f"{st.mae:.4f}",
            f"{st.mse:.4f}",
            f"{st.max_abs:.4f}",
            f"{st.tokens_per_step:.1f}",
            f"{st.vocab_utilization:.0%}",
        )
    console.print(table)
    if best:
        console.print(f"\n[green]Lowest MAE:[/green] {best}")

    if export:
        report.to_json(export)
        console.print(f"[green]Report saved to:[/green] {export}")


@tokenize_app.command("fit")
def tokenize_fit_cmd(
    path: str = typer.Argument(..., help="Path to dataset (local or hf://org/repo)"),
    out: Path = typer.Option(..., "--out", "-o", help="Output path for fitted tokenizer JSON"),
    strategy: str = typer.Option("openvla-bins", "--strategy", help="Tokenizer strategy"),
    num_bins: int = typer.Option(256, "--num-bins", "-b", help="Vocabulary size"),
) -> None:
    """Fit a tokenizer on a dataset's action corpus and save it.

    Examples:
        forge tokenize fit ./dataset --strategy openvla-bins --out tok.json
    """
    from forge.core.exceptions import ForgeError
    from forge.formats.registry import FormatRegistry
    from forge.tokenize import TokenizerRegistry
    from forge.tokenize.writer import _build_corpus

    resolved = _resolve_dataset_path(path)
    if not resolved.exists():
        console.print(f"[red]Error:[/red] Dataset not found: {resolved}")
        raise typer.Exit(1)

    try:
        fmt = FormatRegistry.detect_format(resolved)
        reader = FormatRegistry.get_reader(fmt)
        with console.status("[bold green]Fitting tokenizer..."):
            corpus = _build_corpus(reader.read_episodes(resolved))
            tok = TokenizerRegistry.create(strategy, num_bins=num_bins).fit(corpus)
            tok.save(out)
    except ForgeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(
        f"[green]Fitted[/green] {strategy} (vocab={tok.vocab_size}) on "
        f"{corpus.shape[0]} frames → [cyan]{out}[/cyan]"
    )


@tokenize_app.command("write")
def tokenize_write_cmd(
    source: str = typer.Argument(..., help="Source dataset (local or hf://org/repo)"),
    output: Path = typer.Argument(..., help="Output dataset directory"),
    strategy: str = typer.Option("openvla-bins", "--strategy", help="Tokenizer strategy"),
    num_bins: int = typer.Option(256, "--num-bins", "-b", help="Vocabulary size when fitting"),
    tokenizer: Path | None = typer.Option(
        None, "--tokenizer", help="Use a pre-fitted tokenizer JSON instead of fitting"
    ),
    keep_actions: bool = typer.Option(
        False, "--keep-actions", help="Retain the float action column alongside tokens"
    ),
    fps: float = typer.Option(30.0, "--fps", "-f", help="Output dataset FPS"),
) -> None:
    """Write a LeRobot v3 dataset with an action_tokens column.

    The fitted tokenizer is saved to <output>/meta/action_tokenizer.json for
    inference-time detokenization.

    Examples:
        forge tokenize write ./dataset ./out --strategy openvla-bins
        forge tokenize write pusht ./out --tokenizer tok.json --keep-actions
    """
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

    from forge.core.exceptions import ForgeError
    from forge.tokenize.writer import tokenize_and_write

    resolved = _resolve_dataset_path(source)
    if not resolved.exists():
        console.print(f"[red]Error:[/red] Dataset not found: {resolved}")
        raise typer.Exit(1)

    console.print(f"[cyan]Tokenizing:[/cyan] {source} → {output}")
    try:
        with Progress(
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Writing episodes...", total=None)

            def _cb(idx: int, ep_id: str) -> None:
                progress.advance(task)

            result = tokenize_and_write(
                source=resolved,
                output=output,
                strategy=strategy,
                tokenizer_path=tokenizer,
                num_bins=num_bins,
                keep_actions=keep_actions,
                fps=fps,
                progress_callback=_cb,
            )
    except ForgeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(
        f"[green]Wrote[/green] {result.num_episodes} episodes "
        f"({result.num_frames} frames) using {result.strategy}"
    )
    console.print(f"[green]Tokenizer saved to:[/green] {result.tokenizer_path}")


@app.command("version")
def version_cmd() -> None:
    """Show Forge version."""
    from forge import __version__

    console.print(f"Forge v{__version__}")


@app.callback()
def main() -> None:
    """Forge - The normalization layer for robotics data.

    Convert between RLDS, LeRobot, Zarr, and other robotics dataset formats.
    """
    pass


if __name__ == "__main__":
    app()
