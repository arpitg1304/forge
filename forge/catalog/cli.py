"""Catalog CLI commands, wired into the main ``forge`` app.

Kept import-light at module load (only typer/rich/stdlib) so that wiring these
commands into the base CLI never imports pyarrow/duckdb. Heavy imports and a
clear missing-dependency error live inside the command bodies.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    pass

console = Console()

catalog_app = typer.Typer(
    name="catalog",
    help="The catalog — an append-only, queryable registry of ingested episodes.",
)


def _require_catalog_deps() -> None:
    """Raise a clear install error if the catalog extra isn't installed."""
    missing = []
    for mod in ("pyarrow", "duckdb", "xxhash"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        from forge.core.exceptions import MissingDependencyError

        err = MissingDependencyError(
            dependency=", ".join(missing),
            feature="the Forge catalog",
            install_hint="pip install forge-robotics[catalog]",
        )
        console.print(f"[red]Error:[/red] {err}")
        raise typer.Exit(1)


@catalog_app.command("init")
def catalog_init_cmd(
    path: str = typer.Argument(..., help="Catalog root (local path or s3://, gs:// URI)"),
    exist_ok: bool = typer.Option(
        False, "--exist-ok", help="Reuse an existing catalog instead of failing"
    ),
) -> None:
    """Create a new catalog (writes catalog.json).

    Examples:
        forge catalog init ./forge-catalog
        forge catalog init s3://lab-bucket/forge-catalog
    """
    _require_catalog_deps()
    from forge.catalog import Catalog
    from forge.core.exceptions import ForgeError

    try:
        cat = Catalog.init(path, exist_ok=exist_ok)
    except ForgeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    console.print(
        f"[green]Initialized catalog[/green] at [cyan]{path}[/cyan] "
        f"(schema v{cat.schema_version})"
    )


@catalog_app.command("stats")
def catalog_stats_cmd(
    catalog: str = typer.Option(..., "--catalog", "-c", help="Catalog root"),
) -> None:
    """Show catalog summary statistics (counts, tasks, robots, score spread)."""
    _require_catalog_deps()
    from forge.catalog import Catalog
    from forge.core.exceptions import ForgeError

    try:
        cat = Catalog.open(catalog)
        s = cat.stats()
    except ForgeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[bold]Catalog:[/bold] {catalog}")
    console.print(
        f"  Episodes: [cyan]{s['episodes']}[/cyan]   "
        f"Frames: [cyan]{s['total_frames']:,}[/cyan]   "
        f"Hours: [cyan]{s['total_hours']}[/cyan]"
    )

    score = s["overall_score"]
    if score.get("scored"):
        console.print(
            f"  Quality (latest): mean [cyan]{score['mean']:.2f}[/cyan] "
            f"median [cyan]{score['median']:.2f}[/cyan] "
            f"range [dim]{score['min']:.2f}–{score['max']:.2f}[/dim] "
            f"over {score['scored']} scored"
        )

    def _print_counts(title: str, rows: list, key: str) -> None:
        if not rows:
            return
        table = Table(title=title, show_header=True, header_style="bold")
        table.add_column(key)
        table.add_column("episodes", justify="right")
        for r in rows[:20]:
            table.add_row(str(r[key]), str(r["n"]))
        console.print(table)

    _print_counts("By task", s["per_task"], "task")
    _print_counts("By robot", s["per_robot"], "robot")


def ingest_cmd(
    sources: list[str] = typer.Argument(
        ..., help="Dataset URIs to ingest (local paths, hf://, s3://, gs://)"
    ),
    catalog: str = typer.Option(..., "--catalog", "-c", help="Catalog root"),
    batch_id: str = typer.Option(
        None, "--batch-id", help="Group this run's files (generated if omitted)"
    ),
    batch_size: int = typer.Option(
        200, "--batch-size", help="Flush staged rows every N new episodes"
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Fail on the first episode error instead of skipping"
    ),
    embed: bool = typer.Option(
        False, "--embed", help="Also embed ingested episodes (requires [embed] extra)"
    ),
    embed_model: str = typer.Option(
        None, "--embed-model", help="Model for --embed (default: siglip-so400m)"
    ),
) -> None:
    """Ingest datasets into a catalog: register + quality-score each episode.

    Re-running over the same sources is a no-op (episodes are skipped by content
    hash).

    Examples:
        forge ingest ./my_dataset --catalog ./forge-catalog
        forge ingest s3://lab-bucket/raw/2026-07-18/ -c s3://lab-bucket/forge-catalog
    """
    _require_catalog_deps()
    from forge.catalog import Catalog
    from forge.catalog.ingest import ingest
    from forge.core.exceptions import ForgeError

    try:
        cat = Catalog.open(catalog)
    except ForgeError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("[dim]Run `forge catalog init` first.[/dim]")
        raise typer.Exit(1)

    stats = ingest(
        list(sources),
        cat,
        batch_id=batch_id,
        batch_size=batch_size,
        strict=strict,
        console=console,
    )
    console.print(
        f"[green]Ingest complete[/green] (batch {stats.batch_id[:8]}): "
        f"[cyan]{stats.ingested}[/cyan] ingested, "
        f"[yellow]{stats.skipped}[/yellow] skipped, "
        f"[red]{stats.failed}[/red] failed, "
        f"{stats.frames:,} frames"
    )

    if embed:
        from forge.catalog.embed import embed_catalog
        from forge.core.exceptions import ForgeError

        try:
            estats = embed_catalog(cat, model_name=embed_model, console=console)
        except ForgeError as e:
            console.print(f"[red]Embed error:[/red] {e}")
            raise typer.Exit(1)
        console.print(
            f"[green]Embed complete[/green] ({estats.model_id}): "
            f"[cyan]{estats.embedded}[/cyan] embedded "
            f"({estats.vision_rows} vision + {estats.text_rows} text vectors)"
        )


def query_cmd(
    sql: str = typer.Argument(..., help="SQL over: episodes, quality_scores, v_latest_quality"),
    catalog: str = typer.Option(..., "--catalog", "-c", help="Catalog root"),
    output_format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table | json | csv"
    ),
) -> None:
    """Run a read-only SQL query against the catalog.

    Examples:
        forge query "SELECT task, count(*) FROM episodes GROUP BY task" -c ./cat
        forge query "SELECT * FROM v_latest_quality LIMIT 5" -c ./cat --format json
    """
    _require_catalog_deps()
    from forge.catalog import Catalog
    from forge.core.exceptions import ForgeError

    try:
        result = Catalog.open(catalog).sql(sql)
    except ForgeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:  # surface DuckDB SQL errors cleanly
        console.print(f"[red]Query error:[/red] {e}")
        raise typer.Exit(1)

    rows = result.to_pylist()

    if output_format == "json":
        console.print_json(json.dumps(rows, default=str))
        return
    if output_format == "csv":
        import csv
        import io

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(result.column_names)
        for r in rows:
            writer.writerow([r.get(c) for c in result.column_names])
        # Plain stdout so it pipes cleanly.
        print(buf.getvalue(), end="")
        return

    if not rows:
        console.print("[dim](no rows)[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    for col in result.column_names:
        table.add_column(col)
    for r in rows[:1000]:
        table.add_row(*[_fmt_cell(r.get(c)) for c in result.column_names])
    console.print(table)
    if len(rows) > 1000:
        console.print(f"[dim]… {len(rows) - 1000} more rows (use --format json/csv)[/dim]")


def _fmt_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _render_result(result, output_format: str) -> None:
    """Print a pyarrow query/search result as table / json / csv."""
    rows = result.to_pylist()
    if output_format == "json":
        console.print_json(json.dumps(rows, default=str))
        return
    if output_format == "csv":
        import csv
        import io

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(result.column_names)
        for r in rows:
            writer.writerow([r.get(c) for c in result.column_names])
        print(buf.getvalue(), end="")
        return
    if not rows:
        console.print("[dim](no rows)[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    for col in result.column_names:
        table.add_column(col)
    for r in rows[:1000]:
        table.add_row(*[_fmt_cell(r.get(c)) for c in result.column_names])
    console.print(table)
    if len(rows) > 1000:
        console.print(f"[dim]… {len(rows) - 1000} more rows (use --format json/csv)[/dim]")


def embed_cmd(
    catalog: str = typer.Option(..., "--catalog", "-c", help="Catalog root"),
    model: str = typer.Option(
        None, "--model", "-m", help="Embedding model (default: siglip-so400m)"
    ),
    device: str = typer.Option(
        "auto", "--device", help="cuda | mps | cpu | auto (default: auto)"
    ),
    cameras: str = typer.Option(
        None, "--cameras", help="Comma-separated camera names to embed (default: all)"
    ),
    sample_hz: float = typer.Option(
        1.0, "--sample-hz", help="Frame sampling rate for pooling (default: 1.0)"
    ),
    pooling: str = typer.Option(
        "mean", "--pooling", help="mean | first-mid-last"
    ),
    batch_size: int = typer.Option(
        64, "--batch-size", help="Flush embeddings every N episodes"
    ),
) -> None:
    """Compute embeddings for a catalog's episodes (vision per camera + instruction).

    Re-running is a no-op — episodes already embedded for the model are skipped.
    Requires the [embed] extra (torch + transformers).

    Examples:
        forge embed -c ./forge-catalog
        forge embed -c s3://lab-bucket/forge-catalog --device cuda
    """
    _require_catalog_deps()
    from forge.catalog import Catalog
    from forge.catalog.embed import embed_catalog
    from forge.core.exceptions import ForgeError

    cam_list = [c.strip() for c in cameras.split(",")] if cameras else None
    try:
        cat = Catalog.open(catalog)
        stats = embed_catalog(
            cat,
            model_name=model,
            device=device,
            cameras=cam_list,
            sample_hz=sample_hz,
            pooling=pooling,
            batch_size=batch_size,
            console=console,
        )
    except ForgeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(
        f"[green]Embed complete[/green] ({stats.model_id}): "
        f"[cyan]{stats.embedded}[/cyan] episodes embedded, "
        f"[yellow]{stats.skipped}[/yellow] skipped, "
        f"[red]{stats.failed}[/red] failed "
        f"({stats.vision_rows} vision + {stats.text_rows} text vectors)"
    )


def search_cmd(
    query: str = typer.Argument(
        None, help="Text query (omit when using --like)"
    ),
    catalog: str = typer.Option(..., "--catalog", "-c", help="Catalog root"),
    like: str = typer.Option(
        None, "--like", help="Find episodes similar to this episode_id"
    ),
    top: int = typer.Option(20, "--top", "-k", help="Number of results"),
    level: str = typer.Option(
        "episode", "--level", help="episode (vision) | instruction (text)"
    ),
    camera: str = typer.Option(None, "--camera", help="Restrict to one camera view"),
    model: str = typer.Option(None, "--model", "-m", help="model_id (if catalog has several)"),
    device: str = typer.Option("auto", "--device", help="Device for the text encoder"),
    output_format: str = typer.Option(
        "table", "--format", "-f", help="table | json | csv"
    ),
) -> None:
    """Semantic search over a catalog's embeddings.

    Examples:
        forge search "picks up the red cup" -c ./forge-catalog --top 10
        forge search --like <episode_id> -c ./forge-catalog
    """
    _require_catalog_deps()
    from forge.catalog import Catalog
    from forge.core.exceptions import ForgeError

    if query is None and like is None:
        console.print("[red]Error:[/red] provide a text query or --like <episode_id>.")
        raise typer.Exit(1)

    try:
        result = Catalog.open(catalog).search(
            query,
            like=like,
            model_id=model,
            level=level,
            camera=camera,
            top=top,
            device=device,
        )
    except ForgeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    _render_result(result, output_format)


@catalog_app.command("dedup")
def catalog_dedup_cmd(
    catalog: str = typer.Option(..., "--catalog", "-c", help="Catalog root"),
    threshold: float = typer.Option(
        0.97, "--threshold", "-t", help="Cosine similarity threshold (default 0.97)"
    ),
    model: str = typer.Option(None, "--model", "-m", help="Embedding model_id"),
) -> None:
    """Find near-duplicate episode pairs from embeddings; store as dedup_edges.

    Records similarity *facts* (not verdicts) — which episode wins is decided at
    curation time. Re-running is idempotent.

    Examples:
        forge catalog dedup -c ./forge-catalog
        forge catalog dedup -c ./forge-catalog --threshold 0.9
    """
    _require_catalog_deps()
    from forge.catalog import Catalog
    from forge.catalog.dedup import compute_dedup_edges
    from forge.core.exceptions import ForgeError

    try:
        cat = Catalog.open(catalog)
        stats = compute_dedup_edges(cat, model_id=model, threshold=threshold)
    except ForgeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    console.print(
        f"[green]Dedup complete[/green] ({stats.model_id}, ≥ {threshold}): "
        f"[cyan]{stats.pairs_found}[/cyan] pairs found over {stats.episodes} episodes, "
        f"[cyan]{stats.pairs_added}[/cyan] new"
    )


def curate_cmd(
    catalog: str = typer.Option(..., "--catalog", "-c", help="Catalog root"),
    where: str = typer.Option(
        None, "--where", help="SQL filter over episodes + quality (e.g. \"overall_score > 6\")"
    ),
    label: str = typer.Option("approved", "--label", help="Label for survivors"),
    reason: str = typer.Option(None, "--reason", help="Free-text reason"),
    dedup: float = typer.Option(
        None, "--dedup", help="Also drop near-dup losers at this cosine threshold"
    ),
    dedup_policy: str = typer.Option(
        "keep-higher-quality", "--dedup-policy",
        help="keep-higher-quality | keep-longer | keep-first",
    ),
    by: str = typer.Option("cli", "--by", help="Recorded in labeled_by"),
) -> None:
    """Label a selection of episodes, optionally dropping near-duplicates.

    Appends to the curation_labels log (latest-row-wins); never deletes history.

    Examples:
        forge curate -c ./cat --where "overall_score > 6 AND task='pick_place'" --label approved
        forge curate -c ./cat --dedup 0.97 --dedup-policy keep-higher-quality --label approved
    """
    _require_catalog_deps()
    from forge.catalog import Catalog
    from forge.catalog.dedup import compute_dedup_edges, curate
    from forge.core.exceptions import ForgeError

    try:
        cat = Catalog.open(catalog)
        if dedup is not None:
            # Ensure edges exist at this threshold (idempotent).
            compute_dedup_edges(cat, threshold=dedup)
        stats = curate(
            cat, where=where, label=label, reason=reason, labeled_by=f"user:{by}",
            dedup_threshold=dedup, dedup_policy=dedup_policy,
        )
    except (ForgeError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    msg = (
        f"[green]Curated[/green]: {stats.selected} selected → "
        f"[cyan]{stats.approved}[/cyan] {label}"
    )
    if stats.rejected:
        msg += f", [red]{stats.rejected}[/red] rejected ({stats.dedup_policy})"
    console.print(msg)


def studio_cmd(
    catalog: str = typer.Option(..., "--catalog", "-c", help="Catalog root"),
    output: str = typer.Option("studio.html", "--output", "-o", help="Output HTML file"),
    threshold: float = typer.Option(
        0.97, "--threshold", "-t", help="Near-dup cosine threshold for the review tab"
    ),
    max_thumbnails: int = typer.Option(
        120, "--max-thumbnails", help="Cap on thumbnails extracted (0 = none)"
    ),
) -> None:
    """Generate a self-contained Forge Studio HTML app for a catalog.

    Tabs: Overview · Corpus · Dedup review · Snapshot, with real data and video
    thumbnails. Open the file in a browser.

    Examples:
        forge studio -c ./forge-catalog -o studio.html
    """
    _require_catalog_deps()
    from forge.catalog import Catalog
    from forge.catalog.dedup import compute_dedup_edges
    from forge.catalog.studio import generate_studio
    from forge.core.exceptions import ForgeError

    try:
        cat = Catalog.open(catalog)
        if cat.embedding_model_ids():
            compute_dedup_edges(cat, threshold=threshold)  # idempotent
        stats = generate_studio(
            cat, output, threshold=threshold,
            max_thumbnails=max_thumbnails, console=console,
        )
    except ForgeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    console.print(
        f"[green]Studio written[/green] → [cyan]{stats.out_path}[/cyan] "
        f"({stats.episodes} episodes, {stats.pairs} pairs, {stats.thumbnails} thumbnails)"
    )
    console.print(f"[dim]open it:[/dim] open {stats.out_path}")


def register_catalog_cli(app: typer.Typer) -> None:
    """Attach catalog commands to the main forge Typer app."""
    app.add_typer(catalog_app, name="catalog")
    app.command("ingest")(ingest_cmd)
    app.command("query")(query_cmd)
    app.command("embed")(embed_cmd)
    app.command("search")(search_cmd)
    app.command("curate")(curate_cmd)
    app.command("studio")(studio_cmd)
