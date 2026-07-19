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


def register_catalog_cli(app: typer.Typer) -> None:
    """Attach catalog commands to the main forge Typer app."""
    app.add_typer(catalog_app, name="catalog")
    app.command("ingest")(ingest_cmd)
    app.command("query")(query_cmd)
