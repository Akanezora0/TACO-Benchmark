"""Data generation CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from taco.core.experiments import resolve_dataset, run_python_script
from taco.core.paths import (
    BENCHMARK_DATA_DIR,
    CROSS_DATABASE_DIR,
    CROSS_DATABASE_US_DIR,
    GENERATION_DIR,
)

generate_app = typer.Typer(help="Regenerate benchmark data (requires LLM API).")

REGION_ALIASES: dict[str, str] = {
    "beijing": "beijing",
    "us": "us",
    "taco_beijing": "beijing",
    "taco_us": "us",
}


def resolve_region(name: str) -> str:
    """Normalize region shorthand (``beijing`` / ``us``)."""
    key = name.strip().lower().replace("-", "_")
    if key in REGION_ALIASES:
        return REGION_ALIASES[key]
    if key.startswith("taco_"):
        return REGION_ALIASES.get(key, key.replace("taco_", ""))
    return key


def region_data_dir(region: str) -> Path:
    """Return ``benchmark/data/{region}/``."""
    resolved = resolve_region(region)
    return BENCHMARK_DATA_DIR / resolved


@generate_app.command("single-db")
def single_db(
    database: str = typer.Option(..., "--database", "-db", help="Database name."),
    target_count: int = typer.Option(..., "--target-count", "-n", help="Target NL query count."),
    region: str = typer.Option("beijing", "--region", "-r", help="Region: beijing or us."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Output root."),
    skip_skeleton: bool = typer.Option(False, "--skip-skeleton"),
    skip_graph: bool = typer.Option(False, "--skip-graph"),
    skip_fill: bool = typer.Option(False, "--skip-fill"),
    skip_nl: bool = typer.Option(False, "--skip-nl"),
) -> None:
    """Run skeleton → graph → SQL fill → NL for one database."""
    resolved = resolve_region(region)
    data_root = region_data_dir(resolved)
    out = output_dir or (data_root / "output")
    schema_dir = data_root / ("database_chinese" if resolved == "beijing" else "database")
    database_dir = data_root / "database"
    expert_file = BENCHMARK_DATA_DIR / "target" / f"expert_skeletons_{resolved}.json"
    if not expert_file.is_file() and resolved == "beijing":
        expert_file = BENCHMARK_DATA_DIR / "target" / "expert_skeletons_beijing.json"

    args = [
        "--database",
        database,
        "--target_count",
        str(target_count),
        "--output_dir",
        str(out),
        "--schema_dir",
        str(schema_dir),
        "--database_dir",
        str(database_dir),
        "--expert_file",
        str(expert_file),
    ]
    if skip_skeleton:
        args.append("--skip_skeleton")
    if skip_graph:
        args.append("--skip_graph")
    if skip_fill:
        args.append("--skip_fill")
    if skip_nl:
        args.append("--skip_nl")

    typer.echo(f"Region   : {resolved}")
    typer.echo(f"Database : {database}")
    typer.echo(f"Output   : {out}")

    code = run_python_script(GENERATION_DIR / "complete_pipeline_single_db.py", args)
    if code != 0:
        raise typer.Exit(code=code)
    typer.secho("Pipeline finished.", fg=typer.colors.GREEN)


@generate_app.command("cross-db")
def cross_db(
    region: str = typer.Option("beijing", "--region", "-r", help="Region: beijing or us."),
    status: bool = typer.Option(False, "--status", help="Show generation status only."),
    step: Optional[int] = typer.Option(None, "--step", help="Run a single step (1–5)."),
    from_step: int = typer.Option(1, "--from-step", help="Start step (default: 1)."),
    to_step: int = typer.Option(5, "--to-step", help="End step (default: 5)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Run the cross-database JOIN SQL pipeline (or show status)."""
    resolved = resolve_region(region)
    if resolved == "us":
        script = CROSS_DATABASE_US_DIR / "run_all.py"
    else:
        script = CROSS_DATABASE_DIR / "run_all.py"

    args: list[str] = []
    if status:
        args.append("--status")
    if step is not None:
        args.extend(["--step", str(step)])
    if from_step != 1:
        args.extend(["--from-step", str(from_step)])
    if to_step != 5:
        args.extend(["--to-step", str(to_step)])
    if yes:
        args.append("-y")

    typer.echo(f"Region : {resolved}")
    typer.echo(f"Script : {script}")

    code = run_python_script(script, args, cwd=script.parent)
    if code != 0:
        raise typer.Exit(code=code)


@generate_app.command("status")
def generation_status(
    region: str = typer.Option("beijing", "--region", "-r", help="Region: beijing or us."),
) -> None:
    """Show cross-database SQL generation progress."""
    cross_db(region=region, status=True, step=None, from_step=1, to_step=5, yes=True)
