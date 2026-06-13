"""Dataset management CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from taco.core.dataset import (
    GOOGLE_DRIVE_VIEW_URL,
    download_and_extract,
    verify_dataset,
)
from taco.core.paths import BENCHMARK_DATA_DIR

data_app = typer.Typer(help="Download and verify the TACO-Benchmark dataset.")


@data_app.command("download")
def download(
    output: Path = typer.Option(
        BENCHMARK_DATA_DIR,
        "--output",
        "-o",
        help="Target directory for extracted data.",
    ),
    force: bool = typer.Option(False, "--force", help="Re-download and re-extract."),
) -> None:
    """Download and extract the dataset from Google Drive."""
    typer.echo(f"Source: {GOOGLE_DRIVE_VIEW_URL}")
    try:
        data_dir = download_and_extract(data_dir=output, force=force)
    except Exception as exc:
        typer.secho(f"Download failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    missing = verify_dataset(data_dir)
    if missing:
        typer.secho("Extraction completed with warnings. Missing paths:", fg=typer.colors.YELLOW)
        for rel in missing:
            typer.echo(f"  - {data_dir / rel}")
        raise typer.Exit(code=1)

    typer.secho(f"Dataset ready at {data_dir.resolve()}", fg=typer.colors.GREEN)


@data_app.command("verify")
def verify(
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data-dir",
        help="Data directory to verify (default: benchmark/data).",
    ),
) -> None:
    """Check that the local dataset layout is complete."""
    root = data_dir or BENCHMARK_DATA_DIR
    missing = verify_dataset(root)
    if missing:
        typer.secho(f"Dataset incomplete at {root.resolve()}", fg=typer.colors.RED)
        for rel in missing:
            typer.echo(f"  missing: {rel}")
        typer.echo(f"\nDownload: {GOOGLE_DRIVE_VIEW_URL}")
        typer.echo("Run: taco data download")
        raise typer.Exit(code=1)
    typer.secho(f"Dataset OK at {root.resolve()}", fg=typer.colors.GREEN)
