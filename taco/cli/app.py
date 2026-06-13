"""
TACO-Benchmark CLI entry point.
"""

from __future__ import annotations

import typer

from taco import __version__
from taco.cli.data_cmd import data_app
from taco.cli.eval_cmd import eval_app
from taco.cli.exp_cmd import exp_app
from taco.cli.generate_cmd import generate_app

app = typer.Typer(
    name="taco",
    help="TACO-Benchmark: open-domain Text-to-SQL benchmark toolkit.",
    no_args_is_help=True,
)

app.add_typer(data_app, name="data")
app.add_typer(generate_app, name="generate")
app.add_typer(eval_app, name="eval")
app.add_typer(exp_app, name="exp")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """TACO-Benchmark toolkit."""


@app.command("info")
def info() -> None:
    """Print project paths and environment hints."""
    from taco.core.paths import BENCHMARK_DATA_DIR, CONFIGS_DIR, PROJECT_ROOT

    typer.echo(f"TACO-Benchmark v{__version__}")
    typer.echo(f"Project root : {PROJECT_ROOT}")
    typer.echo(f"Data dir     : {BENCHMARK_DATA_DIR}")
    typer.echo(f"Configs dir  : {CONFIGS_DIR}")
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo("  1. python scripts/setup_env.py")
    typer.echo("  2. taco data download")
    typer.echo("  3. taco data verify")
    typer.echo("  4. bash examples/quick_eval.sh   # or: taco eval run --model gpt-4o --dataset beijing")
    typer.echo("  5. taco generate single-db --database Housing --target-count 5 --region beijing")
    typer.echo("  6. taco exp ablation --setting qr_tl_qp --model gpt-4o")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
