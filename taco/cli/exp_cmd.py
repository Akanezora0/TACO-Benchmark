"""Experiment CLI commands (baselines and TACO-SQL ablations)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from taco.core.experiments import (
    default_ablation_output,
    default_baseline_output,
    default_test_data,
    ensure_exists,
    resolve_dataset,
    run_python_script,
)
from taco.core.paths import EXPERIMENTS_DIR

exp_app = typer.Typer(help="Run baseline and TACO-SQL ablation experiments.")

ABLATION_SETTINGS = ("origin", "qr", "qr_tl", "qr_tl_qp")


@exp_app.command("baseline")
def baseline(
    model: str = typer.Option(..., "--model", "-m", help="Model name."),
    dataset: str = typer.Option("taco_beijing", "--dataset", "-d", help="Dataset name."),
    test_data: Optional[Path] = typer.Option(None, "--test-data", help="Test split JSON."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON path."),
    max_tables: Optional[int] = typer.Option(None, "--max-tables"),
    max_columns: Optional[int] = typer.Option(None, "--max-columns"),
) -> None:
    """Run a base-LLM baseline experiment (alias of ``taco eval run``)."""
    from taco.cli.eval_cmd import run as eval_run

    eval_run(
        model=model,
        dataset=dataset,
        test_data=test_data,
        output=output,
        max_tables=max_tables,
        max_columns=max_columns,
    )


@exp_app.command("ablation")
def ablation(
    setting: str = typer.Option(
        ...,
        "--setting",
        "-s",
        help="Ablation setting: origin, qr, qr_tl, qr_tl_qp.",
    ),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Model name."),
    dataset: str = typer.Option("taco_beijing", "--dataset", "-d", help="Dataset name."),
    test_data: Optional[Path] = typer.Option(None, "--test-data", help="Test split JSON."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON path."),
    qr_temperature: float = typer.Option(0.3, "--qr-temperature"),
    tl_top_k: int = typer.Option(5, "--tl-top-k"),
    qp_temperature: float = typer.Option(0.3, "--qp-temperature"),
) -> None:
    """Run a TACO-SQL ablation experiment."""
    if setting not in ABLATION_SETTINGS:
        typer.secho(
            f"Invalid setting '{setting}'. Choose from: {', '.join(ABLATION_SETTINGS)}",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    resolved = resolve_dataset(dataset)
    test_path = test_data or default_test_data(resolved)
    out_path = output or default_ablation_output(setting, model, resolved)

    try:
        ensure_exists(test_path, "Test data")
    except FileNotFoundError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        typer.echo("Run: taco data download && taco data verify")
        raise typer.Exit(code=1) from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "--setting",
        setting,
        "--model",
        model,
        "--dataset",
        resolved,
        "--test_data",
        str(test_path),
        "--output",
        str(out_path),
        "--qr_temperature",
        str(qr_temperature),
        "--tl_top_k",
        str(tl_top_k),
        "--qp_temperature",
        str(qp_temperature),
    ]

    code = run_python_script(EXPERIMENTS_DIR / "taco_sql_exp/run_ablation.py", args)
    if code != 0:
        raise typer.Exit(code=code)
    typer.secho(f"Results saved to {out_path}", fg=typer.colors.GREEN)


@exp_app.command("run-all")
def run_all(
    dataset: str = typer.Option("taco_beijing", "--dataset", "-d", help="Dataset name."),
    output_dir: Path = typer.Option(
        EXPERIMENTS_DIR / "results",
        "--output-dir",
        help="Output directory for all baseline runs.",
    ),
    base_llm: bool = typer.Option(False, "--base-llm", help="Run base LLM baselines."),
    llm_based: bool = typer.Option(False, "--llm-based", help="Run LLM-based baselines."),
    sft_based: bool = typer.Option(False, "--sft-based", help="Run SFT baselines."),
    hybrid: bool = typer.Option(False, "--hybrid", help="Run hybrid baselines."),
    all_types: bool = typer.Option(
        False,
        "--all",
        help="Run every baseline category (may require local model weights).",
    ),
) -> None:
    """Run selected baseline experiment suites."""
    resolved = resolve_dataset(dataset)
    args = [
        "--dataset",
        resolved,
        "--output_dir",
        str(output_dir),
    ]
    if all_types:
        args.append("--all")
    else:
        if base_llm:
            args.append("--base_llm")
        if llm_based:
            args.append("--llm_based")
        if sft_based:
            args.append("--sft_based")
        if hybrid:
            args.append("--hybrid")
        if len(args) == 2:
            typer.secho(
                "Specify at least one suite: --base-llm, --llm-based, --sft-based, --hybrid, or --all",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

    code = run_python_script(EXPERIMENTS_DIR / "baselines/run_all_baselines.py", args)
    if code != 0:
        raise typer.Exit(code=code)
