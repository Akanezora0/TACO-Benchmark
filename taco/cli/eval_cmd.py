"""Evaluation CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from taco.core.experiments import (
    default_baseline_output,
    default_test_data,
    ensure_exists,
    load_results_json,
    resolve_dataset,
    run_python_script,
    summarize_results,
)
from taco.core.paths import EXPERIMENTS_DIR, PROJECT_ROOT

eval_app = typer.Typer(help="Run baseline evaluation and compute metrics.")


@eval_app.command("run")
def run(
    model: str = typer.Option(..., "--model", "-m", help="Model name (e.g. gpt-4o)."),
    dataset: str = typer.Option(
        "taco_beijing",
        "--dataset",
        "-d",
        help="Dataset name or shorthand (beijing, us, taco_beijing).",
    ),
    test_data: Optional[Path] = typer.Option(
        None,
        "--test-data",
        help="Test split JSON (default: benchmark/data/final/{dataset}/test.json).",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output results JSON path.",
    ),
    max_tables: Optional[int] = typer.Option(None, "--max-tables", help="Max tables in schema."),
    max_columns: Optional[int] = typer.Option(
        None,
        "--max-columns",
        help="Max columns per table.",
    ),
) -> None:
    """Run a base-LLM baseline experiment on the official test split."""
    resolved = resolve_dataset(dataset)
    test_path = test_data or default_test_data(resolved)
    out_path = output or default_baseline_output(model, resolved)

    try:
        ensure_exists(test_path, "Test data")
    except FileNotFoundError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        typer.echo("Run: taco data download && taco data verify")
        raise typer.Exit(code=1) from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "--model",
        model,
        "--test_data",
        str(test_path),
        "--output",
        str(out_path),
    ]
    if max_tables is not None:
        args.extend(["--max_tables", str(max_tables)])
    if max_columns is not None:
        args.extend(["--max_columns", str(max_columns)])

    typer.echo(f"Dataset  : {resolved}")
    typer.echo(f"Test data: {test_path}")
    typer.echo(f"Output   : {out_path}")

    code = run_python_script(
        EXPERIMENTS_DIR / "baselines/base_llm/run_experiment.py",
        args,
    )
    if code != 0:
        raise typer.Exit(code=code)
    typer.secho(f"Results saved to {out_path}", fg=typer.colors.GREEN)


@eval_app.command("batch")
def batch(
    models: str = typer.Option(
        "gpt-4o,gpt-4o-mini",
        "--models",
        help="Comma-separated model names, or 'all'.",
    ),
    dataset: str = typer.Option("taco_beijing", "--dataset", "-d", help="Dataset name."),
    test_data: Optional[Path] = typer.Option(None, "--test-data", help="Test split JSON."),
    output_dir: Path = typer.Option(
        EXPERIMENTS_DIR / "results" / "baselines",
        "--output-dir",
        help="Directory for per-model result files.",
    ),
    max_workers: int = typer.Option(3, "--max-workers", help="Parallel model workers."),
    aggregate: Optional[Path] = typer.Option(
        None,
        "--aggregate",
        help="Optional path to write aggregated summary JSON.",
    ),
) -> None:
    """Batch-evaluate multiple base-LLM models."""
    resolved = resolve_dataset(dataset)
    test_path = test_data or default_test_data(resolved)

    try:
        ensure_exists(test_path, "Test data")
    except FileNotFoundError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if len(model_list) == 1 and model_list[0].lower() == "all":
        model_args = ["--models", "all"]
    else:
        model_args = ["--models", *model_list]

    args = [
        *model_args,
        "--test_data",
        str(test_path),
        "--output_dir",
        str(output_dir),
        "--max_workers",
        str(max_workers),
    ]
    if aggregate:
        args.extend(["--aggregate", str(aggregate)])

    code = run_python_script(
        EXPERIMENTS_DIR / "baselines/base_llm/batch_evaluate.py",
        args,
    )
    if code != 0:
        raise typer.Exit(code=code)


@eval_app.command("report")
def report(
    pred: Path = typer.Option(..., "--pred", "-p", help="Predictions / results JSON."),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write metrics summary JSON to this path.",
    ),
    gold: Optional[Path] = typer.Option(
        None,
        "--gold",
        "-g",
        help="Gold queries JSON (for Spider-style evaluation.py).",
    ),
    db_dir: Optional[Path] = typer.Option(
        None,
        "--db",
        help="Database directory for evaluation.py execution match.",
    ),
    etype: str = typer.Option(
        "exec",
        "--etype",
        help="evaluation.py mode: exec, match, or all.",
    ),
) -> None:
    """Summarize experiment results or run execution evaluation."""
    if not pred.is_file():
        typer.secho(f"Results file not found: {pred}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if gold and db_dir:
        args = [
            "--gold",
            str(gold),
            "--pred",
            str(pred),
            "--db",
            str(db_dir),
            "--etype",
            etype,
        ]
        code = run_python_script(EXPERIMENTS_DIR / "evaluation/evaluation.py", args)
        raise typer.Exit(code=code)

    results = load_results_json(pred)
    summary = summarize_results(results)

    try:
        ensure_project_on_path()
        from experiments.evaluation.metrics_calculator import calculate_metrics

        if any("is_correct" in r for r in results):
            summary["metrics"] = calculate_metrics(results)
    except Exception:
        pass

    typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        typer.secho(f"Summary written to {output}", fg=typer.colors.GREEN)


@eval_app.command("legacy-db")
def legacy_db(
    database: str = typer.Option(..., "--database", help="Database directory name."),
    model: str = typer.Option(..., "--model", "-m", help="Model name."),
    region: str = typer.Option(
        "beijing",
        "--region",
        help="Dataset region: beijing or us.",
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON path."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Limit evaluated queries."),
    max_workers: int = typer.Option(5, "--max-workers", help="API concurrency."),
) -> None:
    """
    Per-database baseline evaluation (legacy script wrapper).

    Uses ``evaluate_baseline.py`` with explicit NL/SQL/DB paths.
    """
    data_root = PROJECT_ROOT / "benchmark/data" / region
    db_root = data_root / ("database_chinese" if region == "beijing" else "database")
    nl_dir = data_root / "output/nl_query" / database
    sql_dir = data_root / "output/single" / database
    db_path = db_root / database / f"{database}.db"
    schema_file = db_root / database / f"{database}.json"
    out_path = output or (
        EXPERIMENTS_DIR / "baselines/base_llm/results" / f"{region}_{database}_{model}.json"
    )

    for label, path in [
        ("NL dir", nl_dir),
        ("SQL dir", sql_dir),
        ("DB file", db_path),
        ("Schema", schema_file),
    ]:
        if not path.exists():
            typer.secho(f"{label} not found: {path}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)

    args = [
        "--nl_query_dir",
        str(nl_dir),
        "--sql_dir",
        str(sql_dir),
        "--db_path",
        str(db_path),
        "--schema_file",
        str(schema_file),
        "--model",
        model,
        "--output_file",
        str(out_path),
        "--max_workers",
        str(max_workers),
    ]
    if limit is not None:
        args.extend(["--limit", str(limit)])

    code = run_python_script(
        EXPERIMENTS_DIR / "baselines/base_llm/evaluate_baseline.py",
        args,
    )
    if code != 0:
        raise typer.Exit(code=code)
