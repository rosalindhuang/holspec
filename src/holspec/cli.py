"""
Command-line interface for holspec.

This module defines the installed ``holspec`` command. CLI commands handle
terminal-facing concerns such as argument parsing, config loading,
project-root resolution, and concise status/error messages, then delegate to
the public holspec API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
import yaml

from holspec.pipeline import run_pipeline


app = typer.Typer(
    help="Run holspec workflows from configuration files.",
    no_args_is_help=True,
)


@app.callback()
def cli() -> None:
    """Run holspec workflows from configuration files."""


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config file and ensure it contains a mapping."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            f"Config must contain a YAML mapping at the top level: "
            f"{config_path}"
        )

    return config


def _count_output_files(results: dict[str, dict[str, dict[str, Path]]]) -> int:
    """Count output files in a pipeline result mapping."""
    return sum(
        len(output_files)
        for stage_results in results.values()
        for output_files in stage_results.values()
    )


def _print_run_summary(
    *,
    config_path: Path,
    project_root: Path,
    config: dict[str, Any],
    results: dict[str, dict[str, dict[str, Path]]],
) -> None:
    """Print a concise post-run summary for CLI users."""
    output_dir = config.get("outputs", {}).get("data_dir", "(not configured)")
    stages_completed = ", ".join(results)
    output_file_count = _count_output_files(results)

    typer.echo()
    typer.secho("holspec pipeline run complete.", bold=True)
    typer.echo(f"\nconfig:       {config_path}")
    typer.echo(f"project root: {project_root}")
    typer.echo(f"output dir:   {output_dir}")
    typer.echo(f"stages:       {stages_completed}")
    typer.echo(f"output files: {output_file_count}")


@app.command()
def run(
    config_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        metavar="CONFIG",
        help="Path to a holspec pipeline YAML config.",
    ),
    project_root: Path = typer.Option(
        Path("."),
        "--project-root",
        "-r",
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Project root for resolving relative paths in the config.",
    ),
) -> None:
    """Run the full holspec pipeline from a YAML config."""
    try:
        config = _load_yaml_config(config_path)
        results = run_pipeline(config=config, project_root=project_root)
    except (OSError, yaml.YAMLError, ValueError, KeyError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    _print_run_summary(
        config_path=config_path,
        project_root=project_root,
        config=config,
        results=results,
    )


def main() -> None:
    """Console-script entry point."""
    app()
