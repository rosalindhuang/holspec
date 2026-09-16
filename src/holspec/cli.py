"""
Command-line interface for holspec.

This module defines the installed ``holspec`` command. CLI commands handle
terminal-facing concerns such as argument parsing, config loading,
project-root resolution, and concise status/error messages, then delegate to
the public holspec API.
"""

from __future__ import annotations

from contextlib import nullcontext, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import typer
import yaml

from holspec.pipeline import run_pipeline
from holspec.point_data import run_data_import, run_data_generation
from holspec.utilities import inspect_h5


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
            f"Config must contain a YAML mapping at the top level: {config_path}"
        )

    return config


def _count_output_files(results: dict[str, dict[str, dict[str, Path]]]) -> int:
    """Count output files in a pipeline result mapping."""
    return sum(
        len(output_files)
        for stage_results in results.values()
        for output_files in stage_results.values()
    )


def _count_nested_files(results: dict[str, dict[str, Path]]) -> int:
    """Count output files in a two-level result mapping."""
    return sum(len(output_files) for output_files in results.values())


def _display_path(path: Path, relative_to: Path) -> str:
    """Return a path relative to a root when possible."""
    try:
        return str(path.relative_to(relative_to))
    except ValueError:
        return str(path)


def _validate_output_options(verbose: bool, quiet: bool) -> None:
    """Validate mutually exclusive output controls."""
    if verbose and quiet:
        raise ValueError("Use either --verbose or --quiet, not both.")


def _set_config_verbose(
    config: dict[str, Any],
    verbose: bool,
    *,
    include_stages: bool = False,
) -> None:
    """Override runtime verbosity in a config dictionary."""
    config.setdefault("runtime", {})["verbose"] = verbose

    if include_stages:
        for stage_config in config.get("stages", {}).values():
            if isinstance(stage_config, dict):
                stage_config.setdefault("runtime", {})["verbose"] = verbose


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
    typer.echo(f"\nconfig:       {_display_path(config_path, project_root)}")
    typer.echo(f"project root: {project_root}")
    typer.echo(f"output dir:   {output_dir}")
    typer.echo(f"stages:       {stages_completed}")
    typer.echo(f"output files: {output_file_count}")


def _print_generate_data_summary(
    *,
    config_path: Path,
    project_root: Path,
    config: dict[str, Any],
    select_datasets: list[str] | None,
    results: dict[str, dict[str, Path]],
) -> None:
    """Print a concise data-generation summary for CLI users."""
    output_dir = config.get("outputs", {}).get("data_dir", "(not configured)")
    selected = ", ".join(select_datasets) if select_datasets else "(all)"
    generated = ", ".join(results) if results else "(none)"
    output_file_count = _count_nested_files(results)

    typer.echo()
    typer.secho("holspec data generation complete.", bold=True)
    typer.echo(f"\nconfig:               {_display_path(config_path, project_root)}")
    typer.echo(f"project root:         {project_root}")
    typer.echo(f"output dir:           {output_dir}")
    typer.echo(f"selected datasets:    {selected}")
    typer.echo(f"generated datasets:   {generated}")
    typer.echo(f"output files:         {output_file_count}")


def _print_import_data_summary(
    *,
    config_path: Path,
    project_root: Path,
    config: dict[str, Any],
    select_datasets: list[str] | None,
    results: dict[str, dict[str, Path]],
) -> None:
    """Print a concise data-import summary for CLI users."""
    output_dir = config.get("outputs", {}).get("data_dir", "(not configured)")
    selected = ", ".join(select_datasets) if select_datasets else "(all)"
    imported = ", ".join(results) if results else "(none)"
    output_file_count = _count_nested_files(results)

    typer.echo()
    typer.secho("holspec data import complete.", bold=True)
    typer.echo(f"\nconfig:               {_display_path(config_path, project_root)}")
    typer.echo(f"project root:         {project_root}")
    typer.echo(f"output dir:           {output_dir}")
    typer.echo(f"selected datasets:    {selected}")
    typer.echo(f"imported datasets:    {imported}")
    typer.echo(f"output files:         {output_file_count}")


@app.command()
def import_data(
    config_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        metavar="CONFIG",
        help="Path to a holspec data-import YAML config.",
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
    select_datasets: list[str] | None = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Dataset glob to import. Can be supplied multiple times.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Override the config and print detailed import output.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help="Suppress import progress output and print only the summary.",
    ),
) -> None:
    """Import external point data ensembles from a YAML config."""
    try:
        _validate_output_options(verbose, quiet)
        config = _load_yaml_config(config_path)
        if verbose or quiet:
            _set_config_verbose(config, verbose=verbose and not quiet)

        output_context = redirect_stdout(StringIO()) if quiet else nullcontext()
        with output_context:
            results = run_data_import(
                config=config,
                project_root=project_root,
                select_datasets=select_datasets,
            )
    except (OSError, yaml.YAMLError, ValueError, KeyError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    _print_import_data_summary(
        config_path=config_path,
        project_root=project_root,
        config=config,
        select_datasets=select_datasets,
        results=results,
    )


@app.command()
def generate_data(
    config_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        metavar="CONFIG",
        help="Path to a holspec data-generation YAML config.",
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
    select_datasets: list[str] | None = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Dataset glob to generate. Can be supplied multiple times.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Override the config and print detailed generation output.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help="Suppress generation progress output and print only the summary.",
    ),
) -> None:
    """Generate synthetic point cloud data ensembles from a YAML config."""
    try:
        _validate_output_options(verbose, quiet)
        config = _load_yaml_config(config_path)
        if verbose or quiet:
            _set_config_verbose(config, verbose=verbose and not quiet)

        output_context = redirect_stdout(StringIO()) if quiet else nullcontext()
        with output_context:
            results = run_data_generation(
                config=config,
                project_root=project_root,
                select_datasets=select_datasets,
            )
    except (OSError, yaml.YAMLError, ValueError, KeyError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    _print_generate_data_summary(
        config_path=config_path,
        project_root=project_root,
        config=config,
        select_datasets=select_datasets,
        results=results,
    )


@app.command()
def inspect(
    filepath: Path = typer.Argument(
        ...,
        metavar="PATH",
        help="Path to an HDF5 file to inspect.",
    ),
    project_root: Path = typer.Option(
        Path("."),
        "--project-root",
        "-r",
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Project root for resolving relative paths.",
    ),
    max_depth: int | None = typer.Option(
        None,
        "--max-depth",
        min=0,
        help="Maximum HDF5 tree depth to print.",
    ),
) -> None:
    """Inspect the structure and attributes of an HDF5 file."""
    inspect_path = filepath if filepath.is_absolute() else project_root / filepath

    try:
        inspect_h5(inspect_path, relative_to=project_root, max_depth=max_depth)
    except (OSError, ValueError, KeyError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


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
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Override the config and print detailed pipeline output.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help="Suppress pipeline progress output and print only the summary.",
    ),
) -> None:
    """Run the full holspec pipeline from a YAML config."""
    try:
        _validate_output_options(verbose, quiet)
        config = _load_yaml_config(config_path)
        if verbose or quiet:
            _set_config_verbose(
                config,
                verbose=verbose and not quiet,
                include_stages=True,
            )

        output_context = redirect_stdout(StringIO()) if quiet else nullcontext()
        with output_context:
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
