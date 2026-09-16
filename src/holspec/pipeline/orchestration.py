"""
End-to-end orchestration for the holspec pipeline.

Provides ``run_pipeline`` (files-first) and ``run_pipeline_stages``
(stages-first) for running the full pipeline, plus internal helpers for
assembling per-stage configs from a unified pipeline config and saving
those configs to YAML.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from holspec.utilities import (
    convert_relative_to_paths,
    convert_paths_to_relative,
)

from .stages import (
    PIPELINE_STAGE_NAMES,
    run_topology_simplicial,
    run_geometry_metric,
    run_hodge_laplacian,
    run_spectra,
)


PIPELINE_STAGE_FUNCTIONS: dict[int, callable] = {
    1: run_topology_simplicial,
    2: run_geometry_metric,
    3: run_hodge_laplacian,
    4: run_spectra,
}


def _assemble_stage_config(
    pipeline_config: dict,
    stage_num: int,
    input_filepaths: dict,
    project_root: Path,
) -> dict:
    """
    Build a per-stage config dict from the pipeline config.

    Assembles the standard five-key per-stage schema (``summary``,
    ``inputs``, ``configs``, ``runtime``, ``outputs``) by combining
    stage settings from the pipeline config with dynamically-wired
    input filepaths from the previous stage's output.

    Pipeline-level ``error_handling``, ``output_retention``, and
    ``save_error_report`` values are propagated into the stage runtime
    when the stage does not already specify them. Stage-level values
    take precedence.

    Parameters
    ----------
    pipeline_config : dict
        Full pipeline config dict.
    stage_num : int
        Stage number (1--4).
    input_filepaths : dict
        Input filepaths for this stage. Flat ``{ptd_label: path}`` for
        Stage 1, nested ``{ptd_label: {label: path}}`` for Stages 2--4.
        Values may be Path objects or relative path strings; Path objects
        are converted to relative strings via ``convert_paths_to_relative``.
    project_root : Path
        Project root for converting absolute paths to relative strings.

    Returns
    -------
    dict
        Per-stage config dict with keys: ``summary``, ``inputs``,
        ``configs``, ``runtime``, ``outputs``.
    """
    stage_name = PIPELINE_STAGE_NAMES[stage_num]
    prev_stage_name = PIPELINE_STAGE_NAMES[stage_num - 1]
    base_data_dir = pipeline_config["outputs"]["data_dir"]

    # Input data directory: raw data for Stage 1, previous output dir
    # for Stages 2--4
    if stage_num == 1:
        input_data_dir = pipeline_config["inputs"]["data_dir"]
    else:
        input_data_dir = f"{base_data_dir}/{prev_stage_name}"

    # Build runtime: stage-level takes precedence, pipeline-level fills gaps
    runtime = dict(pipeline_config["stages"][stage_name]["runtime"])
    pipeline_runtime = pipeline_config.get("runtime", {})
    for key in ("error_handling", "output_retention", "save_error_report"):
        if key not in runtime:
            value = pipeline_runtime.get(key)
            if value is not None:
                runtime[key] = value

    return {
        "summary": {
            "created_by": pipeline_config["summary"]["created_by"],
            "creation_time": datetime.now().isoformat(),
        },
        "inputs": {
            "stage_name": prev_stage_name,
            "data_dir": input_data_dir,
            "filepaths": convert_paths_to_relative(
                input_filepaths,
                project_root,
            ),
        },
        "configs": pipeline_config["stages"][stage_name]["configs"],
        "runtime": runtime,
        "outputs": {
            "stage_name": stage_name,
            "data_dir": f"{base_data_dir}/{stage_name}",
        },
    }


def _save_stage_configs(
    pipeline_config: dict,
    results: dict,
    project_root: Path,
) -> None:
    """
    Save assembled per-stage configs as YAML files.

    For each stage, assembles a complete standalone config dict
    containing input filepaths for all point data labels, then writes
    it as a YAML file. Saved configs can be passed directly to
    ``run_{stage}`` functions for independent re-execution.

    Parameters
    ----------
    pipeline_config : dict
        Full pipeline config dict. Must have ``outputs.configs_dir``
        specifying the directory for saved config files.
    results : dict
        Accumulated pipeline results with shape
        ``{stage_name: {ptd_label: {output_label: Path}}}``.
    project_root : Path
        Project root for path resolution.
    """
    configs_dir = project_root / pipeline_config["outputs"]["configs_dir"]
    configs_dir.mkdir(parents=True, exist_ok=True)

    for stage_num in (1, 2, 3, 4):
        stage_name = PIPELINE_STAGE_NAMES[stage_num]

        # Collect full input filepaths for this stage
        if stage_num == 1:
            input_filepaths = convert_relative_to_paths(
                pipeline_config["inputs"]["filepaths"],
                project_root,
            )
        else:
            prev_stage_name = PIPELINE_STAGE_NAMES[stage_num - 1]
            input_filepaths = results[prev_stage_name]

        # Assemble complete per-stage config and save
        stage_config = _assemble_stage_config(
            pipeline_config,
            stage_num,
            input_filepaths,
            project_root,
        )

        config_path = configs_dir / f"{stage_name}.yml"
        with open(config_path, "w") as f:
            yaml.safe_dump(
                stage_config,
                f,
                default_flow_style=False,
                sort_keys=False,
            )


def run_pipeline(
    config: dict,
    project_root: str | Path,
) -> dict[str, dict[str, dict[str, Path]]]:
    """
    Run the full pipeline end-to-end for all point data inputs.

    Iterates over point data inputs (outer loop) and stages (inner loop),
    wiring each stage's output filepaths into the next stage's input.
    Reuses the ``run_{stage}`` functions for all computation; the
    orchestrator handles config assembly and inter-stage wiring only.

    All intermediate results are written to disk by the ``run_{stage}``
    functions. Assembled per-stage configs can optionally be saved as
    YAML files for reproducibility.

    Parameters
    ----------
    config : dict
        Pipeline config dict with keys: ``summary``, ``inputs``,
        ``stages``, ``runtime``, ``outputs``.
    project_root : str or Path
        Project root for resolving relative paths.

    Returns
    -------
    dict[str, dict[str, dict[str, Path]]]
        Nested mapping ``{stage_name: {ptd_label: {output_label: path}}}``
        with absolute paths to all output files.

    Raises
    ------
    ValueError
        If the pipeline config is missing required stage entries or
        ``configs_dir`` when ``save_stage_configs`` is enabled.
    FileNotFoundError
        If any point data input file does not exist.
    """
    project_root = Path(project_root)

    # --- Extract settings ---

    input_filepaths = convert_relative_to_paths(
        config["inputs"]["filepaths"],
        project_root,
    )
    verbose = config["runtime"]["verbose"]
    save_stage_configs = config["runtime"].get("save_stage_configs", False)

    # --- Validate inputs ---

    # All four stages must be present in config
    expected_stages = {PIPELINE_STAGE_NAMES[n] for n in (1, 2, 3, 4)}
    provided_stages = set(config.get("stages", {}).keys())
    missing_stages = expected_stages - provided_stages
    if missing_stages:
        raise ValueError(
            f"Pipeline config 'stages' is missing entries: {sorted(missing_stages)}"
        )

    # All input files must exist
    for ptd_label, ptd_filepath in input_filepaths.items():
        if not ptd_filepath.exists():
            raise FileNotFoundError(
                f"Input file for '{ptd_label}' does not exist: {ptd_filepath}"
            )

    # configs_dir required when save_stage_configs is enabled
    if save_stage_configs and "configs_dir" not in config.get("outputs", {}):
        raise ValueError(
            "'outputs.configs_dir' must be specified when "
            "'runtime.save_stage_configs' is true."
        )

    # --- Run pipeline ---

    results: dict[str, dict[str, dict[str, Path]]] = {}

    for ptd_label, ptd_filepath in input_filepaths.items():
        if verbose:
            print(f"{'=' * 60}")
            print(f"{ptd_label}")
            print(f"{'=' * 60}")

        # Stage 1 input: flat {ptd_label: filepath}
        prev_output = {ptd_label: ptd_filepath}

        for stage_num in (1, 2, 3, 4):
            stage_name = PIPELINE_STAGE_NAMES[stage_num]

            if verbose:
                print()
                print(f"{'-' * 60}")
                print(f"Stage {stage_num}: {stage_name}")
                print(f"{'-' * 60}")
                print()

            # Assemble per-stage config for this single input
            stage_config = _assemble_stage_config(
                config,
                stage_num,
                prev_output,
                project_root,
            )

            # Run the stage
            stage_output = PIPELINE_STAGE_FUNCTIONS[stage_num](
                stage_config,
                project_root,
            )

            # Accumulate results and wire output to next stage
            results.setdefault(stage_name, {}).update(stage_output)
            prev_output = stage_output

        if not stage_config["runtime"].get("verbose", False):
            print()

    # --- Save assembled per-stage configs ---

    if save_stage_configs and results:
        _save_stage_configs(config, results, project_root)

    # --- Completion ---

    if verbose:
        num_inputs = len(input_filepaths)
        print(
            f"Pipeline complete. Outputs saved for "
            f"{len(PIPELINE_STAGE_FUNCTIONS)} stages, "
            f"{num_inputs} point data input"
            f"{'s' if num_inputs != 1 else ''}."
        )

    return results


def run_pipeline_stages(
    config: dict,
    project_root: str | Path,
    stage_nums: tuple[int, ...] = (1, 2, 3, 4),
    input_filepaths: dict | None = None,
) -> dict[str, dict[str, dict[str, Path]]]:
    """
    Run the full pipeline stage-by-stage for all point data inputs.

    Iterates over stages (outer loop) and processes all point data
    inputs within each stage (inner loop handled by ``run_{stage}``).
    Each stage completes for all inputs before the next stage begins.
    Wires each stage's output filepaths into the next stage's input.

    This is the stages-first complement to :func:`run_pipeline`, which
    iterates files-first. Both produce identical results when run with
    default arguments; the difference is execution order and verbose
    output structure.

    A subset of stages can be selected via ``stage_nums``. When the
    first selected stage is not stage 1, ``input_filepaths`` must
    supply the previous stage's output filepaths so the first selected
    stage has inputs to consume. The return value contains results for
    the selected stages only and can be chained across calls::

        results_12 = run_pipeline_stages(cfg, root, stage_nums=(1, 2))
        results_34 = run_pipeline_stages(
            cfg, root, stage_nums=(3, 4),
            input_filepaths=results_12['geometry_metric'],
        )

    All intermediate results are written to disk by the ``run_{stage}``
    functions. Assembled per-stage configs can optionally be saved as
    YAML files for reproducibility.

    Parameters
    ----------
    config : dict
        Pipeline config dict with keys: ``summary``, ``inputs``,
        ``stages``, ``runtime``, ``outputs``.
    project_root : str or Path
        Project root for resolving relative paths.
    stage_nums : tuple of int, optional
        Stage numbers to run, by default ``(1, 2, 3, 4)``.
        Must be a subset of ``{1, 2, 3, 4}``.
    input_filepaths : dict or None, optional
        Input filepaths for the first selected stage. Required when
        ``stage_nums`` does not start at 1. For stage 1, this should
        be flat ``{ptd_label: path}``; for stages 2--4, nested
        ``{ptd_label: {output_label: path}}``. Ignored (and
        overridden by the pipeline config's input filepaths) when
        the first selected stage is 1.

    Returns
    -------
    dict[str, dict[str, dict[str, Path]]]
        Nested mapping ``{stage_name: {ptd_label: {output_label: path}}}``
        with absolute paths to all output files, for selected stages
        only.

    Raises
    ------
    ValueError
        If ``stage_nums`` contains invalid entries, if the pipeline
        config is missing required stage entries, if ``input_filepaths``
        is not provided when the first stage is not 1, or if
        ``configs_dir`` is missing when ``save_stage_configs`` is
        enabled.
    FileNotFoundError
        If any point data input file does not exist (checked only
        when starting from stage 1).
    """
    project_root = Path(project_root)

    # --- Extract settings ---

    verbose = config["runtime"]["verbose"]
    save_stage_configs = config["runtime"].get("save_stage_configs", False)

    # --- Validate stage_nums ---

    valid_stage_nums = set(PIPELINE_STAGE_FUNCTIONS)
    invalid_stage_nums = set(stage_nums) - valid_stage_nums
    if invalid_stage_nums:
        raise ValueError(
            f"Invalid stage numbers: {sorted(invalid_stage_nums)}. "
            f"Valid values are {sorted(valid_stage_nums)}."
        )

    # --- Validate stage configs ---

    expected_stages = {PIPELINE_STAGE_NAMES[n] for n in stage_nums}
    provided_stages = set(config.get("stages", {}).keys())
    missing_stages = expected_stages - provided_stages
    if missing_stages:
        raise ValueError(
            f"Pipeline config 'stages' is missing entries: {sorted(missing_stages)}"
        )

    # --- Resolve initial input filepaths ---

    first_stage = stage_nums[0]

    if first_stage == 1:
        prev_output = convert_relative_to_paths(
            config["inputs"]["filepaths"],
            project_root,
        )

        # All input files must exist
        for ptd_label, ptd_filepath in prev_output.items():
            if not ptd_filepath.exists():
                raise FileNotFoundError(
                    f"Input file for '{ptd_label}' does not exist: {ptd_filepath}"
                )
    else:
        if input_filepaths is None:
            raise ValueError(
                f"'input_filepaths' is required when the first "
                f"selected stage is not 1 (got stage_nums={stage_nums})."
            )
        prev_output = input_filepaths

    # --- Validate save_stage_configs ---

    if save_stage_configs and "configs_dir" not in config.get("outputs", {}):
        raise ValueError(
            "'outputs.configs_dir' must be specified when "
            "'runtime.save_stage_configs' is true."
        )

    # --- Run pipeline stages-first ---

    results: dict[str, dict[str, dict[str, Path]]] = {}

    for stage_num in stage_nums:
        stage_name = PIPELINE_STAGE_NAMES[stage_num]

        if verbose:
            print()
            print(f"{'=' * 60}")
            print(f"Stage {stage_num}: {stage_name}")
            print(f"{'=' * 60}")
            print()

        # Assemble per-stage config for all inputs
        stage_config = _assemble_stage_config(
            config,
            stage_num,
            prev_output,
            project_root,
        )

        # Run the stage on all inputs at once
        stage_output = PIPELINE_STAGE_FUNCTIONS[stage_num](
            stage_config,
            project_root,
        )

        # Store results and wire output to next stage
        results[stage_name] = stage_output
        prev_output = stage_output

    # --- Save assembled per-stage configs ---

    if save_stage_configs:
        _save_stage_configs(config, results, project_root)

    # --- Completion ---

    if verbose:
        print(
            f"Pipeline complete. Outputs saved for "
            f"{len(stage_nums)} stage"
            f"{'s' if len(stage_nums) != 1 else ''}."
        )

    return results
