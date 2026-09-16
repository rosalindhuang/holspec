"""
Caller-facing pipeline helpers.

Convenience functions for selecting pipeline input/output filepaths via
glob filters, and for splitting a unified pipeline config into per-stage
config dicts.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

from .stages import PIPELINE_STAGE_NAMES
from .orchestration import _assemble_stage_config


def select_pipeline_inputs(
    project_root: str | Path,
    select_point_data: list[str] | None = None,
    select_datasets: list[str] | None = None,
    raw_data_dir: str | Path | None = None,
) -> dict[str, Path]:
    """
    Select raw point data files by dataset and glob filters.

    Scans a raw data directory for HDF5 files matching
    the selection criteria. This is the entry point data for the pipeline
    (stage 0 outputs / stage 1 inputs).

    Parameters
    ----------
    project_root : str or Path
        Project root directory.
    select_datasets : list of str, optional
        Glob patterns for raw point data dataset names (e.g.
        ``['test_examples', 'exp_noise_trilatt_nr*']``). Datasets correspond
        to subdirectory names under ``raw_data_dir``. If None or empty, all
        datasets are included. Exact names are valid patterns.
    select_point_data : list of str, optional
        Glob patterns for point data labels (e.g. ``['trilatthex*',
        'randunif*']``). If None or empty, all point data labels are included.
    raw_data_dir : str or Path, optional
        Raw point data directory to scan. Relative paths are resolved from
        ``project_root``. Defaults to ``data/raw``.

    Returns
    -------
    dict[str, Path]
        Dictionary ``{ptd_label: filepath}`` mapping point data labels
        (file stems) to their absolute file paths.
    """
    project_root = Path(project_root)
    raw_data_dir = _resolve_raw_data_dir(project_root, raw_data_dir)

    filepaths: dict[str, Path] = {}

    if not raw_data_dir.is_dir():
        return filepaths

    for dataset_dir in sorted(raw_data_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        # Filter by dataset
        if select_datasets and not any(
            fnmatch(dataset_dir.name, pat) for pat in select_datasets
        ):
            continue

        for filepath in sorted(dataset_dir.glob("*.h5")):
            ptd_label = filepath.stem

            # Filter by point data label
            if select_point_data and not any(
                fnmatch(ptd_label, g) for g in select_point_data
            ):
                continue

            filepaths[ptd_label] = filepath

    return filepaths


def select_stage_outputs(
    stage_num: int,
    project_root: str | Path,
    base_dir: str | Path | None = None,
    select_point_data: list[str] | None = None,
    select_simplicial_complex: list[str] | None = None,
    select_cochain_metric: list[str] | None = None,
    select_datasets: list[str] | None = None,
    raw_data_dir: str | Path | None = None,
) -> dict[str, dict[str, Path]]:
    """
    Select pipeline stage output files by dataset and glob filters.

    Scans the output directory for the given stage and returns file paths
    matching the selection criteria. Supports stages 1--4.

    Parameters
    ----------
    stage_num : int
        Pipeline stage number (1--4).
    project_root : str or Path
        Project root directory.
    base_dir : str or Path, optional
        Base directory for stage outputs, relative to ``project_root``.
        When provided, scans ``project_root / base_dir / stage_name``
        instead of the default ``data/interim/stage_name``. Use this to
        select outputs from a specific per-dataset pipeline run (e.g.
        ``'data/interim/exp_noise_trilatt'``).
    select_datasets : list of str, optional
        Glob patterns for raw point data dataset names. Datasets are derived
        from subdirectory names under ``raw_data_dir``. If None or empty, all
        datasets are included. Not needed when ``base_dir`` already isolates
        outputs by dataset. Exact names are valid patterns.
    select_point_data : list of str, optional
        Glob patterns for point data labels (e.g. ``['trilatthex*',
        'randunif*']``). If None or empty, all point data labels are included.
    select_simplicial_complex : list of str, optional
        Glob patterns for simplicial complex labels (e.g. ``['delaunay*']``).
        Applies to stages 1--4. If None or empty, all are included.
    select_cochain_metric : list of str, optional
        Glob patterns for cochain metric labels (e.g. ``['combinatorial*']``).
        Applies to stages 2--4 (ignored for stage 1). If None or empty, all
        are included.
    raw_data_dir : str or Path, optional
        Raw point data directory used only for dataset filtering. Relative
        paths are resolved from ``project_root``. Defaults to ``data/raw``.

    Returns
    -------
    dict[str, dict[str, Path]]
        Nested dictionary ``{ptd_label: {output_label: filepath}}``, where
        ``ptd_label`` is the point data subdirectory name and
        ``output_label`` is the HDF5 file stem.
    """
    if stage_num not in PIPELINE_STAGE_NAMES or stage_num == 0:
        raise ValueError(f"stage_num must be 1--4, got {stage_num}")

    project_root = Path(project_root)
    stage_name = PIPELINE_STAGE_NAMES[stage_num]
    if base_dir is not None:
        output_data_dir = project_root / base_dir / stage_name
    else:
        output_data_dir = project_root / "data" / "interim" / stage_name
    raw_data_dir = _resolve_raw_data_dir(project_root, raw_data_dir)

    if not output_data_dir.is_dir():
        return {}

    # Build dataset mapping: point_data_label -> set of datasets
    # (a label can appear in multiple raw datasets)
    ptd_datasets: dict[str, set[str]] = {}
    if select_datasets and raw_data_dir.is_dir():
        for dataset_dir in sorted(raw_data_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for raw_file in dataset_dir.glob("*.h5"):
                ptd_datasets.setdefault(raw_file.stem, set()).add(dataset_dir.name)

    # Whether filenames have the {sc}__{cm} format (stage 2+)
    has_metric_suffix = stage_num >= 2

    filepaths: dict[str, dict[str, Path]] = {}

    for ptd_subdir in sorted(output_data_dir.iterdir()):
        if not ptd_subdir.is_dir():
            continue
        ptd_label = ptd_subdir.name

        # Filter by dataset
        if select_datasets:
            ptd_dataset_set = ptd_datasets.get(ptd_label, set())
            if not any(
                fnmatch(dataset, pat)
                for dataset in ptd_dataset_set
                for pat in select_datasets
            ):
                continue

        # Filter by point data label
        if select_point_data and not any(
            fnmatch(ptd_label, g) for g in select_point_data
        ):
            continue

        # Collect matching output files
        ptd_filepaths: dict[str, Path] = {}
        for filepath in sorted(ptd_subdir.glob("*.h5")):
            stem = filepath.stem

            if has_metric_suffix:
                parts = stem.split("__", 1)
                sc_label = parts[0]
                cm_label = parts[1] if len(parts) == 2 else ""
            else:
                sc_label = stem
                cm_label = None

            # Filter by simplicial complex label
            if select_simplicial_complex and not any(
                fnmatch(sc_label, g) for g in select_simplicial_complex
            ):
                continue

            # Filter by cochain metric label (stages 2+ only)
            if (
                has_metric_suffix
                and select_cochain_metric
                and not any(fnmatch(cm_label, g) for g in select_cochain_metric)
            ):
                continue

            ptd_filepaths[stem] = filepath

        if ptd_filepaths:
            filepaths[ptd_label] = ptd_filepaths

    return filepaths


def split_pipeline_config(
    pipeline_config: dict,
    project_root: str | Path,
) -> dict[int, dict]:
    """
    Split a unified pipeline config into per-stage config dicts.

    Assembles standalone per-stage configs from the unified pipeline
    config without writing any files. Stage 1 receives the pipeline's
    input filepaths; Stages 2--4 receive empty filepaths (to be wired
    at runtime from the previous stage's output).

    Parameters
    ----------
    pipeline_config : dict
        Full pipeline config dict (as loaded from ``pipeline.yml``).
    project_root : str or Path
        Project root for path resolution.

    Returns
    -------
    dict[int, dict]
        Mapping ``{stage_num: stage_config_dict}`` for stages 1--4.
        Each value has the standard five-key per-stage schema
        (``summary``, ``inputs``, ``configs``, ``runtime``, ``outputs``).
    """
    project_root = Path(project_root)
    stage_configs = {}

    for stage_num in (1, 2, 3, 4):
        if stage_num == 1:
            input_filepaths = pipeline_config["inputs"]["filepaths"]
        else:
            input_filepaths = {}

        stage_configs[stage_num] = _assemble_stage_config(
            pipeline_config,
            stage_num,
            input_filepaths,
            project_root,
        )

    return stage_configs


# Private Helpers


def _resolve_raw_data_dir(
    project_root: Path,
    raw_data_dir: str | Path | None,
) -> Path:
    """Resolve a raw point data directory relative to the project root."""
    if raw_data_dir is None:
        return project_root / "data" / "raw"
    raw_data_dir = Path(raw_data_dir)
    if raw_data_dir.is_absolute():
        return raw_data_dir
    return project_root / raw_data_dir
