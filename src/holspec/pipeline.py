"""
Pipeline orchestration for holspec.

Provides stage metadata, provenance tracing, loading functions for 
objects that require live references to upstream objects.

Provides pipeline stage functions ``run_{stage}`` that formalize the
pipeline computations into callable functions.
"""
from __future__ import annotations

from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
import warnings

import numpy as np
import yaml

from holspec.utilities import (
    read_h5, save_h5, convert_relative_to_paths, convert_paths_to_relative,
    get_keys_h5,
)
from holspec.point_data import PointDataEnsemble
from holspec.simplicial import SimplicialComplex
from holspec.cochain_metric import CochainMetric
from holspec.hodge_laplacian import HodgeLaplacian, LAPLACIAN_COMPONENT_NAMES
from holspec.spectra import HodgeLaplacianSpectra


# =============================================================================
# Pipeline Metadata
# =============================================================================

PIPELINE_STAGE_NAMES: dict[int, str] = {
    0: 'point_data',
    1: 'topology_simplicial',
    2: 'geometry_metric',
    3: 'hodge_laplacian',
    4: 'spectra',
}


# =============================================================================
# Pipeline Provenance
# =============================================================================

def trace_provenance(
    filepath: str | Path,
    project_root: str | Path,
) -> dict[str, Path]:
    """
    Trace the provenance chain of a pipeline output file.

    Follows ``input_file`` root-level HDF5 attributes backward through
    the chain of pipeline output files. Each file in the chain must have
    a ``stage_name`` root-level attribute (part of the uniform file-level
    metadata schema). Tracing stops when a file lacks ``stage_name``
    or ``input_file``.

    Parameters
    ----------
    filepath : str or Path
        Starting file (e.g., a HodgeLaplacian or CochainMetric file).
    project_root : str or Path
        Project root directory for resolving the relative paths stored
        in ``input_file`` attributes.

    Returns
    -------
    dict[str, Path]
        Mapping from stage name to resolved file path, ordered from the
        starting file backward through the chain. Includes the starting
        file itself.

    Raises
    ------
    FileNotFoundError
        If any file in the chain does not exist.
    ValueError
        If a stage name appears more than once (malformed provenance).

    Examples
    --------
    >>> chain = trace_provenance(hl_filepath, PROJECT_ROOT)
    >>> chain
    {'hodge_laplacian': PosixPath('.../hodge_laplacian/...h5'),
     'geometry_metric': PosixPath('.../geometry_metric/...h5'),
     'topology_simplicial': PosixPath('.../topology_simplicial/...h5'),
     'point_data': PosixPath('.../point_data/...h5')}
    """
    filepath = Path(filepath)
    project_root = Path(project_root)

    # Resolve relative paths against project_root, not cwd
    if not filepath.is_absolute():
        filepath = project_root / filepath

    provenance_chain: dict[str, Path] = {}
    current_filepath = filepath

    while True:
        # Read root-level attributes only (no datasets)
        _, root_attributes = read_h5(current_filepath, dataset_names=[])

        stage_name = root_attributes.get('stage_name')
        if stage_name is None:
            break

        # Guard against malformed provenance (cycle or misconfiguration)
        if stage_name in provenance_chain:
            raise ValueError(
                f"Duplicate stage name '{stage_name}' in provenance chain. "
                f"First: {provenance_chain[stage_name]}, "
                f"second: {current_filepath}"
            )

        provenance_chain[stage_name] = current_filepath

        # Follow the input_file link to the next file in the chain
        input_file_relative = root_attributes.get('input_file')
        if input_file_relative is None:
            break

        current_filepath = project_root / input_file_relative

    return provenance_chain


# =============================================================================
# Pipeline Stage Functions
# =============================================================================

def _initialize_pipeline_file(
    output_filepath: Path,
    created_by: str,
    input_filepath: Path,
    project_root: Path,
    stage_name: str,
    stage_config: dict,
) -> None:
    """
    Write root-level pipeline metadata to a new output file.

    Creates (or replaces) the output HDF5 file with the five standard
    file-level metadata attributes that every pipeline output carries.
    Called once per output file at the start of each ``run_{stage}``
    function, before any data is written.

    Parameters
    ----------
    output_filepath : Path
        Path to the output HDF5 file.
    created_by : str
        Identifier for the notebook or script that produced this file.
    input_filepath : Path
        Path to the immediate input file for this stage.
    project_root : Path
        Project root for computing the relative input path.
    stage_name : str
        Name of the pipeline stage producing this file.
    stage_config : dict
        Stage-specific configuration recorded for provenance. Empty dict
        for stages with no mathematical parameters.
    """
    save_h5(
        output_filepath,
        attributes={
            'created_by': created_by,
            'creation_time': datetime.now().isoformat(),
            'input_file': str(input_filepath.relative_to(project_root)),
            'stage_name': stage_name,
            'stage_config': stage_config,
        },
        group=None,
        mode='replace',
    )


def _build_failure_record(
    member_key: str,
    input_filepath: Path,
    output_filepath: Path,
    exc: Exception,
    project_root: Path,
) -> dict:
    """
    Build a per-member failure record dict.

    Centralizes the failure record schema used by all four pipeline
    stage functions. Each record corresponds to one entry in the
    ``exceptions`` list of the YAML error report.

    Parameters
    ----------
    member_key : str
        HDF5 group name of the failed member (e.g. ``'member_0003'``).
    input_filepath : Path
        Absolute path to the input file for this stage.
    output_filepath : Path
        Absolute path to the output file being written.
    exc : Exception
        The caught exception.
    project_root : Path
        Project root for computing relative paths.

    Returns
    -------
    dict
        Failure record with keys: ``member_key``, ``input_file``,
        ``output_file``, ``exception_type``, ``exception_message``,
        ``timestamp``.
    """
    return {
        'member_key': member_key,
        'input_file': str(input_filepath.relative_to(project_root)),
        'output_file': str(output_filepath.relative_to(project_root)),
        'exception_type': type(exc).__name__,
        'exception_message': str(exc),
        'timestamp': datetime.now().isoformat(),
    }


def _save_error_report(
    failures: list[dict],
    num_succeeded: int,
    num_output_files_deleted: int,
    num_output_files_retained: int,
    output_data_dir: Path,
    stage_name: str,
    created_by: str,
    project_root: Path,
    error_handling: str,
    output_retention: str,
    save_error_report: bool,
) -> Path:
    """
    Write a YAML error report for a pipeline stage run.

    Called at the end of a ``run_{stage}`` function when at least one
    member failed and ``save_error_report`` is True. The report is
    written to ``data/reports/`` (derived from the stage output
    directory) with a timestamped filename.

    Parameters
    ----------
    failures : list[dict]
        List of failure records from ``_build_failure_record``.
    num_succeeded : int
        Total members that succeeded across all config combinations.
    num_output_files_deleted : int
        Number of output files deleted by the retention policy.
    num_output_files_retained : int
        Number of output files retained.
    output_data_dir : Path
        Stage output data directory (e.g. ``data/interim/spectra``).
    stage_name : str
        Pipeline stage name.
    created_by : str
        Identifier for the notebook or script that ran the stage.
    project_root : Path
        Project root for computing relative paths.
    error_handling : str
        The ``error_handling`` runtime setting used for this run.
    output_retention : str
        The ``output_retention`` runtime setting used for this run.
    save_error_report : bool
        The ``save_error_report`` runtime setting used for this run.

    Returns
    -------
    Path
        Absolute path to the written report file.
    """
    reports_dir = output_data_dir.parent.parent / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat().replace(':', '-')
    report_filename = f"{timestamp}__{stage_name}__report.yml"
    report_path = reports_dir / report_filename

    report = {
        'summary': {
            'created_by': created_by,
            'creation_time': datetime.now().isoformat(),
            'stage_name': stage_name,
        },
        'runtime': {
            'error_handling': error_handling,
            'output_retention': output_retention,
            'save_error_report': save_error_report,
        },
        'results': {
            'num_failed': len(failures),
            'num_succeeded': num_succeeded,
            'num_output_files_deleted': num_output_files_deleted,
            'num_output_files_retained': num_output_files_retained,
        },
        'exceptions': failures,
    }

    with open(report_path, 'w') as f:
        yaml.safe_dump(
            report, f,
            default_flow_style=False, sort_keys=False,
        )

    print(
        f"Error report saved: "
        f"{report_path.relative_to(project_root)}"
    )

    return report_path


def run_topology_simplicial(
    config: dict,
    project_root: str | Path,
) -> dict[str, dict[str, Path]]:
    """
    Run pipeline Stage 1: Topological Structure via Simplicial Complexes.
 
    Constructs a SimplicialComplex for each (point data ensemble, simplicial
    construction) pair, and writes the results to HDF5 files.
 
    Which incidence matrices are cached is controlled by ``cache_incidence``
    in the config runtime: True caches all degrees, a list caches only the
    specified degrees, and False skips caching. Validation independently
    populates the cache for the degrees it checks.
 
    Parameters
    ----------
    config : dict
        Stage config dictionary.
    project_root : str or Path
        Project root for resolving relative paths in the config.
 
    Returns
    -------
    dict[str, dict[str, Path]]
        Nested mapping ``{ptd_label: {sc_label: output_filepath}}``.
        Same shape as ``inputs.filepaths`` in the next stage's config.
 
    Raises
    ------
    FileNotFoundError
        If any input file does not exist.
    ValueError
        If boundary property validation fails (when enabled) and
        ``error_handling`` is ``'raise'``. Under ``'skip'``, per-member
        failures are caught, recorded, and optionally saved as a YAML
        error report.
    """
    project_root = Path(project_root)

    input_filepaths = convert_relative_to_paths(
        config['inputs']['filepaths'], project_root,
    )
    output_data_dir = project_root / config['outputs']['data_dir']
    created_by = config['summary']['created_by']
    verbose = config['runtime']['verbose']
    stage_name = config['outputs']['stage_name']

    simplicial_constructions = config['configs']['simplicial_constructions']
    cache_incidence = config['runtime']['cache_incidence']
    validate_boundary_property = config['runtime']['validate_boundary_property']
    error_handling = config['runtime'].get('error_handling', 'skip')
    output_retention = config['runtime'].get('output_retention', 'partial')
    save_error_report = config['runtime'].get('save_error_report', True)

    failures: list[dict] = []
    total_succeeded = 0
    total_output_files_deleted = 0
    total_output_files_retained = 0

    # --- Loop over inputs ---
    output_filepaths: dict[str, dict[str, Path]] = {}

    for ptd_label, ptd_filepath in input_filepaths.items():

        # Load ensemble once per point data label
        point_data_ensemble = PointDataEnsemble.load(ptd_filepath)

        # Discover ensemble members from input file
        member_keys = [
            key for key in get_keys_h5(ptd_filepath)
            if key.startswith('member_')
        ]

        for sc_label, sc_config in simplicial_constructions.items():

            # Output file path
            output_filepath = output_data_dir / ptd_label / f"{sc_label}.h5"

            # Initialize file with pipeline metadata
            _initialize_pipeline_file(
                output_filepath, created_by, ptd_filepath,
                project_root, stage_name, sc_config,
            )

            num_succeeded = 0
            num_failed = 0

            # Iterate computation over ensemble members
            for member_key in member_keys:

                try:
                    member_index = int(member_key.split('_')[1])
                    ptd = point_data_ensemble[member_index]

                    # Construct simplicial complex
                    sc = SimplicialComplex.from_point_data(
                        ptd, sc_config,
                        metadata={'member_index': member_index},
                    )

                    # Cache incidence matrices for requested degrees
                    if cache_incidence is True:
                        for k in range(sc.max_dim + 1):
                            sc.incidence_matrix(k)
                    elif isinstance(cache_incidence, list):
                        for k in cache_incidence:
                            sc.incidence_matrix(k)
                    # cache_incidence is False: no caching

                    # Validate boundary property (also populates cache)
                    if validate_boundary_property:
                        sc.validate_boundary_property()

                    # Save simplicial complex to file
                    sc.save(
                        output_filepath,
                        save_incidence=True,
                        mode='replace',
                        group=member_key,
                    )

                    num_succeeded += 1

                except Exception as exc:
                    num_failed += 1
                    failure_record = _build_failure_record(
                        member_key, ptd_filepath, output_filepath,
                        exc, project_root,
                    )
                    failures.append(failure_record)

                    if num_failed == 1: print()
                    print(f"[SKIP] {member_key} | {type(exc).__name__}: {exc}")
                    print(f"    input_file:  {failure_record['input_file']}")
                    print(f"    output_file: {failure_record['output_file']}")
                    print(f"    timestamp:   {failure_record['timestamp']}")

                    if error_handling == 'raise':
                        raise

            # Accumulate stage-level counts
            total_succeeded += num_succeeded

            # Completion
            skip_suffix = (
                f" ({num_failed} skipped)" if num_failed > 0 else ""
            )
            print(
                f"Constructed {num_succeeded} simplicial "
                f"complexes for {ptd_label} / {sc_label}{skip_suffix}"
            )
            if verbose and num_succeeded > 0:
                print(f"  {sc}")
                print(f"  simplicial construction: {sc_label}")
                print(f"  incidence matrices:")
                for k in range(sc.max_dim + 1):
                    if k in sc._incidence_cache:
                        D_k = sc._incidence_cache[k]
                        print(f"    D_{k}: shape={D_k.shape}, nnz={D_k.nnz}")
                    else:
                        print(f"    D_{k}: (not cached)")
                print(f"  file path: "
                      f"{output_filepath.relative_to(project_root)}")
                print(f"  file size: "
                      f"{output_filepath.stat().st_size / 1024:.2f} KB")
                print()
            if num_succeeded == 0: print()

            # Output retention
            file_deleted = False
            if num_failed > 0:
                delete_file = False
                if num_succeeded == 0 and output_retention in (
                    'partial', 'complete',
                ):
                    delete_file = True
                elif num_succeeded > 0 and output_retention == 'complete':
                    delete_file = True

                if delete_file:
                    output_filepath.unlink(missing_ok=True)
                    file_deleted = True
                    total_output_files_deleted += 1

            if not file_deleted:
                output_filepaths.setdefault(ptd_label, {})[sc_label] = (
                    output_filepath
                )
                total_output_files_retained += 1

    if save_error_report and failures:
        _save_error_report(
            failures=failures,
            num_succeeded=total_succeeded,
            num_output_files_deleted=total_output_files_deleted,
            num_output_files_retained=total_output_files_retained,
            output_data_dir=output_data_dir,
            stage_name=stage_name,
            created_by=created_by,
            project_root=project_root,
            error_handling=error_handling,
            output_retention=output_retention,
            save_error_report=save_error_report,
        )

    return output_filepaths


def run_geometry_metric(
    config: dict,
    project_root: str | Path,
) -> dict[str, dict[str, Path]]:
    """
    Run pipeline Stage 2: Geometric Structure via Discrete Cochain Metrics.

    Constructs a CochainMetric for each (simplicial complex, metric model)
    pair, and writes the results to HDF5 files. Upstream point data is
    resolved via provenance tracing.

    Parameters
    ----------
    config : dict
        Stage config dictionary.
    project_root : str or Path
        Project root for resolving relative paths in the config.

    Returns
    -------
    dict[str, dict[str, Path]]
        Nested mapping
        ``{ptd_label: {sc_label__cm_label: output_filepath}}``.
        Same shape as ``inputs.filepaths`` in the next stage's config.

    Raises
    ------
    FileNotFoundError
        If any input file does not exist.
    ValueError
        If metric construction or validation fails (when enabled) and
        ``error_handling`` is ``'raise'``. Under ``'skip'``, per-member
        failures are caught, recorded, and optionally saved as a YAML
        error report.
    """
    project_root = Path(project_root)

    input_filepaths = convert_relative_to_paths(
        config['inputs']['filepaths'], project_root,
    )
    output_data_dir = project_root / config['outputs']['data_dir']
    created_by = config['summary']['created_by']
    verbose = config['runtime']['verbose']
    stage_name = config['outputs']['stage_name']

    metric_models = config['configs']['metric_models']
    validate_metric = config['runtime']['validate_metric']
    error_handling = config['runtime'].get('error_handling', 'skip')
    output_retention = config['runtime'].get('output_retention', 'partial')
    save_error_report = config['runtime'].get('save_error_report', True)

    failures: list[dict] = []
    total_succeeded = 0
    total_output_files_deleted = 0
    total_output_files_retained = 0

    # --- Loop over inputs ---
    output_filepaths: dict[str, dict[str, Path]] = {}

    for ptd_label, sc_files in input_filepaths.items():

        for sc_label, sc_filepath in sc_files.items():

            # Trace provenance chain for file paths
            provenance_chain = trace_provenance(sc_filepath, project_root)
            ptd_filepath = provenance_chain['point_data']

            # Load point data from file
            point_data_ensemble = PointDataEnsemble.load(ptd_filepath)

            # Discover ensemble members from input file
            member_keys = [
                key for key in get_keys_h5(sc_filepath)
                if key.startswith('member_')
            ]

            for cm_label, cm_config in metric_models.items():

                # Output file path
                output_label = f"{sc_label}__{cm_label}"
                output_filepath = (
                    output_data_dir / ptd_label / f"{output_label}.h5"
                )

                # Initialize file with pipeline metadata
                # (input_file points to SC file, the immediate predecessor)
                _initialize_pipeline_file(
                    output_filepath, created_by, sc_filepath,
                    project_root, stage_name, cm_config,
                )

                num_succeeded = 0
                num_failed = 0

                # Iterate computation over ensemble members
                for member_key in member_keys:

                    try:
                        member_index = int(member_key.split('_')[1])
                        ptd = point_data_ensemble[member_index]

                        # Load simplicial complex for this member
                        sc = SimplicialComplex.load(
                            sc_filepath, group=member_key,
                            load_incidence=True,
                        )

                        # Construct cochain metric
                        cm = CochainMetric.from_simplicial_complex_and_point_data(
                            sc, ptd, cm_config,
                            metadata={'member_index': member_index},
                        )

                        # Validate metric dimensions against simplicial complex
                        if validate_metric:
                            cm.validate(sc.num_simplices)

                        # Save cochain metric to file
                        cm.save(
                            output_filepath,
                            mode='replace',
                            group=member_key,
                        )

                        num_succeeded += 1

                    except Exception as exc:
                        num_failed += 1
                        failure_record = _build_failure_record(
                            member_key, sc_filepath, output_filepath,
                            exc, project_root,
                        )
                        failures.append(failure_record)

                        if num_failed == 1: print()
                        print(f"[SKIP] {member_key} | {type(exc).__name__}: {exc}")
                        print(f"    input_file:  {failure_record['input_file']}")
                        print(f"    output_file: {failure_record['output_file']}")
                        print(f"    timestamp:   {failure_record['timestamp']}")

                        if error_handling == 'raise':
                            raise

                # Accumulate stage-level counts
                total_succeeded += num_succeeded

                # Completion
                skip_suffix = (
                    f" ({num_failed} skipped)" if num_failed > 0 else ""
                )
                print(
                    f"Constructed {num_succeeded} cochain "
                    f"metrics for {ptd_label} / {output_label}{skip_suffix}"
                )
                if verbose and num_succeeded > 0:
                    print(f"  {cm}")
                    print(f"  cochain metric model: {cm_label}")
                    print(f"  metric tensors:")
                    for k in range(cm.max_dim + 1):
                        G_k = cm[k]
                        print(
                            f"    G_{k}: size={G_k.size}, "
                            f"is_diagonal={G_k.is_diagonal}"
                        )
                    print(f"  file path: "
                          f"{output_filepath.relative_to(project_root)}")
                    print(f"  file size: "
                          f"{output_filepath.stat().st_size / 1024:.2f} KB")
                    print()
                if num_succeeded == 0: print()

                # Output retention
                file_deleted = False
                if num_failed > 0:
                    delete_file = False
                    if num_succeeded == 0 and output_retention in (
                        'partial', 'complete',
                    ):
                        delete_file = True
                    elif (
                        num_succeeded > 0
                        and output_retention == 'complete'
                    ):
                        delete_file = True

                    if delete_file:
                        output_filepath.unlink(missing_ok=True)
                        file_deleted = True
                        total_output_files_deleted += 1

                if not file_deleted:
                    output_filepaths.setdefault(
                        ptd_label, {},
                    )[output_label] = output_filepath
                    total_output_files_retained += 1

    if save_error_report and failures:
        _save_error_report(
            failures=failures,
            num_succeeded=total_succeeded,
            num_output_files_deleted=total_output_files_deleted,
            num_output_files_retained=total_output_files_retained,
            output_data_dir=output_data_dir,
            stage_name=stage_name,
            created_by=created_by,
            project_root=project_root,
            error_handling=error_handling,
            output_retention=output_retention,
            save_error_report=save_error_report,
        )

    return output_filepaths


def run_hodge_laplacian(
    config: dict,
    project_root: str | Path,
) -> dict[str, dict[str, Path]]:
    """
    Run pipeline Stage 3: Hodge Laplacians and Discrete Differential Operators.
 
    Constructs a HodgeLaplacian for each cochain metric file, and writes
    the results to HDF5 files. Upstream SimplicialComplex files are
    resolved via provenance tracing.
 
    Which Laplacian matrices are cached is controlled by
    ``cache_laplacians`` in the config runtime: True caches all
    (k, component) pairs, a list caches only the specified pairs, and
    False skips caching. Validation independently populates the cache
    for the pairs it checks.
 
    When neither ``cache_laplacians`` nor ``validate_laplacians``
    triggers computation, no Laplacian matrices are computed. The output
    file serves as a provenance waypoint containing only metadata and
    content hashes.
 
    Parameters
    ----------
    config : dict
        Stage config dictionary.
    project_root : str or Path
        Project root for resolving relative paths in the config.
 
    Returns
    -------
    dict[str, dict[str, Path]]
        Nested mapping ``{ptd_label: {cm_label: output_filepath}}``.
        Same shape as ``inputs.filepaths`` in the next stage's config.
 
    Raises
    ------
    FileNotFoundError
        If any input file does not exist.
    ValueError
        If Laplacian property validation fails (when enabled) and
        ``error_handling`` is ``'raise'``. Under ``'skip'``, per-member
        failures are caught, recorded, and optionally saved as a YAML
        error report.
    """
    project_root = Path(project_root)

    input_filepaths = convert_relative_to_paths(
        config['inputs']['filepaths'], project_root,
    )
    output_data_dir = project_root / config['outputs']['data_dir']
    created_by = config['summary']['created_by']
    verbose = config['runtime']['verbose']
    stage_name = config['outputs']['stage_name']

    cache_laplacians = config['runtime']['cache_laplacians']
    validate_laplacians = config['runtime']['validate_laplacians']
    error_handling = config['runtime'].get('error_handling', 'skip')
    output_retention = config['runtime'].get('output_retention', 'partial')
    save_error_report = config['runtime'].get('save_error_report', True)

    failures: list[dict] = []
    total_succeeded = 0
    total_output_files_deleted = 0
    total_output_files_retained = 0

    # --- Loop over inputs ---
    output_filepaths: dict[str, dict[str, Path]] = {}

    for ptd_label, cm_files in input_filepaths.items():

        for cm_label, cm_filepath in cm_files.items():

            # Trace provenance chain for file paths
            provenance_chain = trace_provenance(cm_filepath, project_root)
            sc_filepath = provenance_chain['topology_simplicial']

            # Output file path
            output_label = cm_label
            output_filepath = (
                output_data_dir / ptd_label / f"{output_label}.h5"
            )

            # Initialize file with pipeline metadata
            _initialize_pipeline_file(
                output_filepath, created_by, cm_filepath,
                project_root, stage_name, {},
            )

            # Discover ensemble members from input file
            member_keys = [
                key for key in get_keys_h5(cm_filepath)
                if key.startswith('member_')
            ]

            num_succeeded = 0
            num_failed = 0

            # Iterate computation over ensemble members
            for member_key in member_keys:

                try:
                    # Load upstream objects for this member
                    sc = SimplicialComplex.load(
                        sc_filepath, group=member_key,
                        load_incidence=True,
                    )
                    cm = CochainMetric.load(
                        cm_filepath, group=member_key,
                    )

                    # Construct Hodge Laplacian
                    hl = HodgeLaplacian(sc, cm)

                    # Cache Laplacian matrices for requested
                    # (k, component) pairs
                    if cache_laplacians is True:
                        for k in range(hl.max_dim + 1):
                            _ = hl[k]
                    elif isinstance(cache_laplacians, list):
                        for k, comp in cache_laplacians:
                            hl.to_matrix(k, comp)
                    # cache_laplacians is False: no caching

                    # Validate Laplacian properties (also populates cache)
                    if validate_laplacians:
                        hl.validate_laplacians()

                    # Save Hodge Laplacian to file
                    hl.save(
                        output_filepath,
                        save_laplacians=True,
                        mode='replace',
                        group=member_key,
                    )

                    num_succeeded += 1

                except Exception as exc:
                    num_failed += 1
                    failure_record = _build_failure_record(
                        member_key, cm_filepath, output_filepath,
                        exc, project_root,
                    )
                    failures.append(failure_record)

                    if num_failed == 1: print()
                    print(f"[SKIP] {member_key} | {type(exc).__name__}: {exc}")
                    print(f"    input_file:  {failure_record['input_file']}")
                    print(f"    output_file: {failure_record['output_file']}")
                    print(f"    timestamp:   {failure_record['timestamp']}")

                    if error_handling == 'raise':
                        raise

            # Accumulate stage-level counts
            total_succeeded += num_succeeded

            # Completion
            skip_suffix = (
                f" ({num_failed} skipped)" if num_failed > 0 else ""
            )
            print(
                f"Constructed {num_succeeded} Hodge Laplacians "
                f"for {ptd_label} / {cm_label}{skip_suffix}"
            )
            if verbose and num_succeeded > 0:
                print(f"  {hl}")
                print(f"  hodge laplacian matrices:")
                for k in range(hl.max_dim + 1):
                    for comp in LAPLACIAN_COMPONENT_NAMES:
                        if (k, comp) in hl._laplacian_cache:
                            L = hl._laplacian_cache[(k, comp)]
                            print(
                                f"    L^{k},{comp}: "
                                f"shape={L.shape}, nnz={L.nnz}"
                            )
                        else:
                            print(f"    L^{k},{comp}: (not cached)")
                print(f"  file path: "
                      f"{output_filepath.relative_to(project_root)}")
                print(f"  file size: "
                      f"{output_filepath.stat().st_size / 1024:.2f} KB")
                print()
            if num_succeeded == 0: print()

            # Output retention
            file_deleted = False
            if num_failed > 0:
                delete_file = False
                if num_succeeded == 0 and output_retention in (
                    'partial', 'complete',
                ):
                    delete_file = True
                elif num_succeeded > 0 and output_retention == 'complete':
                    delete_file = True

                if delete_file:
                    output_filepath.unlink(missing_ok=True)
                    file_deleted = True
                    total_output_files_deleted += 1

            if not file_deleted:
                output_filepaths.setdefault(ptd_label, {})[output_label] = (
                    output_filepath
                )
                total_output_files_retained += 1

    if save_error_report and failures:
        _save_error_report(
            failures=failures,
            num_succeeded=total_succeeded,
            num_output_files_deleted=total_output_files_deleted,
            num_output_files_retained=total_output_files_retained,
            output_data_dir=output_data_dir,
            stage_name=stage_name,
            created_by=created_by,
            project_root=project_root,
            error_handling=error_handling,
            output_retention=output_retention,
            save_error_report=save_error_report,
        )

    return output_filepaths


def run_spectra(
    config: dict,
    project_root: str | Path,
) -> dict[str, dict[str, Path]]:
    """
    Run pipeline Stage 4: Hodge Laplacian Spectra and Spectral Observables.

    Computes eigendecompositions for each HodgeLaplacian file, and writes
    the results to HDF5 files. Upstream objects are reconstructed via
    ``load_hodge_laplacian`` per member.

    Which spectra are computed is controlled by ``compute_spectra`` in
    the config: True computes all (k, component) pairs, a list computes
    only the specified pairs, and False skips computation. Eigenvector
    computation is further controlled by ``compute_eigenvectors``.

    Parameters
    ----------
    config : dict
        Stage config dictionary.
    project_root : str or Path
        Project root for resolving relative paths in the config.

    Returns
    -------
    dict[str, dict[str, Path]]
        Nested mapping ``{ptd_label: {hl_label: output_filepath}}``.

    Raises
    ------
    FileNotFoundError
        If any input file does not exist.
    """
    project_root = Path(project_root)

    input_filepaths = convert_relative_to_paths(
        config['inputs']['filepaths'], project_root,
    )
    output_data_dir = project_root / config['outputs']['data_dir']
    created_by = config['summary']['created_by']
    verbose = config['runtime']['verbose']
    stage_name = config['outputs']['stage_name']

    compute_spectra = config['configs'].get('compute_spectra', True)
    solver = config['configs']['solver']
    solver_params = config['configs'].get('solver_params', None)
    compute_eigenvectors = config['configs']['compute_eigenvectors']
    error_handling = config['runtime'].get('error_handling', 'skip')
    output_retention = config['runtime'].get('output_retention', 'partial')
    save_error_report = config['runtime'].get('save_error_report', True)

    # Warn if compute_eigenvectors requests pairs outside compute_spectra
    if (isinstance(compute_spectra, list)
            and isinstance(compute_eigenvectors, list)):
        spectra_keys = {(k, comp) for k, comp in compute_spectra}
        eigenvector_keys = {(k, comp) for k, comp in compute_eigenvectors}
        unreachable = eigenvector_keys - spectra_keys
        if unreachable:
            warnings.warn(
                f"compute_eigenvectors contains pairs not in "
                f"compute_spectra: {sorted(unreachable)}. "
                f"Eigenvectors for these pairs will not be computed.",
                stacklevel=2,
            )

    failures: list[dict] = []
    total_succeeded = 0
    total_output_files_deleted = 0
    total_output_files_retained = 0

    # --- Loop over inputs ---
    output_filepaths: dict[str, dict[str, Path]] = {}

    for ptd_label, hl_files in input_filepaths.items():

        for hl_label, hl_filepath in hl_files.items():

            # Output file path
            output_label = hl_label
            output_filepath = (
                output_data_dir / ptd_label / f"{output_label}.h5"
            )

            # Initialize file with pipeline metadata
            _initialize_pipeline_file(
                output_filepath, created_by, hl_filepath,
                project_root, stage_name,
                {'compute_spectra': compute_spectra,
                 'solver': solver,
                 'solver_params': solver_params,
                 'compute_eigenvectors': compute_eigenvectors},
            )

            # Discover ensemble members from input file
            member_keys = [
                key for key in get_keys_h5(hl_filepath)
                if key.startswith('member_')
            ]

            num_succeeded = 0
            num_failed = 0

            # Iterate computation over ensemble members
            for member_key in member_keys:

                try:
                    # Load upstream HodgeLaplacian for this member
                    hl = load_hodge_laplacian(
                        hl_filepath, project_root, group=member_key,
                    )

                    # Construct HodgeLaplacianSpectra
                    hlsp = HodgeLaplacianSpectra(
                        hl,
                        solver=solver,
                        solver_params=solver_params,
                        compute_eigenvectors=compute_eigenvectors,
                    )

                    # Compute spectra for requested (k, component) pairs
                    if compute_spectra is True:
                        for k in range(hlsp.max_dim + 1):
                            _ = hlsp[k]
                    elif isinstance(compute_spectra, list):
                        for k, comp in compute_spectra:
                            hlsp.spectrum(k, comp)
                    # compute_spectra is False: no computation

                    # Save spectra to file
                    hlsp.save(
                        output_filepath,
                        save_eigenvectors=True,
                        mode='replace',
                        group=member_key,
                    )

                    num_succeeded += 1

                except Exception as exc:
                    num_failed += 1
                    failure_record = _build_failure_record(
                        member_key, hl_filepath, output_filepath,
                        exc, project_root,
                    )
                    failures.append(failure_record)

                    if num_failed == 1: print()
                    print(f"[SKIP] {member_key} | {type(exc).__name__}: {exc}")
                    print(f"    input_file:  {failure_record['input_file']}")
                    print(f"    output_file: {failure_record['output_file']}")
                    print(f"    timestamp:   {failure_record['timestamp']}")

                    if error_handling == 'raise':
                        raise

            # Accumulate stage-level counts
            total_succeeded += num_succeeded

            # Completion
            skip_suffix = (
                f" ({num_failed} skipped)" if num_failed > 0 else ""
            )
            print(
                f"Computed {num_succeeded} spectra "
                f"for {ptd_label} / {hl_label}{skip_suffix}"
            )
            if verbose and num_succeeded > 0:
                print(f"  {hlsp}")
                print(f"  compute_spectra: {compute_spectra}")
                print(f"  solver: {solver}")
                if solver_params:
                    print(f"  solver_params: {solver_params}")
                print(f"  compute_eigenvectors: {compute_eigenvectors}")
                print(f"  hodge laplacian spectra:")
                for k in hlsp.degrees:
                    for comp in LAPLACIAN_COMPONENT_NAMES:
                        if (k, comp) in hlsp._spectrum_cache:
                            spc = hlsp._spectrum_cache[(k, comp)]
                            print(
                                f"    L^{k},{comp}: "
                                f"num_eig={spc.num_eigenvalues}, "
                                f"dim_ker={spc.dim_ker()}"
                            )
                        else:
                            print(f"    L^{k},{comp}: (not computed)")
                print(f"  file path: "
                      f"{output_filepath.relative_to(project_root)}")
                print(f"  file size: "
                      f"{output_filepath.stat().st_size / 1024:.2f} KB")
                print()
            if num_succeeded == 0: print()

            # Output retention
            file_deleted = False
            if num_failed > 0:
                delete_file = False
                if num_succeeded == 0 and output_retention in (
                    'partial', 'complete',
                ):
                    delete_file = True
                elif num_succeeded > 0 and output_retention == 'complete':
                    delete_file = True

                if delete_file:
                    output_filepath.unlink(missing_ok=True)
                    file_deleted = True
                    total_output_files_deleted += 1

            if not file_deleted:
                output_filepaths.setdefault(ptd_label, {})[output_label] = (
                    output_filepath
                )
                total_output_files_retained += 1

    if save_error_report and failures:
        _save_error_report(
            failures=failures,
            num_succeeded=total_succeeded,
            num_output_files_deleted=total_output_files_deleted,
            num_output_files_retained=total_output_files_retained,
            output_data_dir=output_data_dir,
            stage_name=stage_name,
            created_by=created_by,
            project_root=project_root,
            error_handling=error_handling,
            output_retention=output_retention,
            save_error_report=save_error_report,
        )

    return output_filepaths


# =============================================================================
# Pipeline Orchestration
# =============================================================================

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
    base_data_dir = pipeline_config['outputs']['data_dir']

    # Input data directory: raw data for Stage 1, previous output dir
    # for Stages 2--4
    if stage_num == 1:
        input_data_dir = pipeline_config['inputs']['data_dir']
    else:
        input_data_dir = f"{base_data_dir}/{prev_stage_name}"

    # Build runtime: stage-level takes precedence, pipeline-level fills gaps
    runtime = dict(pipeline_config['stages'][stage_name]['runtime'])
    pipeline_runtime = pipeline_config.get('runtime', {})
    for key in ('error_handling', 'output_retention', 'save_error_report'):
        if key not in runtime:
            value = pipeline_runtime.get(key)
            if value is not None:
                runtime[key] = value

    return {
        'summary': {
            'created_by': pipeline_config['summary']['created_by'],
            'creation_time': datetime.now().isoformat(),
        },
        'inputs': {
            'stage_name': prev_stage_name,
            'data_dir': input_data_dir,
            'filepaths': convert_paths_to_relative(
                input_filepaths, project_root,
            ),
        },
        'configs': pipeline_config['stages'][stage_name]['configs'],
        'runtime': runtime,
        'outputs': {
            'stage_name': stage_name,
            'data_dir': f"{base_data_dir}/{stage_name}",
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
    configs_dir = project_root / pipeline_config['outputs']['configs_dir']
    configs_dir.mkdir(parents=True, exist_ok=True)

    for stage_num in (1, 2, 3, 4):
        stage_name = PIPELINE_STAGE_NAMES[stage_num]
 
        # Collect full input filepaths for this stage
        if stage_num == 1:
            input_filepaths = convert_relative_to_paths(
                pipeline_config['inputs']['filepaths'], project_root,
            )
        else:
            prev_stage_name = PIPELINE_STAGE_NAMES[stage_num - 1]
            input_filepaths = results[prev_stage_name]
 
        # Assemble complete per-stage config and save
        stage_config = _assemble_stage_config(
            pipeline_config, stage_num, input_filepaths, project_root,
        )
 
        config_path = configs_dir / f"{stage_name}.yml"
        with open(config_path, 'w') as f:
            yaml.safe_dump(
                stage_config, f,
                default_flow_style=False, sort_keys=False,
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
        config['inputs']['filepaths'], project_root,
    )
    verbose = config['runtime']['verbose']
    save_stage_configs = config['runtime'].get('save_stage_configs', False)

    # --- Validate inputs ---

    # All four stages must be present in config
    expected_stages = {PIPELINE_STAGE_NAMES[n] for n in (1, 2, 3, 4)}
    provided_stages = set(config.get('stages', {}).keys())
    missing_stages = expected_stages - provided_stages
    if missing_stages:
        raise ValueError(
            f"Pipeline config 'stages' is missing entries: "
            f"{sorted(missing_stages)}"
        )

    # All input files must exist
    for ptd_label, ptd_filepath in input_filepaths.items():
        if not ptd_filepath.exists():
            raise FileNotFoundError(
                f"Input file for '{ptd_label}' does not exist: "
                f"{ptd_filepath}"
            )

    # configs_dir required when save_stage_configs is enabled
    if save_stage_configs and 'configs_dir' not in config.get('outputs', {}):
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
                config, stage_num, prev_output, project_root,
            )

            # Run the stage
            stage_output = PIPELINE_STAGE_FUNCTIONS[stage_num](
                stage_config, project_root,
            )

            # Accumulate results and wire output to next stage
            results.setdefault(stage_name, {}).update(stage_output)
            prev_output = stage_output

        if not stage_config['runtime'].get('verbose', False):
            print()

    # --- Save assembled per-stage configs ---

    if save_stage_configs:
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

    verbose = config['runtime']['verbose']
    save_stage_configs = config['runtime'].get('save_stage_configs', False)

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
    provided_stages = set(config.get('stages', {}).keys())
    missing_stages = expected_stages - provided_stages
    if missing_stages:
        raise ValueError(
            f"Pipeline config 'stages' is missing entries: "
            f"{sorted(missing_stages)}"
        )

    # --- Resolve initial input filepaths ---

    first_stage = stage_nums[0]

    if first_stage == 1:
        prev_output = convert_relative_to_paths(
            config['inputs']['filepaths'], project_root,
        )

        # All input files must exist
        for ptd_label, ptd_filepath in prev_output.items():
            if not ptd_filepath.exists():
                raise FileNotFoundError(
                    f"Input file for '{ptd_label}' does not exist: "
                    f"{ptd_filepath}"
                )
    else:
        if input_filepaths is None:
            raise ValueError(
                f"'input_filepaths' is required when the first "
                f"selected stage is not 1 (got stage_nums={stage_nums})."
            )
        prev_output = input_filepaths

    # --- Validate save_stage_configs ---

    if save_stage_configs and 'configs_dir' not in config.get('outputs', {}):
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
            config, stage_num, prev_output, project_root,
        )

        # Run the stage on all inputs at once
        stage_output = PIPELINE_STAGE_FUNCTIONS[stage_num](
            stage_config, project_root,
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


# =============================================================================
# Pipeline Loading
# =============================================================================

def load_hodge_laplacian(
    filepath: str | Path,
    project_root: str | Path,
    group: str | None = None,
    validate_hash: bool = True,
) -> HodgeLaplacian:
    """
    Load a HodgeLaplacian by retracing the provenance chain.

    Reads the HodgeLaplacian pipeline file to locate the upstream
    CochainMetric and SimplicialComplex files via provenance attributes,
    loads both upstream objects, constructs a HodgeLaplacian from them,
    and populates its cache from the file.

    Parameters
    ----------
    filepath : str or Path
        Path to an HDF5 pipeline file containing HodgeLaplacian data.
        Must have root-level ``input_file`` and ``stage_name`` attributes.
    project_root : str or Path
        Project root for resolving relative provenance paths.
    group : str, optional
        HDF5 group for all three objects (HL, CM, SC). Typically
        ``'member_{i:04d}'`` for pipelines based on point data ensembles.
    validate_hash : bool, default=True
        Whether to verify content hashes during loading. Passed through
        to ``SimplicialComplex.load()``, ``CochainMetric.load()``, and
        ``HodgeLaplacian.load_cache()``.

    Returns
    -------
    HodgeLaplacian
        Fully constructed instance with cache populated from file.

    Raises
    ------
    KeyError
        If the provenance chain does not contain the required upstream
        stages (``'geometry_metric'`` and ``'topology_simplicial'``).
    FileNotFoundError
        If any file in the provenance chain does not exist.
    ValueError
        If hash validation fails at any loading step.
    """
    filepath = Path(filepath)
    project_root = Path(project_root)

    # Trace provenance to locate upstream files
    provenance_chain = trace_provenance(filepath, project_root)

    required_stages = ('geometry_metric', 'topology_simplicial')
    missing_stages = [s for s in required_stages if s not in provenance_chain]
    if missing_stages:
        raise KeyError(
            f"Provenance chain missing required stages: {missing_stages}. "
            f"Found stages: {list(provenance_chain.keys())}"
        )

    cm_filepath = provenance_chain['geometry_metric']
    sc_filepath = provenance_chain['topology_simplicial']

    # Load upstream objects
    sc = SimplicialComplex.load(
        sc_filepath, group=group, validate_hash=validate_hash, load_incidence=True,
    )
    cm = CochainMetric.load(
        cm_filepath, group=group, validate_hash=validate_hash,
    )

    # Read group-level attributes for constructor params
    _, group_attributes = read_h5(filepath, group=group, dataset_names=[])
    metadata = group_attributes.get('metadata', {})

    # Construct HodgeLaplacian and populate cache from file
    hl = HodgeLaplacian(sc, cm, metadata=metadata)
    hl.load_cache(filepath, group=group, validate_hash=validate_hash)

    return hl


def load_spectra(
    filepath: str | Path,
    project_root: str | Path,
    group: str | None = None,
    validate_hash: bool = True,
) -> HodgeLaplacianSpectra:
    """
    Load a HodgeLaplacianSpectra by retracing the provenance chain.

    Reads the spectra pipeline file to locate the upstream HodgeLaplacian
    file via provenance attributes, calls ``load_hodge_laplacian`` to
    reconstruct the live HodgeLaplacian, reads solver configuration from
    the file, constructs a HodgeLaplacianSpectra from the live HL, and
    populates its cache from the file.

    Parameters
    ----------
    filepath : str or Path
        Path to an HDF5 pipeline file containing HodgeLaplacianSpectra
        data.  Must have root-level ``input_file`` and ``stage_name``
        attributes pointing back to the upstream HodgeLaplacian file.
    project_root : str or Path
        Project root for resolving relative provenance paths.
    group : str, optional
        HDF5 group for all objects (spectra, HL, CM, SC). Typically
        ``'member_{i:04d}'`` for pipelines based on point data ensembles.
    validate_hash : bool, default=True
        Whether to verify content hashes during loading. Passed through
        to ``load_hodge_laplacian`` and ``HodgeLaplacianSpectra.load_cache``.

    Returns
    -------
    HodgeLaplacianSpectra
        Fully constructed instance with cache populated from file.

    Raises
    ------
    KeyError
        If the provenance chain does not contain ``'hodge_laplacian'``.
    FileNotFoundError
        If any file in the provenance chain does not exist.
    ValueError
        If hash validation fails at any loading step.
    """
    filepath = Path(filepath)
    project_root = Path(project_root)

    # Trace provenance to locate the upstream HodgeLaplacian file
    provenance_chain = trace_provenance(filepath, project_root)
    if 'hodge_laplacian' not in provenance_chain:
        raise KeyError(
            f"Provenance chain missing required stage 'hodge_laplacian'. "
            f"Found stages: {list(provenance_chain.keys())}"
        )
    hl_filepath = provenance_chain['hodge_laplacian']

    # Load upstream HodgeLaplacian object
    hl = load_hodge_laplacian(
        hl_filepath, project_root, group=group, validate_hash=validate_hash,
    )
    
    # Read group-level attributes for constructor params
    _, group_attributes = read_h5(filepath, group=group, dataset_names=[])
    solver = str(group_attributes.get('solver', 'dense'))
    solver_params = group_attributes.get('solver_params', None)
    raw_eigvec = group_attributes.get('compute_eigenvectors', False)
    if isinstance(raw_eigvec, (bool, np.bool_)):
        compute_eigenvectors = bool(raw_eigvec)
    elif hasattr(raw_eigvec, '__iter__'):
        compute_eigenvectors = [(int(k), str(comp)) for k, comp in raw_eigvec]
    else:
        compute_eigenvectors = bool(raw_eigvec)
    metadata = group_attributes.get('metadata', {})

    # Construct HodgeLaplacianSpectra and populate cache from file
    hlsp = HodgeLaplacianSpectra(
        hl,
        solver=solver,
        solver_params=solver_params,
        compute_eigenvectors=compute_eigenvectors,
        metadata=metadata,
    )
    hlsp.load_cache(filepath, group=group, validate_hash=validate_hash)

    return hlsp


# =============================================================================
# Pipeline Helpers
# =============================================================================

def select_pipeline_inputs(
    project_root: str | Path,
    select_categories: list[str] | None = None,
    select_point_data: list[str] | None = None,
) -> dict[str, Path]:
    """
    Select raw point data files by category and glob filters.

    Scans the raw data directory (``data/raw/``) for HDF5 files matching
    the selection criteria. This is the entry point data for the pipeline
    (stage 0 outputs / stage 1 inputs).

    Parameters
    ----------
    project_root : str or Path
        Project root directory.
    select_categories : list of str, optional
        Raw data category names to include (e.g. ``['lattice_2d', 'random']``).
        Categories correspond to subdirectory names under ``data/raw/``.
        If None or empty, all categories are included.
    select_point_data : list of str, optional
        Glob patterns for point data labels (e.g. ``['trilatthex*', 'randunif*']``).
        If None or empty, all point data labels are included.

    Returns
    -------
    dict[str, Path]
        Dictionary ``{ptd_label: filepath}`` mapping point data labels
        (file stems) to their absolute file paths.
    """
    raw_data_dir = Path(project_root) / 'data' / 'raw'

    filepaths: dict[str, Path] = {}

    for category_dir in sorted(raw_data_dir.iterdir()):
        if not category_dir.is_dir():
            continue

        # Filter by category
        if select_categories and category_dir.name not in select_categories:
            continue

        for filepath in sorted(category_dir.glob('*.h5')):
            ptd_label = filepath.stem

            # Filter by point data label
            if select_point_data and not any(fnmatch(ptd_label, g) for g in select_point_data):
                continue

            filepaths[ptd_label] = filepath

    return filepaths


def select_stage_outputs(
    stage_num: int,
    project_root: str | Path,
    select_categories: list[str] | None = None,
    select_point_data: list[str] | None = None,
    select_simplicial_complex: list[str] | None = None,
    select_cochain_metric: list[str] | None = None,
) -> dict[str, dict[str, Path]]:
    """
    Select pipeline stage output files by category and glob filters.

    Scans the output directory for the given stage and returns file paths
    matching the selection criteria. Supports stages 1--4.

    Parameters
    ----------
    stage_num : int
        Pipeline stage number (1--4).
    project_root : str or Path
        Project root directory.
    select_categories : list of str, optional
        Raw data category names to include (e.g. ``['lattice_2d', 'random']``).
        Categories are derived from subdirectory names under ``data/raw/``.
        If None or empty, all categories are included.
    select_point_data : list of str, optional
        Glob patterns for point data labels (e.g. ``['trilatthex*', 'randunif*']``).
        If None or empty, all point data labels are included.
    select_simplicial_complex : list of str, optional
        Glob patterns for simplicial complex labels (e.g. ``['delaunay*']``).
        Applies to stages 1--4. If None or empty, all are included.
    select_cochain_metric : list of str, optional
        Glob patterns for cochain metric labels (e.g. ``['combinatorial*']``).
        Applies to stages 2--4 (ignored for stage 1). If None or empty, all
        are included.

    Returns
    -------
    dict[str, dict[str, Path]]
        Nested dictionary ``{ptd_label: {output_label: filepath}}``, where
        ``ptd_label`` is the point data subdirectory name and
        ``output_label`` is the HDF5 file stem.
    """
    if stage_num not in PIPELINE_STAGE_NAMES or stage_num == 0:
        raise ValueError(
            f"stage_num must be 1--4, got {stage_num}"
        )

    project_root = Path(project_root)
    stage_name = PIPELINE_STAGE_NAMES[stage_num]
    output_data_dir = project_root / 'data' / 'interim' / stage_name
    raw_data_dir = project_root / 'data' / 'raw'

    # Build category mapping: point_data_label -> set of categories
    # (a label can appear in multiple raw categories)
    ptd_categories: dict[str, set[str]] = {}
    if select_categories:
        for category_dir in sorted(raw_data_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            for raw_file in category_dir.glob('*.h5'):
                ptd_categories.setdefault(raw_file.stem, set()).add(category_dir.name)

    # Whether filenames have the {sc}__{cm} format (stage 2+)
    has_metric_suffix = stage_num >= 2

    filepaths: dict[str, dict[str, Path]] = {}

    for ptd_subdir in sorted(output_data_dir.iterdir()):
        if not ptd_subdir.is_dir():
            continue
        ptd_label = ptd_subdir.name

        # Filter by category
        if select_categories:
            if not ptd_categories.get(ptd_label, set()) & set(select_categories):
                continue

        # Filter by point data label
        if select_point_data and not any(fnmatch(ptd_label, g) for g in select_point_data):
            continue

        # Collect matching output files
        ptd_filepaths: dict[str, Path] = {}
        for filepath in sorted(ptd_subdir.glob('*.h5')):
            stem = filepath.stem

            if has_metric_suffix:
                parts = stem.split('__', 1)
                sc_label = parts[0]
                cm_label = parts[1] if len(parts) == 2 else ''
            else:
                sc_label = stem
                cm_label = None

            # Filter by simplicial complex label
            if select_simplicial_complex and not any(fnmatch(sc_label, g) for g in select_simplicial_complex):
                continue

            # Filter by cochain metric label (stages 2+ only)
            if has_metric_suffix and select_cochain_metric and not any(fnmatch(cm_label, g) for g in select_cochain_metric):
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
            input_filepaths = pipeline_config['inputs']['filepaths']
        else:
            input_filepaths = {}

        stage_configs[stage_num] = _assemble_stage_config(
            pipeline_config, stage_num, input_filepaths, project_root,
        )

    return stage_configs

