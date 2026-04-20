"""
Pipeline stage functions for the holspec pipeline.

Defines the four stage runners ``run_topology_simplicial``,
``run_geometry_metric``, ``run_hodge_laplacian``, and ``run_spectra``,
plus shared helpers for output-file initialization and per-stage failure
reporting. Stage names are exposed via ``PIPELINE_STAGE_NAMES``.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import warnings

import yaml

from holspec.utilities import (
    save_h5, convert_relative_to_paths, get_keys_h5,
)
from holspec.point_data import PointDataEnsemble
from holspec.simplicial import SimplicialComplex
from holspec.cochain_metric import CochainMetric
from holspec.hodge_laplacian import HodgeLaplacian, LAPLACIAN_COMPONENT_NAMES
from holspec.spectra import HodgeLaplacianSpectra

from .provenance import trace_provenance, load_hodge_laplacian


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
