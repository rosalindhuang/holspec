"""
Pipeline orchestration for holspec.

Provides stage metadata, provenance tracing, loading functions for 
objects that require live references to upstream objects.

Provides pipeline stage functions ``run_{stage}`` that formalize the
pipeline computations into callable functions.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from holspec.utilities import (
    read_h5, save_h5, convert_relative_to_paths, get_keys_h5,
)
from holspec.point_data import PointDataEnsemble
from holspec.simplicial import SimplicialComplex
from holspec.cochain_metric import CochainMetric
from holspec.hodge_laplacian import HodgeLaplacian
from holspec.spectra import HodgeLaplacianSpectra


# =============================================================================
# Pipeline Stage Metadata
# =============================================================================

PIPELINE_STAGES: dict[int, str] = {
    0: 'point_data',
    1: 'topology_simplicial',
    2: 'geometry_metric',
    3: 'hodge_laplacian',
    4: 'spectra',
}


# =============================================================================
# Provenance
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
        compute_eigenvectors=compute_eigenvectors,
        metadata=metadata,
    )
    hlsp.load_cache(filepath, group=group, validate_hash=validate_hash)

    return hlsp


# =============================================================================
# Pipeline File Initialization
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


# =============================================================================
# Pipeline Stage Functions
# =============================================================================

def run_topology_simplicial(
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, dict[str, Path]]:
    """
    Run pipeline Stage 1: Topological Structure via Simplicial Complexes.

    Reads the stage config YAML, constructs a SimplicialComplex for each
    (point data ensemble, simplicial construction) pair, and writes the
    results to HDF5 files. 

    Parameters
    ----------
    config_path : str or Path
        Path to a YAML config file
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
        If the config file or any input file does not exist.
    AssertionError
        If boundary property validation fails (when enabled).
    """
    config_path = Path(config_path)
    project_root = Path(project_root)

    # --- Validate config path ---
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # --- Read and unpack config ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    input_filepaths = convert_relative_to_paths(
        config['inputs']['filepaths'], project_root,
    )
    output_data_dir = project_root / config['outputs']['data_dir']
    created_by = config['summary']['created_by']
    verbose = config['runtime']['verbose']
    stage_name = config['outputs']['stage_name']

    simplicial_constructions = config['configs']['simplicial_constructions']
    validate_boundary_property = config['runtime']['validate_boundary_property']
    cache_incidence = config['runtime']['cache_incidence']

    # --- Loop over inputs ---
    output_filepaths: dict[str, dict[str, Path]] = {}

    for ptd_label, ptd_filepath in input_filepaths.items():

        # Load ensemble once per point data label
        point_data_ensemble = PointDataEnsemble.load(ptd_filepath)

        if verbose:
            print(f"{'-'*60}")
            print(f"{ptd_label}")
            print(f"{'-'*60}")
            print()

        for sc_label, sc_config in simplicial_constructions.items():

            # Output file path
            output_filepath = output_data_dir / ptd_label / f"{sc_label}.h5"

            # Initialize file with pipeline metadata
            _initialize_pipeline_file(
                output_filepath, created_by, ptd_filepath,
                project_root, stage_name, sc_config,
            )

            # Iterate computation over ensemble members
            for member_index, member in enumerate(point_data_ensemble):

                # Construct simplicial complex
                sc = SimplicialComplex.from_point_data(
                    member, sc_config,
                    metadata={'member_index': member_index},
                )

                # Compute incidence matrices (if caching)
                if cache_incidence:
                    for k in range(sc.max_dim + 1):
                        sc.incidence_matrix(k)

                # Validate boundary property (also populates cache)
                if validate_boundary_property:
                    boundary_results = sc.validate_boundary_property()
                    failed = [
                        k for k, passed in boundary_results.items()
                        if not passed
                    ]
                    assert not failed, (
                        f"Boundary property D_{{k-1}} @ D_k = 0 failed:\n"
                        f"  point data             : {ptd_label}\n"
                        f"  simplicial construction: {sc_label}\n"
                        f"  member                 : member_{member_index:04d}\n"
                        f"  file                   : "
                        f"{output_filepath.relative_to(project_root)}\n"
                        f"  failed at k            : {failed}"
                    )

                # Save simplicial complex to file
                sc.save(
                    output_filepath,
                    save_incidence=cache_incidence,
                    mode='replace',
                    group=f"member_{member_index:04d}",
                )

            # Verbose output
            if verbose:
                print(
                    f"Constructed {point_data_ensemble.size} simplicial "
                    f"complexes for {ptd_label} / {sc_label}:"
                )
                print(f"  {sc}")
                print(f"  simplicial construction: {sc_label}")
                print(f"  incidence matrices:")
                for k in range(sc.max_dim + 1):
                    if k in sc._incidence_cache:
                        D_k = sc._incidence_cache[k]
                        print(
                            f"    D_{k}: shape={D_k.shape}, nnz={D_k.nnz}"
                        )
                    else:
                        print(f"    D_{k}: (not cached)")
                print(f"  file path: "
                      f"{output_filepath.relative_to(project_root)}")
                print(f"  file size: "
                      f"{output_filepath.stat().st_size / 1024:.2f} KB")
                print()

            # Collect output filepaths
            output_filepaths.setdefault(ptd_label, {})[sc_label] = (
                output_filepath
            )

    return output_filepaths


def run_geometry_metric(
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, dict[str, Path]]:
    """
    Run pipeline Stage 2: Geometric Structure via Discrete Cochain Metrics.

    Reads the stage config YAML, constructs a CochainMetric for each
    (simplicial complex, metric model) pair, and writes the results to
    HDF5 files. Upstream point data is resolved via provenance tracing.

    Parameters
    ----------
    config_path : str or Path
        Path to a YAML config file.
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
        If the config file or any input file does not exist.
    ValueError
        If metric validation fails (when enabled).
    """
    config_path = Path(config_path)
    project_root = Path(project_root)

    # --- Validate config path ---
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # --- Read and unpack config ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    input_filepaths = convert_relative_to_paths(
        config['inputs']['filepaths'], project_root,
    )
    output_data_dir = project_root / config['outputs']['data_dir']
    created_by = config['summary']['created_by']
    verbose = config['runtime']['verbose']
    stage_name = config['outputs']['stage_name']

    metric_models = config['configs']['metric_models']
    validate_metric = config['runtime']['validate_metric']

    # --- Loop over inputs ---
    output_filepaths: dict[str, dict[str, Path]] = {}

    for ptd_label, sc_files in input_filepaths.items():

        if verbose:
            print(f"{'-'*60}")
            print(f"{ptd_label}")
            print(f"{'-'*60}")
            print()

        for sc_label, sc_filepath in sc_files.items():

            # Trace provenance chain for file paths
            provenance_chain = trace_provenance(sc_filepath, project_root)
            ptd_filepath = provenance_chain['point_data']

            # Load point data from file
            point_data_ensemble = PointDataEnsemble.load(ptd_filepath)

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

                # Iterate computation over ensemble members
                for member_index, ptd in enumerate(point_data_ensemble):

                    member_group = f"member_{member_index:04d}"

                    # Load simplicial complex for this member
                    sc = SimplicialComplex.load(
                        sc_filepath, group=member_group,
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
                        group=member_group,
                    )

                # Verbose output
                if verbose:
                    print(
                        f"Constructed {point_data_ensemble.size} cochain "
                        f"metrics for {ptd_label} / {output_label}:"
                    )
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

                # Collect output filepaths
                output_filepaths.setdefault(ptd_label, {})[output_label] = (
                    output_filepath
                )

    return output_filepaths


def run_hodge_laplacian(
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, dict[str, Path]]:
    """
    Run pipeline Stage 3: Hodge Laplacians and Discrete Differential Operators.

    Reads the stage config YAML, constructs a HodgeLaplacian for each
    cochain metric file, and writes the results to HDF5 files. Upstream
    SimplicialComplex files are resolved via provenance tracing.

    When both ``cache_laplacians`` and ``validate_laplacians`` are False,
    no Laplacian matrices are computed. The output file serves as a
    provenance waypoint containing only metadata and content hashes.

    Parameters
    ----------
    config_path : str or Path
        Path to a YAML config file.
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
        If the config file or any input file does not exist.
    AssertionError
        If Laplacian property validation fails (when enabled).
    """
    config_path = Path(config_path)
    project_root = Path(project_root)

    # --- Validate config path ---
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # --- Read and unpack config ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    input_filepaths = convert_relative_to_paths(
        config['inputs']['filepaths'], project_root,
    )
    output_data_dir = project_root / config['outputs']['data_dir']
    created_by = config['summary']['created_by']
    verbose = config['runtime']['verbose']
    stage_name = config['outputs']['stage_name']

    validate_laplacians = config['runtime']['validate_laplacians']
    cache_laplacians = config['runtime']['cache_laplacians']

    # --- Loop over inputs ---
    output_filepaths: dict[str, dict[str, Path]] = {}

    for ptd_label, cm_files in input_filepaths.items():

        if verbose:
            print(f"{'-'*60}")
            print(f"{ptd_label}")
            print(f"{'-'*60}")
            print()

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

            # Iterate computation over ensemble members
            for member_key in member_keys:

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

                # Compute Laplacian matrices (if caching)
                if cache_laplacians:
                    for k in range(hl.max_dim + 1):
                        _ = hl[k]

                # Validate Laplacian properties (also populates cache)
                if validate_laplacians:
                    validation_results = hl.validate_laplacians()
                    for (k, comp), result_dict in validation_results.items():
                        failed = [
                            prop for prop, passed in result_dict.items()
                            if not passed
                        ]
                        assert not failed, (
                            f"Laplacian validation failed:\n"
                            f"  point data        : {ptd_label}\n"
                            f"  cochain metric    : {cm_label}\n"
                            f"  member            : {member_key}\n"
                            f"  degree            : k={k}\n"
                            f"  component         : {comp}\n"
                            f"  file              : "
                            f"{output_filepath.relative_to(project_root)}\n"
                            f"  failed properties : {failed}"
                        )

                # Save Hodge Laplacian to file
                hl.save(
                    output_filepath,
                    save_laplacians=cache_laplacians,
                    mode='replace',
                    group=member_key,
                )

            # Verbose output
            if verbose:
                print(
                    f"Constructed {len(member_keys)} Hodge Laplacians "
                    f"for {ptd_label} / {cm_label}:"
                )
                print(f"  {hl}")
                print(f"  hodge laplacian matrices:")
                for k in range(hl.max_dim + 1):
                    if (k, 'full') in hl._laplacian_cache:
                        L_full = hl._laplacian_cache[(k, 'full')]
                        print(
                            f"    L^{k}: shape={L_full.shape}, "
                            f"nnz={L_full.nnz}"
                        )
                    else:
                        print(f"    L^{k}: (not cached)")
                print(f"  file path: "
                      f"{output_filepath.relative_to(project_root)}")
                print(f"  file size: "
                      f"{output_filepath.stat().st_size / 1024:.2f} KB")
                print()

            # Collect output filepaths
            output_filepaths.setdefault(ptd_label, {})[output_label] = (
                output_filepath
            )

    return output_filepaths


def run_spectra(
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, dict[str, Path]]:
    """
    Run pipeline Stage 4: Hodge Laplacian Spectra and Spectral Observables.

    Reads the stage config YAML, computes eigendecompositions for each
    HodgeLaplacian file, and writes the results to HDF5 files. Upstream
    objects are reconstructed via ``load_hodge_laplacian`` per member.

    This is the terminal pipeline stage. All spectra are computed eagerly
    (no conditional computation flags).

    Parameters
    ----------
    config_path : str or Path
        Path to a YAML config file.
    project_root : str or Path
        Project root for resolving relative paths in the config.

    Returns
    -------
    dict[str, dict[str, Path]]
        Nested mapping ``{ptd_label: {hl_label: output_filepath}}``.

    Raises
    ------
    FileNotFoundError
        If the config file or any input file does not exist.
    """
    config_path = Path(config_path)
    project_root = Path(project_root)

    # --- Validate config path ---
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # --- Read and unpack config ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    input_filepaths = convert_relative_to_paths(
        config['inputs']['filepaths'], project_root,
    )
    output_data_dir = project_root / config['outputs']['data_dir']
    created_by = config['summary']['created_by']
    verbose = config['runtime']['verbose']
    stage_name = config['outputs']['stage_name']

    solver = config['configs']['solver']
    compute_eigenvectors = config['configs']['compute_eigenvectors']

    # --- Loop over inputs ---
    output_filepaths: dict[str, dict[str, Path]] = {}

    for ptd_label, hl_files in input_filepaths.items():

        if verbose:
            print(f"{'-'*60}")
            print(f"{ptd_label}")
            print(f"{'-'*60}")
            print()

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
                {'solver': solver,
                 'compute_eigenvectors': compute_eigenvectors},
            )

            # Discover ensemble members from input file
            member_keys = [
                key for key in get_keys_h5(hl_filepath)
                if key.startswith('member_')
            ]

            # Iterate computation over ensemble members
            for member_key in member_keys:

                # Load upstream HodgeLaplacian for this member
                hl = load_hodge_laplacian(
                    hl_filepath, project_root, group=member_key,
                )

                # Construct HodgeLaplacianSpectra
                hlsp = HodgeLaplacianSpectra(
                    hl,
                    solver=solver,
                    compute_eigenvectors=compute_eigenvectors,
                )

                # Compute all components at every degree
                for k in range(hlsp.max_dim + 1):
                    _ = hlsp[k]

                # Save spectra to file
                hlsp.save(
                    output_filepath,
                    mode='replace',
                    group=member_key,
                    save_eigenvectors=True,
                )

            # Verbose output
            if verbose:
                print(
                    f"Computed {len(member_keys)} spectra "
                    f"for {ptd_label} / {hl_label}:"
                )
                print(f"  {hlsp}")
                print(f"  solver: {solver}")
                print(f"  compute_eigenvectors: {compute_eigenvectors}")
                print(f"  hodge laplacian spectra:")
                for k in hlsp.degrees:
                    spc = hlsp[k]['full']
                    print(
                        f"    L^{k},full: num_eig={spc.num_eigenvalues}, "
                        f"dim_ker={spc.dim_ker()}"
                    )
                print(f"  file path: "
                      f"{output_filepath.relative_to(project_root)}")
                print(f"  file size: "
                      f"{output_filepath.stat().st_size / 1024:.2f} KB")
                print()

            # Collect output filepaths
            output_filepaths.setdefault(ptd_label, {})[output_label] = (
                output_filepath
            )

    return output_filepaths