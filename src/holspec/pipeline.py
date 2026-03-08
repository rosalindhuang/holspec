"""
Pipeline orchestration for holspec.

Provides stage metadata, provenance tracing across pipeline output files,
and loading functions for derived-quantity objects that require live
references to upstream objects.
"""
from __future__ import annotations

from pathlib import Path

from holspec.utilities import read_h5
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
    compute_eigenvectors = bool(group_attributes.get('compute_eigenvectors', False))
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