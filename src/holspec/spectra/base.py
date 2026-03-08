"""
Spectra of Hodge Laplacian operators.

Provides HodgeLaplacianSpectra, the primary Stage 4 output object. A lazy
computation wrapper over a HodgeLaplacian that produces eigendecompositions
per (k, component) pair on demand, caching Spectrum instances to avoid
redundant computation.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from holspec.utilities import save_h5, read_h5, join_h5_group
from holspec.hodge_laplacian.base import LAPLACIAN_COMPONENT_NAMES

from .spectrum import Spectrum
from .eigensolvers import compute_eigendecomposition, VALID_SOLVER_NAMES
from .validation import ZERO_EIGENVALUE_TOL

if TYPE_CHECKING:
    from holspec.hodge_laplacian import HodgeLaplacian


# =============================================================================
# HodgeLaplacianSpectra
# =============================================================================

class HodgeLaplacianSpectra:
    """
    Spectra of the Hodge Laplacian operators on a simplicial complex.

    The primary output object of pipeline Stage 4. A lazy computation
    wrapper that holds a live reference to a HodgeLaplacian, producing
    Spectrum instances per (k, component) pair on demand and caching
    results. 

    For each degree k and Laplacian component (lower, upper, full), the
    eigendecomposition is performed by symmetrizing each Laplacian matrix 
    via the cochain metric and applying a standard symmetric eigensolver. 
    Eigenvectors, if requested, are back-transformed to the cochain basis.

    Parameters
    ----------
    hl : HodgeLaplacian
        Source Hodge Laplacian defining the operators to decompose.
    solver : {'dense', 'sparse'}, default='dense'
        Eigensolver backend. 'sparse' is accepted at construction but
        raises NotImplementedError when computation is attempted.
    compute_eigenvectors : bool, default=False
        Whether to compute eigenvectors alongside eigenvalues.
    metadata : dict, optional
        Provenance metadata. 'creation_time' is auto-populated if not
        provided.

    Notes
    -----
    - Validates the solver string at construction. No spectra are
      computed until requested.
    - Lazy computation piggybacks on upstream caching: hl.to_matrix(k)
      and hl.cm[k] are themselves lazy and cached.
    - Content hash is derived from hl.content_hash alone, not from
      solver or compute_eigenvectors. These affect computation strategy
      and data storage, not the mathematical content. When sparse solver
      support is added, solver parameters affecting which eigenvalues
      are computed will need to be incorporated.
    - Persistence follows the derived-quantity pattern: save() writes
      the cache; load_cache() populates the cache of an existing
      instance. No standalone load() classmethod exists. Chain-loading is
      delegated to pipeline.load_spectra.
    """

    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(
        self,
        hl: HodgeLaplacian,
        solver: str = 'dense',
        compute_eigenvectors: bool = False,
        metadata: dict | None = None,
    ):
        # Validate solver
        if solver not in VALID_SOLVER_NAMES:
            raise ValueError(
                f"Unknown solver '{solver}'. "
                f"Valid solvers: {VALID_SOLVER_NAMES}."
            )

        self._hl = hl
        self._solver = solver
        self._compute_eigenvectors = compute_eigenvectors

        # Initialize metadata
        self.metadata: dict = metadata if metadata is not None else {}
        if 'creation_time' not in self.metadata:
            self.metadata['creation_time'] = datetime.now().isoformat()

        # Lazy cache: keyed by (k, component) tuples
        self._spectrum_cache: dict[tuple[int, str], Spectrum] = {}

        self._hash_cache: str | None = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def hodge_laplacian(self) -> HodgeLaplacian:
        """Live reference to the source HodgeLaplacian."""
        return self._hl

    hl = hodge_laplacian    # Alias

    @property
    def max_dim(self) -> int:
        """Maximum simplex dimension n."""
        return self._hl.max_dim

    @property
    def degrees(self) -> list[int]:
        """Sorted list of degrees [0, 1, ..., n]."""
        return self._hl.degrees

    @property
    def f_vector(self) -> list[int]:
        """Face vector [N_0, N_1, ..., N_n] from the simplicial complex."""
        return self._hl.f_vector

    @property
    def dimensions(self) -> dict[int, int]:
        """Cochain space dimensions {k: N_k} at each degree."""
        return self._hl.dimensions

    @property
    def solver(self) -> str:
        """Eigensolver backend name."""
        return self._solver

    @property
    def compute_eigenvectors(self) -> bool:
        """Whether eigenvectors are computed alongside eigenvalues."""
        return self._compute_eigenvectors

    @property
    def content_hash(self) -> str:
        """
        SHA-256 hash derived from the HodgeLaplacian content hash.

        Computed lazily and cached. Uniquely identifies the spectra since
        they are a deterministic function of the Hodge Laplacian inputs.
        """
        if self._hash_cache is None:
            self._hash_cache = self._compute_content_hash()
        return self._hash_cache

    # =========================================================================
    # Spectrum Access Methods
    # =========================================================================

    def spectrum(self, k: int, component: str = 'full') -> Spectrum:
        """
        Return the spectrum at degree k for the given Laplacian component.

        Triggers lazy eigendecomposition and caching on first access for
        each (k, component) pair.

        Parameters
        ----------
        k : int
            Degree. Valid range: 0 <= k <= max_dim.
        component : {'full', 'lower', 'upper'}, default='full'
            Which Laplacian component to decompose.

        Returns
        -------
        Spectrum
            Spectrum with dimension N_k.
        """
        self._validate_degree(k)
        self._validate_component(component)

        if (k, component) not in self._spectrum_cache:
            self._compute_spectrum(k, component)

        return self._spectrum_cache[(k, component)]

    def eigenvalues(self, k: int, component: str = 'full') -> np.ndarray:
        """
        Return eigenvalues at degree k for the given component.

        Convenience wrapper for ``self.spectrum(k, component).eigenvalues``.

        Parameters
        ----------
        k : int
            Degree. Valid range: 0 <= k <= max_dim.
        component : {'full', 'lower', 'upper'}, default='full'
            Which Laplacian component.

        Returns
        -------
        ndarray, shape (N_k,)
            Eigenvalues sorted ascending, non-negative.
        """
        return self.spectrum(k, component).eigenvalues

    def eigenvectors(self, k: int, component: str = 'full') -> np.ndarray | None:
        """
        Return eigenvectors at degree k for the given component.

        Convenience wrapper for ``self.spectrum(k, component).eigenvectors``.

        Parameters
        ----------
        k : int
            Degree. Valid range: 0 <= k <= max_dim.
        component : {'full', 'lower', 'upper'}, default='full'
            Which Laplacian component.

        Returns
        -------
        ndarray or None
            Eigenvector matrix of shape (N_k, N_k), or None if
            compute_eigenvectors is False.
        """
        return self.spectrum(k, component).eigenvectors

    # =========================================================================
    # I/O Methods
    # =========================================================================

    def save(
        self,
        filepath: str | Path,
        mode: str = 'replace',
        group: str | None = None,
        hdf5_options: dict | None = None,
    ) -> str:
        """
        Save computed spectra to HDF5.

        Writes whatever is currently in the spectrum cache. Spectra that
        have not been computed are not saved.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        mode : {'replace', 'update', 'create'}, default='replace'
            How to handle an existing file/group.
        group : str, optional
            HDF5 group path for the data. If None, saves at root level.
        hdf5_options : dict, optional
            HDF5 compression options. Default: gzip level 4.

        Returns
        -------
        content_hash : str
            Content hash of the HodgeLaplacianSpectra for verification.

        Format
        ------
        - Root attributes: max_dim, content_hash, hl_content_hash, solver,
          compute_eigenvectors, cached_keys, metadata.
        - degree_{k}/component_{comp}/: subgroups for each cached
          spectrum, containing eigenvalues dataset, dimension and
          num_eigenvalues attributes, and optionally eigenvectors
          dataset.
        """
        filepath = Path(filepath)
        if hdf5_options is None:
            hdf5_options = {'compression': 'gzip', 'compression_opts': 4}

        # Serialize cached keys for metadata
        cached_keys = [
            f'{k}_{comp}' for k, comp in sorted(self._spectrum_cache.keys())
        ]

        # Root-level attributes
        root_attributes = {
            'max_dim': self.max_dim,
            'content_hash': self.content_hash,
            'hl_content_hash': self._hl.content_hash,
            'solver': self._solver,
            'compute_eigenvectors': self._compute_eigenvectors,
            'cached_keys': cached_keys,
            'metadata': self.metadata,
        }
        save_h5(
            filepath,
            datasets=None,
            attributes=root_attributes,
            mode=mode,
            group=group,
            hdf5_options=hdf5_options,
        )

        # Per-degree, per-component subgroups
        for (k, comp), spc in self._spectrum_cache.items():
            subgroup = join_h5_group(group, f'degree_{k}/component_{comp}')

            spc_datasets = {'eigenvalues': spc.eigenvalues}
            spc_attributes = {
                'dimension': spc.dimension,
                'num_eigenvalues': spc.num_eigenvalues,
            }

            if spc.eigenvectors is not None:
                spc_datasets['eigenvectors'] = spc.eigenvectors

            save_h5(
                filepath,
                datasets=spc_datasets,
                attributes=spc_attributes,
                mode=mode,
                group=subgroup,
                hdf5_options=hdf5_options,
            )

        return self.content_hash

    def load_cache(
        self,
        filepath: str | Path,
        group: str | None = None,
        validate_hash: bool = True,
    ) -> None:
        """
        Populate spectrum cache from a previously saved file.

        Reads eigenvalue and eigenvector data and reconstructs Spectrum
        instances into the lazy cache. Does not modify the live
        HodgeLaplacian reference or any other state. Existing cache
        entries are preserved unless a loaded entry shares the same key,
        in which case the loaded entry takes precedence.

        Parameters
        ----------
        filepath : str or Path
            Path to HDF5 file created by save().
        group : str, optional
            HDF5 group path for the data.
        validate_hash : bool, default=True
            Whether to verify that the stored hl_content_hash matches
            the live HodgeLaplacian, ensuring cached data corresponds
            to the same inputs.

        Raises
        ------
        ValueError
            If hash validation fails (cached data was computed from a
            different HodgeLaplacian than the current one).

        Notes
        -----
        Tolerates mismatches between the stored compute_eigenvectors
        setting and the current instance setting. Cache population loads
        whatever data is present; extra or missing eigenvector data is
        harmless. The important validation is the hl_content_hash.
        """
        filepath = Path(filepath)
        _, root_attributes = read_h5(filepath, group=group)

        # Hash validation
        if validate_hash:
            stored_hl_hash = str(root_attributes.get('hl_content_hash', ''))
            if stored_hl_hash != self._hl.content_hash:
                raise ValueError(
                    f"HL content hash mismatch: stored {stored_hl_hash[:8]}, "
                    f"live {self._hl.content_hash[:8]}"
                )

        # Discover and load cached spectra from the stored key list
        cached_keys = root_attributes.get('cached_keys', [])
        for key_str in cached_keys:
            # Parse "k_component" format
            k_str, comp = key_str.split('_', 1)
            k = int(k_str)

            subgroup = join_h5_group(group, f'degree_{k}/component_{comp}')
            spc_datasets, spc_attributes = read_h5(filepath, group=subgroup)

            dimension = int(spc_attributes['dimension'])
            eigenvalues = spc_datasets['eigenvalues']

            eigenvectors = spc_datasets.get('eigenvectors', None)

            self._spectrum_cache[(k, comp)] = Spectrum(
                eigenvalues, dimension, eigenvectors=eigenvectors,
            )

    # =========================================================================
    # Protocols and Utilities
    # =========================================================================

    def __getitem__(self, k: int) -> dict[str, Spectrum]:
        """
        Return all Hodge Laplacian component spectra at degree k.

        Triggers lazy computation for any spectra not yet cached.

        Parameters
        ----------
        k : int
            Degree. Valid range: 0 <= k <= max_dim.

        Returns
        -------
        dict with keys 'lower', 'upper', 'full', each mapping to a
        Spectrum with dimension N_k.
        """
        return {comp: self.spectrum(k, comp) for comp in LAPLACIAN_COMPONENT_NAMES}

    def __len__(self) -> int:
        """Number of degrees (= max_dim + 1)."""
        return self.max_dim + 1

    def __iter__(self):
        """Iterate over degrees."""
        return iter(range(self.max_dim + 1))

    def __repr__(self) -> str:
        n_cached = len(self._spectrum_cache)
        sizes = list(self.dimensions.values())
        return (
            f"HodgeLaplacianSpectra(max_dim={self.max_dim}, "
            f"dimensions={sizes}, "
            f"solver='{self._solver}', "
            f"cached={n_cached}, "
            f"hash={self.content_hash[:8]})"
        )

    def summary(self, indent: str = '') -> str:
        """
        Generate human-readable summary of the spectra.

        Shows per-degree information including cochain space dimensions
        and which component spectra are cached, along with key
        observables (dim_ker, lambda_min, lambda_max) for cached entries.

        Parameters
        ----------
        indent : str, optional
            String prepended to every line.

        Returns
        -------
        summary : str
        """
        lines = []

        lines.append('Hodge Laplacian Spectra:')
        lines.append('-' * 80)
        lines.append(
            f"{'k':<4} {'N_k':<6} "
            f"{'lower':<20} {'upper':<20} {'full':<20}"
        )
        lines.append('-' * 80)

        for k in self.degrees:
            N_k = self.dimensions[k]
            parts = []
            for comp in LAPLACIAN_COMPONENT_NAMES:
                key = (k, comp)
                if key in self._spectrum_cache:
                    spc = self._spectrum_cache[key]
                    dk = spc.dim_ker()
                    parts.append(f"num_eig={spc.num_eigenvalues}, dim_ker={spc.dim_ker()}")
                else:
                    parts.append('--')

            lines.append(
                f"{k:<4} {N_k:<6} "
                f"{parts[0]:<20} {parts[1]:<20} {parts[2]:<20}"
            )

        lines.append('-' * 80)
        lines.append(
            f"solver: {self._solver}, "
            f"eigenvectors: {self._compute_eigenvectors}"
        )
        lines.append(f"content_hash: {self.content_hash[:16]}")
        lines.append('')

        return '\n'.join(indent + line for line in lines)

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _validate_degree(self, k: int) -> None:
        """Validate that k is a valid degree."""
        if k < 0 or k > self.max_dim:
            raise ValueError(
                f"Degree k={k} out of range. "
                f"Valid range: 0 <= k <= {self.max_dim}."
            )

    @staticmethod
    def _validate_component(component: str) -> None:
        """Validate that component is a recognized name."""
        if component not in LAPLACIAN_COMPONENT_NAMES:
            raise ValueError(
                f"Unknown component '{component}'. "
                f"Valid components: {LAPLACIAN_COMPONENT_NAMES}."
            )

    def _compute_spectrum(self, k: int, component: str) -> None:
        """Compute eigendecomposition at (k, component) and cache the result."""
        L = self._hl.to_matrix(k, component)
        G_k = self._hl.cm[k]

        eigenvalues, eigenvectors = compute_eigendecomposition(
            L, G_k,
            solver=self._solver,
            compute_eigenvectors=self._compute_eigenvectors,
        )

        N_k = self.dimensions[k]
        self._spectrum_cache[(k, component)] = Spectrum(
            eigenvalues, N_k, eigenvectors=eigenvectors,
        )

    def _compute_content_hash(self) -> str:
        """Compute SHA-256 hash from the HodgeLaplacian content hash."""
        hasher = hashlib.sha256()
        hasher.update(self._hl.content_hash.encode('utf-8'))
        return hasher.hexdigest()