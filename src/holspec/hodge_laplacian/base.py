"""
Hodge Laplacian on cochain spaces.

Provides HodgeLaplacian, the primary Stage 3 output object. A lazy
computation wrapper over SimplicialComplex and CochainMetric that produces
Laplacian matrices and differential operators on demand, caching results
to avoid redundant computation.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from scipy import sparse

from holspec.utilities import save_h5, read_h5, join_h5_group

from .operators import (
    compute_coboundary_matrix,
    compute_dual_coboundary_matrix,
    compute_laplacian_lower_matrix,
    compute_laplacian_upper_matrix,
    symmetrize_matrix,
)
from .validation import validate_hodge_laplacian_inputs

if TYPE_CHECKING:
    from holspec.simplicial import SimplicialComplex
    from holspec.cochain_metric import CochainMetric


# Valid component names for Laplacian access methods
_VALID_COMPONENTS = ('lower', 'upper', 'full')


# =============================================================================
# HodgeLaplacian
# =============================================================================

class HodgeLaplacian:
    """
    Hodge Laplacian of a simplicial complex equipped with a cochain metric.

    The primary output object of pipeline Stage 3. A lazy computation wrapper
    that holds live references to a SimplicialComplex and CochainMetric,
    producing Laplacian matrices {L^k}_{k=0}^n on demand and caching results.

    The Hodge Laplacian at degree k decomposes as:

        L^k = L^{k,low} + L^{k,upp}

    where L^{k,low} captures (k-1)--(k) coupling and L^{k,upp} captures
    (k)--(k+1) coupling. For non-identity metrics, these are G^k-self-adjoint
    but not symmetric as matrices; symmetric variants for eigsh are available
    via to_symmetric_matrix.

    Parameters
    ----------
    sc : SimplicialComplex
        Source simplicial complex defining the topology.
    cm : CochainMetric
        Cochain metric defining the geometry.
    metadata : dict, optional
        Provenance metadata. 'creation_time' is auto-populated if not
        provided.

    Notes
    -----
    - The constructor validates cross-compatibility of sc and cm, then
      stores live references. No matrices are computed until requested.
    - Lazy computation piggybacks on upstream caching: sc.incidence_matrix(k)
      and cm[k] are themselves lazy and cached.
    - Content hash is derived from sc.content_hash and cm.content_hash,
      not from materialized Laplacian matrices. This is cheap, avoids
      defeating lazy computation, and is stable since the Laplacians
      are a deterministic function of the inputs.
    - Persistence follows the derived-quantity pattern: save() writes
      the cache; load_cache() populates the cache of an existing instance.
      No standalone load() classmethod exists.
    """

    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(
        self,
        sc: SimplicialComplex,
        cm: CochainMetric,
        metadata: dict | None = None,
    ):
        validate_hodge_laplacian_inputs(sc, cm)

        self._sc = sc
        self._cm = cm

        # Initialize metadata
        self.metadata: dict = metadata if metadata is not None else {}
        if 'creation_time' not in self.metadata:
            self.metadata['creation_time'] = datetime.now().isoformat()

        # Lazy caches: keyed by (k, component) tuples
        self._laplacian_cache: dict[tuple[int, str], sparse.csr_matrix] = {}
        self._symmetric_laplacian_cache: dict[tuple[int, str], sparse.csr_matrix] = {}

        self._hash_cache: str | None = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def simplicial_complex(self) -> SimplicialComplex:
        """Live reference to the source SimplicialComplex."""
        return self._sc

    sc = simplicial_complex    # Alias

    @property
    def cochain_metric(self) -> CochainMetric:
        """Live reference to the source CochainMetric."""
        return self._cm

    cm = cochain_metric    # Alias

    @property
    def max_dim(self) -> int:
        """Maximum simplex dimension n."""
        return self._sc.max_dim

    @property
    def degrees(self) -> list[int]:
        """Sorted list of degrees [0, 1, ..., n]."""
        return list(range(self.max_dim + 1))

    @property
    def f_vector(self) -> list[int]:
        """Face vector [N_0, N_1, ..., N_n] from the simplicial complex."""
        return self._sc.f_vector

    @property
    def dimensions(self) -> list[int]:
        """Dimension N_k = dim(C^k) of the cochain spaces at each degree."""
        return self._cm.dimensions

    @property
    def content_hash(self) -> str:
        """SHA-256 hash derived from sc and cm content hashes.

        Computed lazily and cached. Uniquely identifies the Hodge Laplacian
        since it is a deterministic function of the topological and metric
        inputs.
        """
        if self._hash_cache is None:
            self._hash_cache = self._compute_content_hash()
        return self._hash_cache

    # =========================================================================
    # Primary Access Methods
    # =========================================================================

    def to_matrix(
        self, k: int, component: str = 'full',
    ) -> sparse.csr_matrix:
        """
        Return a Hodge Laplacian matrix at degree k.

        Parameters
        ----------
        k : int
            Degree. Valid range: 0 <= k <= max_dim.
        component : {'full', 'lower', 'upper'}, default='full'
            Which component to return.

        Returns
        -------
        sparse.csr_matrix, shape (N_k, N_k)
        """
        self._validate_degree(k)
        self._validate_component(component)

        if (k, component) not in self._laplacian_cache:
            self._compute_laplacian(k, component)

        return self._laplacian_cache[(k, component)]
    
    laplacian = to_matrix    # Alias

    def to_symmetric_matrix(
        self, k: int, component: str = 'full',
    ) -> sparse.csr_matrix:
        """
        Return a symmetrized Hodge Laplacian matrix at degree k.

        Computes the similarity transform:

            L̃ = (G^k)^{1/2} L (G^k)^{-1/2}

        which has the same eigenvalues as L but is symmetric as a matrix,
        suitable for eigsh.

        Parameters
        ----------
        k : int
            Degree. Valid range: 0 <= k <= max_dim.
        component : {'full', 'lower', 'upper'}, default='full'
            Which component to return.

        Returns
        -------
        sparse.csr_matrix, shape (N_k, N_k)
            Symmetric matrix with the same spectrum as to_matrix(k, component).
        """
        self._validate_degree(k)
        self._validate_component(component)

        if (k, component) not in self._symmetric_laplacian_cache:
            L = self.to_matrix(k, component)
            self._symmetric_laplacian_cache[(k, component)] = (
                symmetrize_matrix(L, self._cm[k])
            )

        return self._symmetric_laplacian_cache[(k, component)]

    # =========================================================================
    # Differential Operator Inspection
    # =========================================================================

    def coboundary(self, k: int) -> sparse.csr_matrix:
        """
        Return the coboundary matrix d^k: C^k -> C^{k+1}.

        The coboundary (discrete exterior derivative) is purely topological:
        d^k = D_{k+1}^T. At k=n, returns a zero matrix of shape (0, N_n)
        per the boundary-degree convention d^n = 0.

        Parameters
        ----------
        k : int
            Degree. Valid range: 0 <= k <= max_dim.
        """
        self._validate_degree(k)
        return compute_coboundary_matrix(self._sc.incidence_matrix(k + 1))

    def dual_coboundary(self, k: int) -> sparse.csr_matrix:
        """
        Return the dual coboundary (codifferential) matrix δ^k: C^k -> C^{k-1}.

        The dual coboundary is the formal adjoint of d^{k-1} with respect to
        the cochain metric: δ^k = (G^{k-1})^{-1} D_k G^k. At k=0, returns
        a zero matrix of shape (0, N_0) per the boundary-degree convention
        δ^0 = 0.

        Parameters
        ----------
        k : int
            Degree. Valid range: 0 <= k <= max_dim.
        """
        self._validate_degree(k)
        return compute_dual_coboundary_matrix(
            self._sc.incidence_matrix(k), self._cm[k - 1], self._cm[k]
        )

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
        Save computed Laplacian matrices to HDF5.

        Writes whatever is currently in the Laplacian cache. Matrices that
        have not been computed are not saved. Symmetrized matrices are not
        saved (cheaply recomputed from cached Laplacians and the metric).

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
            Content hash of the HodgeLaplacian for verification.

        Format
        ------
        - Root attributes: max_dim, content_hash, sc_content_hash,
          cm_content_hash, cached_keys, metadata.
        - degree_{k}/component_{comp}/: subgroups for each cached matrix,
          containing CSR arrays (data, indices, indptr) and shape attribute.
        """
        filepath = Path(filepath)
        if hdf5_options is None:
            hdf5_options = {'compression': 'gzip', 'compression_opts': 4}

        # Serialize cached keys for metadata
        cached_keys = [
            f'{k}_{comp}' for k, comp in sorted(self._laplacian_cache.keys())
        ]

        # Root-level attributes
        root_attributes = {
            'max_dim': self.max_dim,
            'content_hash': self.content_hash,
            'sc_content_hash': self._sc.content_hash,
            'cm_content_hash': self._cm.content_hash,
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
        for (k, comp), L in self._laplacian_cache.items():
            L_csr = L.tocsr()
            csr_datasets = {
                'data': L_csr.data,
                'indices': L_csr.indices,
                'indptr': L_csr.indptr,
            }
            csr_attributes = {
                'shape': L_csr.shape,
                'nnz': L_csr.nnz,
            }
            save_h5(
                filepath,
                datasets=csr_datasets,
                attributes=csr_attributes,
                mode=mode,
                group=join_h5_group(group, f'degree_{k}/component_{comp}'),
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
        Populate Laplacian cache from a previously saved file.

        Reads Laplacian matrices and inserts them into the lazy cache.
        Does not modify the live sc/cm references or any other state.
        Existing cache entries are preserved unless a loaded entry shares
        the same key, in which case the loaded entry takes precedence.

        Parameters
        ----------
        filepath : str or Path
            Path to HDF5 file created by save().
        group : str, optional
            HDF5 group path for the data.
        validate_hash : bool, default=True
            Whether to verify that stored sc/cm content hashes match the
            live objects, ensuring cached data corresponds to the same inputs.

        Raises
        ------
        ValueError
            If hash validation fails (cached data was computed from different
            inputs than the current sc and cm).
        """
        filepath = Path(filepath)
        _, root_attributes = read_h5(filepath, group=group)

        # Hash validation
        if validate_hash:
            stored_sc_hash = str(root_attributes.get('sc_content_hash', ''))
            stored_cm_hash = str(root_attributes.get('cm_content_hash', ''))
            if stored_sc_hash != self._sc.content_hash:
                raise ValueError(
                    f"SC content hash mismatch: stored {stored_sc_hash[:8]}, "
                    f"live {self._sc.content_hash[:8]}"
                )
            if stored_cm_hash != self._cm.content_hash:
                raise ValueError(
                    f"CM content hash mismatch: stored {stored_cm_hash[:8]}, "
                    f"live {self._cm.content_hash[:8]}"
                )

        # Discover and load cached matrices from the stored key list
        cached_keys = root_attributes.get('cached_keys', [])
        for key_str in cached_keys:
            # Parse "k_component" format
            k_str, comp = key_str.split('_', 1)
            k = int(k_str)

            subgroup = join_h5_group(group, f'degree_{k}/component_{comp}')
            csr_datasets, csr_attributes = read_h5(filepath, group=subgroup)

            shape = tuple(csr_attributes['shape'])
            L = sparse.csr_matrix(
                (csr_datasets['data'], csr_datasets['indices'], csr_datasets['indptr']),
                shape=shape,
            )
            self._laplacian_cache[(k, comp)] = L

    # =========================================================================
    # Protocols and Utilities
    # =========================================================================

    def __getitem__(self, k: int) -> dict[str, sparse.csr_matrix]:
        """
        Return all Laplacian components at degree k.

        Triggers lazy computation for any components not yet cached.

        Parameters
        ----------
        k : int
            Degree. Valid range: 0 <= k <= max_dim.

        Returns
        -------
        dict with keys 'lower', 'upper', 'full', each mapping to a
        sparse.csr_matrix of shape (N_k, N_k).
        """
        return {comp: self.to_matrix(k, comp) for comp in _VALID_COMPONENTS}

    def __len__(self) -> int:
        """Number of degrees (= max_dim + 1)."""
        return self.max_dim + 1

    def __iter__(self):
        """Iterate over degrees."""
        return iter(range(self.max_dim + 1))

    def __repr__(self) -> str:
        n_cached = len(self._laplacian_cache)
        sizes = list(self.dimensions.values())
        return (
            f"HodgeLaplacian(max_dim={self.max_dim}, "
            f"dimensions={sizes}, "
            f"cached={n_cached}, "
            f"hash={self.content_hash[:8]})"
        )

    def summary(self, indent: str = '') -> str:
        """
        Generate human-readable summary of the Hodge Laplacian.

        Shows per-degree information including matrix dimensions and which
        components are cached, along with nnz counts for cached matrices.

        Parameters
        ----------
        indent : str, optional
            String prepended to every line.

        Returns
        -------
        summary : str
        """
        lines = []

        lines.append('Hodge Laplacian:')
        lines.append('-' * 60)
        lines.append(
            f"{'k':<4} {'N_k':<6} "
            f"{'lower':<14} {'upper':<14} {'full':<14}"
        )
        lines.append('-' * 60)

        for k in self.degrees:
            N_k = self._sc.num_simplices[k]
            parts = []
            for comp in _VALID_COMPONENTS:
                key = (k, comp)
                if key in self._laplacian_cache:
                    parts.append(f"nnz={self._laplacian_cache[key].nnz}")
                else:
                    parts.append('--')

            lines.append(
                f"{k:<4} {N_k:<6} "
                f"{parts[0]:<14} {parts[1]:<14} {parts[2]:<14}"
            )

        lines.append('-' * 60)
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
        if component not in _VALID_COMPONENTS:
            raise ValueError(
                f"Unknown component '{component}'. "
                f"Valid components: {_VALID_COMPONENTS}."
            )

    def _compute_laplacian(self, k: int, component: str) -> None:
        """
        Compute and cache Laplacian component(s) at degree k.

        When 'full' is requested, lower and upper are computed and cached
        as well, since they are produced as intermediates. When 'lower'
        or 'upper' is requested individually, only that component is computed.
        """
        D_k = self._sc.incidence_matrix(k)
        D_kp1 = self._sc.incidence_matrix(k + 1)
        G_km1 = self._cm[k - 1]
        G_k = self._cm[k]
        G_kp1 = self._cm[k + 1]

        if component == 'lower':
            if (k, 'lower') not in self._laplacian_cache:
                self._laplacian_cache[(k, 'lower')] = (
                    compute_laplacian_lower_matrix(D_k, G_km1, G_k)
                )

        elif component == 'upper':
            if (k, 'upper') not in self._laplacian_cache:
                self._laplacian_cache[(k, 'upper')] = (
                    compute_laplacian_upper_matrix(D_kp1, G_k, G_kp1)
                )

        elif component == 'full':
            # Compute both components if not already cached
            if (k, 'lower') not in self._laplacian_cache:
                self._laplacian_cache[(k, 'lower')] = (
                    compute_laplacian_lower_matrix(D_k, G_km1, G_k)
                )
            if (k, 'upper') not in self._laplacian_cache:
                self._laplacian_cache[(k, 'upper')] = (
                    compute_laplacian_upper_matrix(D_kp1, G_k, G_kp1)
                )
            self._laplacian_cache[(k, 'full')] = (
                (self._laplacian_cache[(k, 'lower')]
                 + self._laplacian_cache[(k, 'upper')]).tocsr()
            )

    def _compute_content_hash(self) -> str:
        """
        Compute SHA-256 hash from sc and cm content hashes.

        The Hodge Laplacian is a deterministic function of the topology
        (SimplicialComplex) and the metric (CochainMetric), so its identity
        is uniquely determined by their content hashes.
        """
        hasher = hashlib.sha256()
        hasher.update(self._sc.content_hash.encode('utf-8'))
        hasher.update(self._cm.content_hash.encode('utf-8'))
        return hasher.hexdigest()