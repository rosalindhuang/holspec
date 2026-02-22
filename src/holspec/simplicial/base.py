"""
Base simplicial complex class.

Provides the core SimplicialComplex class for representing
and manipulating oriented abstract simplicial complexes.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import hashlib
from typing import TYPE_CHECKING
import numpy as np
from scipy import sparse

from holspec.utilities import save_h5, read_h5, convert_numpy_to_python
from .simplicial_constructions import construct_complex_from_config

if TYPE_CHECKING:
    from holspec.point_data import PointData

class SimplicialComplex:
    """
    Oriented abstract simplicial complex.

    An immutable container storing simplices at all dimensions with methods
    for computing incidence matrices and validating structure.

    Parameters
    ----------
    simplices : dict[int, list[tuple]]
        Dictionary mapping dimension k to list of k-simplices.
        Each k-simplex is a tuple of vertex indices in sorted order.
    metadata : dict, optional
        Provenance and construction information.
    validate : bool, default=True
        Whether to validate structure on construction.

    Attributes
    ----------
    simplices : dict[int, list[tuple]]
        The simplices at each dimension (read-only).
    metadata : dict
        Construction metadata.

    Notes
    -----
    - Simplices are stored in canonical form (sorted vertex tuples).
    - Vertex indices must be consecutive integers from 0 to num_vertices-1.
    - Incidence matrices are computed lazily and cached.
    - Content hash computed lazily from simplex structure.
    """

    # =========================================================================
    # Construction and Validation
    # =========================================================================

    def __init__(
        self,
        simplices: dict[int, list[tuple]],
        metadata: dict | None = None,
        validate: bool = True
    ):
        # Store simplices in native Python types (defensive copy)
        self._simplices = {k: list(simps) for k, simps in simplices.items()}
        self._simplices = convert_numpy_to_python(self._simplices)
        
        # Initialize metadata
        self.metadata = metadata if metadata is not None else {}
        if 'creation_time' not in self.metadata:
            self.metadata['creation_time'] = datetime.now().isoformat()
        
        # Initialize caches
        self._incidence_cache = {}
        self._hash_cache = None
        
        # Validate if requested
        if validate:
            self.validate()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def simplices(self) -> dict[int, list[tuple]]:
        """Simplices at each dimension (read-only)."""
        return {k: list(simps) for k, simps in self._simplices.items()}

    @property
    def max_dim(self) -> int:
        """Maximal simplex dimension."""
        return max(self._simplices.keys())

    @property
    def num_vertices(self) -> int:
        """Number of vertices (0-simplices)."""
        return len(self._simplices[0])

    @property
    def num_simplices(self) -> dict[int, int]:
        """Dictionary of simplex counts at each dimension."""
        return {k: len(simps) for k, simps in self._simplices.items()}

    @property
    def f_vector(self) -> list[int]:
        """
        Face vector: [n_0, n_1, ..., n_dim].
        Standard combinatorial topology notation.
        """
        return [len(self._simplices[k]) for k in range(self.max_dim + 1)]

    @property
    def euler_characteristic(self) -> int:
        """
        Euler characteristic: χ = Σ (-1)^k n_k.
        Topological invariant of the simplicial complex.
        """
        return sum((-1)**k * len(self._simplices[k]) 
                   for k in range(self.max_dim + 1))

    @property
    def content_hash(self) -> str:
        """
        SHA-256 hash of simplex structure.

        Computed lazily and cached. Enables reproducibility tracking
        and verification of simplicial complex identity across saves/loads.

        Notes
        -----
        Hash is based on canonical representation of all simplices,
        independent of metadata or incidence matrix cache.
        """
        if self._hash_cache is None:
            self._hash_cache = self._compute_content_hash()
        return self._hash_cache

    # =========================================================================
    # Core Mathematical Methods
    # =========================================================================

    def incidence_matrix(self, k: int) -> sparse.csr_matrix:
        """
        Get incidence matrix D_k: C_k -> C_{k-1} with lazy computation.

        Parameters
        ----------
        k : int
            Dimension of domain (k-simplices).

        Returns
        -------
        D_k : sparse.csr_matrix, shape (n_{k-1}, n_k)
            Boundary operator matrix.

        Notes
        -----
        - Returns cached result if available.
        - D_k[i, j] = ±1 if simplex j has face i, 0 otherwise.
        - Sign determined by orientation convention.
        - This is the primary method; use boundary_matrix() for alternative naming.
        """
        if k not in self._incidence_cache:
            from .incidence import compute_incidence_matrix
            self._incidence_cache[k] = compute_incidence_matrix(
                self._simplices, k
            )
        return self._incidence_cache[k]

    def boundary_matrix(self, k: int) -> sparse.csr_matrix:
        """
        Get boundary operator matrix (alias for incidence_matrix).

        Provided for users familiar with boundary operator notation.
        Calls incidence_matrix(k) internally.
        """
        return self.incidence_matrix(k)

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self) -> None:
        """
        Comprehensive validation of simplicial complex structure.

        Raises
        ------
        ValueError
            If any structural invariant is violated.

        Notes
        -----
        - Called automatically on construction if validate=True.
        - Can be called manually after loading from file.
        - Does NOT check boundary property (D_k @ D_{k+1} = 0) as that requires
          computing all incidence matrices (expensive).
        """
        from .validation import (
            validate_simplices_structure,
            validate_face_closure
        )
        validate_simplices_structure(self._simplices)
        validate_face_closure(self._simplices)

    def check_boundary_property(self, tol: float = 1e-10) -> dict[int, bool]:
        """
        Verify D_k @ D_{k+1} = 0 for all k.

        Parameters
        ----------
        tol : float, default=1e-10
            Numerical tolerance for zero comparison.

        Returns
        -------
        results : dict[int, bool]
            {k: passed} for each applicable dimension.

        Notes
        -----
        - Fundamental property of simplicial complexes.
        - Computationally expensive: computes all incidence matrices.
        - Primarily useful for testing correctness of implementation.
        - NOT called during validate() due to computational cost.
        - Pre-populates cache before validation to avoid redundant computation.
        """
        from .validation import check_boundary_property
        
        # Pre-compute all incidence matrices to populate cache and pass to validation
        for k in range(self.max_dim + 1):
            _ = self.incidence_matrix(k)
        
        # Pass cache to validation function
        return check_boundary_property(
            self._simplices, 
            tol,
            incidence_matrices=self._incidence_cache
        )

    # =========================================================================
    # I/O Methods
    # =========================================================================
    
    def save(
        self,
        filepath: str | Path,
        save_incidence: bool = False,
        mode: str = 'replace',
        group: str | None = None,
        hdf5_options: dict | None = None
    ) -> str:
        """
        Save simplicial complex to HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        save_incidence : bool, default=False
            Whether to save cached incidence matrices.
            Only saves matrices that have been computed (in cache).
        mode : {'replace', 'update', 'create'}, default='replace'
            How to handle existing file/group (see save_h5 for details).
        group : str, optional
            HDF5 group path for the data. If None, saves at root level.
        hdf5_options : dict, optional
            Additional HDF5 options. Default: {'compression': 'gzip', 'compression_opts': 4}.
            Pass {'compression': None} to disable compression.

        Returns
        -------
        content_hash : str
            Content hash of saved simplicial complex for verification.

        Format
        ------
        - Root attributes: max_dim, f_vector, content_hash, has_incidence, metadata
        - /simplices/{k}-simplices: datasets containing k-simplices as (n_k, k+1) arrays
        - /incidence/k/: subgroups with CSR components (if save_incidence=True)
        """
        filepath = Path(filepath)
        
        # Set default compression if not specified
        if hdf5_options is None:
            hdf5_options = {'compression': 'gzip', 'compression_opts': 4}
        
        # Helper to construct subgroup paths
        def subgroup_path(subpath: str) -> str:
            return f"{group}/{subpath}" if group else subpath
        
        # Prepare root-level attributes
        root_attributes = {
            'max_dim': self.max_dim,
            'f_vector': self.f_vector,
            'content_hash': self.content_hash,
            'has_incidence': save_incidence and bool(self._incidence_cache),
            'metadata': self.metadata
        }
        
        # Prepare simplices datasets with subgroup paths
        simplices_datasets = {
            f'simplices/{k}-simplices': np.array(simps, dtype=int)
            for k, simps in self._simplices.items()
        }
        
        # Save root attributes and simplices
        save_h5(
            filepath,
            datasets=simplices_datasets,
            attributes=root_attributes,
            mode=mode,
            group=group,
            hdf5_options=hdf5_options
        )
        
        # Save incidence matrices to subgroups (if requested)
        if save_incidence and self._incidence_cache:
            for k, D_k in self._incidence_cache.items():
                # Convert to CSR format
                D_k_csr = D_k.tocsr()
                
                # Prepare datasets and attributes for this incidence matrix
                inc_datasets = {
                    'data': D_k_csr.data,
                    'indices': D_k_csr.indices,
                    'indptr': D_k_csr.indptr
                }
                inc_attributes = {
                    'shape': D_k_csr.shape,
                    'nnz': D_k_csr.nnz
                }
                
                save_h5(
                    filepath,
                    datasets=inc_datasets,
                    attributes=inc_attributes,
                    mode=mode,
                    group=subgroup_path(f'incidence/{k}'),
                    hdf5_options=hdf5_options
                )
        
        return self.content_hash

    @classmethod
    def load(
        cls,
        filepath: str | Path,
        group: str | None = None,
        validate_hash: bool = True,
        load_incidence: bool = True
    ) -> 'SimplicialComplex':
        """
        Load simplicial complex from HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Path to HDF5 file created by save().
        group : str, optional
            HDF5 group path for the data.
        validate_hash : bool, default=True
            Whether to verify content hash after loading.
        load_incidence : bool, default=True
            Whether to load and cache incidence matrices if available.
            If False, incidence matrices are skipped and computed on demand.

        Returns
        -------
        sc : SimplicialComplex
            Loaded simplicial complex instance.

        Raises
        ------
        ValueError
            If file format is invalid or hash validation fails.
        """
        filepath = Path(filepath)
        
        # Helper to construct subgroup paths
        def subgroup_path(subpath: str) -> str:
            return f"{group}/{subpath}" if group else subpath
        
        # Read root-level attributes
        _, attributes = read_h5(filepath, group=group)
        
        # Extract metadata and max_dim
        metadata = attributes.get('metadata', {})
        
        if 'max_dim' not in attributes:
            raise ValueError(f"Missing 'max_dim' in {filepath}")
        max_dim = attributes['max_dim']
        
        # Read all simplices from /simplices group
        datasets, _ = read_h5(filepath, group=subgroup_path('simplices'))
        
        simplices = {}
        for key, data in datasets.items():
            if key.endswith('-simplices'):
                k = int(key.split('-')[0])
                # Convert (n_k, k+1) array to list of tuples
                simplices[k] = [tuple(row) for row in data]
        
        if not simplices:
            raise ValueError(f"No simplices found in {filepath}")
        
        # Create simplicial complex without validation (data already validated when saved)
        sc = cls(simplices, metadata=metadata, validate=False)
        
        # Validate content hash if requested
        if validate_hash:
            if 'content_hash' not in attributes:
                raise ValueError(f"Missing 'content_hash' in {filepath}, cannot validate hash")
            else:
                stored_hash = str(attributes['content_hash'])
                computed_hash = sc.content_hash
                if computed_hash != stored_hash:
                    raise ValueError(
                        f"Content hash mismatch in {filepath}: "
                        f"expected {stored_hash}, got {computed_hash}"
                    )
        
        # Load incidence matrices if available and requested
        has_incidence = attributes.get('has_incidence', False)
        if has_incidence and load_incidence:
            # Try loading each possible incidence matrix
            for k in range(max_dim + 1):
                try:
                    datasets, inc_attrs = read_h5(
                        filepath, 
                        group=subgroup_path(f'incidence/{k}')
                    )
                    
                    # Extract CSR components
                    if 'data' not in datasets or 'indices' not in datasets or 'indptr' not in datasets:
                        print(f"Warning: Incomplete incidence matrix D_{k}, skipping")
                        continue
                    
                    data = datasets['data']
                    indices = datasets['indices']
                    indptr = datasets['indptr']
                    shape = tuple(inc_attrs['shape'])
                    
                    # Validate shape matches expected dimensions
                    expected_shape = (len(sc._simplices.get(k-1, [])), 
                                    len(sc._simplices.get(k, [])))
                    
                    if shape != expected_shape:
                        print(f"Warning: Incidence matrix D_{k} shape mismatch. "
                            f"Expected {expected_shape}, got {shape}. Skipping.")
                        continue
                    
                    # Reconstruct sparse matrix
                    D_k = sparse.csr_matrix((data, indices, indptr), shape=shape)
                    
                    # Store in cache
                    sc._incidence_cache[k] = D_k
                    
                except (KeyError, OSError):
                    # Group doesn't exist - this is fine, not all k may have been saved
                    continue
        
        return sc

    @classmethod
    def from_point_data(
        cls,
        point_data: 'PointData',
        config: dict,
        metadata: dict | None = None,
        validate: bool = True,
    ) -> 'SimplicialComplex':
        """
        Construct a SimplicialComplex from a PointData object and a method config.

        Parameters
        ----------
        point_data : PointData
            Input data containing positions or distances.
        config : dict
            Must contain 'method' (str) and 'params' (dict). E.g.:
              {'method': 'delaunay', 'params': {'max_dim': 2}}
              {'method': 'alpha', 'params': {'alpha': 1.5, 'max_dim': 2}}
              {'method': 'vietoris_rips', 'params': {'epsilon': 1.2, 'max_dim': 2}}
        metadata : dict, optional
            Additional metadata to store with the complex. Construction parameters
            and provenance from point_data are added automatically and will not
            overwrite keys already present in metadata.
        validate : bool, default=True
            Whether to validate the resulting complex structure.

        Returns
        -------
        sc : SimplicialComplex

        Notes
        -----
        - For 'delaunay' and 'alpha', point_data.has_positions must be True.
        - For 'vietoris_rips', either positions or distances are accepted;
          positions are preferred when available.
        - point_data.content_hash is recorded in the complex metadata for
          provenance tracking.
        """
        # Determine input arrays and record shape/type for metadata
        if point_data.has_positions:
            positions = point_data.get_positions()
            N, d = positions.shape
            input_type = 'positions'
            input_shape = (N, d)
            distances = None
        else:
            distances = point_data.get_distances()
            N = distances.shape[0]
            input_type = 'distances'
            input_shape = (N, N)
            positions = None

        # Construct raw simplices dict (pure computation, no framework objects)
        simplices = construct_complex_from_config(config, positions=positions, distances=distances)

        # Assemble metadata: caller-supplied values take precedence
        complex_metadata = {
            'construction_method': config['method'],
            'construction_config': {
                'input_type': input_type,
                'input_shape': input_shape,
                **config.get('params', {}),
            },
            'input_hash': point_data.content_hash,
        }
        if metadata is not None:
            complex_metadata.update(metadata)

        return cls(simplices, metadata=complex_metadata, validate=validate)

    # =========================================================================
    # Utilities and Protocols
    # =========================================================================

    def _compute_content_hash(self) -> str:
        """
        Compute SHA-256 hash of simplex structure.

        Returns
        -------
        hash : str
            Hexadecimal hash string.

        Notes
        -----
        Hash is based on canonical string representation of all simplices,
        ensuring consistency across different orderings.
        """
        
        # Build canonical string representation
        # Format: dimension -> sorted list of sorted simplices
        hash_parts = []
        
        for k in sorted(self._simplices.keys()):
            # Sort simplices at this dimension for canonical ordering
            sorted_simplices = sorted(self._simplices[k])
            # Convert to string representation
            hash_parts.append(f"{k}:{sorted_simplices}")
        
        # Combine into single string
        canonical_str = "|".join(hash_parts)
        
        # Compute SHA-256 hash
        return hashlib.sha256(canonical_str.encode()).hexdigest()
    
    def summary(self) -> str:
        """
        Generate human-readable summary of the complex.

        Returns
        -------
        summary : str
            Multi-line string with dimension, counts, and Euler characteristic.

        Notes
        -----
        Useful for debugging and quick inspection in notebooks.
        """
        lines = [] 
        
        # Simplices summary
        simplex_names = {
            0: "vertices",
            1: "edges",
            2: "triangles",
            3: "tetrahedra",
        }
        
        lines.append('Simplices:')
        lines.append("-" * 40)
        lines.append(f"{'k':<4} {'k-simplex':<15} {'count':<10}")
        lines.append("-" * 40)
        
        for k in range(self.max_dim + 1):
            name = simplex_names.get(k, f"{k}-simplices")
            count = len(self._simplices[k])
            lines.append(f"{k:<4} {name:<15} {count:<10}")
        
        lines.append("-" * 40)

        # Incidence matrix summary
        lines.append('\nIncidence Matrices:')
        lines.append("-" * 40)
        lines.append(f"{'k':<4} {'D_k computed':<15} {'shape':<20}")
        lines.append("-" * 40)
        
        for k in range(1, self.max_dim + 1):
            computed = k in self._incidence_cache
            if computed:
                shape = self._incidence_cache[k].shape
                lines.append(f"{k:<4} {str(computed):<15} {str(shape):<20}")
            else:
                lines.append(f"{k:<4} {str(computed):<15} {'':<20}")
        
        lines.append("-" * 40)
        lines.append("")
        
        return '\n'.join(lines)

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        return (f"SimplicialComplex(dim={self.max_dim}, "
                f"f_vector={self.f_vector}, χ={self.euler_characteristic}, hash={self.content_hash[:8]})")
    
    def __eq__(self, other: 'SimplicialComplex') -> bool:
        """
        Equality comparison based on simplices.

        Two complexes are equal if they have the same simplices
        at each dimension (regardless of ordering or metadata).
        """
        if not isinstance(other, SimplicialComplex):
            return False
        
        # Check if dimensions match
        if self._simplices.keys() != other._simplices.keys():
            return False
        
        # Check if simplices match at each dimension (order-independent)
        for k in self._simplices.keys():
            if set(self._simplices[k]) != set(other._simplices[k]):
                return False
        
        return True

    def __hash__(self) -> int:
        """
        Hash based on simplex structure.

        Enables use in sets and as dict keys for testing.

        Notes
        -----
        - Based on content_hash for consistency.
        - Can be expensive for large complexes.
        - Primarily intended for testing and small examples.
        """
        # Use first 16 hex digits (64 bits) of content hash
        return int(self.content_hash[:16], 16)
