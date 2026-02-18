"""
Base simplicial complex class.

Provides the core SimplicialComplex class for representing
and manipulating oriented abstract simplicial complexes.
"""

from datetime import datetime
from pathlib import Path
import numpy as np
from scipy import sparse


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
        Construction metadata (read-only).

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
        # Store simplices (defensive copy)
        self._simplices = {k: list(simps) for k, simps in simplices.items()}
        
        # Initialize metadata
        self._metadata = metadata if metadata is not None else {}
        if 'creation_time' not in self._metadata:
            self._metadata['creation_time'] = datetime.now().isoformat()
        
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
        return self._simplices

    @property
    def metadata(self) -> dict:
        """Construction metadata (read-only)."""
        return self._metadata

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
        Topological invariant of the complex.
        """
        return sum((-1)**k * len(self._simplices[k]) 
                   for k in range(self.max_dim + 1))

    @property
    def content_hash(self) -> str:
        """
        SHA-256 hash of simplex structure.

        Computed lazily and cached. Enables reproducibility tracking
        and verification of complex identity across saves/loads.

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
    # Validation and Testing Utilities
    # =========================================================================

    def validate(self) -> None:
        """
        Comprehensive validation of complex structure.

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
        
        for k in range(1, self.max_dim + 2):
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

    # =========================================================================
    # I/O Methods
    # =========================================================================

    def save(self, filepath: str | Path) -> None:
        """
        Save complex to HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Output file path.

        Format
        ------
        - /simplices/k: datasets containing k-simplices as (n_k, k+1) arrays
        - /metadata: attributes containing metadata dict
        - /f_vector: dataset with face counts
        - /content_hash: attribute with structure hash
        """
        pass

    @classmethod
    def load(
        cls,
        filepath: str | Path,
        validate_hash: bool = True
    ) -> 'SimplicialComplex':
        """
        Load complex from HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Path to HDF5 file created by save().
        validate_hash : bool, default=True
            Whether to verify content hash after loading.

        Returns
        -------
        complex : SimplicialComplex
            Loaded complex instance.

        Raises
        ------
        ValueError
            If validate_hash=True and hash mismatch detected.
        """
        pass

    # =========================================================================
    # Private Utilities
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
        import hashlib
        
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