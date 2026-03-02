"""
Cochain metric tensors on a simplicial complex.

Provides CochainMetric, the primary Stage 2 output object. Bundles per-degree
MetricTensor instances {G^k}_{k=0}^n with provenance metadata and HDF5 I/O.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from scipy import sparse

from holspec.utilities import save_h5, read_h5, join_h5_group

from .metric_tensor import MetricTensor
from .metric_models import construct_cochain_metric_from_config
from .validation import validate_cochain_metric

if TYPE_CHECKING:
    from holspec.simplicial import SimplicialComplex
    from holspec.point_data import PointData


# =============================================================================
# CochainMetric
# =============================================================================

class CochainMetric:
    """
    Collection of metric tensors {G^k}_{k=0}^n on cochain spaces.

    The primary output object of pipeline Stage 2. Bundles per-degree metric
    tensors with provenance metadata and supports HDF5 serialization. Validates
    the cochain metric contract at construction and enforces it through all
    access methods.

    Parameters
    ----------
    metric_tensors : dict[int, MetricTensor]
        Mapping from degree k to metric tensor G^k. Degrees must form a
        consecutive sequence 0, 1, ..., n.
    metadata : dict, optional
        Provenance and construction information. 'creation_time' is
        auto-populated if not provided.

    Notes
    -----
    - Calls validate_cochain_metric at construction; raises on invalid input.
    - Metric tensors are stored internally sorted by degree.
    - Content hash is computed lazily from per-degree matrix data and cached.
    - Use from_simplicial_complex_and_point_data for pipeline construction;
      the constructor is for direct assembly (e.g. loading from file).
    """

    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(
        self,
        metric_tensors: dict[int, MetricTensor],
        metadata: dict | None = None,
    ):
        # Store sorted by degree
        self._metric_tensors: dict[int, MetricTensor] = dict(sorted(metric_tensors.items()))

        # Extract dimensions for validation and quick access
        self._dimensions: dict[int, int] = {
            k: G_k.size for k, G_k in self._metric_tensors.items()
        }
        validate_cochain_metric(self._metric_tensors, self._dimensions)

        # Metadata
        self.metadata: dict = metadata if metadata is not None else {}
        if 'creation_time' not in self.metadata:
            self.metadata['creation_time'] = datetime.now().isoformat()

        self._hash_cache: str | None = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def degrees(self) -> list[int]:
        """Sorted list of degrees k for which G^k is defined."""
        return list(self._metric_tensors.keys())

    @property
    def max_dim(self) -> int:
        """Maximum degree n."""
        return max(self._metric_tensors.keys())

    @property
    def dimensions(self) -> dict[int, int]:
        """Dimension N_k = dim(C^k) at each degree."""
        return dict(self._dimensions)

    @property
    def all_diagonal(self) -> bool:
        """True if every stored MetricTensor is diagonal.

        Provides a global optimization flag for downstream operators: when True,
        all metric applications reduce to elementwise multiplication.
        """
        return all(G_k.is_diagonal for G_k in self._metric_tensors.values())

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of per-degree metric data.

        Computed lazily and cached. Based on diagonal arrays for diagonal
        metrics; uniquely identifies the metric collection independent of
        metadata.
        """
        if self._hash_cache is None:
            self._hash_cache = self._compute_content_hash()
        return self._hash_cache

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self, cochain_dimensions: dict[int, int] | None = None) -> None:
        """
        Validate the cochain metric collection.

        Parameters
        ----------
        cochain_dimensions : dict[int, int], optional
            Expected dimension N_k at each degree k. When provided, cross-validates
            each MetricTensor.size against the supplied dimensions, which is useful
            for checking consistency with a (re-)loaded SimplicialComplex.
            When omitted, validates self-consistency only (degrees consecutive,
            sizes internally consistent).
        """
        dimensions = cochain_dimensions if cochain_dimensions is not None else self._dimensions
        validate_cochain_metric(self._metric_tensors, dimensions)

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
        Save cochain metric to HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        mode : {'replace', 'update', 'create'}, default='replace'
            How to handle an existing file/group.
        group : str, optional
            HDF5 group path for the data. If None, saves at root level.
            Useful for co-locating with a SimplicialComplex in one file.
        hdf5_options : dict, optional
            HDF5 compression options. Default: gzip level 4.

        Returns
        -------
        content_hash : str
            Content hash of the saved collection for verification.

        Format
        ------
        - Root attributes: max_dim, all_diagonal, content_hash, metadata.
        - degree_{k}/ subgroup per degree:
            - attributes: size, is_diagonal.
            - dataset diagonal_elements (N_k,): diagonal of G^k.

        Raises
        ------
        NotImplementedError
            If any metric tensor is not diagonal.
        """
        filepath = Path(filepath)
        if hdf5_options is None:
            hdf5_options = {'compression': 'gzip', 'compression_opts': 4}

        # Root-level attributes
        root_attributes = {
            'max_dim': self.max_dim,
            'all_diagonal': self.all_diagonal,
            'content_hash': self.content_hash,
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

        # Per-degree subgroups
        for k, G_k in self._metric_tensors.items():
            if not G_k.is_diagonal:
                raise NotImplementedError(
                    f"save is not yet implemented for non-diagonal metric tensors "
                    f"(degree {k})."
                )
            G_k_attributes = {'size': G_k.size, 'is_diagonal': G_k.is_diagonal}
            G_k_datasets = {'diagonal_elements': G_k.matrix.diagonal()}
            save_h5(
                filepath,
                datasets=G_k_datasets,
                attributes=G_k_attributes,
                mode=mode,
                group=join_h5_group(group, f'degree_{k}'),
                hdf5_options=hdf5_options,
            )

        return self.content_hash

    @classmethod
    def load(
        cls,
        filepath: str | Path,
        validate_hash: bool = True,
        group: str | None = None,
    ) -> CochainMetric:
        """
        Load cochain metric from HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Path to HDF5 file created by save().
        validate_hash : bool, default=True
            Whether to verify content hash after loading.
        group : str, optional
            HDF5 group path for the data. Must match the group used in save().

        Returns
        -------
        cochain_metric : CochainMetric

        Raises
        ------
        ValueError
            If file format is invalid or hash validation fails.
        NotImplementedError
            If any stored metric tensor is not diagonal.
        """
        filepath = Path(filepath)

        # Read root-level attributes
        _, root_attributes = read_h5(filepath, group=group)
        if 'max_dim' not in root_attributes:
            raise ValueError(f"Missing 'max_dim' in {filepath}")

        max_dim = root_attributes['max_dim']
        metadata = root_attributes.get('metadata', {})

        # Reconstruct per-degree metric tensors
        from .metric_models import construct_diagonal_metric
        metric_tensors: dict[int, MetricTensor] = {}
        for k in range(max_dim + 1):
            metric_tensor_datasets, metric_tensor_attributes = read_h5(
                filepath, group=join_h5_group(group, f'degree_{k}')
            )
            if 'size' not in metric_tensor_attributes:
                raise ValueError(
                    f"Missing 'size' attribute in degree_{k} group of {filepath}"
                )
            if 'is_diagonal' not in metric_tensor_attributes:
                raise ValueError(
                    f"Missing 'is_diagonal' attribute in degree_{k} group of {filepath}"
                )
            if not metric_tensor_attributes['is_diagonal']:
                raise NotImplementedError(
                    f"load is not yet implemented for non-diagonal metric tensors "
                    f"(degree {k})."
                )
            if 'diagonal_elements' not in metric_tensor_datasets:
                raise ValueError(
                    f"Missing 'diagonal_elements' dataset in degree_{k} "
                    f"group of {filepath}"
                )

            metric_tensors[k] = construct_diagonal_metric(
                metric_tensor_datasets['diagonal_elements']
            )

        cm = cls(metric_tensors, metadata=metadata)

        if validate_hash:
            if 'content_hash' not in root_attributes:
                raise ValueError(
                    f"Missing 'content_hash' in {filepath}, cannot validate"
                )
            stored_hash = str(root_attributes['content_hash'])
            if cm.content_hash != stored_hash:
                raise ValueError(
                    f"Content hash mismatch in {filepath}: "
                    f"stored {stored_hash}, computed {cm.content_hash}"
                )

        return cm

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_simplicial_complex_and_point_data(
        cls,
        sc: SimplicialComplex,
        point_data: PointData,
        config: dict,
        metadata: dict | None = None,
    ) -> CochainMetric:
        """
        Construct a CochainMetric from a SimplicialComplex, PointData, and config.

        Parameters
        ----------
        sc : SimplicialComplex
            Source simplicial complex defining the cochain spaces.
        point_data : PointData
            Point cloud data. Required by geometric metric models; unused in
            the combinatorial special case.
        config : dict
            Metric model config. Must contain 'model' (str) and 'params' (dict).
            Example: {'model': 'combinatorial', 'params': {}}
        metadata : dict, optional
            Additional provenance metadata. Caller-supplied values take
            precedence over auto-populated keys. 

        Returns
        -------
        cochain_metric : CochainMetric

        Notes
        -----
        Auto-populated metadata keys (overridden by caller if present):

        - 'metric_model': config['model']
        - 'metric_model_config': config
        - 'input_hash': sc.content_hash, identifying the source complex
        
        To record the file path of the source complex for downstream
        provenance, pass ``metadata={'input_file': str(sc_filepath)}``.
        """
        metric_tensors = construct_cochain_metric_from_config(config, sc, point_data)

        cm_metadata = {
            'metric_model': config['model'],
            'metric_model_config': config,
            'input_hash': sc.content_hash,
        }
        if metadata is not None:
            cm_metadata.update(metadata)

        return cls(metric_tensors, metadata=cm_metadata)

    # =========================================================================
    # Protocols and Utilities
    # =========================================================================

    def __getitem__(self, k: int) -> MetricTensor:
        """Return metric tensor G^k at degree k.

        Parameters
        ----------
        k : int
            Degree. Valid range: -1 <= k <= max_dim+1.

        Returns
        -------
        MetricTensor

        Raises
        ------
        KeyError
            If k is not in [-1, max_dim+1].

        Notes
        -----
        For interior degrees 0 <= k <= max_dim, returns the stored metric tensor.
        For boundary degrees k=-1 and k=max_dim+1, returns the unique 0x0 metric
        tensor on the zero vector space, constructed on demand.
        """
        if k == -1 or k == self.max_dim + 1:
            # Boundary-degree metric: unique metric on the zero cochain space.
            # Constructed on demand; trivially cheap (no diagonal to cache).
            return MetricTensor(sparse.csr_matrix((0, 0)), is_diagonal=True)
        if k not in self._metric_tensors:
            raise KeyError(
                f"No metric tensor at degree {k}. Available degrees: {self.degrees}"
            )
        return self._metric_tensors[k]

    def __iter__(self):
        """Iterate over degrees."""
        return iter(self._metric_tensors)

    def __len__(self) -> int:
        """Number of degrees (= max_dim + 1)."""
        return len(self._metric_tensors)

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        sizes = list(self._dimensions.values())
        return (
            f"CochainMetric(max_dim={self.max_dim}, "
            f"dimensions={sizes}, "
            f"all_diagonal={self.all_diagonal}, "
            f"hash={self.content_hash[:8]})"
        )

    def summary(self, indent: str = '') -> str:
        """
        Generate human-readable summary of the metric collection.

        Parameters
        ----------
        indent : str, optional
            String prepended to every line. Default is '' (no indent).
            Common values: '  ' (two spaces) or '    ' (four spaces).

        Returns
        -------
        summary : str
            Multi-line string with degree, dimension, diagonality, and diagonal elements.

        Notes
        -----
        Useful for debugging and quick inspection in notebooks.
        """
        lines = []

        lines.append('Metric Tensors:')
        lines.append("-" * 48)
        lines.append(f"{'k':<4} {'N_k':<6} {'is_diagonal':<14} {'diagonal elements'}")
        lines.append("-" * 48)

        for k in self.degrees:
            G_k = self._metric_tensors[k]
            diag = G_k.matrix.diagonal()
            if G_k.size > 3:
                vals = ', '.join(f'{v:g}' for v in diag[:3])
                diag_str = f'[{vals}, ...]'
            else:
                vals = ', '.join(f'{v:g}' for v in diag)
                diag_str = f'[{vals}]'
            lines.append(f"{k:<4} {G_k.size:<6} {str(G_k.is_diagonal):<14} {diag_str}")

        lines.append("-" * 48)
        lines.append(f"content_hash: {self.content_hash[:16]}")
        lines.append("")

        return '\n'.join(indent + line for line in lines)

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _compute_content_hash(self) -> str:
        """
        Compute SHA-256 hash of per-degree metric data.

        For diagonal metrics, hashes the diagonal array at each degree in
        sorted order. Degree index is included as a 4-byte little-endian
        integer to prevent collisions across different degree structures.

        Notes
        -----
        Non-diagonal metrics are not yet supported; only the matrix diagonal
        is used, which would be insufficient for a general SPD matrix.
        """
        hasher = hashlib.sha256()
        for k in sorted(self._metric_tensors.keys()):
            hasher.update(k.to_bytes(4, byteorder='little'))
            hasher.update(self._metric_tensors[k].matrix.diagonal().tobytes())
        return hasher.hexdigest()