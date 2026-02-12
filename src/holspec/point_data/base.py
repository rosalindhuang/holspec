"""
Point cloud data container.

Provides PointData class for managing point cloud data in the form 
of positions or pairwise distances.
"""

import numpy as np
from datetime import datetime
import json
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

from holspec.utilities import save_h5, read_h5, compute_content_hash
from .generators import generate_from_config

class PointData:
    """
    Container for point cloud data.

    Stores either positions or pairwise distances (not both). Provides lazy
    distance computation, content hashing, and metadata tracking for provenance.
    
    Parameters
    ----------
    positions : ndarray, shape (N, d), optional
        Point positions in d-dimensional space.
    distances : ndarray, shape (N, N), optional
        Pairwise distance matrix.
    metadata : dict, optional
        Metadata for provenance and context.
    
    Notes
    -----
    - Exactly one of positions or distances must be provided.
    - The object is immutable once created; data cannot be changed.
    - Metadata is auto-populated with 'creation_time' and may include 'config'
      (if generated) plus any custom fields.
    """
    
    # =========================================================================
    # Construction and Validation
    # =========================================================================
    
    def __init__(
        self,
        positions: np.ndarray | None = None,
        distances: np.ndarray | None = None,
        metadata: dict | None = None
    ):
        # Validate exactly one input provided
        if (positions is None) == (distances is None):
            raise ValueError("Provide exactly one of 'positions' or 'distances'")
        
        # Store and validate primary data
        if positions is not None:
            self._validate_positions(positions)
            self._positions = positions
            self._distances = None
        else:
            self._validate_distances(distances)
            self._positions = None
            self._distances = distances
        
        # Initialize caches
        self._distances_cache = self._distances  # Cache if provided directly
        self._hash_cache = None
        
        # Initialize metadata
        self.metadata = metadata if metadata is not None else {}

        # Auto-set creation time if not already present
        if 'creation_time' not in self.metadata:
            self.metadata['creation_time'] = datetime.now().isoformat()
    
    @staticmethod
    def _validate_positions(positions: np.ndarray) -> None:
        """Validate position array."""
        if positions.ndim != 2:
            raise ValueError(f"Positions must be 2D array, got shape {positions.shape}")
        if positions.shape[0] < 1:
            raise ValueError("Must have at least 1 point")
    
    @staticmethod
    def _validate_distances(distances: np.ndarray) -> None:
        """Validate distance matrix."""
        if distances.ndim != 2:
            raise ValueError(f"Distances must be 2D array, got shape {distances.shape}")
        
        N = distances.shape[0]
        if distances.shape[1] != N:
            raise ValueError("Distance matrix must be square")
        
        # Use reasonable tolerance for floating point comparisons
        if not np.allclose(distances, distances.T, rtol=1e-10, atol=1e-10):
            raise ValueError("Distance matrix must be symmetric")
        if not np.allclose(np.diag(distances), 0, atol=1e-10):
            raise ValueError("Diagonal must be zero")
        if np.any(distances < -1e-10):
            raise ValueError("Distances must be non-negative")
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def num_points(self) -> int:
        """Number of points in the cloud."""
        return len(self._positions) if self._positions is not None else len(self._distances)
    
    @property
    def dimension(self) -> int | None:
        """Ambient dimension (None if only distances available)."""
        return self._positions.shape[1] if self._positions is not None else None
    
    @property
    def has_positions(self) -> bool:
        """Whether position data is available."""
        return self._positions is not None
    
    @property
    def data_type(self) -> str:
        """Type of data stored ('positions' or 'distances')."""
        return 'positions' if self.has_positions else 'distances'
    
    @property
    def content_hash(self) -> str:
        """Unique content identifier for reproducibility."""
        if self._hash_cache is None:
            source = self._positions if self.has_positions else self._distances
            self._hash_cache = compute_content_hash(source)
        return self._hash_cache
    
    # =========================================================================
    # Data Access
    # =========================================================================
    
    def get_positions(self) -> np.ndarray:
        """
        Return positions array (N, d).
        
        Returns
        -------
        positions : ndarray, shape (N, d)
            Point positions.
        
        Raises
        ------
        ValueError
            If position data is not available.
        """
        if self._positions is None:
            raise ValueError("Position data not available")
        return self._positions
    
    def get_distances(self) -> np.ndarray:
        """
        Return or compute pairwise distance matrix (N, N).
        
        Returns
        -------
        distances : ndarray, shape (N, N)
            Pairwise distance matrix. Computed lazily from positions
            if not provided at construction.
        
        Raises
        ------
        ValueError
            If distance data is not available and cannot be computed.
        """
        if self._distances_cache is None:
            if self._positions is None:
                raise ValueError("Distance data not available")
            self._distances_cache = self._compute_distances()
        return self._distances_cache
    
    def _compute_distances(self) -> np.ndarray:
        """Compute pairwise Euclidean distances from positions."""
        return squareform(pdist(self._positions, metric='euclidean'))

    # =========================================================================
    # I/O and Factory Methods
    # =========================================================================
    
    def save(
        self,
        filepath: str | Path,
        mode: str = 'replace',
        group: str | None = None,
        hdf5_options: dict | None = None
    ) -> str:
        """
        Save to HDF5 file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to HDF5 file.
        mode : {'replace', 'append'}, default='replace'
            How to handle existing file.
        group : str, optional
            HDF5 group path for the data.
        hdf5_options : dict, optional
            Additional HDF5 options (compression, etc.).
        
        Returns
        -------
        content_hash : str
            Content hash of saved data for verification.
        
        Notes
        -----
        Saves positions if available, otherwise saves distances.
        Does not save both to avoid redundancy.
        """
        # Get data to save
        data_array = self._positions if self.has_positions else self._distances
        
        # File-level attributes (minimal technical info)
        attributes = {
            'data_type': self.data_type,
            'content_hash': self.content_hash,
            'shape': tuple(data_array.shape),
        }
        
        # Include instance metadata (creation_time, config, user fields)
        if self.metadata:
            attributes['metadata'] = json.dumps(self.metadata)
        
        # Save using utility
        save_h5(
            filepath,
            datasets={self.data_type: data_array},
            attributes=attributes,
            mode=mode,
            group=group,
            hdf5_options=hdf5_options
        )
        
        return self.content_hash

    @classmethod
    def load(
        cls,
        filepath: str | Path,
        group: str | None = None,
        validate_hash: bool = True
    ) -> 'PointData':
        """
        Load from HDF5 file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to HDF5 file.
        group : str, optional
            HDF5 group path for the data.
        validate_hash : bool, default=True
            Whether to validate content hash against data.
        
        Returns
        -------
        point_data : PointData
            Loaded PointData object.
        
        Raises
        ------
        ValueError
            If file format is invalid or hash validation fails.
        """
        # Read all datasets and attributes
        datasets, attributes = read_h5(filepath, group=group)
        
        # Determine data type
        if 'data_type' not in attributes:
            raise ValueError(f"Missing 'data_type' in attributes for {filepath}")
        data_type = attributes['data_type']
        
        # Extract appropriate data
        if data_type == 'positions':
            if 'positions' not in datasets:
                raise ValueError(f"No 'positions' dataset found in {filepath}")
            data_array = datasets['positions']
            positions = data_array
            distances = None
        elif data_type == 'distances':
            if 'distances' not in datasets:
                raise ValueError(f"No 'distances' dataset found in {filepath}")
            data_array = datasets['distances']
            positions = None
            distances = data_array
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
        
        # Validate content hash if requested
        if validate_hash:
            if 'content_hash' not in attributes:
                print(f"Warning: No 'content_hash' in attributes for {filepath}, skipping validation")
            else:
                stored_hash = attributes['content_hash']
                computed_hash = compute_content_hash(data_array, length=len(stored_hash))
                if computed_hash != stored_hash:
                    raise ValueError(
                        f"Content hash mismatch in {filepath}: "
                        f"expected {stored_hash}, got {computed_hash}"
                    )
        
        # Extract metadata
        metadata = attributes.get('metadata', {})
        
        return cls(positions=positions, distances=distances, metadata=metadata)

    @classmethod
    def from_config(cls, config: dict) -> 'PointData':
        """
        Generate point data from configuration.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary with 'generator' and 'params' keys.
        
        Returns
        -------
        point_data : PointData
            Generated PointData object with config stored in metadata.
        """
        positions = generate_from_config(config)
        return cls(positions=positions, metadata={'config': config})

    
    # =========================================================================
    # Special Methods
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation of PointData object."""
        if self.has_positions:
            return (f"PointData(N={self.num_points}, d={self.dimension}, "
                    f"hash={self.content_hash})")
        else:
            return (f"PointData(N={self.num_points}, distances_only, "
                    f"hash={self.content_hash})")
    
    def __len__(self) -> int:
        """Return number of points."""
        return self.num_points
