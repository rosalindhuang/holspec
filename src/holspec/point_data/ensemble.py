# holspec/point_data/ensemble.py
"""
Ensemble container for related point clouds.

Provides PointDataEnsemble class for managing collections of PointData objects
that share a common base configuration with variations (e.g., noise realizations).
"""

from pathlib import Path
from datetime import datetime

from holspec.point_data.base import PointData
from holspec.point_data.point_generators import generate_points_from_config
from holspec.utilities import save_h5, read_h5, add_noise


class PointDataEnsemble:
    """
    Container for ensemble of related point clouds.
    
    Stores multiple PointData objects that share a common base configuration
    with variations (e.g., different noise realizations).
    
    Parameters
    ----------
    members : list[PointData]
        List of PointData objects in the ensemble.
    base_config : dict, optional
        Base configuration for generating the point cloud.
    noise_config : dict, optional
        Configuration for noise applied to base.
    metadata : dict, optional
        Additional ensemble-level metadata.

    Notes
    -----
    - All members are stored in memory; for very large ensembles (>1000 members),
      consider processing in batches.
    - Members are typically generated with same structure but different random seeds.
    - Ensemble metadata is auto-populated with 'creation_time'.
    - Use `from_base_config()` class method for standard workflow of generating
      multiple noise realizations.
    """
    
    def __init__(
        self,
        members: list[PointData],
        base_config: dict | None = None,
        noise_config: dict | None = None,
        metadata: dict | None = None
    ):
        # Validation
        if not members:
            raise ValueError("Ensemble must contain at least one member")
        if not all(isinstance(m, PointData) for m in members):
            raise TypeError("All members must be PointData objects")
        
        # Store data
        self.members = members
        self.base_config = base_config
        self.noise_config = noise_config
        self.metadata = metadata if metadata is not None else {}
        
        # Auto-set creation time if not present
        if 'creation_time' not in self.metadata:
            self.metadata['creation_time'] = datetime.now().isoformat()
    

    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def size(self) -> int:
        """Number of ensemble members."""
        return len(self.members)
    
    @property
    def num_points(self) -> int:
        """Number of points per member (assumes all members have same N)."""
        return self.members[0].num_points
    
    @property
    def dimension(self) -> int | None:
        """Ambient dimension (None if only distances available)."""
        return self.members[0].dimension
    

    # =========================================================================
    # I/O and Factory Methods
    # =========================================================================
    
    def save(
        self,
        filepath: str | Path,
        mode: str = 'replace',
    ) -> None:
        """
        Save ensemble to HDF5 file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to HDF5 file.
        mode : str, default='replace'
            Write mode passed to save_h5. One of 'replace', 'update', or 'create'.
        
        Notes
        -----
        Creates hierarchical structure:
        - Root attributes: ensemble_size, base_config, noise_config, metadata
        - Groups: member_0000, member_0001, ... (one per ensemble member)
        
        Each member saved using PointData.save() with mode='replace'.
        """
        # Prepare ensemble-level attributes
        attributes = {
            'ensemble_size': self.size,
        }
        
        if self.base_config is not None:
            attributes['base_config'] = self.base_config
        
        if self.noise_config is not None:
            attributes['noise_config'] = self.noise_config
        
        if self.metadata:
            attributes['metadata'] = self.metadata
        
        # Save root-level attributes
        save_h5(filepath, attributes=attributes, group=None, mode=mode)
        
        # Save each member to a numbered group
        for i, point_data in enumerate(self.members):
            group_name = f"member_{i:04d}"
            point_data.save(filepath, group=group_name, mode='replace')
    

    @classmethod
    def load(cls, filepath: str | Path) -> 'PointDataEnsemble':
        """
        Load ensemble from HDF5 file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to ensemble HDF5 file.
        
        Returns
        -------
        ensemble : PointDataEnsemble
            Loaded ensemble with all members.
        
        Raises
        ------
        ValueError
            If file format is invalid or required attributes missing.
        """
        filepath = Path(filepath)
        
        # Read root-level attributes 
        _, attributes = read_h5(filepath, group=None)
        
        # Extract and validate ensemble metadata
        if 'ensemble_size' not in attributes:
            raise ValueError("Missing 'ensemble_size' in root attributes")
        
        ensemble_size = attributes['ensemble_size']
        base_config = attributes.get('base_config')  # Already parsed from JSON by read_h5
        noise_config = attributes.get('noise_config')
        metadata = attributes.get('metadata', {})
        
        # Load all members
        members = []
        for i in range(ensemble_size):
            group_name = f"member_{i:04d}"
            point_data = PointData.load(filepath, group=group_name)
            members.append(point_data)
        
        # Construct and return ensemble
        return cls(
            members=members,
            base_config=base_config,
            noise_config=noise_config,
            metadata=metadata
        )


    @classmethod
    def from_base_config(
        cls,
        base_config: dict,
        noise_config: dict,
        num_realizations: int,
        base_seed: int = 42,
        metadata: dict | None = None,
        member_metadata: dict | None = None,
    ) -> 'PointDataEnsemble':
        """
        Generate ensemble from base config with noise.
        
        Parameters
        ----------
        base_config : dict
            Base point cloud configuration (for generate_from_config).
            Must have 'generator' and 'params' keys.
        noise_config : dict
            Noise parameters. Must include 'scale'.
            Optional: 'distribution' (default 'uniform').
        num_realizations : int
            Number of ensemble members to generate.
        base_seed : int, default=42
            Starting seed. Member i uses seed = base_seed + i.
        member_metadata : dict, optional
            Additional metadata applied uniformly to all members.
            Merged with the per-member metadata (seed, member_index, etc.).
        
        Returns
        -------
        ensemble : PointDataEnsemble
            Generated ensemble with noise realizations.
        
        Examples
        --------
        >>> base_config = {
        ...     'generator': 'trilatthex',
        ...     'params': {'n_rings': 4, 'spacing': 1.0}
        ... }
        >>> noise_config = {'scale': 0.1, 'distribution': 'uniform'}
        >>> ensemble = PointDataEnsemble.from_base_config(
        ...     base_config, noise_config, num_realizations=100
        ... )
        """
        # Validate inputs
        if num_realizations < 1:
            raise ValueError("num_realizations must be at least 1")
        if 'scale' not in noise_config:
            raise ValueError("noise_config must include 'scale'")
        
        # Generate base positions once
        base_positions = generate_points_from_config(base_config)
        
        # Generate noise realizations
        members = []
        for i in range(num_realizations):
            seed = base_seed + i
            
            # Add noise
            noisy_positions = add_noise(
                base_positions,
                scale=noise_config['scale'],
                distribution=noise_config.get('distribution', 'uniform'),
                seed=seed
            )
            
            # Create PointData with comprehensive metadata
            member_metadata_all = dict(member_metadata) if member_metadata is not None else {}
            member_metadata_all.update({
                'base_config': base_config,
                'noise_config': noise_config,
                'seed': seed,
                'member_index': i,
            })
            point_data = PointData(positions=noisy_positions, metadata=member_metadata_all)
            members.append(point_data)
        
        # Construct and return ensemble
        ensemble_metadata = dict(metadata) if metadata is not None else {}
        ensemble_metadata['base_seed'] = base_seed

        return cls(
            members=members,
            base_config=base_config,
            noise_config=noise_config,
            metadata=ensemble_metadata,
        )
    

    # =========================================================================
    # Utilities and Protocols
    # =========================================================================
    
    def __len__(self) -> int:
        """Return number of ensemble members."""
        return self.size
    
    def __getitem__(self, index: int) -> PointData:
        """
        Get ensemble member by index.
        
        Parameters
        ----------
        index : int
            Member index (0 to size-1).
        
        Returns
        -------
        PointData
            The requested ensemble member.
        """
        return self.members[index]
    
    def __iter__(self):
        """Iterate over ensemble members."""
        return iter(self.members)
    
    def __repr__(self) -> str:
        """String representation of ensemble."""
        return (f"PointDataEnsemble(size={self.size}, "
                f"N={self.num_points}, d={self.dimension})")
