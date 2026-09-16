# holspec/point_data/ensemble.py
"""
Ensemble container for related point clouds.

Provides PointDataEnsemble class for managing compatible collections of
PointData objects.
"""

from pathlib import Path
from datetime import datetime

import numpy as np

from holspec.point_data.base import PointData
from holspec.point_data.data_generators import generate_points_from_config
from holspec.utilities import save_h5, read_h5, add_noise


class PointDataEnsemble:
    """
    Container for ensemble of related point clouds.

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
    reference_point_data : PointData, optional
        Reference/base point cloud for the ensemble (e.g. the unperturbed
        positions underlying a noisy ensemble). When provided, it must match
        the member data type and, for position data, the ambient dimension.

    Notes
    -----
    - All members are stored in memory; for very large ensembles (>1000 members),
      consider processing in batches.
    - Members must share a data type. Position-data members must also share
      an ambient dimension, but may have different point counts.
    - Generated members are typically the same size with different random seeds.
    - Ensemble metadata is auto-populated with 'creation_time'.
    - Use `from_base_config()` class method for standard workflow of generating
      multiple noise realizations.
    """

    def __init__(
        self,
        members: list[PointData],
        base_config: dict | None = None,
        noise_config: dict | None = None,
        metadata: dict | None = None,
        reference_point_data: PointData | None = None,
    ):
        # Validation
        if not members:
            raise ValueError("Ensemble must contain at least one member")
        if not all(isinstance(m, PointData) for m in members):
            raise TypeError("All members must be PointData objects")
        _validate_member_compatibility(members, context="Ensemble")

        # Validate reference compatibility
        if reference_point_data is not None:
            _validate_reference_compatibility(reference_point_data, members[0])

        # Store data
        self.members = members
        self.base_config = base_config
        self.noise_config = noise_config
        self.metadata = metadata if metadata is not None else {}
        self.reference_point_data = reference_point_data

        # Auto-set creation time if not present
        if "creation_time" not in self.metadata:
            self.metadata["creation_time"] = datetime.now().isoformat()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def size(self) -> int:
        """Number of ensemble members."""
        return len(self.members)

    @property
    def num_points(self) -> int:
        """
        Shared number of points per member.

        Raises
        ------
        ValueError
            If ensemble members have variable point counts. Use
            ``num_points_per_member`` for per-member sizes.
        """
        if not self.has_uniform_num_points:
            raise ValueError(
                "Ensemble members have variable numbers of points; use "
                "num_points_per_member for per-member sizes"
            )
        return self.members[0].num_points

    @property
    def num_points_per_member(self) -> tuple[int, ...]:
        """Number of points in each ensemble member."""
        return tuple(member.num_points for member in self.members)

    @property
    def has_uniform_num_points(self) -> bool:
        """Whether all ensemble members have the same number of points."""
        return len(set(self.num_points_per_member)) == 1

    @property
    def dimension(self) -> int | None:
        """Ambient dimension (None if only distances available)."""
        return self.members[0].dimension

    @property
    def has_reference(self) -> bool:
        """Whether a reference/base point cloud is available."""
        return self.reference_point_data is not None

    def get_reference_positions(self) -> np.ndarray:
        """
        Return the reference/base positions for the ensemble.

        Returns
        -------
        positions : ndarray, shape (N, d)
            Reference point positions.

        Raises
        ------
        ValueError
            If no reference point data exists, or if the reference is
            distance-only and positions are unavailable.
        """
        if self.reference_point_data is None:
            raise ValueError("No reference point data available for this ensemble")
        if not self.reference_point_data.has_positions:
            raise ValueError(
                "Reference point data is distance-only; positions are unavailable"
            )
        return self.reference_point_data.get_positions()

    # =========================================================================
    # I/O and Factory Methods
    # =========================================================================

    def save(
        self,
        filepath: str | Path,
        mode: str = "replace",
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
        - Root attributes: ensemble_size, has_reference, base_config,
          noise_config, metadata
        - Groups: member_0000, member_0001, ... (one per ensemble member)
        - Group: reference (only when a reference point cloud is present)

        Each member saved using PointData.save() with mode='replace'.
        """
        # Prepare ensemble-level attributes
        attributes = {
            "ensemble_size": self.size,
            "has_reference": self.has_reference,
        }

        if self.base_config is not None:
            attributes["base_config"] = self.base_config

        if self.noise_config is not None:
            attributes["noise_config"] = self.noise_config

        if self.metadata:
            attributes["metadata"] = self.metadata

        # Save root-level attributes
        save_h5(filepath, attributes=attributes, group=None, mode=mode)

        # Save each member to a numbered group
        for i, point_data in enumerate(self.members):
            group_name = f"member_{i:04d}"
            point_data.save(filepath, group=group_name, mode="replace")

        # Save reference point data
        if self.reference_point_data is not None:
            self.reference_point_data.save(filepath, group="reference", mode="replace")

    @classmethod
    def load(cls, filepath: str | Path) -> "PointDataEnsemble":
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
        if "ensemble_size" not in attributes:
            raise ValueError("Missing 'ensemble_size' in root attributes")

        ensemble_size = attributes["ensemble_size"]
        base_config = attributes.get(
            "base_config"
        )  # Already parsed from JSON by read_h5
        noise_config = attributes.get("noise_config")
        metadata = attributes.get("metadata", {})
        has_reference = attributes.get(
            "has_reference", False
        )  # Default for legacy files

        # Load all members
        members = []
        for i in range(ensemble_size):
            group_name = f"member_{i:04d}"
            point_data = PointData.load(filepath, group=group_name)
            members.append(point_data)

        # Load reference when declared by file metadata
        reference_point_data = None
        if has_reference:
            reference_point_data = PointData.load(filepath, group="reference")

        # Construct and return ensemble
        return cls(
            members=members,
            base_config=base_config,
            noise_config=noise_config,
            metadata=metadata,
            reference_point_data=reference_point_data,
        )

    @classmethod
    def from_files(
        cls,
        configs: list[dict],
        base_dir: str | Path | None = None,
        metadata: dict | None = None,
        member_metadata: dict | None = None,
        reference_index: int | None = 0,
    ) -> "PointDataEnsemble":
        """
        Load an ensemble from external point data file configurations.

        Each config is passed to ``PointData.from_file`` and becomes one
        ensemble member. Imported ensembles must have one data type. Position
        data members must share an ambient dimension, but may have different
        numbers of points.

        Parameters
        ----------
        configs : list of dict
            File loading configurations. Each config must have ``data_type``
            and ``filepath`` keys, plus optional ``file_format`` and ``params``.
        base_dir : str or Path, optional
            Base directory for resolving relative filepaths.
        metadata : dict, optional
            Ensemble-level metadata.
        member_metadata : dict, optional
            Additional metadata applied to every member. The per-member
            ``member_index`` field is added automatically.
        reference_index : int or None, default=0
            Index of the member to use as the ensemble reference point cloud.
            Defaults to the first member. Set to ``None`` for no reference.

        Returns
        -------
        ensemble : PointDataEnsemble
            Imported point data ensemble.
        """
        if not configs:
            raise ValueError("configs must contain at least one file config")

        members = []
        for i, config in enumerate(configs):
            member_metadata_all = (
                dict(member_metadata) if member_metadata is not None else {}
            )
            member_metadata_all["member_index"] = i
            point_data = PointData.from_file(
                config,
                base_dir=base_dir,
                metadata=member_metadata_all,
            )
            members.append(point_data)

        _validate_imported_members(members)

        if reference_index is None:
            reference_point_data = None
        elif reference_index < 0 or reference_index >= len(members):
            raise ValueError(
                f"reference_index {reference_index} is out of range for "
                f"ensemble of size {len(members)}"
            )
        else:
            reference_point_data = members[reference_index]

        ensemble_metadata = dict(metadata) if metadata is not None else {}
        ensemble_metadata.update(
            {
                "import_mode": "files",
                "data_type": members[0].data_type,
                "source_files": [config["filepath"] for config in configs],
            }
        )
        if base_dir is not None:
            ensemble_metadata["base_dir"] = str(Path(base_dir))

        return cls(
            members=members,
            metadata=ensemble_metadata,
            reference_point_data=reference_point_data,
        )

    @classmethod
    def from_file_with_noise(
        cls,
        config: dict,
        noise_config: dict,
        num_realizations: int,
        base_dir: str | Path | None = None,
        base_seed: int = 42,
        include_base: bool = False,
        metadata: dict | None = None,
        member_metadata: dict | None = None,
    ) -> "PointDataEnsemble":
        """
        Load one positions file and construct a noisy ensemble from it.

        The imported file provides the base coordinates. Each noisy realization
        is generated with absolute coordinate noise and seed ``base_seed + i``.
        Distance-matrix perturbations are intentionally not supported.

        Parameters
        ----------
        config : dict
            File loading configuration for position data.
        noise_config : dict
            Noise parameters. Must include ``scale``. Optional
            ``distribution`` defaults to ``"uniform"``.
        num_realizations : int
            Number of noisy realizations to generate. If ``include_base=True``,
            the unperturbed imported data is prepended in addition to these
            noisy realizations.
        base_dir : str or Path, optional
            Base directory for resolving relative filepaths.
        base_seed : int, default=42
            Starting seed. Noisy realization i uses seed = base_seed + i.
        include_base : bool, default=False
            Whether to include the unperturbed imported point data as the first
            ensemble member.
        metadata : dict, optional
            Ensemble-level metadata.
        member_metadata : dict, optional
            Additional metadata applied to every member.

        Returns
        -------
        ensemble : PointDataEnsemble
            Imported noisy point data ensemble.
        """
        if num_realizations < 1:
            raise ValueError("num_realizations must be at least 1")
        if "scale" not in noise_config:
            raise ValueError("noise_config must include 'scale'")

        base_point_data = PointData.from_file(config, base_dir=base_dir)
        if not base_point_data.has_positions:
            raise ValueError("from_file_with_noise only supports position data")

        base_positions = base_point_data.get_positions()
        members = []

        if include_base:
            base_member_metadata = _imported_noise_member_metadata(
                config=config,
                noise_config=noise_config,
                member_index=0,
                noise_index=None,
                seed=None,
                is_base=True,
                base_dir=base_dir,
                member_metadata=member_metadata,
            )
            members.append(
                PointData(
                    positions=base_positions.copy(),
                    metadata=base_member_metadata,
                )
            )

        for i in range(num_realizations):
            seed = base_seed + i
            member_index = i + 1 if include_base else i
            noisy_positions = add_noise(
                base_positions,
                scale=noise_config["scale"],
                distribution=noise_config.get("distribution", "uniform"),
                seed=seed,
            )
            member_metadata_all = _imported_noise_member_metadata(
                config=config,
                noise_config=noise_config,
                member_index=member_index,
                noise_index=i,
                seed=seed,
                is_base=False,
                base_dir=base_dir,
                member_metadata=member_metadata,
            )
            members.append(
                PointData(positions=noisy_positions, metadata=member_metadata_all)
            )

        ensemble_metadata = dict(metadata) if metadata is not None else {}
        ensemble_metadata.update(
            {
                "import_mode": "file_with_noise",
                "num_realizations": num_realizations,
                "base_seed": base_seed,
                "include_base": include_base,
            }
        )
        if base_dir is not None:
            ensemble_metadata["base_dir"] = str(Path(base_dir))

        # The imported base is the ensemble reference.
        return cls(
            members=members,
            base_config=config,
            noise_config=noise_config,
            metadata=ensemble_metadata,
            reference_point_data=base_point_data,
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
    ) -> "PointDataEnsemble":
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
        if "scale" not in noise_config:
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
                scale=noise_config["scale"],
                distribution=noise_config.get("distribution", "uniform"),
                seed=seed,
            )

            # Create PointData with comprehensive metadata
            member_metadata_all = (
                dict(member_metadata) if member_metadata is not None else {}
            )
            member_metadata_all.update(
                {
                    "base_config": base_config,
                    "noise_config": noise_config,
                    "seed": seed,
                    "member_index": i,
                }
            )
            point_data = PointData(
                positions=noisy_positions, metadata=member_metadata_all
            )
            members.append(point_data)

        # Construct and return ensemble
        ensemble_metadata = dict(metadata) if metadata is not None else {}
        ensemble_metadata["base_seed"] = base_seed

        # Store the unperturbed base as reference.
        reference_point_data = PointData(positions=base_positions)

        return cls(
            members=members,
            base_config=base_config,
            noise_config=noise_config,
            metadata=ensemble_metadata,
            reference_point_data=reference_point_data,
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
        counts = self.num_points_per_member
        if self.has_uniform_num_points:
            num_points_repr = str(counts[0])
        else:
            num_points_repr = f"variable({min(counts)}-{max(counts)})"
        return (
            f"PointDataEnsemble(size={self.size}, "
            f"N={num_points_repr}, d={self.dimension})"
        )


# =============================================================================
# Helpers
# =============================================================================


def _validate_imported_members(members: list[PointData]) -> None:
    """Validate imported ensemble members have compatible data arrays."""
    _validate_member_compatibility(members, context="Imported ensemble")


def _validate_member_compatibility(
    members: list[PointData],
    context: str,
) -> None:
    """Validate ensemble members share data type and position dimension."""
    first = members[0]
    data_type = first.data_type
    dimension = first.dimension

    for i, member in enumerate(members[1:], start=1):
        if member.data_type != data_type:
            raise ValueError(
                f"{context} members must have the same data_type; "
                f"member 0 has {data_type!r}, member {i} has {member.data_type!r}"
            )
        if data_type == "positions" and member.dimension != dimension:
            raise ValueError(
                f"{context} position members must have the same ambient "
                f"dimension; member 0 has dimension {dimension}, member {i} "
                f"has dimension {member.dimension}"
            )


def _validate_reference_compatibility(
    reference_point_data: PointData,
    member: PointData,
) -> None:
    """Validate reference point data against ensemble member compatibility."""
    if not isinstance(reference_point_data, PointData):
        raise TypeError("reference_point_data must be a PointData object")
    if reference_point_data.data_type != member.data_type:
        raise ValueError(
            "reference_point_data data_type "
            f"{reference_point_data.data_type!r} does not match ensemble "
            f"member data_type {member.data_type!r}"
        )
    if (
        reference_point_data.has_positions
        and reference_point_data.dimension != member.dimension
    ):
        raise ValueError(
            "reference_point_data ambient dimension "
            f"{reference_point_data.dimension} does not match ensemble "
            f"member dimension {member.dimension}"
        )


def _imported_noise_member_metadata(
    config: dict,
    noise_config: dict,
    member_index: int,
    noise_index: int | None,
    seed: int | None,
    is_base: bool,
    base_dir: str | Path | None,
    member_metadata: dict | None,
) -> dict:
    """Build member metadata for noisy ensembles from imported data."""
    metadata = dict(member_metadata) if member_metadata is not None else {}
    metadata.update(
        {
            "base_config": config,
            "noise_config": noise_config,
            "member_index": member_index,
            "noise_index": noise_index,
            "is_base": is_base,
        }
    )
    if seed is not None:
        metadata["seed"] = seed
    if base_dir is not None:
        metadata["base_dir"] = str(Path(base_dir))
    return metadata
