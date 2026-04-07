"""
Analysis utilities for spectral data.

Provides standalone mathematical functions for distribution estimation and
comparison, and the EnsembleSpectraAnalysis class for computing, caching,
and persisting spectral observables and eigenvalue distributions from
pipeline outputs.
"""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from holspec.utilities import save_h5, read_h5, get_keys_h5, join_h5_group
from holspec.pipeline import trace_provenance
from holspec.spectra import Spectrum
from holspec.hodge_laplacian import LAPLACIAN_COMPONENT_NAMES

if TYPE_CHECKING:
    from holspec.spectra import HodgeLaplacianSpectra


# =============================================================================
# Constants
# =============================================================================

# Default set of scalar observables computed eagerly by EnsembleSpectraAnalysis
STANDARD_OBSERVABLES = (
    'dim_ker',
    'eigval_min_nz',
    'eigval_max',
    'eigval_mean',
    'eigval_mean_nz',
    'eigval_var',
    'eigval_var_nz',
    'eigval_sum',
    'num_nonzero',
)

_HISTOGRAM_DEFAULTS = {
    'bins': 50,
    'range': None,
    'density': True,
}


# =============================================================================
# Distributions
# =============================================================================

def empirical_distribution(
    values: np.ndarray,
    method: str = 'histogram',
    method_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the empirical distribution of a 1D sample.

    Parameters
    ----------
    values : ndarray
        1D array of sample values.
    method : {'histogram'}, default='histogram'
        Distribution estimation method. 'kde' is planned but not yet
        implemented.
    method_params : dict, optional
        Method-specific parameters. If None, method defaults are used.
        Keys not present in the dict fall back to their defaults.

        Histogram parameters:
            bins : int or ndarray, default=50
                Number of bins or explicit bin edges.
            range : tuple of float, optional
                Lower and upper value range. None uses data range.
                Ignored when bins is an ndarray.
            density : bool, default=True
                If True, return probability density; if False, return
                counts.

    Returns
    -------
    x : ndarray, shape (m,)
        Support points. For histogram, these are bin centers.
    density : ndarray, shape (m,)
        Estimated density (or counts if density=False).
    """
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(
            f"values must be 1D, got shape {values.shape}."
        )

    if method == 'histogram':
        params = {**_HISTOGRAM_DEFAULTS, **(method_params or {})}
        hist, bin_edges = np.histogram(
            values,
            bins=params['bins'],
            range=params['range'],
            density=params['density'],
        )
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return bin_centers, hist

    elif method == 'kde':
        raise NotImplementedError(
            "KDE distribution estimation is planned but not yet implemented."
        )

    else:
        raise ValueError(
            f"Unknown method '{method}'. Supported methods: 'histogram'."
        )


def distribution_distance(
    density_a: np.ndarray,
    density_b: np.ndarray,
    x: np.ndarray | None = None,
    metric: str = 'l2',
    p: float | None = None,
) -> float:
    """
    Compute distance between two distributions on a shared support.

    Parameters
    ----------
    density_a, density_b : ndarray
        1D density arrays of equal length, defined on a shared support.
    x : ndarray, optional
        Shared support points (e.g., bin centers). Required for
        'wasserstein'; ignored for Lp metrics.
    metric : {'l1', 'l2', 'lp', 'wasserstein'}, default='l2'
        Distance metric.
    p : float, optional
        Exponent for the Lp distance. Required when metric='lp'.
        Ignored for 'l1', 'l2', and 'wasserstein'.

    Returns
    -------
    float
        Non-negative distance.
    """
    density_a = np.asarray(density_a, dtype=np.float64)
    density_b = np.asarray(density_b, dtype=np.float64)

    if density_a.ndim != 1 or density_b.ndim != 1:
        raise ValueError(
            f"Densities must be 1D. Got shapes {density_a.shape} "
            f"and {density_b.shape}."
        )
    if len(density_a) != len(density_b):
        raise ValueError(
            f"Densities must have the same length. Got {len(density_a)} "
            f"and {len(density_b)}."
        )

    if metric in ('l1', 'l2', 'lp'):
        if metric == 'l1':
            p_val = 1.0
        elif metric == 'l2':
            p_val = 2.0
        else:
            if p is None:
                raise ValueError(
                    "Parameter p is required when metric='lp'."
                )
            p_val = float(p)

        diff = np.abs(density_a - density_b)
        return float(np.sum(diff ** p_val) ** (1.0 / p_val))

    elif metric == 'wasserstein':
        if x is None:
            raise ValueError(
                "Parameter x (support points) is required for "
                "metric='wasserstein'."
            )
        from scipy.stats import wasserstein_distance
        x = np.asarray(x, dtype=np.float64)
        return float(wasserstein_distance(x, x, u_weights=density_a, v_weights=density_b))

    else:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            f"Supported metrics: 'l1', 'l2', 'lp', 'wasserstein'."
        )


# =============================================================================
# EnsembleSpectraAnalysis
# =============================================================================

class EnsembleSpectraAnalysis:
    """
    Analysis of spectra across ensemble members from a single pipeline output.

    Wraps a collection of Spectrum objects organized by (k, component) pairs,
    eagerly computes scalar observables (eigenvalue statistics, kernel
    dimension, etc.), and provides lazy-cached eigenvalue distribution
    estimation. Custom observables can be added via compute_member_observable.

    Parameters
    ----------
    member_spectra : dict[tuple[int, str], list[Spectrum]]
        Spectra keyed by (degree, component). Each list contains one
        Spectrum per ensemble member; all lists must have the same length.
    metadata : dict, optional
        Provenance metadata. 'creation_time' is auto-populated if absent.

    Notes
    -----
    - Scalar observables in STANDARD_OBSERVABLES are computed eagerly at
      construction for all (k, component) pairs.
    - Eigenvalue distributions are computed lazily via
      eigenvalue_distribution() and cached per (k, component, nonzero).
    - The primary factory from_file() reads eigenvalues directly from
      pipeline HDF5 files without reconstructing upstream objects.
    """

    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(
        self,
        member_spectra: dict[tuple[int, str], list[Spectrum]],
        metadata: dict | None = None,
    ):
        # Validate member_spectra
        if not member_spectra:
            raise ValueError("member_spectra must be non-empty.")

        # Check consistent list lengths
        lengths = {key: len(spc_list) for key, spc_list in member_spectra.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) != 1:
            raise ValueError(
                f"All spectrum lists must have the same length. "
                f"Got lengths: {lengths}."
            )

        # Validate keys
        for k, component in member_spectra:
            if not isinstance(k, int) or k < 0:
                raise ValueError(
                    f"Degree must be a non-negative integer, got k={k!r}."
                )
            if component not in LAPLACIAN_COMPONENT_NAMES:
                raise ValueError(
                    f"Unknown component '{component}'. "
                    f"Valid components: {LAPLACIAN_COMPONENT_NAMES}."
                )

        self._member_spectra = member_spectra
        self._num_members: int = unique_lengths.pop()

        # Derive structural properties
        all_degrees = sorted({k for k, _ in member_spectra})
        all_components = sorted({comp for _, comp in member_spectra})
        self._degrees: list[int] = all_degrees
        self._components: tuple[str, ...] = tuple(all_components)
        self._max_dim: int = max(all_degrees)

        # Dimensions from first member per degree
        self._dimensions: dict[int, int] = {}
        for k in all_degrees:
            for comp in all_components:
                if (k, comp) in member_spectra:
                    self._dimensions[k] = member_spectra[(k, comp)][0].dimension
                    break

        # Initialize metadata
        self.metadata: dict = metadata if metadata is not None else {}
        if 'creation_time' not in self.metadata:
            self.metadata['creation_time'] = datetime.now().isoformat()

        # Eagerly compute observables
        self._observables_cache: dict[tuple[int, str], dict[str, np.ndarray]] = {}
        self._compute_all_observables()

        # Distribution cache stores one active result per
        # (k, component, nonzero). Returns cached on hit when
        # method_params is None; recomputes when explicit params
        # are provided.
        self._distribution_cache: dict[tuple[int, str, bool], dict] = {}

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_file(
        cls,
        filepath: str | Path,
        project_root: str | Path | None = None,
        degrees: list[int] | None = None,
        components: tuple[str, ...] = ('full',),
        load_eigenvectors: bool = False,
        metadata: dict | None = None,
    ) -> EnsembleSpectraAnalysis:
        """
        Construct from a pipeline spectra HDF5 file.

        Reads eigenvalue (and optionally eigenvector) arrays directly
        from the file without reconstructing upstream objects, making
        this much faster than loading via load_spectra.

        Parameters
        ----------
        filepath : str or Path
            Path to a Stage 4 pipeline HDF5 file.
        project_root : str or Path, optional
            Project root, stored in metadata for provenance.
        degrees : list of int, optional
            Degrees to load. None loads all available degrees.
        components : tuple of str, default=('full',)
            Laplacian components to load.
        load_eigenvectors : bool, default=False
            Whether to load eigenvector arrays. False saves memory when
            only eigenvalue analysis is needed.
        metadata : dict, optional
            Override metadata. If None, metadata is built from file
            attributes.

        Returns
        -------
        EnsembleSpectraAnalysis
        """
        filepath = Path(filepath)

        # Discover member groups
        all_keys = get_keys_h5(filepath)
        member_keys = [key for key in all_keys if key.startswith('member_')]
        if not member_keys:
            raise ValueError(
                f"No member groups found in {filepath}."
            )

        # Read first member attributes to discover available (k, component) pairs
        _, first_attrs = read_h5(filepath, group=member_keys[0], dataset_names=[])
        cached_keys_raw = first_attrs.get('cached_keys', [])

        # Parse "k_component" strings and filter
        available_keys: list[tuple[int, str]] = []
        for key_str in cached_keys_raw:
            k_str, comp = key_str.split('_', 1)
            k = int(k_str)
            if degrees is not None and k not in degrees:
                continue
            if comp not in components:
                continue
            available_keys.append((k, comp))

        if not available_keys:
            raise ValueError(
                f"No matching (degree, component) pairs found in {filepath}. "
                f"Available: {cached_keys_raw}. "
                f"Requested degrees={degrees}, components={components}."
            )

        # Read spectra for all members
        member_spectra: dict[tuple[int, str], list[Spectrum]] = {
            key: [] for key in available_keys
        }

        for member_key in member_keys:
            for k, comp in available_keys:
                subgroup = join_h5_group(member_key, f'degree_{k}/component_{comp}')
                spc_datasets, spc_attributes = read_h5(filepath, group=subgroup)

                eigenvalues = spc_datasets['eigenvalues']
                dimension = int(spc_attributes['dimension'])

                eigenvectors = None
                if load_eigenvectors:
                    eigenvectors = spc_datasets.get('eigenvectors', None)

                member_spectra[(k, comp)].append(
                    Spectrum(eigenvalues, dimension, eigenvectors=eigenvectors)
                )

        # Build metadata
        if metadata is None:
            metadata = {
                'source_file': str(filepath),
            }
            if project_root is not None:
                metadata['project_root'] = str(project_root)

        return cls(member_spectra, metadata=metadata)

    @classmethod
    def from_hlsp_list(
        cls,
        hlsp_list: list[HodgeLaplacianSpectra],
        degrees: list[int] | None = None,
        components: tuple[str, ...] = ('full',),
        metadata: dict | None = None,
    ) -> EnsembleSpectraAnalysis:
        """
        Construct from a list of live HodgeLaplacianSpectra objects.

        Parameters
        ----------
        hlsp_list : list of HodgeLaplacianSpectra
            One per ensemble member.
        degrees : list of int, optional
            Degrees to include. None uses all degrees from the first
            member.
        components : tuple of str, default=('full',)
            Laplacian components to include.
        metadata : dict, optional
            Provenance metadata.

        Returns
        -------
        EnsembleSpectraAnalysis
        """
        if not hlsp_list:
            raise ValueError("hlsp_list must be non-empty.")

        # Validate consistent max_dim
        max_dims = {hlsp.max_dim for hlsp in hlsp_list}
        if len(max_dims) != 1:
            raise ValueError(
                f"All HodgeLaplacianSpectra must have the same max_dim. "
                f"Got: {max_dims}."
            )

        if degrees is None:
            degrees = hlsp_list[0].degrees

        # Collect spectra
        member_spectra: dict[tuple[int, str], list[Spectrum]] = {}
        for k in degrees:
            for comp in components:
                key = (k, comp)
                member_spectra[key] = [
                    hlsp.spectrum(k, comp) for hlsp in hlsp_list
                ]

        return cls(member_spectra, metadata=metadata)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def num_members(self) -> int:
        """Number of ensemble members."""
        return self._num_members

    @property
    def degrees(self) -> list[int]:
        """Sorted list of degrees present in the analysis."""
        return self._degrees

    @property
    def components(self) -> tuple[str, ...]:
        """Laplacian components present in the analysis."""
        return self._components

    @property
    def max_dim(self) -> int:
        """Maximum degree."""
        return self._max_dim

    @property
    def dimensions(self) -> dict[int, int]:
        """Cochain space dimensions {k: N_k}."""
        return self._dimensions

    @property
    def keys(self) -> list[tuple[int, str]]:
        """Sorted (k, component) pairs present in the analysis."""
        return sorted(self._member_spectra.keys())

    # =========================================================================
    # Spectrum Access
    # =========================================================================

    def member_spectra(self, k: int, component: str = 'full') -> list[Spectrum]:
        """
        Per-member Spectrum objects at (k, component).

        Parameters
        ----------
        k : int
            Degree.
        component : str, default='full'
            Laplacian component.

        Returns
        -------
        list of Spectrum
        """
        self._validate_key(k, component)
        return self._member_spectra[(k, component)]

    def member_eigenvalues(self, k: int, component: str = 'full') -> list[np.ndarray]:
        """
        Per-member eigenvalue arrays at (k, component).

        Parameters
        ----------
        k : int
            Degree.
        component : str, default='full'
            Laplacian component.

        Returns
        -------
        list of ndarray
        """
        return [spc.eigenvalues for spc in self.member_spectra(k, component)]

    def member_eigenvectors(
        self, k: int, component: str = 'full',
    ) -> list[np.ndarray | None]:
        """
        Per-member eigenvector matrices at (k, component).

        Parameters
        ----------
        k : int
            Degree.
        component : str, default='full'
            Laplacian component.

        Returns
        -------
        list of ndarray or None
        """
        return [spc.eigenvectors for spc in self.member_spectra(k, component)]

    # =========================================================================
    # Observable Access
    # =========================================================================

    def member_observables(
        self, k: int, component: str = 'full',
    ) -> dict[str, np.ndarray]:
        """
        Per-member scalar observables at (k, component).

        Returns
        -------
        dict
            {name: array of shape (num_members,)} for all currently cached
            observables at this (k, component) pair, including the standard
            eager observables and any custom observables added via
            ``compute_member_observable(..., cache_name=...)``.
        """
        self._validate_key(k, component)
        return self._observables_cache[(k, component)]

    def observables_summary(
        self,
        k: int,
        component: str = 'full',
        ci: float | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Summary statistics of observables across ensemble members.

        Parameters
        ----------
        k : int
            Degree.
        component : str, default='full'
            Laplacian component.
        ci : float, optional
            Confidence interval width (e.g. 0.95). When provided,
            includes 'ci_low' and 'ci_high' computed as percentile-based
            confidence intervals.

        Returns
        -------
        dict
            {name: {'mean': float, 'std': float, ...}} for each
            observable.
        """
        if ci is not None and not (0.0 < ci < 1.0):
            raise ValueError(
                f"ci must be in (0, 1), got {ci}."
            )

        obs = self.member_observables(k, component)
        summary: dict[str, dict[str, float]] = {}

        for name, arr in obs.items():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                entry: dict[str, float] = {
                    'mean': float(np.nanmean(arr)),
                    'std': float(np.nanstd(arr)),
                }
            if ci is not None:
                alpha = (1.0 - ci) / 2.0
                entry['ci_low'] = float(np.nanpercentile(arr, 100.0 * alpha))
                entry['ci_high'] = float(np.nanpercentile(arr, 100.0 * (1.0 - alpha)))
            summary[name] = entry

        return summary

    def compute_member_observable(
        self,
        k: int,
        component: str = 'full',
        *,
        func: Callable[[Spectrum], float],
        cache_name: str | None = None,
    ) -> np.ndarray:
        """
        Compute a custom per-member observable from a function on Spectrum.

        Parameters
        ----------
        k : int
            Degree.
        component : str, default='full'
            Laplacian component.
        func : callable (keyword-only)
            Function mapping a Spectrum to a float.
        cache_name : str, optional (keyword-only)
            If provided, store the result in the observables cache under
            this name. Raises ValueError if the name already exists.

        Returns
        -------
        ndarray, shape (num_members,)
        """
        self._validate_key(k, component)

        spectra_list = self._member_spectra[(k, component)]
        result = np.array(
            [func(spc) for spc in spectra_list], dtype=np.float64,
        )

        if cache_name is not None:
            obs = self._observables_cache[(k, component)]
            if cache_name in obs:
                raise ValueError(
                    f"Observable '{cache_name}' already exists for "
                    f"(k={k}, component='{component}')."
                )
            obs[cache_name] = result

        return result

    # =========================================================================
    # Distribution Methods
    # =========================================================================

    def eigenvalue_distribution(
        self,
        k: int,
        component: str = 'full',
        nonzero: bool = True,
        method: str = 'histogram',
        method_params: dict | None = None,
    ) -> dict:
        """
        Compute ensemble-averaged eigenvalue distribution.

        Parameters
        ----------
        k : int
            Degree.
        component : str, default='full'
            Laplacian component.
        nonzero : bool, default=True
            If True, restrict to nonzero eigenvalues.
        method : str, default='histogram'
            Distribution estimation method (passed to
            empirical_distribution).
        method_params : dict, optional
            Method-specific parameters (passed to
            empirical_distribution). For histogram method, shared bin
            edges are computed automatically from pooled data.

        Returns
        -------
        dict
            Keys: 'x', 'density_mean', 'density_std', 'method',
            'method_params', 'num_members'.

        Notes
        -----
        - Cache semantics: cached distributions are keyed by
          (k, component, nonzero). When ``method_params`` is None,
          a cached result is returned if one exists and its method
          matches. When ``method_params`` is explicitly provided,
          the distribution is recomputed and the cache entry replaced.
        - This means the cache stores at most one "active"
          distribution per (k, component, nonzero) selection.
        """
        self._validate_key(k, component)

        # Return cached result if available and no specific params requested
        cache_key = (k, component, nonzero)
        if method_params is None and cache_key in self._distribution_cache:
            cached = self._distribution_cache[cache_key]
            if cached['method'] == method:
                return cached

        # Extract eigenvalue arrays
        spectra_list = self._member_spectra[(k, component)]
        if nonzero:
            eig_arrays = [spc.nonzero_eigenvalues() for spc in spectra_list]
        else:
            eig_arrays = [spc.eigenvalues for spc in spectra_list]

        # Resolve method params
        if method == 'histogram':
            params = {**_HISTOGRAM_DEFAULTS, **(method_params or {})}

            # Compute shared bin edges from pooled data
            pooled = np.concatenate(eig_arrays) if eig_arrays else np.array([])
            if pooled.size > 0:
                n_bins = params['bins']
                # When bins is an int and no explicit range is set,
                # clamp to available unique values to avoid errors
                if isinstance(n_bins, (int, np.integer)) and params.get('range') is None:
                    n_bins = min(int(n_bins), max(len(np.unique(pooled)), 1))
                shared_edges = np.histogram_bin_edges(
                    pooled, bins=n_bins, range=params['range'],
                )
                params['bins'] = shared_edges
            else:
                params['bins'] = np.array([0.0, 1.0])

            params['density'] = True
        else:
            params = method_params or {}

        # Compute per-member distributions
        # Determine number of bins for zero-density fallback
        if method == 'histogram' and isinstance(params.get('bins'), np.ndarray):
            n_output = len(params['bins']) - 1
        else:
            n_output = None

        densities = []
        x = None
        for eig in eig_arrays:
            if eig.size == 0:
                # Empty spectrum contributes a zero-density row so the
                # ensemble average reflects all members, not just those
                # with non-empty spectra.
                if n_output is not None:
                    densities.append(np.zeros(n_output))
                continue
            xi, di = empirical_distribution(eig, method=method, method_params=params)
            if x is None:
                x = xi
                if n_output is None:
                    n_output = len(xi)
            densities.append(di)

        if x is None:
            # All members had empty spectra
            n_output = n_output or 1
            x = np.zeros(n_output)
            density_mean = np.zeros(n_output)
            density_std = np.zeros(n_output)
        else:
            density_stack = np.stack(densities, axis=0)
            density_mean = np.mean(density_stack, axis=0)
            density_std = np.std(density_stack, axis=0)

        result = {
            'x': x,
            'density_mean': density_mean,
            'density_std': density_std,
            'nonzero': nonzero,
            'method': method,
            'method_params': params,
            'num_members': self._num_members,
        }

        # Cache result keyed by (k, component, nonzero)
        self._distribution_cache[(k, component, nonzero)] = result
        return result

    # =========================================================================
    # I/O Methods
    # =========================================================================

    def save(
        self,
        filepath: str | Path,
        mode: str = 'replace',
        group: str | None = None,
        hdf5_options: dict | None = None,
    ) -> None:
        """
        Save analysis results to HDF5.

        Writes observables and any cached distributions. Does not save
        underlying eigenvalue arrays (those live in the pipeline spectra
        file).

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        mode : {'replace', 'update', 'create'}, default='replace'
            How to handle an existing file/group.
        group : str, optional
            HDF5 group path.
        hdf5_options : dict, optional
            HDF5 compression options.
        """
        filepath = Path(filepath)
        if hdf5_options is None:
            hdf5_options = {'compression': 'gzip', 'compression_opts': 4}

        # Root attributes
        root_attributes = {
            'num_members': self._num_members,
            'degrees': self._degrees,
            'components': list(self._components),
            'max_dim': self._max_dim,
            'observable_names': sorted({
                name
                for obs_dict in self._observables_cache.values()
                for name in obs_dict
            }),
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

        # Save observables
        for (k, comp), obs_dict in self._observables_cache.items():
            subgroup = join_h5_group(
                group, f'observables/degree_{k}/component_{comp}',
            )
            save_h5(
                filepath,
                datasets=obs_dict,
                attributes=None,
                mode='update',
                group=subgroup,
                hdf5_options=hdf5_options,
            )

        # Save cached distributions
        for (k, comp, nz), dist in self._distribution_cache.items():
            nz_label = 'nonzero_true' if nz else 'nonzero_false'
            subgroup = join_h5_group(
                group,
                f'distributions/degree_{k}/component_{comp}/{nz_label}',
            )
            dist_datasets = {
                'x': dist['x'],
                'density_mean': dist['density_mean'],
                'density_std': dist['density_std'],
            }
            dist_attributes = {
                'method': dist['method'],
                'method_params': dist['method_params'],
                'num_members': dist['num_members'],
                'nonzero': nz,
            }
            save_h5(
                filepath,
                datasets=dist_datasets,
                attributes=dist_attributes,
                mode='update',
                group=subgroup,
                hdf5_options=hdf5_options,
            )

    def load_cache(
        self,
        filepath: str | Path,
        group: str | None = None,
    ) -> None:
        """
        Populate caches from a previously saved analysis file.

        Parameters
        ----------
        filepath : str or Path
            Path to HDF5 file created by save().
        group : str, optional
            HDF5 group path.
        """
        filepath = Path(filepath)

        # Read root attributes for validation
        _, root_attrs = read_h5(filepath, group=group, dataset_names=[])

        stored_num_members = int(root_attrs.get('num_members', -1))
        if stored_num_members != self._num_members:
            raise ValueError(
                f"num_members mismatch: stored {stored_num_members}, "
                f"live {self._num_members}."
            )
        stored_degrees = list(root_attrs.get('degrees', []))
        if sorted(stored_degrees) != sorted(self._degrees):
            raise ValueError(
                f"degrees mismatch: stored {stored_degrees}, "
                f"live {self._degrees}."
            )
        stored_components = list(root_attrs.get('components', []))
        if sorted(stored_components) != sorted(self._components):
            raise ValueError(
                f"components mismatch: stored {stored_components}, "
                f"live {list(self._components)}."
            )

        # Restore metadata
        stored_metadata = root_attrs.get('metadata')
        if isinstance(stored_metadata, dict):
            self.metadata.update(stored_metadata)

        # Load observables
        for k in self._degrees:
            for comp in self._components:
                if (k, comp) not in self._member_spectra:
                    continue
                obs_group = join_h5_group(
                    group, f'observables/degree_{k}/component_{comp}',
                )
                try:
                    obs_datasets, _ = read_h5(filepath, group=obs_group)
                    self._observables_cache[(k, comp)] = obs_datasets
                except KeyError:
                    pass

        # Load distributions
        for k in self._degrees:
            for comp in self._components:
                if (k, comp) not in self._member_spectra:
                    continue
                for nz, nz_label in ((True, 'nonzero_true'), (False, 'nonzero_false')):
                    dist_group = join_h5_group(
                        group,
                        f'distributions/degree_{k}/component_{comp}/{nz_label}',
                    )
                    try:
                        dist_datasets, dist_attrs = read_h5(
                            filepath, group=dist_group,
                        )
                        self._distribution_cache[(k, comp, nz)] = {
                            'x': dist_datasets['x'],
                            'density_mean': dist_datasets['density_mean'],
                            'density_std': dist_datasets['density_std'],
                            'nonzero': nz,
                            'method': str(dist_attrs.get('method', 'histogram')),
                            'method_params': dist_attrs.get('method_params', {}),
                            'num_members': int(dist_attrs.get(
                                'num_members', self._num_members,
                            )),
                        }
                    except KeyError:
                        pass

    # =========================================================================
    # Utilities
    # =========================================================================

    def summary(self, indent: str = '') -> str:
        """
        Generate human-readable summary of ensemble-averaged observables.

        Parameters
        ----------
        indent : str, optional
            String prepended to every line.

        Returns
        -------
        str
        """
        lines = []
        lines.append('Ensemble Spectra Analysis:')
        lines.append('-' * 80)
        lines.append(
            f"num_members: {self._num_members}, "
            f"degrees: {self._degrees}, "
            f"components: {self._components}"
        )
        lines.append('-' * 80)

        # Header row — show a compact subset of observables
        display_obs = ('dim_ker', 'eigval_min_nz', 'eigval_max', 'eigval_mean_nz')
        multi_comp = len(self._components) > 1
        if multi_comp:
            header = f"{'k':<4} {'comp':<8} {'N_k':<6} "
        else:
            header = f"{'k':<4} {'N_k':<6} "
        header += '  '.join(f'{name:<16}' for name in display_obs)
        lines.append(header)
        if multi_comp:
            lines.append(f"{'':4} {'':8} {'':6} " + '  '.join(
                f'{"(mean +/- std)":<16}' for _ in display_obs
            ))
        else:
            lines.append(f"{'':4} {'':6} " + '  '.join(
                f'{"(mean +/- std)":<16}' for _ in display_obs
            ))
        lines.append('-' * 80)

        for k in self._degrees:
            N_k = self._dimensions.get(k, 0)
            for comp in self._components:
                key = (k, comp)
                if key not in self._observables_cache:
                    continue
                obs = self._observables_cache[key]
                parts = []
                for obs_name in display_obs:
                    if obs_name in obs:
                        arr = obs[obs_name]
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', RuntimeWarning)
                            mean = float(np.nanmean(arr))
                            std = float(np.nanstd(arr))
                        if np.isnan(mean):
                            parts.append('--')
                        else:
                            parts.append(f'{mean:.2f} +/- {std:.2f}')
                    else:
                        parts.append('--')

                if multi_comp:
                    lines.append(
                        f"{k:<4} {comp:<8} {N_k:<6} "
                        + '  '.join(f'{p:<16}' for p in parts)
                    )
                else:
                    lines.append(
                        f"{k:<4} {N_k:<6} "
                        + '  '.join(f'{p:<16}' for p in parts)
                    )

        lines.append('-' * 80)
        lines.append('')

        return '\n'.join(indent + line for line in lines)

    def __repr__(self) -> str:
        return (
            f"EnsembleSpectraAnalysis("
            f"num_members={self._num_members}, "
            f"max_dim={self._max_dim}, "
            f"degrees={self._degrees}, "
            f"components={self._components})"
        )

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _validate_key(self, k: int, component: str) -> None:
        """Validate that (k, component) is present in member_spectra."""
        if (k, component) not in self._member_spectra:
            raise KeyError(
                f"(k={k}, component='{component}') not present. "
                f"Available keys: {sorted(self._member_spectra.keys())}."
            )

    def _compute_all_observables(self) -> None:
        """Compute scalar observables for all (k, component) pairs."""
        for key in self._member_spectra:
            k, component = key
            self._observables_cache[key] = self._compute_member_observables(
                k, component,
            )

    def _compute_member_observables(
        self, k: int, component: str,
    ) -> dict[str, np.ndarray]:
        """
        Compute standard scalar observables for one (k, component) pair.

        Returns
        -------
        dict
            {name: array of shape (num_members,)} for each name in
            STANDARD_OBSERVABLES.
        """
        spectra_list = self._member_spectra[(k, component)]
        n = self._num_members

        dim_ker = np.empty(n, dtype=np.float64)
        eigval_min_nz = np.empty(n, dtype=np.float64)
        eigval_max = np.empty(n, dtype=np.float64)
        eigval_mean = np.empty(n, dtype=np.float64)
        eigval_mean_nz = np.empty(n, dtype=np.float64)
        eigval_var = np.empty(n, dtype=np.float64)
        eigval_var_nz = np.empty(n, dtype=np.float64)
        eigval_sum = np.empty(n, dtype=np.float64)
        num_nonzero = np.empty(n, dtype=np.float64)

        for i, spc in enumerate(spectra_list):
            evals = spc.eigenvalues
            nz = spc.nonzero_eigenvalues()

            dim_ker[i] = spc.dim_ker()
            eigval_min_nz[i] = nz[0] if len(nz) > 0 else np.nan
            eigval_max[i] = evals[-1] if len(evals) > 0 else np.nan
            eigval_mean[i] = spc.moment(1, normalized=True, nonzero=False) if len(evals) > 0 else np.nan
            eigval_mean_nz[i] = spc.moment(1, normalized=True, nonzero=True) if len(nz) > 0 else np.nan
            eigval_var[i] = float(np.var(evals)) if len(evals) > 0 else np.nan
            eigval_var_nz[i] = float(np.var(nz)) if len(nz) > 0 else np.nan
            eigval_sum[i] = spc.moment(1, normalized=False, nonzero=False)
            num_nonzero[i] = len(nz)

        return {
            'dim_ker': dim_ker,
            'eigval_min_nz': eigval_min_nz,
            'eigval_max': eigval_max,
            'eigval_mean': eigval_mean,
            'eigval_mean_nz': eigval_mean_nz,
            'eigval_var': eigval_var,
            'eigval_var_nz': eigval_var_nz,
            'eigval_sum': eigval_sum,
            'num_nonzero': num_nonzero,
        }


# =============================================================================
# Helpers
# =============================================================================

def extract_exp_params(provenance: dict, exp_params: list[str]) -> dict:
    """
    Extract experimental parameters from a provenance chain.

    Reads upstream file attributes to recover parameter values.
    Returns a dict with keys for all requested parameters; missing
    parameters are stored as None.

    Parameters
    ----------
    provenance : dict
        Provenance dict mapping stage names to file paths.
    exp_params : list of str
        Names of experimental parameters to extract (e.g. ['noise', 'alpha']).

    Returns
    -------
    dict
        {param_name: value or None}.
    """
    result = {name: None for name in exp_params}

    for name in exp_params:
        if name == 'noise':
            ptd_filepath = provenance.get('point_data')
            if ptd_filepath is None:
                continue
            _, ptd_attrs = read_h5(ptd_filepath, dataset_names=[])
            noise_config = ptd_attrs.get('noise_config')
            if noise_config is not None:
                result['noise'] = noise_config.get('scale')

        elif name == 'alpha':
            sc_filepath = provenance.get('topology_simplicial')
            if sc_filepath is None:
                continue
            _, sc_attrs = read_h5(sc_filepath, dataset_names=[])
            stage_config = sc_attrs.get('stage_config', {})
            result['alpha'] = stage_config.get('params', {}).get('alpha')

    return result


def build_spectra_file_records(
    spectra_filepaths_nested: dict,
    project_root: str | Path,
) -> dict[Path, dict]:
    """
    Convert nested selection output into a flat provenance-based file index.

    Parameters
    ----------
    spectra_filepaths_nested : dict[str, dict[str, Path]]
        Output of select_stage_outputs: {ptd_label: {output_label: filepath}}.
    project_root : str or Path
        Project root for resolving provenance paths.

    Returns
    -------
    dict[Path, dict]
        Flat mapping {filepath: file_record}.
    """
    records = {}

    for ptd_dir_label, output_dict in spectra_filepaths_nested.items():
        for output_label, filepath in output_dict.items():
            filepath = Path(filepath).resolve()

            provenance = trace_provenance(filepath, project_root)
            provenance = {k: Path(v).resolve() for k, v in provenance.items()}

            ptd_label = provenance['point_data'].stem if 'point_data' in provenance else None
            sc_label = provenance['topology_simplicial'].stem if 'topology_simplicial' in provenance else None
            cm_stem = provenance['geometry_metric'].stem if 'geometry_metric' in provenance else None
            if cm_stem is not None and sc_label is not None and cm_stem.startswith(sc_label + '__'):
                cm_label = cm_stem[len(sc_label) + 2:]
            else:
                cm_label = cm_stem

            records[filepath] = {
                'filepath': filepath,
                'ptd_label': ptd_label,
                'sc_label': sc_label,
                'cm_label': cm_label,
                'output_label': output_label,
                'provenance': provenance,
            }

    return records


def save_exp_series(
    filepath: str | Path,
    dataset_name: str,
    group_key: tuple[str, ...],
    series: dict,
    analysis_config: dict,
) -> None:
    """
    Save an experiment series to an HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    dataset_name : str
        Name of the dataset (e.g. 'trilatt_noise').
    group_key : tuple of str
        Fields identifying this series (e.g. ('delaunay', 'combinatorial')).
    series : dict
        Series data with keys: 'exp_param', 'exp_values',
        'observable_series', 'distribution_series', 'distance_series'.
    analysis_config : dict
        Analysis configuration (native Python types; HDF5 conversion
        is handled by save_h5 / to_h5_attribute).
    """
    filepath = Path(filepath)
    if filepath.exists():
        filepath.unlink()

    ac = analysis_config
    root_attributes = {
        'dataset_name': dataset_name,
        'exp_param': series['exp_param'],
        'group_key': group_key,
        **ac,
    }
    save_h5(filepath, datasets={'exp_values': series['exp_values']}, attributes=root_attributes, mode='create')

    analysis_keys = ac['analysis_keys']
    observable_names = ac['observable_names']

    for k, comp in analysis_keys:
        obs = series['observable_series'][(k, comp)]
        obs_datasets = {}
        for obs_name in observable_names:
            obs_datasets[f'{obs_name}_mean'] = obs[obs_name]['mean']
            obs_datasets[f'{obs_name}_std'] = obs[obs_name]['std']
        save_h5(filepath, datasets=obs_datasets, group=f'observables/degree_{k}_{comp}', mode='create')

    for k, comp in analysis_keys:
        dist = series['distribution_series'][(k, comp)]
        save_h5(filepath, datasets={'x': dist['x'], 'density_stack': dist['density_stack']}, group=f'distributions/degree_{k}_{comp}', mode='create')

    for k, comp in analysis_keys:
        save_h5(filepath, datasets={f'degree_{k}_{comp}': series['distance_series'][(k, comp)]}, group='distances', mode='update')
