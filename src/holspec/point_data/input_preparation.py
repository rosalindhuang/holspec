# holspec/point_data/input_preparation.py
"""
Input preparation utilities for point data ensembles.

Provides config construction, labeling, and orchestration for preparing point
data ensembles from configuration dictionaries.
"""

from pathlib import Path
from datetime import datetime
from fnmatch import fnmatch

from holspec.point_data.data_generators import create_point_generator_label
from holspec.point_data.ensemble import PointDataEnsemble
from holspec.utilities import create_noise_label, save_h5


# =============================================================================
# Constants
# =============================================================================

SUPPORTED_IMPORT_MODES = {"files", "file_with_noise"}


# =============================================================================
# Utilities
# =============================================================================


def make_ensemble_config(
    generator: str,
    params: dict,
    scale: float = 0,
    distribution: str = "normal",
    num_realizations: int = 1,
    base_seed: int = 42,
) -> dict:
    """
    Create ensemble config dict.

    Parameters
    ----------
    generator : str
        Generator name (e.g., 'trilatthex', 'bcclatt').
    params : dict
        Generator-specific parameters.
    scale : float, default=0
        Noise scale. Use 0 for clean (BASE) data.
    distribution : str, default='normal'
        Noise distribution type.
    num_realizations : int, default=1
        Number of ensemble members.
    base_seed : int, default=42
        Base random seed for reproducibility.

    Returns
    -------
    dict
        Ensemble config with keys: base_config, noise_config,
        num_realizations, base_seed.
    """
    return {
        "base_config": {"generator": generator, "params": params},
        "noise_config": {"scale": scale, "distribution": distribution},
        "num_realizations": num_realizations,
        "base_seed": base_seed,
    }


def create_ensemble_label(
    ensemble_config: dict,
    dimension: int | None = None,
    float_fmt: str | None = None,
    strip_zeros: bool = True,
) -> str:
    """
    Create label from ensemble config.

    Label format: ``{base_label}__BASE`` for clean data,
    ``{base_label}__{noise_label}__E{num}`` for noisy ensembles.

    Parameters
    ----------
    ensemble_config : dict
        Ensemble configuration containing base_config, noise_config,
        num_realizations.
    dimension : int or None, optional
        Dimension override for the label.
    float_fmt : str or None, optional
        Format string for floating point numbers.
    strip_zeros : bool, default=True
        If True, trailing zeros after the decimal point are removed.

    Returns
    -------
    str
        Label string for the ensemble.
    """
    label = create_point_generator_label(
        ensemble_config["base_config"],
        dimension=dimension,
        float_fmt=float_fmt,
        strip_zeros=strip_zeros,
    )
    if ensemble_config["noise_config"]["scale"] > 0:
        noise_label = create_noise_label(
            ensemble_config["noise_config"],
            float_fmt=float_fmt,
            strip_zeros=strip_zeros,
        )
        label += f"__{noise_label}__E{ensemble_config['num_realizations']}"
    else:
        label += "__BASE"
    return label


# =============================================================================
# Orchestration
# =============================================================================


def run_data_generation(
    config: dict,
    project_root: str | Path,
    select_datasets: list[str] | None = None,
) -> dict[str, dict[str, Path]]:
    """
    Generate point data ensembles from a config dictionary.

    Iterates over dataset configs, generates each PointDataEnsemble, and saves
    to HDF5 files with metadata.

    Parameters
    ----------
    config : dict
        Data generation config dict with keys: ``summary``, ``configs``,
        ``runtime``, ``outputs``. The ``configs`` value is a nested dict
        ``{dataset: {ensemble_label: ensemble_config}}``. By default,
        outputs are saved under ``{data_dir}/{dataset}/``; set
        ``outputs.dataset_subdirs`` to false to save directly under
        ``{data_dir}/``.
    project_root : str or Path
        Project root for resolving relative paths.
    select_datasets : list of str or None, optional
        Glob patterns for dataset names (e.g. ``['test_examples',
        'exp_noise_trilatt_nr*']``). If None, generates all datasets
        present in the config. Exact names are valid patterns.

    Returns
    -------
    dict[str, dict[str, Path]]
        Nested mapping ``{dataset: {ensemble_label: output_filepath}}``.
    """
    project_root = Path(project_root)

    # --- Extract settings ---

    output_data_dir = project_root / config["outputs"]["data_dir"]
    dataset_subdirs = _get_dataset_subdirs(config["outputs"])
    stage_name = config["outputs"]["stage_name"]
    created_by = config["summary"]["created_by"]
    verbose = config["runtime"]["verbose"]
    dataset_configs = config["configs"]

    # --- Generate ensembles ---

    output_filepaths: dict[str, dict[str, Path]] = {}

    for dataset, configs_dict in dataset_configs.items():
        if select_datasets and not any(
            fnmatch(dataset, pat) for pat in select_datasets
        ):
            continue

        if verbose:
            print("-" * 60)
            print(f"{dataset}")
            print("-" * 60)
            print()

        output_filepaths[dataset] = {}

        for ensemble_label, ensemble_config in configs_dict.items():
            # Generate PointDataEnsemble
            ptd_ensemble = PointDataEnsemble.from_base_config(
                base_config=ensemble_config["base_config"],
                noise_config=ensemble_config["noise_config"],
                num_realizations=ensemble_config["num_realizations"],
                base_seed=ensemble_config["base_seed"],
            )

            # Save to HDF5
            if dataset_subdirs:
                output_filepath = output_data_dir / dataset / f"{ensemble_label}.h5"
            else:
                output_filepath = output_data_dir / f"{ensemble_label}.h5"
            output_filepath.parent.mkdir(parents=True, exist_ok=True)
            ptd_ensemble.save(output_filepath)

            # Add metadata
            save_h5(
                output_filepath,
                attributes={
                    "created_by": created_by,
                    "creation_time": datetime.now().isoformat(),
                    "stage_name": stage_name,
                    "stage_config": ensemble_config,
                },
                group=None,
                mode="update",
            )

            output_filepaths[dataset][ensemble_label] = output_filepath

            # Completion
            print(
                f"Generated {ptd_ensemble.size} members "
                f"for {dataset} / {ensemble_label}"
            )
            if verbose:
                print(f"  {ptd_ensemble}")
                print(f"  data type: {ptd_ensemble.members[0].data_type}")
                print(
                    f"  noise config: "
                    f"scale={ensemble_config['noise_config']['scale']}, "
                    f"distribution={ensemble_config['noise_config'].get('distribution', 'normal')}"
                )
                print(f"  file path: {output_filepath.relative_to(project_root)}")
                print(f"  file size: {output_filepath.stat().st_size / 1024:.2f} KB")
                print()

    return output_filepaths


def run_data_import(
    config: dict,
    project_root: str | Path,
    select_datasets: list[str] | None = None,
) -> dict[str, dict[str, Path]]:
    """
    Import point data ensembles from a config dictionary.

    Iterates over dataset configs, constructs each ``PointDataEnsemble`` from
    external files, and saves to HDF5 files with provenance metadata. Input
    filepaths in import configs are resolved relative to ``project_root``.

    Parameters
    ----------
    config : dict
        Data import config dict with keys: ``summary``, ``configs``,
        ``runtime``, ``outputs``. The ``configs`` value is a nested dict
        ``{dataset: {ensemble_label: import_config}}``. Supported import modes
        are ``"files"`` and ``"file_with_noise"``.
    project_root : str or Path
        Project root for resolving relative input and output paths.
    select_datasets : list of str or None, optional
        Glob patterns for dataset names. If None, imports all datasets
        present in the config. Exact names are valid patterns.

    Returns
    -------
    dict[str, dict[str, Path]]
        Nested mapping ``{dataset: {ensemble_label: output_filepath}}``.
    """
    project_root = Path(project_root)

    # --- Extract settings ---

    output_data_dir = project_root / config["outputs"]["data_dir"]
    dataset_subdirs = _get_dataset_subdirs(config["outputs"])
    stage_name = config["outputs"]["stage_name"]
    created_by = config["summary"]["created_by"]
    verbose = config.get("runtime", {}).get("verbose", False)
    dataset_configs = config["configs"]

    # --- Import ensembles ---

    output_filepaths: dict[str, dict[str, Path]] = {}

    for dataset, configs_dict in dataset_configs.items():
        if select_datasets and not any(
            fnmatch(dataset, pat) for pat in select_datasets
        ):
            continue

        if verbose:
            print("-" * 60)
            print(f"{dataset}")
            print("-" * 60)
            print()

        output_filepaths[dataset] = {}

        for ensemble_label, import_config in configs_dict.items():
            # Construct PointDataEnsemble
            ptd_ensemble = _import_ensemble_from_config(
                import_config,
                project_root=project_root,
            )

            # Save to HDF5
            if dataset_subdirs:
                output_filepath = output_data_dir / dataset / f"{ensemble_label}.h5"
            else:
                output_filepath = output_data_dir / f"{ensemble_label}.h5"
            output_filepath.parent.mkdir(parents=True, exist_ok=True)
            ptd_ensemble.save(output_filepath)

            # Add metadata
            save_h5(
                output_filepath,
                attributes={
                    "created_by": created_by,
                    "creation_time": datetime.now().isoformat(),
                    "stage_name": stage_name,
                    "stage_config": import_config,
                },
                group=None,
                mode="update",
            )

            output_filepaths[dataset][ensemble_label] = output_filepath

            # Completion
            print(
                f"Imported {ptd_ensemble.size} members for {dataset} / {ensemble_label}"
            )
            if verbose:
                print(f"  {ptd_ensemble}")
                print(f"  import mode: {import_config['import_mode']}")
                print(f"  data type: {ptd_ensemble.members[0].data_type}")
                print(f"  file path: {output_filepath.relative_to(project_root)}")
                print(f"  file size: {output_filepath.stat().st_size / 1024:.2f} KB")
                print()

    return output_filepaths


# =============================================================================
# Helpers
# =============================================================================


def _get_dataset_subdirs(outputs_config: dict) -> bool:
    """Return whether point data outputs should be grouped by dataset."""
    if "category_subdirs" in outputs_config:
        raise ValueError(
            "Use outputs.dataset_subdirs instead of outputs.category_subdirs."
        )
    return outputs_config.get("dataset_subdirs", True)


def _import_ensemble_from_config(
    import_config: dict,
    project_root: Path,
) -> PointDataEnsemble:
    """Construct a PointDataEnsemble from one import config."""
    import_mode = import_config.get("import_mode")

    if import_mode == "files":
        return PointDataEnsemble.from_files(
            import_config["file_configs"],
            base_dir=project_root,
            metadata=import_config.get("metadata"),
        )
    if import_mode == "file_with_noise":
        return PointDataEnsemble.from_file_with_noise(
            import_config["base_config"],
            noise_config=import_config["noise_config"],
            num_realizations=import_config["num_realizations"],
            base_dir=project_root,
            base_seed=import_config.get("base_seed", 42),
            include_base=import_config.get("include_base", False),
            metadata=import_config.get("metadata"),
        )

    raise ValueError(
        f"Unsupported import_mode {import_mode!r}. "
        f"Supported modes: {sorted(SUPPORTED_IMPORT_MODES)}"
    )
