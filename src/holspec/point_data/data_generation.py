# holspec/point_data/data_generation.py
"""
Data generation utilities for point data ensembles.

Provides config construction, labeling, and orchestration for generating
point data ensembles from configuration dictionaries.
"""

from pathlib import Path
from datetime import datetime
from fnmatch import fnmatch

from holspec.point_data.point_generators import create_point_generator_label
from holspec.point_data.ensemble import PointDataEnsemble
from holspec.utilities import create_noise_label, save_h5


# =============================================================================
# Utilities
# =============================================================================

def make_ensemble_config(
    generator: str,
    params: dict,
    scale: float = 0,
    distribution: str = 'normal',
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
        'base_config': {'generator': generator, 'params': params},
        'noise_config': {'scale': scale, 'distribution': distribution},
        'num_realizations': num_realizations,
        'base_seed': base_seed,
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
        ensemble_config['base_config'], dimension=dimension,
        float_fmt=float_fmt, strip_zeros=strip_zeros,
    )
    if ensemble_config['noise_config']['scale'] > 0:
        noise_label = create_noise_label(
            ensemble_config['noise_config'],
            float_fmt=float_fmt, strip_zeros=strip_zeros,
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
    select_categories: list[str] | None = None,
) -> dict[str, dict[str, Path]]:
    """
    Generate point data ensembles from a config dictionary.

    Iterates over dataset categories and ensemble configs, generates
    each PointDataEnsemble, and saves to HDF5 files with metadata.

    Parameters
    ----------
    config : dict
        Data generation config dict with keys: ``summary``, ``configs``,
        ``runtime``, ``outputs``. The ``configs`` value is a nested dict
        ``{category: {ensemble_label: ensemble_config}}``. By default,
        outputs are saved under ``{data_dir}/{category}/``; set
        ``outputs.category_subdirs`` to false to save directly under
        ``{data_dir}/``.
    project_root : str or Path
        Project root for resolving relative paths.
    select_categories : list of str or None, optional
        Glob patterns for category names (e.g. ``['test_examples',
        'exp_noise_trilatt_nr*']``). If None, generates all categories
        present in the config. Exact names are valid patterns.

    Returns
    -------
    dict[str, dict[str, Path]]
        Nested mapping ``{category: {ensemble_label: output_filepath}}``.
    """
    project_root = Path(project_root)

    # --- Extract settings ---

    output_data_dir = project_root / config['outputs']['data_dir']
    category_subdirs = config['outputs'].get('category_subdirs', True)
    stage_name = config['outputs']['stage_name']
    created_by = config['summary']['created_by']
    verbose = config['runtime']['verbose']
    dataset_configs = config['configs']

    # --- Generate ensembles ---

    output_filepaths: dict[str, dict[str, Path]] = {}

    for category, configs_dict in dataset_configs.items():
        if select_categories and not any(fnmatch(category, pat) for pat in select_categories):
            continue

        print(f"-" * 60)
        print(f"{category}")
        print(f"-" * 60)
        print()

        output_filepaths[category] = {}

        for ensemble_label, ensemble_config in configs_dict.items():

            # Generate PointDataEnsemble
            ptd_ensemble = PointDataEnsemble.from_base_config(
                base_config=ensemble_config['base_config'],
                noise_config=ensemble_config['noise_config'],
                num_realizations=ensemble_config['num_realizations'],
                base_seed=ensemble_config['base_seed'],
            )

            # Save to HDF5
            if category_subdirs:
                output_filepath = (
                    output_data_dir / category / f"{ensemble_label}.h5"
                )
            else:
                output_filepath = output_data_dir / f"{ensemble_label}.h5"
            output_filepath.parent.mkdir(parents=True, exist_ok=True)
            ptd_ensemble.save(output_filepath)

            # Add metadata
            save_h5(
                output_filepath,
                attributes={
                    'created_by': created_by,
                    'creation_time': datetime.now().isoformat(),
                    'stage_name': stage_name,
                    'stage_config': ensemble_config,
                },
                group=None,
                mode='update',
            )

            output_filepaths[category][ensemble_label] = output_filepath

            # Completion
            print(
                f"Generated {ptd_ensemble.size} members "
                f"for {category} / {ensemble_label}"
            )
            if verbose:
                print(f"  {ptd_ensemble}")
                print(f"  data type: "
                      f"{ptd_ensemble.members[0].data_type}")
                print(f"  noise config: "
                      f"scale={ensemble_config['noise_config']['scale']}, "
                      f"distribution={ensemble_config['noise_config'].get('distribution', 'normal')}")
                print(f"  file path: "
                      f"{output_filepath.relative_to(project_root)}")
                print(f"  file size: "
                      f"{output_filepath.stat().st_size / 1024:.2f} KB")
                print()

    return output_filepaths
