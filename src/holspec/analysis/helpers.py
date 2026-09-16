"""
Helpers for working with spectra analyses across datasets.

Provides entry-label utilities, provenance-based file indexing,
experiment-series persistence, and loaders for saved
:class:`EnsembleSpectraAnalysis` results and experiment series.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from holspec.utilities import save_h5, read_h5
from holspec.pipeline import trace_provenance, select_stage_outputs

from .spectra_analysis import EnsembleSpectraAnalysis


# =============================================================================
# Helpers
# =============================================================================


def entry_group_key(
    entry: dict,
    group_by: list[str],
) -> tuple[str, ...]:
    """Build a group-key tuple from a spectra analysis entry.

    Parses the entry's ``output_label`` to extract ``sc_label`` and
    ``cm_label`` subfields, then returns the tuple of field values
    requested by *group_by*.

    Parameters
    ----------
    entry : dict
        An entry dict as produced by :func:`load_spectra_analyses`, with
        at least keys ``'ptd_label'`` and ``'output_label'``.
    group_by : list of str
        Field names to include in the key.  Valid names are
        ``'ptd_label'``, ``'sc_label'``, and ``'cm_label'``.

    Returns
    -------
    tuple of str
    """
    parts = entry["output_label"].split("__", 1)
    sc_label = parts[0]
    cm_label = parts[1] if len(parts) == 2 else ""
    field_map = {
        "ptd_label": entry["ptd_label"],
        "sc_label": sc_label,
        "cm_label": cm_label,
    }
    return tuple(field_map[f] for f in group_by)


def entry_fixed_varying(
    entry: dict,
    group_by: list[str],
) -> tuple[list[str], str]:
    """Split an entry's label fields into fixed and varying values.

    Partitions the three label fields ``ptd_label``, ``sc_label``,
    ``cm_label`` into those listed in *group_by* (fixed across an
    experiment series) and the single remaining field (the varying one,
    i.e. the experiment parameter).  ``sc_label`` and ``cm_label`` are
    parsed from the entry's ``output_label`` the same way as
    :func:`entry_group_key`.

    Parameters
    ----------
    entry : dict
        An entry dict as produced by :func:`load_spectra_analyses`, with
        at least keys ``'ptd_label'`` and ``'output_label'``.
    group_by : list of str
        Field names that are fixed across the series.  Must be a subset
        of ``{'ptd_label', 'sc_label', 'cm_label'}`` that leaves exactly
        one varying field.

    Returns
    -------
    fixed_values : list of str
        Values of the fixed fields, in *group_by* order.
    varying_value : str
        Value of the field not in *group_by*.

    Raises
    ------
    ValueError
        If *group_by* does not leave exactly one varying field.
    """
    parts = entry["output_label"].split("__", 1)
    sc_label = parts[0]
    cm_label = parts[1] if len(parts) == 2 else ""
    field_map = {
        "ptd_label": entry["ptd_label"],
        "sc_label": sc_label,
        "cm_label": cm_label,
    }

    all_fields = ["ptd_label", "sc_label", "cm_label"]
    varying = [f for f in all_fields if f not in group_by]
    if len(varying) != 1:
        raise ValueError(
            f"group_by must leave exactly one varying field; "
            f"got group_by={group_by}, varying={varying}"
        )

    fixed_values = [field_map[f] for f in group_by]
    varying_value = field_map[varying[0]]
    return fixed_values, varying_value


def extract_exp_params(provenance: dict, exp_params: list[str]) -> dict:
    """
    Extract experimental parameters from a provenance chain.

    Reads upstream file attributes to recover parameter values. External
    imported datasets can expose arbitrary experiment parameters through
    point-data metadata fields ``exp_param`` and ``exp_value``. Built-in
    generated-data parameters such as ``noise`` and ``alpha`` are recovered
    from their stage-specific provenance metadata.
    Returns a dict with keys for all requested parameters; missing
    parameters are stored as None.

    Parameters
    ----------
    provenance : dict
        Provenance dict mapping stage names to file paths.
    exp_params : list of str
        Names of experimental parameters to extract (e.g.
        ``['noise', 'alpha', 'area_fraction']``).

    Returns
    -------
    dict
        {param_name: value or None}.
    """
    result = {name: None for name in exp_params}

    for name in exp_params:
        ptd_filepath = provenance.get("point_data")
        ptd_attrs = None

        if ptd_filepath is not None:
            _, ptd_attrs = read_h5(ptd_filepath, dataset_names=[])
            metadata = ptd_attrs.get("metadata", {})
            if isinstance(metadata, dict) and metadata.get("exp_param") == name:
                result[name] = metadata.get("exp_value")
                continue

        if name == "noise":
            if ptd_attrs is None:
                continue
            noise_config = ptd_attrs.get("noise_config")
            if noise_config is not None:
                result["noise"] = noise_config.get("scale")

        elif name == "alpha":
            sc_filepath = provenance.get("topology_simplicial")
            if sc_filepath is None:
                continue
            _, sc_attrs = read_h5(sc_filepath, dataset_names=[])
            stage_config = sc_attrs.get("stage_config", {})
            result["alpha"] = stage_config.get("params", {}).get("alpha")

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

            ptd_label = (
                provenance["point_data"].stem if "point_data" in provenance else None
            )
            sc_label = (
                provenance["topology_simplicial"].stem
                if "topology_simplicial" in provenance
                else None
            )
            cm_stem = (
                provenance["geometry_metric"].stem
                if "geometry_metric" in provenance
                else None
            )
            if (
                cm_stem is not None
                and sc_label is not None
                and cm_stem.startswith(sc_label + "__")
            ):
                cm_label = cm_stem[len(sc_label) + 2 :]
            else:
                cm_label = cm_stem

            records[filepath] = {
                "filepath": filepath,
                "ptd_label": ptd_label,
                "sc_label": sc_label,
                "cm_label": cm_label,
                "output_label": output_label,
                "provenance": provenance,
            }

    return records


def compute_transition_points(
    series: dict,
) -> dict[tuple[int, str], float]:
    """
    Per-(k, component) phase-transition point for an experiment series.

    For each (k, component) key, returns the midpoint between the two
    consecutive ``exp_values`` at which ``distance_series[(k, comp)]``
    peaks.  Returns NaN for keys whose distance series is empty.

    Parameters
    ----------
    series : dict
        Experiment series dict with keys ``'exp_values'``,
        ``'analysis_keys'``, and ``'distance_series'``.

    Returns
    -------
    dict[tuple[int, str], float]
        Mapping ``{(k, component): transition_point}``.
    """
    exp_values = np.asarray(series["exp_values"])
    analysis_keys = series["analysis_keys"]
    distance_series = series["distance_series"]

    midpoints = 0.5 * (exp_values[:-1] + exp_values[1:])

    transition_points = {}
    for k, comp in analysis_keys:
        dists = np.asarray(distance_series[(k, comp)])
        if dists.size == 0:
            transition_points[(k, comp)] = float("nan")
        else:
            transition_points[(k, comp)] = float(midpoints[np.argmax(dists)])
    return transition_points


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
        Name of the dataset (e.g. 'exp_noise_trilatt').
    group_key : tuple of str
        Fields identifying this series (e.g. ('delaunay', 'combinatorial')).
    series : dict
        Series data with keys: 'exp_param', 'exp_values',
        'observable_series', 'distribution_series', 'distance_series',
        'transition_points'.
    analysis_config : dict
        Analysis configuration (native Python types; HDF5 conversion
        is handled by save_h5 / to_h5_attribute).
    """
    filepath = Path(filepath)
    if filepath.exists():
        filepath.unlink()

    ac = analysis_config
    analysis_keys = ac.get(
        "analysis_keys", [(k, comp) for k in ac["degrees"] for comp in ac["components"]]
    )
    observable_names = ac["observable_names"]

    root_attributes = {
        "dataset_name": dataset_name,
        "exp_param": series["exp_param"],
        "group_key": group_key,
        "group_label": "__".join(group_key),
        "analysis_keys": analysis_keys,
        **{k: v for k, v in ac.items() if k != "analysis_keys"},
    }
    save_h5(
        filepath,
        datasets={"exp_values": series["exp_values"]},
        attributes=root_attributes,
        mode="create",
    )

    for k, comp in analysis_keys:
        obs = series["observable_series"][(k, comp)]
        obs_datasets = {}
        for obs_name in observable_names:
            obs_datasets[f"{obs_name}_mean"] = obs[obs_name]["mean"]
            obs_datasets[f"{obs_name}_std"] = obs[obs_name]["std"]
        save_h5(
            filepath,
            datasets=obs_datasets,
            group=f"observables/degree_{k}_{comp}",
            mode="create",
        )

    for k, comp in analysis_keys:
        dist = series["distribution_series"][(k, comp)]
        save_h5(
            filepath,
            datasets={"x": dist["x"], "density_stack": dist["density_stack"]},
            group=f"distributions/degree_{k}_{comp}",
            mode="create",
        )

    for k, comp in analysis_keys:
        save_h5(
            filepath,
            datasets={f"degree_{k}_{comp}": series["distance_series"][(k, comp)]},
            group="distances",
            mode="update",
        )

    for k, comp in analysis_keys:
        save_h5(
            filepath,
            datasets={f"degree_{k}_{comp}": series["transition_points"][(k, comp)]},
            group="transitions",
            mode="update",
        )


# =============================================================================
# Loading
# =============================================================================


def load_spectra_analyses(
    spectra_datasets: dict,
    spectra_analysis_dir: Path | str,
    project_root: Path | str,
    plot_datasets: list[str] | None = None,
    verbose: bool = False,
    float_fmt: str = ".4g",
) -> dict[str, dict[str, dict]]:
    """Load per-file spectra analyses from saved results.

    Uses *spectra_datasets* configuration to select pipeline files per
    dataset, then loads the corresponding saved
    :class:`EnsembleSpectraAnalysis` from *spectra_analysis_dir*.  For
    experiment datasets (those with a non-null ``exp_param``), entries
    are sorted by experimental parameter value.

    Parameters
    ----------
    spectra_datasets : dict
        YAML-loaded dataset definitions mapping dataset name to config
        dict.  Each config may have keys ``base_dir``,
        ``select_point_data``, ``select_simplicial_complex``,
        ``select_cochain_metric``, and optionally ``exp_param``.
    spectra_analysis_dir : Path or str
        Directory containing saved ESA HDF5 files (typically
        ``DATA_PROCESSED_DIR / 'spectra_analysis'``).
    project_root : Path or str
        Project root for resolving provenance paths.
    plot_datasets : list of str or None, default None
        Dataset names to load.  ``None`` loads all datasets.
    verbose : bool, default False
        If True, print per-file detail lines.
    float_fmt : str, default '.4g'
        Format specifier for experimental parameter values in summary.

    Returns
    -------
    dict[str, dict[str, dict]]
        ``{dataset_name: {label: entry_dict}}``.  Each *entry_dict*
        contains keys ``'esa'``, ``'metadata'``, ``'esa_path'``,
        ``'source_file'``, ``'ptd_label'``, ``'output_label'``,
        ``'exp_value'``.
    """
    spectra_analysis_dir = Path(spectra_analysis_dir)
    project_root = Path(project_root)

    result = {}

    for dataset_name, dataset_config in spectra_datasets.items():
        if plot_datasets is not None and dataset_name not in plot_datasets:
            continue

        spectra_filepaths = select_stage_outputs(
            stage_num=4,
            project_root=project_root,
            base_dir=dataset_config.get("base_dir"),
            select_point_data=dataset_config.get("select_point_data", []),
            select_simplicial_complex=dataset_config.get(
                "select_simplicial_complex", []
            ),
            select_cochain_metric=dataset_config.get("select_cochain_metric", []),
        )

        exp_param = dataset_config.get("exp_param")
        entries_list = []

        for ptd_dir_label, output_dict in spectra_filepaths.items():
            for output_label, pipeline_filepath in output_dict.items():
                pipeline_filepath = Path(pipeline_filepath).resolve()
                subdir = pipeline_filepath.parent.name
                stem = pipeline_filepath.stem

                esa_path = spectra_analysis_dir / subdir / f"{stem}.h5"
                if not esa_path.exists():
                    print(f"  WARNING: no saved ESA for {subdir}/{stem}, skipping")
                    continue

                _, attrs = read_h5(esa_path, dataset_names=[])
                degrees = list(attrs.get("degrees", [0, 1, 2]))
                components = tuple(attrs.get("components", ["full"]))
                metadata = attrs.get("metadata", {})

                esa = EnsembleSpectraAnalysis.from_file(
                    pipeline_filepath,
                    project_root,
                    degrees=degrees,
                    components=components,
                )
                esa.load_cache(esa_path)

                exp_value = None
                if exp_param is not None:
                    provenance = trace_provenance(pipeline_filepath, project_root)
                    provenance = {k: Path(v).resolve() for k, v in provenance.items()}
                    exp_params_dict = extract_exp_params(provenance, [exp_param])
                    exp_value = exp_params_dict.get(exp_param)

                label = f"{subdir}/{stem}"
                entries_list.append(
                    (
                        label,
                        {
                            "esa": esa,
                            "metadata": metadata,
                            "esa_path": esa_path,
                            "source_file": pipeline_filepath,
                            "ptd_label": ptd_dir_label,
                            "output_label": output_label,
                            "exp_value": exp_value,
                        },
                    )
                )

        if exp_param is not None:
            entries_list.sort(
                key=lambda x: (x[1]["output_label"], x[1]["exp_value"] or 0)
            )

        result[dataset_name] = dict(entries_list)

    # Print summary
    total_files = sum(len(analyses) for analyses in result.values())
    print(
        f"Loaded spectra analyses for {total_files} files "
        f"across {len(result)} dataset(s)."
    )

    if verbose:
        print()
        for dataset_name, analyses in result.items():
            print("=" * 80)
            print(f"Dataset: {dataset_name} ({len(analyses)} files)")
            print("=" * 80)
            print()
            for label, entry in analyses.items():
                esa = entry["esa"]
                exp_param = spectra_datasets[dataset_name].get("exp_param")
                exp_str = (
                    f" ({exp_param}={entry['exp_value']:{float_fmt}})"
                    if entry.get("exp_value") is not None
                    else ""
                )
                print(f"  {entry['ptd_label']} / {entry['output_label']}{exp_str}")
                print(
                    f"    EnsembleSpectraAnalysis(num_members={esa.num_members}, "
                    f"degrees={esa.degrees}, components={esa.components})"
                )
                print(f"    source: {entry['source_file'].relative_to(project_root)}")
                print(f"    cache:  {entry['esa_path'].relative_to(project_root)}")
            print()

    return result


def load_experiment_series(
    exp_series_dir: Path | str,
    plot_datasets: list[str] | None = None,
    verbose: bool = False,
) -> dict[tuple, dict]:
    """Load experiment series from saved HDF5 results.

    Reads each HDF5 file in *exp_series_dir* and reconstructs the
    experiment series dict.

    Parameters
    ----------
    exp_series_dir : Path or str
        Directory containing experiment series HDF5 files (typically
        ``DATA_PROCESSED_DIR / 'exp_series'``).
    plot_datasets : list of str or None, default None
        Dataset names to load.  ``None`` loads all.
    verbose : bool, default False
        If True, print per-series detail lines.

    Returns
    -------
    dict[tuple, dict]
        ``{(dataset_name, group_key_tuple): series_dict}``.  Each
        *series_dict* contains keys ``'exp_param'``, ``'exp_values'``,
        ``'group_label'``, ``'observable_series'``,
        ``'distribution_series'``, ``'distance_series'``,
        ``'transition_points'``, ``'analysis_keys'``,
        ``'observable_names'``.

        ``'transition_points'`` is ``{}`` for files saved before
        transition points were persisted.
    """
    exp_series_dir = Path(exp_series_dir)
    result = {}

    for filepath in sorted(exp_series_dir.glob("*.h5")):
        ds_root, attrs = read_h5(filepath, dataset_names=["exp_values"])
        exp_values = ds_root["exp_values"]
        exp_param = str(attrs["exp_param"])
        dataset_name = str(attrs["dataset_name"])

        if plot_datasets is not None and dataset_name not in plot_datasets:
            continue

        group_key = tuple(str(g) for g in attrs["group_key"])
        group_label = str(attrs.get("group_label", "__".join(group_key)))
        analysis_keys = [tuple(ak) for ak in attrs["analysis_keys"]]
        observable_names = list(attrs["observable_names"])

        observable_series = {}
        for k, comp in analysis_keys:
            obs_ds, _ = read_h5(filepath, group=f"observables/degree_{k}_{comp}")
            obs_dict = {}
            for obs_name in observable_names:
                obs_dict[obs_name] = {
                    "mean": obs_ds[f"{obs_name}_mean"],
                    "std": obs_ds[f"{obs_name}_std"],
                }
            observable_series[(int(k), comp)] = obs_dict

        distribution_series = {}
        for k, comp in analysis_keys:
            dist_ds, _ = read_h5(filepath, group=f"distributions/degree_{k}_{comp}")
            distribution_series[(int(k), comp)] = {
                "x": dist_ds["x"],
                "density_stack": dist_ds["density_stack"],
            }

        distance_series = {}
        for k, comp in analysis_keys:
            dist_ds, _ = read_h5(
                filepath, group="distances", dataset_names=[f"degree_{k}_{comp}"]
            )
            distance_series[(int(k), comp)] = dist_ds[f"degree_{k}_{comp}"]

        transition_points = {}
        try:
            for k, comp in analysis_keys:
                trans_ds, _ = read_h5(
                    filepath, group="transitions", dataset_names=[f"degree_{k}_{comp}"]
                )
                transition_points[(int(k), comp)] = float(
                    trans_ds[f"degree_{k}_{comp}"]
                )
        except KeyError:
            transition_points = {}

        series_key = (dataset_name, group_key)
        result[series_key] = {
            "exp_param": exp_param,
            "exp_values": exp_values,
            "group_label": group_label,
            "observable_series": observable_series,
            "distribution_series": distribution_series,
            "distance_series": distance_series,
            "transition_points": transition_points,
            "analysis_keys": [(int(k), comp) for k, comp in analysis_keys],
            "observable_names": observable_names,
        }

    # Print summary
    print(f"Loaded {len(result)} experiment series from {exp_series_dir}/")

    if verbose:
        print()
        for series_key, series in result.items():
            dataset_name, group_key = series_key
            group_label = series["group_label"]
            fp = exp_series_dir / f"{dataset_name}__{group_label}.h5"
            size_kb = fp.stat().st_size / 1024 if fp.exists() else 0
            exp_param = series["exp_param"]
            exp_values = series["exp_values"]
            print(f"  {series_key}")
            print(f"    exp_param: {exp_param}")
            print(f"    n_params: {len(exp_values)}")
            print(f"    {exp_param} range: [{exp_values[0]:.4g}, {exp_values[-1]:.4g}]")
            print(f"    file: {fp.name} ({size_kb:.1f} KB)")

    return result
