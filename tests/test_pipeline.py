"""
API-level smoke tests for pipeline orchestration.

These tests exercise the public Python pipeline path directly, verifying that a
tiny persisted input can flow through all stages and be loaded back by
provenance-aware helpers.
"""

from pathlib import Path

import numpy as np

from holspec.pipeline import (
    load_spectra,
    run_pipeline,
    run_pipeline_stages,
    select_pipeline_inputs,
    select_stage_outputs,
    trace_provenance,
)
from holspec.point_data import PointData, PointDataEnsemble
from holspec.utilities import save_h5


def test_run_pipeline_tiny_vietoris_rips_workflow(tmp_path: Path):
    project_root = tmp_path
    input_path = project_root / "data" / "raw" / "pipeline" / "triangle.h5"

    _save_triangle_point_data_input(input_path)
    config = _tiny_pipeline_config()

    results = run_pipeline(config=config, project_root=project_root)

    expected_stages = [
        "topology_simplicial",
        "geometry_metric",
        "hodge_laplacian",
        "spectra",
    ]
    assert list(results) == expected_stages

    for stage_name in expected_stages:
        assert list(results[stage_name]) == ["triangle"]
        assert len(results[stage_name]["triangle"]) == 1
        for output_path in results[stage_name]["triangle"].values():
            assert output_path.exists()
            assert output_path.is_relative_to(project_root)

    output_root = project_root / "data" / "interim" / "pipeline_smoke"
    topology_path = output_root / "topology_simplicial" / "triangle" / "vr.h5"
    metric_path = (
        output_root / "geometry_metric" / "triangle" / "vr__combinatorial.h5"
    )
    hodge_laplacian_path = (
        output_root / "hodge_laplacian" / "triangle" / "vr__combinatorial.h5"
    )
    spectra_path = output_root / "spectra" / "triangle" / "vr__combinatorial.h5"

    for output_path in (
        topology_path,
        metric_path,
        hodge_laplacian_path,
        spectra_path,
    ):
        assert output_path.exists()

    provenance = trace_provenance(spectra_path, project_root)
    assert list(provenance) == [
        "spectra",
        "hodge_laplacian",
        "geometry_metric",
        "topology_simplicial",
        "point_data",
    ]
    assert provenance["spectra"] == spectra_path
    assert provenance["hodge_laplacian"] == hodge_laplacian_path
    assert provenance["geometry_metric"] == metric_path
    assert provenance["topology_simplicial"] == topology_path
    assert provenance["point_data"] == input_path

    spectra = load_spectra(spectra_path, project_root, group="member_0000")

    assert spectra.dimensions == {0: 3, 1: 3, 2: 1}
    full_spectrum = spectra.spectrum(1, "full")
    assert full_spectrum.dimension == 3
    assert full_spectrum.is_complete
    assert full_spectrum.eigenvectors is None
    assert np.all(np.diff(full_spectrum.eigenvalues) >= 0)
    assert np.all(full_spectrum.eigenvalues >= 0)


def test_run_pipeline_stages_tiny_vietoris_rips_workflow(tmp_path: Path):
    project_root = tmp_path
    input_path = project_root / "data" / "raw" / "pipeline" / "triangle.h5"

    _save_triangle_point_data_input(input_path)
    config = _tiny_pipeline_config()

    results_12 = run_pipeline_stages(
        config=config,
        project_root=project_root,
        stage_nums=(1, 2),
    )
    results_34 = run_pipeline_stages(
        config=config,
        project_root=project_root,
        stage_nums=(3, 4),
        input_filepaths=results_12["geometry_metric"],
    )

    assert list(results_12) == ["topology_simplicial", "geometry_metric"]
    assert list(results_34) == ["hodge_laplacian", "spectra"]

    for stage_results in (*results_12.values(), *results_34.values()):
        assert list(stage_results) == ["triangle"]
        assert len(stage_results["triangle"]) == 1
        for output_path in stage_results["triangle"].values():
            assert output_path.exists()
            assert output_path.is_relative_to(project_root)

    final_spectra_path = next(iter(results_34["spectra"]["triangle"].values()))
    spectra = load_spectra(final_spectra_path, project_root, group="member_0000")

    assert spectra.dimensions == {0: 3, 1: 3, 2: 1}
    assert spectra.spectrum(0, "full").is_complete
    assert spectra.spectrum(0, "full").eigenvectors is None


def test_select_pipeline_inputs_with_generated_raw_data_dir(tmp_path: Path):
    project_root = tmp_path
    generated_raw_dir = project_root / "data" / "raw" / "generated"
    selected_path = generated_raw_dir / "smoke" / "triangle.h5"
    skipped_path = generated_raw_dir / "skip" / "square.h5"
    selected_path.parent.mkdir(parents=True)
    skipped_path.parent.mkdir(parents=True)
    selected_path.touch()
    skipped_path.touch()

    results = select_pipeline_inputs(
        project_root,
        raw_data_dir="data/raw/generated",
        select_datasets=["smoke"],
    )

    assert results == {"triangle": selected_path}

    alias_results = select_pipeline_inputs(
        project_root,
        raw_data_dir="data/raw/generated",
        select_categories=["smoke"],
    )

    assert alias_results == results


def test_select_stage_outputs_with_generated_raw_data_dir(tmp_path: Path):
    project_root = tmp_path

    raw_selected = (
        project_root / "data" / "raw" / "generated" / "smoke" / "triangle.h5"
    )
    raw_skipped = (
        project_root / "data" / "raw" / "generated" / "skip" / "square.h5"
    )
    raw_selected.parent.mkdir(parents=True)
    raw_skipped.parent.mkdir(parents=True)
    raw_selected.touch()
    raw_skipped.touch()

    selected_output = (
        project_root
        / "data"
        / "interim"
        / "stage_smoke"
        / "spectra"
        / "triangle"
        / "vr__combinatorial.h5"
    )
    skipped_output = (
        project_root
        / "data"
        / "interim"
        / "stage_smoke"
        / "spectra"
        / "square"
        / "vr__combinatorial.h5"
    )
    selected_output.parent.mkdir(parents=True)
    skipped_output.parent.mkdir(parents=True)
    selected_output.touch()
    skipped_output.touch()

    results = select_stage_outputs(
        stage_num=4,
        project_root=project_root,
        base_dir="data/interim/stage_smoke",
        raw_data_dir="data/raw/generated",
        select_datasets=["smoke"],
    )

    assert results == {
        "triangle": {
            "vr__combinatorial": selected_output,
        },
    }


def _save_triangle_point_data_input(input_path: Path) -> None:
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.8],
        ],
    )
    ensemble = PointDataEnsemble(
        members=[PointData(positions=positions)],
        metadata={"source": "tests/test_pipeline.py"},
    )
    ensemble.save(input_path)
    save_h5(
        input_path,
        attributes={
            "created_by": "tests/test_pipeline.py",
            "creation_time": "2026-05-13T00:00:00",
            "stage_name": "point_data",
            "stage_config": {"source": "inline test fixture"},
        },
        group=None,
        mode="update",
    )


def _tiny_pipeline_config() -> dict:
    return {
        "summary": {
            "created_by": "tests/test_pipeline.py",
            "creation_time": "2026-05-13T00:00:00",
        },
        "inputs": {
            "data_dir": "data/raw",
            "filepaths": {
                "triangle": "data/raw/pipeline/triangle.h5",
            },
        },
        "stages": {
            "topology_simplicial": {
                "configs": {
                    "simplicial_constructions": {
                        "vr": {
                            "method": "vietoris_rips",
                            "params": {
                                "epsilon": 1.1,
                                "max_dim": 2,
                            },
                        },
                    },
                },
                "runtime": {
                    "cache_incidence": False,
                    "validate_boundary_property": False,
                    "verbose": False,
                },
            },
            "geometry_metric": {
                "configs": {
                    "metric_models": {
                        "combinatorial": {
                            "model": "combinatorial",
                            "params": {},
                        },
                    },
                },
                "runtime": {
                    "validate_metric": True,
                    "verbose": False,
                },
            },
            "hodge_laplacian": {
                "configs": {},
                "runtime": {
                    "cache_laplacians": False,
                    "validate_laplacians": False,
                    "verbose": False,
                },
            },
            "spectra": {
                "configs": {
                    "compute_spectra": True,
                    "solver": "dense",
                    "compute_eigenvectors": False,
                },
                "runtime": {
                    "verbose": False,
                },
            },
        },
        "runtime": {
            "verbose": False,
            "save_stage_configs": False,
            "error_handling": "raise",
        },
        "outputs": {
            "data_dir": "data/interim/pipeline_smoke",
            "configs_dir": "configs/pipeline_smoke_stages",
        },
    }
