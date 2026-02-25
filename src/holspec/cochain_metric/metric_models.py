"""
Cochain metric model construction.

Factory functions for constructing MetricTensor instances and collections
from geometric data, with config-driven dispatch via the metric model registry.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from .metric_tensor import MetricTensor
from holspec.utilities import format_float_str

if TYPE_CHECKING:
    from holspec.simplicial import SimplicialComplex
    from holspec.point_data import PointData


# =============================================================================
# Per-Degree Builders
# =============================================================================

def construct_identity_metric(size: int) -> MetricTensor:
    pass


def construct_diagonal_metric(diagonal_elements: np.ndarray) -> MetricTensor:
    pass


# =============================================================================
# Collection-Level Builders
# =============================================================================

def construct_combinatorial_metrics(sc: SimplicialComplex) -> dict[int, MetricTensor]:
    pass


# =============================================================================
# Registry and Config-Driven Utilities
# =============================================================================

COCHAIN_METRIC_MODEL_REGISTRY: dict[str, callable] = {
    'combinatorial': construct_combinatorial_metrics,
    # 'hodge_star': construct_hodge_star_metrics,  # future
}


def construct_cochain_metric_from_config(
    config: dict,
    sc: SimplicialComplex,
    point_data: PointData,
) -> dict[int, MetricTensor]:
    pass


def create_metric_model_label(
    config: dict,
    float_fmt: str | None = 'g',
    strip_zeros: bool = True,
) -> str:
    pass
