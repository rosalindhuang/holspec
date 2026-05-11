# holspec/cochain_metric/__init__.py
"""
Cochain metrics for geometry on simplicial complexes.

This subpackage defines the geometric layer of the holspec pipeline: metric
tensors on cochain spaces and metric models that assign inner products to
simplicial complexes.
"""

from .base import CochainMetric
from .metric_tensor import MetricTensor
from .metric_models import (
    construct_diagonal_metric,
    construct_combinatorial_cochain_metric,
    construct_hodge_star_cochain_metric,
    construct_cochain_metric_from_config,
    create_metric_model_label,
    COCHAIN_METRIC_MODEL_REGISTRY,
)
from .validation import (
    validate_metric_tensor,
    validate_cochain_metric,
    METRIC_POSITIVITY_TOL,
)

__all__ = [
    # Core classes
    'CochainMetric',
    'MetricTensor',

    # metric_models.py
    'construct_diagonal_metric',
    'construct_combinatorial_cochain_metric',
    'construct_hodge_star_cochain_metric',
    'construct_cochain_metric_from_config',
    'create_metric_model_label',
    'COCHAIN_METRIC_MODEL_REGISTRY',

    # validation.py
    'validate_metric_tensor',
    'validate_cochain_metric',
    'METRIC_POSITIVITY_TOL',
]
