# Copyright 2026 Shibo Zhao
# Contact: shibowing@gmail.com, shiboz@andrew.cmu.edu
# Please keep the above information when modifying this file.

"""Evaluation: trajectory segmentation, metrics, plotting, and the tester.

Migrated out of the monolithic top-level test.py.
"""

from . import postprocess
from .evaluators import (
    BaseEvaluator,
    BenchmarkEvaluator,
    CrossValidationEvaluator,
    SequenceEvaluator,
    TrajectoryEvaluator,
)
from .metrics import (
    PoseMetrics,
    SequenceMetrics,
    TrajectoryMetrics,
    compute_accruacy_metrics,
    compute_ate,
    compute_ate_rte,
    compute_orientation_error,
    compute_rpe,
    compute_translation_error,
)

__all__ = [
    # Metrics
    "TrajectoryMetrics",
    "PoseMetrics",
    "SequenceMetrics",
    "compute_ate",
    "compute_rpe",
    "compute_orientation_error",
    "compute_translation_error",
    "compute_ate_rte",
    "compute_accruacy_metrics",
    # Evaluators
    "BaseEvaluator",
    "TrajectoryEvaluator",
    "SequenceEvaluator",
    "CrossValidationEvaluator",
    "BenchmarkEvaluator",
    # Postprocess
    "postprocess",
]
