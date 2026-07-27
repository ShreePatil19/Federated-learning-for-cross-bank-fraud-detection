"""Federated triage-and-defer decision layer (calibration -> conformal -> cost -> defer).

Inference-time only: consumes per-bank expert probabilities, never retrains an expert.
See triage/README.md for usage and the statistical-hygiene rules the layer enforces.
"""

from .core import (
    CLEAR,
    DEFER,
    FLAG,
    MIN_FRAUD_FOR_FIT,
    CostMatrix,
    apply_calibrators,
    band_for_defer_budget,
    brier_gate_weights,
    capacity_defer_masks,
    chow_actions,
    combine,
    conformal_fnr_threshold,
    coverage_check,
    deferral_fairness,
    expected_cost,
    fit_calibrators,
    flag_threshold,
    jain_index,
    policy_miss_rate,
    pooled_miss_rate,
    quantile_of_quantiles,
    risk_coverage_curve,
    split_val_indices,
    triage_actions,
    workload_reduction_at_recall,
)
from .integration import (
    DEPLOYABLE_EXPERTS,
    TriageConfig,
    capture_expert_probs,
    load_probs_npz,
    run_from_npz,
    run_triage,
    save_probs_npz,
)

__all__ = [
    "CLEAR", "DEFER", "FLAG", "MIN_FRAUD_FOR_FIT", "CostMatrix",
    "apply_calibrators", "band_for_defer_budget", "brier_gate_weights",
    "capacity_defer_masks", "chow_actions", "combine", "conformal_fnr_threshold",
    "coverage_check", "deferral_fairness", "expected_cost", "fit_calibrators",
    "flag_threshold", "jain_index", "policy_miss_rate", "pooled_miss_rate",
    "quantile_of_quantiles", "risk_coverage_curve", "split_val_indices",
    "triage_actions", "workload_reduction_at_recall",
    "DEPLOYABLE_EXPERTS", "TriageConfig", "capture_expert_probs", "load_probs_npz",
    "run_from_npz", "run_triage", "save_probs_npz",
]
