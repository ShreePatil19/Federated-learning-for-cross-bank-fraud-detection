"""Tests for the triage decision layer.

The regression tests encode the failure modes found in the prototype audit: each one fails
on the prototype's semantics and passes on this implementation.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from triage import core as T
from triage.integration import (
    DEPLOYABLE_EXPERTS, TriageConfig, load_probs_npz, run_triage, save_probs_npz,
)

RNG = np.random.default_rng(1234)


def synth_scores(rng, n, base_rate=0.03, mu1=0.7, mu0=0.2, sd=0.15):
    y = (rng.random(n) < base_rate).astype(int)
    s = np.clip(rng.normal(np.where(y == 1, mu1, mu0), sd), 0, 1)
    return s, y


# ----------------------------------------------------------------------------------
# Conformal layer
# ----------------------------------------------------------------------------------

def test_conformal_marginal_guarantee():
    """Mean test miss-rate over calibration draws stays at/below the target."""
    rng = np.random.default_rng(7)
    target, misses = 0.10, []
    for _ in range(200):
        s, y = synth_scores(rng, 6000)
        h = len(s) // 2
        t = T.conformal_fnr_threshold(s[:h], y[:h], target)
        c = T.coverage_check(s[h:], y[h:], t, target)
        if not np.isnan(c["miss_rate"]):
            misses.append(c["miss_rate"])
    assert np.mean(misses) <= target + 0.01


def test_conformal_insufficient_fraud_returns_none():
    """Too little validation fraud -> None (federated fallback), never a flag-all -inf."""
    rng = np.random.default_rng(0)
    s = rng.random(200)
    # below min_fraud
    y3 = np.zeros(200, int); y3[:3] = 1
    assert T.conformal_fnr_threshold(s, y3, 0.10) is None
    # n1 = 8: k = floor(0.1 * 9) = 0 -> cannot support the target, fall back (None)
    y8 = np.zeros(200, int); y8[:8] = 1
    assert T.conformal_fnr_threshold(s, y8, 0.10) is None
    # n1 = 9: k = floor(0.1 * 10) = 1 -> smallest usable fold, real threshold
    y9 = np.zeros(200, int); y9[:9] = 1
    assert T.conformal_fnr_threshold(s, y9, 0.10) is not None


def test_double_dip_regression():
    """Fitting isotonic calibrators and the conformal quantile on the SAME fold breaks the
    guarantee with a multi-expert ensemble; disjoint halves restore it."""
    rng = np.random.default_rng(11)
    target, E, trials = 0.10, 7, 40
    miss_same, miss_split = [], []
    for _ in range(trials):
        n_val, n_te = 400, 4000
        y_val = (rng.random(n_val) < 0.08).astype(int)
        y_te = (rng.random(n_te) < 0.08).astype(int)
        val_p = [np.clip(rng.normal(np.where(y_val == 1, 0.6, 0.35), 0.22), 0, 1)
                 for _ in range(E)]
        te_p = [np.clip(rng.normal(np.where(y_te == 1, 0.6, 0.35), 0.22), 0, 1)
                for _ in range(E)]

        # same-fold (prototype semantics)
        cals = T.fit_calibrators(val_p, y_val, min_fraud=1)
        s_val = T.combine(T.apply_calibrators(cals, val_p))
        s_te = T.combine(T.apply_calibrators(cals, te_p))
        t = T.conformal_fnr_threshold(s_val, y_val, target, min_fraud=1)
        if t is not None:
            miss_same.append(T.coverage_check(s_te, y_te, t, target)["miss_rate"])

        # split-fold (this implementation)
        cal_idx, conf_idx, ok = T.split_val_indices(y_val, rng)
        assert ok
        cals2 = T.fit_calibrators([p[cal_idx] for p in val_p], y_val[cal_idx], min_fraud=1)
        s_val2 = T.combine(T.apply_calibrators(cals2, val_p))
        s_te2 = T.combine(T.apply_calibrators(cals2, te_p))
        t2 = T.conformal_fnr_threshold(s_val2[conf_idx], y_val[conf_idx], target,
                                       min_fraud=1)
        if t2 is not None:
            miss_split.append(T.coverage_check(s_te2, y_te, t2, target)["miss_rate"])

    assert np.mean(miss_split) <= target + 0.02, (
        f"split-fold guarantee broken: {np.mean(miss_split):.3f}")
    assert np.mean(miss_same) > np.mean(miss_split), (
        "same-fold should be anti-conservative relative to split-fold")


def test_quantile_of_quantiles():
    assert T.quantile_of_quantiles([]) is None
    assert T.quantile_of_quantiles([None, None]) is None
    # method='lower' picks an actual element at/below the interpolated median
    assert T.quantile_of_quantiles([0.1, 0.2, 0.3, 0.4]) == 0.2


# ----------------------------------------------------------------------------------
# Action rules
# ----------------------------------------------------------------------------------

def test_flag_threshold_composition_bounds_policy_miss():
    """tau = min(bayes_cut, t): with a high-rho cut ABOVE t, acting on the raw cut would
    clear fraud the conformal layer promised to catch; the composed policy may not."""
    rng = np.random.default_rng(3)
    n = 20000
    y = (rng.random(n) < 0.05).astype(int)
    s = np.clip(rng.normal(np.where(y == 1, 0.6, 0.45), 0.12), 0, 1)
    h = n // 2
    target = 0.10
    t = T.conformal_fnr_threshold(s[:h], y[:h], target)
    cost = T.CostMatrix(c_fn=1, c_fp=1)          # rho=1 -> cut=0.5, typically > t
    tau = T.flag_threshold(cost, t)
    assert tau <= cost.bayes_cut()
    acts = T.triage_actions(s[h:], tau, band=0.0)
    naive = T.triage_actions(s[h:], cost.bayes_cut(), band=0.0)
    assert T.policy_miss_rate(y[h:], acts) <= T.policy_miss_rate(y[h:], naive) + 1e-12
    assert T.policy_miss_rate(y[h:], acts) <= target + 0.03


def test_band_comes_from_reference_scores():
    """The band is a function of reference (val) scores only; scoring batches of any size
    see the same operating parameter."""
    rng = np.random.default_rng(4)
    s_ref = rng.random(5000)
    band = T.band_for_defer_budget(s_ref, 0.5, 0.05)
    assert band == T.band_for_defer_budget(s_ref, 0.5, 0.05)
    frac_ref = np.mean(np.abs(s_ref - 0.5) < band)
    assert frac_ref <= 0.05 + 1e-9   # method='lower' approaches the budget from below
    assert T.band_for_defer_budget(np.array([]), 0.5, 0.05) == 0.0


def test_chow_rule():
    rng = np.random.default_rng(5)
    s = rng.random(4000)
    cost = T.CostMatrix(c_fn=10, c_fp=1)
    # human too expensive -> never defer
    acts = T.chow_actions(s, cost, c_defer=100.0)
    assert not np.any(acts == T.DEFER)
    # defer region is asymmetric around the cut under asymmetric costs
    acts = T.chow_actions(s, cost, c_defer=0.5)
    deferred = s[acts == T.DEFER]
    cut = cost.bayes_cut()
    assert deferred.size > 0
    assert deferred.max() - cut > cut - deferred.min() - 1e-9
    # boundary point is always deferred while c_defer < the boundary risk
    assert T.chow_actions(np.array([cut]), cost, c_defer=0.5)[0] == T.DEFER


def test_capacity_defer_masks():
    rng = np.random.default_rng(6)
    s_by_bank = [rng.random(1000), rng.random(500), rng.random(1500)]
    masks = T.capacity_defer_masks(s_by_bank, [0.5, 0.5, 0.5], 0.05)
    total = sum(m.sum() for m in masks)
    assert total == int(np.floor(0.05 * 3000))
    # deferred items are globally the most ambiguous
    all_margin = np.concatenate([np.abs(s - 0.5) for s in s_by_bank])
    all_mask = np.concatenate(masks)
    assert all_margin[all_mask].max() <= all_margin[~all_mask].min() + 1e-12


# ----------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------

def test_expected_cost_and_human_accuracy():
    y = np.array([1, 1, 0, 0, 1, 0])
    acts = np.array([T.CLEAR, T.FLAG, T.FLAG, T.CLEAR, T.DEFER, T.DEFER], dtype=object)
    cost = T.CostMatrix(c_fn=10, c_fp=1)
    # perfect human: 1 miss (10) + 1 false alarm (1) + 2 reviews (0.5)
    assert T.expected_cost(y, acts, cost, 0.25) == pytest.approx(11.5)
    # imperfect human adds expected residuals on the deferred fraud + legit
    got = T.expected_cost(y, acts, cost, 0.25, human_accuracy=0.8)
    assert got == pytest.approx(11.5 + 0.2 * 10 + 0.2 * 1)


def test_workload_threshold_is_reference_only():
    rng = np.random.default_rng(8)
    s_ref, y_ref = synth_scores(rng, 8000)
    s_ev, y_ev = synth_scores(rng, 8000)
    wl = T.workload_reduction_at_recall(s_ref, y_ref, s_ev, y_ev, 0.80)
    # permuting evaluation labels must not move the threshold (no test-label peeking)
    wl2 = T.workload_reduction_at_recall(s_ref, y_ref, s_ev,
                                         np.zeros_like(y_ev), 0.80)
    assert wl["threshold"] == wl2["threshold"]
    assert wl["achieved_recall"] == pytest.approx(0.80, abs=0.08)


def test_risk_coverage_endpoints_and_cost_weighting():
    rng = np.random.default_rng(9)
    s, y = synth_scores(rng, 6000)
    cost = T.CostMatrix(c_fn=10, c_fp=1)
    tau = cost.bayes_cut()          # 0.091: the cut that made 0/1 AURC explode
    rc01 = T.risk_coverage_curve(s, y, tau)
    rccost = T.risk_coverage_curve(s, y, tau, cost=cost)
    assert rc01["coverage"][0] == 0.0 and rc01["selective_risk"][0] == 0.0
    # cost-weighted risk at full coverage = mean per-item cost, bounded by c_fp + a bit
    assert rccost["full_risk"] < 1.5
    empty = T.risk_coverage_curve(np.array([]), np.array([]), 0.5)
    assert np.isnan(empty["aurc"])


def test_guards():
    with pytest.raises(ValueError):
        T.CostMatrix(c_fn=10, c_fp=0)
    with pytest.raises(ValueError):
        T.combine([np.array([0.5]), np.array([0.5])], weights=np.array([0.0, 0.0]))
    assert np.isnan(T.jain_index([]))
    assert T.jain_index([0.05, 0.05, 0.05]) == pytest.approx(1.0)
    c = T.coverage_check(np.array([0.9]), np.array([0]), 0.5, 0.1)
    assert c["covers"] is None and c["n_fraud"] == 0


def test_calibrator_fallback_on_scarce_fraud():
    rng = np.random.default_rng(10)
    y = np.zeros(300, int); y[:3] = 1          # 3 positives < MIN_FRAUD_FOR_FIT
    probs = [rng.random(300) for _ in range(3)]
    cals = T.fit_calibrators(probs, y)
    assert all(c is None for c in cals)
    out = T.apply_calibrators(cals, probs)
    for a, b in zip(out, probs):
        np.testing.assert_array_equal(a, b)


# ----------------------------------------------------------------------------------
# Integration
# ----------------------------------------------------------------------------------

def _fake_capture(rng, n_banks=3, n_val=600, n_te=1200, experts=DEPLOYABLE_EXPERTS):
    probs, y_val, y_test = {}, {}, {}
    for bid in range(n_banks):
        yv = (rng.random(n_val) < 0.06).astype(int)
        yt = (rng.random(n_te) < 0.06).astype(int)
        y_val[bid], y_test[bid] = yv, yt
        probs[bid] = {}
        for en in experts:
            skill = 0.15 + 0.1 * rng.random()
            pv = np.clip(rng.normal(np.where(yv == 1, 0.4 + skill, 0.3), 0.18), 0, 1)
            pt = np.clip(rng.normal(np.where(yt == 1, 0.4 + skill, 0.3), 0.18), 0, 1)
            probs[bid][en] = (pv, pt)
    return probs, y_val, y_test


def test_run_triage_rows_and_determinism():
    rng = np.random.default_rng(21)
    probs, y_val, y_test = _fake_capture(rng)
    rows = run_triage(y_val, y_test, probs, "synthetic", 0.5, 42)
    cfgd = TriageConfig()
    assert len(rows) == len(cfgd.rhos)
    expected_cols = {"dataset", "alpha", "seed", "rho", "target_fnr", "defer_budget",
                     "defer_rule", "gate", "miss_rate_audit", "miss_rate_policy",
                     "coverage_valid_frac", "expected_cost", "expected_cost_per_txn",
                     "aurc", "auto_clear_at_recall80", "fpr_at_recall80",
                     "achieved_recall80", "defer_jain", "fpr_gap", "defer_frac",
                     "n_banks", "n_banks_with_fraud", "n_banks_fallback_threshold",
                     "n_banks_split_ok", "human_accuracy"}
    assert expected_cols <= set(rows[0])
    # deterministic: same coordinates -> identical rows (resume-safety)
    rows2 = run_triage(y_val, y_test, probs, "synthetic", 0.5, 42)
    assert rows == rows2
    # the policy the guarantee is about: composed tau keeps policy miss near target
    r10 = [r for r in rows if r["rho"] == 10][0]
    assert r10["miss_rate_policy"] <= 0.10 + 0.10  # slack: 3 banks, finite fraud counts


def test_run_triage_defer_rules_differ():
    rng = np.random.default_rng(22)
    probs, y_val, y_test = _fake_capture(rng)
    band = run_triage(y_val, y_test, probs, "synthetic", 0.5, 42,
                      cfg=TriageConfig(defer_rule="band"))
    chow = run_triage(y_val, y_test, probs, "synthetic", 0.5, 42,
                      cfg=TriageConfig(defer_rule="chow"))
    cap = run_triage(y_val, y_test, probs, "synthetic", 0.5, 42,
                     cfg=TriageConfig(defer_rule="capacity"))
    # capacity rule hits the global budget by construction
    assert cap[0]["defer_frac"] == pytest.approx(0.05, abs=0.005)
    # band rule realized defer is an outcome, not pinned to the budget
    assert band[0]["defer_frac"] != pytest.approx(0.05, abs=1e-6)
    assert chow[0]["defer_frac"] != band[0]["defer_frac"]


def test_npz_roundtrip(tmp_path):
    rng = np.random.default_rng(23)
    probs, y_val, y_test = _fake_capture(rng, n_banks=2)
    banks = [{"id": bid, "y_val": y_val[bid], "y_test": y_test[bid]} for bid in probs]
    path = str(tmp_path / "probs_synthetic_alpha0.5_seed42.npz")
    save_probs_npz(path, probs, banks)
    assert not os.path.exists(path + ".tmp")
    p2, yv2, yt2 = load_probs_npz(path)
    assert set(p2) == set(probs)
    for bid in probs:
        np.testing.assert_array_equal(yv2[bid], y_val[bid])
        np.testing.assert_array_equal(yt2[bid], y_test[bid])
        for en in probs[bid]:
            np.testing.assert_allclose(p2[bid][en][0], probs[bid][en][0])
            np.testing.assert_allclose(p2[bid][en][1], probs[bid][en][1])
    rows_a = run_triage(y_val, y_test, probs, "synthetic", 0.5, 42)
    rows_b = run_triage(yv2, yt2, p2, "synthetic", 0.5, 42)
    assert rows_a == rows_b
