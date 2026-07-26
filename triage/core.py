"""
Federated triage-and-defer decision layer for cross-bank AML.

Pure functions over numpy arrays. Pipeline per bank:

    raw expert probs p_e  --calibrate-->  p~_e  --gate weights-->  s(x)   (calibrated risk)
    s(x) + conformal fold --conformal-->  t     (miss-rate guarantee, computed locally)
    cost matrix           --Bayes-->      cut   (expected-loss-optimal flag threshold)
    tau = min(cut, t)     --defer rule--> action in {clear, flag, defer}

Statistical hygiene rules this module enforces (each was a measured failure mode in the
prototype; see triage/README.md for the numbers):

  * Calibrators and the conformal threshold must NOT share a fold. Isotonic calibration
    adapts the score ordering to the fold it is fit on; taking the conformal quantile of
    those same in-sample scores breaks exchangeability anti-conservatively (measured test
    miss-rate 0.14-0.24 against a 0.10 target with 7 experts). Use split_val_indices() to
    give calibration and conformal disjoint halves.
  * The defer band is an operating parameter: derive it from held-out reference (validation)
    scores, never from the batch being scored. The realized test defer rate is an OUTCOME to
    report, not a constraint to enforce by peeking.
  * The conformal threshold participates in the action policy (tau = min(bayes_cut, t)),
    it is not just an audit column.
  * Banks with too little validation fraud return t = None and fall back to the federated
    aggregate; they never emit a flag-everything -inf threshold.
  * quantile_of_quantiles is a HEURISTIC fallback. A plain median of per-bank conformal
    thresholds carries no finite-sample coverage guarantee (a certified one-shot federated
    estimator needs the jointly-corrected order statistics of Humbert et al., 2023).
    Coverage under the fallback must be reported empirically, per bank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

CLEAR, FLAG, DEFER = "clear", "flag", "defer"
MIN_FRAUD_FOR_FIT = 5


# ----------------------------------------------------------------------------------
# Cost model
# ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class CostMatrix:
    """Asymmetric costs. rho = c_fn / c_fp is the ratio the paper already sweeps."""
    c_fn: float
    c_fp: float

    def __post_init__(self):
        if self.c_fp <= 0 or self.c_fn <= 0:
            raise ValueError(f"costs must be positive, got c_fn={self.c_fn} c_fp={self.c_fp}")

    @property
    def rho(self) -> float:
        return self.c_fn / self.c_fp

    def bayes_cut(self) -> float:
        """Expected-cost-optimal flag threshold on calibrated risk s(x) (DESIGN Prop 2)."""
        return 1.0 / (1.0 + self.rho)


def flag_threshold(cost: CostMatrix, t_conformal: Optional[float]) -> float:
    """Deployed flag threshold tau = min(bayes cut, conformal threshold).

    The min makes the miss-rate guarantee binding on the policy itself: flagging at a
    threshold no higher than t preserves P(miss | fraud) <= target_fnr, while the Bayes cut
    can only lower it further. Without this composition the conformal layer is decorative.
    """
    cut = cost.bayes_cut()
    if t_conformal is None or not np.isfinite(t_conformal):
        return cut
    return min(cut, float(t_conformal))


# ----------------------------------------------------------------------------------
# Fold hygiene
# ----------------------------------------------------------------------------------

def split_val_indices(y_val: np.ndarray, rng: np.random.Generator,
                      cal_frac: float = 0.5, min_pos_each: int = 2):
    """Stratified split of the validation fold into (calibration, conformal) index arrays.

    Returns (cal_idx, conf_idx, split_ok). When the fold has fewer than 2*min_pos_each
    fraud cases a clean split is impossible; both index sets become the full fold and
    split_ok=False so the caller can record the degraded guarantee instead of hiding it.
    """
    y_val = np.asarray(y_val)
    pos = np.flatnonzero(y_val == 1)
    neg = np.flatnonzero(y_val == 0)
    if pos.size < 2 * min_pos_each:
        idx = np.arange(y_val.size)
        return idx, idx, False
    pos = pos[rng.permutation(pos.size)]
    neg = neg[rng.permutation(neg.size)]
    np_cal = int(round(cal_frac * pos.size))
    nn_cal = int(round(cal_frac * neg.size))
    np_cal = min(max(np_cal, min_pos_each), pos.size - min_pos_each)
    cal = np.sort(np.concatenate([pos[:np_cal], neg[:nn_cal]]))
    conf = np.sort(np.concatenate([pos[np_cal:], neg[nn_cal:]]))
    return cal, conf, True


# ----------------------------------------------------------------------------------
# Layer 1: per-expert calibration
# ----------------------------------------------------------------------------------

def fit_calibrators(expert_val_probs: Sequence[np.ndarray], y_val: np.ndarray,
                    min_fraud: int = MIN_FRAUD_FOR_FIT):
    """One calibrator per expert, fit on the calibration fold. Entries are fitted
    IsotonicRegression objects or None (identity).

    Identity fallback when the fold lacks both classes OR has fewer than min_fraud fraud
    cases: isotonic on a handful of positives memorises them as steps, which is worthless
    as a probability map and maximally harmful to any downstream quantile.
    """
    from sklearn.isotonic import IsotonicRegression

    y_val = np.asarray(y_val)
    usable = (y_val.size > 0 and y_val.min() == 0 and y_val.max() == 1
              and int(y_val.sum()) >= min_fraud)
    cals = []
    for p in expert_val_probs:
        if not usable:
            cals.append(None)
            continue
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(np.asarray(p, dtype=float), y_val.astype(float))
        cals.append(ir)
    return cals


def apply_calibrators(cals, expert_probs: Sequence[np.ndarray]) -> list:
    """Apply fitted calibrators to expert probabilities (identity where None)."""
    out = []
    for cal, p in zip(cals, expert_probs):
        p = np.asarray(p, dtype=float)
        out.append(p if cal is None else np.clip(cal.predict(p), 0.0, 1.0))
    return out


def combine(cal_probs: Sequence[np.ndarray], weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Calibrated risk s(x) = sum_e w_e * p~_e(x). Uniform weights = the Static gate."""
    P = np.vstack([np.asarray(p, dtype=float) for p in cal_probs])   # (E, N)
    E = P.shape[0]
    if weights is None:
        w = np.full(E, 1.0 / E)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (E,) or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError(f"gate weights must be {E} non-negative finite values with a "
                             f"positive sum, got {w!r}")
        w = w / w.sum()
    return w @ P


def brier_gate_weights(cal_probs: Sequence[np.ndarray], y: np.ndarray,
                       eps: float = 1e-6) -> np.ndarray:
    """Gate weights proportional to 1 / Brier score of each CALIBRATED expert.

    The Brier score measures exactly the probability quality the Bayes cost rule consumes,
    so this is the natural gate for a decision layer (the paper's F-score gates rank by a
    thresholded metric the triage layer never uses). Compute on the calibration fold only.
    """
    y = np.asarray(y, dtype=float)
    briers = np.array([np.mean((np.asarray(p, dtype=float) - y) ** 2) for p in cal_probs])
    w = 1.0 / (briers + eps)
    return w / w.sum()


# ----------------------------------------------------------------------------------
# Layer 2: federated conformal threshold (class-conditional FNR control)
# ----------------------------------------------------------------------------------

def conformal_fnr_threshold(s_val: np.ndarray, y_val: np.ndarray, target_fnr: float,
                            min_fraud: int = MIN_FRAUD_FOR_FIT) -> Optional[float]:
    """Split-conformal threshold t with P(s(X) < t | Y=1) <= target_fnr on exchangeable data.

    t = the k-th smallest validation fraud score, k = floor(target_fnr * (n1 + 1)).

    Returns None (caller must fall back to the federated aggregate) when the fold cannot
    support the guarantee: n1 < min_fraud, or k < 1 (n1 < 1/target_fnr - 1). The prototype
    returned -inf in the k<1 case, which flags 100% of transactions -- vacuously valid,
    operationally absurd, and it biased the cross-bank aggregate because -inf thresholds
    were silently dropped from the median while their banks still reported "covered".
    """
    s_val = np.asarray(s_val, dtype=float)
    y_val = np.asarray(y_val)
    fraud = np.sort(s_val[y_val == 1])
    n1 = fraud.size
    if n1 < max(min_fraud, 1):
        return None
    k = int(np.floor(target_fnr * (n1 + 1)))
    if k < 1:
        return None
    if k > n1:
        return float(fraud[-1])
    return float(fraud[k - 1])


def quantile_of_quantiles(bank_thresholds: Sequence[Optional[float]],
                          q: float = 0.5) -> Optional[float]:
    """HEURISTIC federated fallback: a low-side quantile of the per-bank thresholds.

    No finite-sample guarantee survives this aggregation (a certified one-shot federated
    conformal estimator requires the jointly-corrected order statistics of Humbert et al.
    2023, under cross-bank exchangeability -- which Dirichlet non-IID deliberately breaks).
    Use it only as the fallback for banks that cannot fit their own threshold, report its
    empirical per-bank coverage, and never describe it as guaranteed. method='lower' keeps
    the interpolation bias on the conservative (more-flagging) side.
    """
    vals = [t for t in bank_thresholds if t is not None and np.isfinite(t)]
    if not vals:
        return None
    return float(np.quantile(vals, q, method="lower"))


def coverage_check(s_test: np.ndarray, y_test: np.ndarray, t: float,
                   target_fnr: float) -> dict:
    """Empirical audit: does the miss-rate guarantee hold on this fold at threshold t?"""
    s_test = np.asarray(s_test, dtype=float)
    y_test = np.asarray(y_test)
    fraud = s_test[y_test == 1]
    if fraud.size == 0:
        # no test fraud -> guarantee is vacuous here; covers=None so callers can EXCLUDE it
        return {"n_fraud": 0, "miss_rate": float("nan"), "covers": None, "target": target_fnr}
    miss = float(np.mean(fraud < t))
    return {"n_fraud": int(fraud.size), "miss_rate": miss,
            "covers": miss <= target_fnr + 1e-9, "target": target_fnr}


def pooled_miss_rate(checks) -> float:
    """Fraud-weighted miss-rate across banks (only banks with test fraud)."""
    num = sum(c["miss_rate"] * c["n_fraud"] for c in checks if c["n_fraud"] > 0)
    den = sum(c["n_fraud"] for c in checks if c["n_fraud"] > 0)
    return float(num / den) if den > 0 else float("nan")


# ----------------------------------------------------------------------------------
# Layers 3 + 4: action rules
# ----------------------------------------------------------------------------------

def band_for_defer_budget(s_ref: np.ndarray, tau: float, defer_budget: float) -> float:
    """Half-width band so that ~defer_budget of REFERENCE volume falls in |s - tau| < band.

    s_ref must be held-out reference scores (the validation fold), never the batch the band
    will be applied to: fitting the band on the scored batch makes "defer <= budget" true by
    construction and unattainable in deployment. method='lower' biases the band small, so
    the budget is approached from below on the reference fold.
    """
    s_ref = np.asarray(s_ref, dtype=float)
    if s_ref.size == 0 or defer_budget <= 0:
        return 0.0
    d = np.abs(s_ref - tau)
    if defer_budget >= 1:
        return float(d.max() + 1.0)
    return float(np.quantile(d, defer_budget, method="lower"))


def triage_actions(s: np.ndarray, tau: float, band: float) -> np.ndarray:
    """Band rule: defer inside |s - tau| < band, else flag iff s >= tau."""
    s = np.asarray(s, dtype=float)
    act = np.where(s >= tau, FLAG, CLEAR).astype(object)
    act[np.abs(s - tau) < band] = DEFER
    return act


def chow_actions(s: np.ndarray, cost: CostMatrix, c_defer: float,
                 tau: Optional[float] = None) -> np.ndarray:
    """Chow (1970) three-action Bayes rule: defer exactly when the human is the cheapest
    option, i.e. when min(c_fn * s, c_fp * (1 - s)) > c_defer.

    Unlike the band rule this uses the defer cost inside the decision (the band rule only
    charges it after the fact), and the defer region is naturally asymmetric under
    asymmetric costs. tau overrides the flag threshold (pass flag_threshold(cost, t) to
    keep the conformal composition); default is the Bayes cut.
    """
    s = np.asarray(s, dtype=float)
    cut = cost.bayes_cut() if tau is None else float(tau)
    act = np.where(s >= cut, FLAG, CLEAR).astype(object)
    risk_if_auto = np.minimum(cost.c_fn * s, cost.c_fp * (1.0 - s))
    act[risk_if_auto > c_defer] = DEFER
    return act


def capacity_defer_masks(s_by_bank: Sequence[np.ndarray], tau_by_bank: Sequence[float],
                         capacity_frac: float) -> list:
    """Federation-level review capacity: pool |s - tau| across banks and defer the globally
    most ambiguous capacity_frac of total volume. Returns one boolean defer-mask per bank.

    A per-bank budget pins every bank's defer rate to the budget, which makes any fairness
    index over defer rates vacuous by construction. A shared queue lets deferral concentrate
    where ambiguity actually is -- that concentration is the fairness result to report.
    """
    ds = [np.abs(np.asarray(s, dtype=float) - t) for s, t in zip(s_by_bank, tau_by_bank)]
    total = int(sum(d.size for d in ds))
    n_defer = int(np.floor(capacity_frac * total))
    masks = [np.zeros(d.size, dtype=bool) for d in ds]
    if n_defer <= 0 or total == 0:
        return masks
    flat = np.concatenate(ds)
    order = np.argsort(flat, kind="stable")[:n_defer]      # smallest margin = most ambiguous
    offsets = np.cumsum([0] + [d.size for d in ds[:-1]])
    for pos in order:
        b = int(np.searchsorted(offsets, pos, side="right")) - 1
        masks[b][pos - offsets[b]] = True
    return masks


# ----------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------

def expected_cost(y: np.ndarray, actions: np.ndarray, cost: CostMatrix, c_defer: float,
                  human_accuracy: float = 1.0) -> float:
    """Total operational cost of the action vector.

    human_accuracy models the investigator: a deferred case is resolved correctly with
    probability human_accuracy, so deferred fraud carries an expected residual miss cost
    (1 - a) * c_fn and deferred legit an expected residual false-alarm cost (1 - a) * c_fp,
    on top of the review cost c_defer. a = 1.0 (default) is the perfect-human assumption
    and reproduces the plain accounting.
    """
    y = np.asarray(y)
    actions = np.asarray(actions, dtype=object)
    a = float(human_accuracy)
    missed = np.sum((actions == CLEAR) & (y == 1)) * cost.c_fn
    false_alarm = np.sum((actions == FLAG) & (y == 0)) * cost.c_fp
    deferred = actions == DEFER
    review = np.sum(deferred) * c_defer
    residual = ((1.0 - a) * cost.c_fn * np.sum(deferred & (y == 1))
                + (1.0 - a) * cost.c_fp * np.sum(deferred & (y == 0)))
    return float(missed + false_alarm + review + residual)


def policy_miss_rate(y: np.ndarray, actions: np.ndarray) -> float:
    """Fraction of fraud the DEPLOYED policy auto-clears. This is the number the conformal
    guarantee must bound once tau = min(cut, t); reporting the audit threshold's miss-rate
    while acting on a different threshold describes two different systems in one row."""
    y = np.asarray(y)
    fraud = y == 1
    if not fraud.any():
        return float("nan")
    return float(np.mean(np.asarray(actions, dtype=object)[fraud] == CLEAR))


def risk_coverage_curve(s: np.ndarray, y: np.ndarray, tau: float,
                        cost: Optional[CostMatrix] = None) -> dict:
    """Selective prediction: rank by decision margin |s - tau|, auto-decide the most
    confident fraction, measure risk on that set. Includes the (coverage=0, risk=0) endpoint.

    With cost given, risk is the per-item expected cost (c_fn per missed fraud, c_fp per
    false alarm) instead of 0/1 error. At cost-skewed cuts the 0/1 version is dominated by
    false positives the cost model prices as cheap -- at rho=10 it produced AURC ~0.9 for a
    perfectly reasonable policy -- so cost-weighted is the honest summary for the rho sweep.
    """
    s = np.asarray(s, dtype=float)
    y = np.asarray(y)
    if s.size == 0:
        return {"coverage": np.zeros(1), "selective_risk": np.zeros(1),
                "aurc": float("nan"), "full_risk": float("nan")}
    pred = (s >= tau).astype(int)
    if cost is None:
        item_risk = (pred != y).astype(float)
    else:
        item_risk = np.where((pred == 0) & (y == 1), cost.c_fn, 0.0) \
                  + np.where((pred == 1) & (y == 0), cost.c_fp, 0.0)
    order = np.argsort(-np.abs(s - tau), kind="stable")     # most confident first
    n = s.size
    cum = np.cumsum(item_risk[order]) / np.arange(1, n + 1)
    coverage = np.concatenate([[0.0], np.arange(1, n + 1) / n])
    risk = np.concatenate([[0.0], cum])
    return {"coverage": coverage, "selective_risk": risk,
            "aurc": float(np.trapezoid(risk, coverage)), "full_risk": float(cum[-1])}


def workload_reduction_at_recall(s_ref: np.ndarray, y_ref: np.ndarray,
                                 s_eval: np.ndarray, y_eval: np.ndarray,
                                 target_recall: float = 0.80) -> dict:
    """Choose the flag threshold to hit target_recall on the REFERENCE (validation) fraud,
    then report auto-clear fraction / FPR / achieved recall on the evaluation fold.

    The prototype picked the threshold from the evaluation fold's own fraud labels, which
    makes auto_clear_at_recall an oracle metric no deployed system attains.
    """
    s_ref, y_ref = np.asarray(s_ref, dtype=float), np.asarray(y_ref)
    s_eval, y_eval = np.asarray(s_eval, dtype=float), np.asarray(y_eval)
    fraud_ref = np.sort(s_ref[y_ref == 1])
    if fraud_ref.size == 0:
        return {"threshold": float("nan"), "auto_clear_frac": float("nan"),
                "fpr": float("nan"), "target_recall": target_recall,
                "achieved_recall": float("nan")}
    idx = min(int(np.floor((1.0 - target_recall) * fraud_ref.size)), fraud_ref.size - 1)
    thr = float(fraud_ref[idx])
    legit = y_eval == 0
    fraud = y_eval == 1
    return {
        "threshold": thr,
        "auto_clear_frac": float(np.mean(s_eval < thr)) if s_eval.size else float("nan"),
        "fpr": float(np.mean(s_eval[legit] >= thr)) if legit.any() else float("nan"),
        "target_recall": target_recall,
        "achieved_recall": float(np.mean(s_eval[fraud] >= thr)) if fraud.any() else float("nan"),
    }


def jain_index(values: Sequence[float]) -> float:
    """Jain's fairness index in [1/n, 1]; 1 = perfectly equal. nan for empty input."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan")
    if np.all(v == 0):
        return 1.0
    return float((v.sum() ** 2) / (v.size * np.sum(v ** 2)))


def deferral_fairness(defer_rates_by_bank: Sequence[float],
                      fpr_by_bank: Sequence[float]) -> dict:
    """Jain's index over per-bank defer rates + max FPR gap across banks."""
    fprs = [f for f in fpr_by_bank if f is not None and np.isfinite(f)]
    fpr_gap = float(max(fprs) - min(fprs)) if fprs else float("nan")
    return {"defer_jain": jain_index(defer_rates_by_bank), "fpr_gap": fpr_gap}


if __name__ == "__main__":
    # Smoke check only; the real verification lives in tests/test_triage.py.
    rng = np.random.default_rng(0)
    n = 8000
    y = (rng.random(n) < 0.03).astype(int)
    s = np.clip(rng.normal(np.where(y == 1, 0.7, 0.2), 0.15), 0, 1)
    h = n // 2
    t = conformal_fnr_threshold(s[:h], y[:h], 0.10)
    cov = coverage_check(s[h:], y[h:], t, 0.10)
    cost = CostMatrix(10, 1)
    tau = flag_threshold(cost, t)
    band = band_for_defer_budget(s[:h], tau, 0.05)
    acts = triage_actions(s[h:], tau, band)
    print(f"t={t:.3f} tau={tau:.3f} audit_miss={cov['miss_rate']:.3f} "
          f"policy_miss={policy_miss_rate(y[h:], acts):.3f} "
          f"defer={np.mean(acts == DEFER):.3f} "
          f"cost={expected_cost(y[h:], acts, cost, 0.25):.0f}")
