"""
Wiring between the trained experts and the triage decision layer.

Two entry points:

  1. In the sweep notebook: capture_expert_probs(...) collects, per bank, every expert's
     validation and test probabilities WITHOUT retraining anything, and save_probs_npz(...)
     checkpoints them per (dataset, alpha, seed). The notebook cell added by this branch
     ("Triage probability capture") calls these right before evaluate_all's objects are
     deleted at the end of each combo.

  2. Offline / end of notebook: run_triage(...) applies the decision layer to one captured
     combo and emits one row per rho with the SPEC result columns;
     run_from_npz(...) sweeps a folder of capture files into a results CSV.

The layer never retrains an expert; everything here is predict-and-decide.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from . import core as T

# Experts captured from the sweep notebook, in canonical order. moe_typology_aware is
# deliberately excluded: its routing key is derived from the label on IBM/ULB (a test-time
# oracle), so no deployable triage stack may consume it.
DEPLOYABLE_EXPERTS = ("fedavg", "fedprox", "fednova", "persfl", "xgb", "lgbm", "catboost")


@dataclass(frozen=True)
class TriageConfig:
    target_fnr: float = 0.10
    defer_budget: float = 0.05
    rhos: Tuple[float, ...] = (1, 2, 5, 10, 20, 50)
    c_defer: float = 0.25
    human_accuracy: float = 1.0
    defer_rule: str = "band"            # "band" | "chow" | "capacity"
    gate: str = "brier"                 # "uniform" | "brier"
    cal_frac: float = 0.5
    min_fraud: int = T.MIN_FRAUD_FOR_FIT
    split_seed_offset: int = 7919       # decouples the fold-split RNG from model seeds


def combo_seed(dataset: str, alpha: float, seed: int, bank_id: int, offset: int) -> int:
    """Deterministic RNG seed for one bank's fold split, from coordinates only, so a
    resumed run reproduces an uninterrupted one exactly."""
    return (int(seed) * 1_000_003 + int(round(float(alpha) * 1000)) * 97
            + sum(ord(c) for c in dataset) + int(bank_id) * 131 + offset) % (2**31 - 1)


# ----------------------------------------------------------------------------------
# Capture (called from the notebook, inside the per-combo loop)
# ----------------------------------------------------------------------------------

def capture_expert_probs(banks, fl_results, experts, mlp_proba, expert_names) -> dict:
    """{bank_id: {expert_name: (val_probs, test_probs)}} from the sweep's live objects.

    banks: the notebook's bank dicts (X_val/y_val/X_test/y_test/id).
    fl_results: {strategy: (fl_model, per_bank_models, history)} -- per_bank_models is
        populated for persfl only.
    experts: {bank_id: {name: expert_or_None}} tree experts with .trained/.predict_proba.
    mlp_proba: the notebook's probability helper for MLP models.
    """
    out = {}
    for b in banks:
        bid = b["id"]
        d = {}
        for strat, (fl_model, bw, _hist) in fl_results.items():
            m = bw[bid] if (strat == "persfl" and bid in bw) else fl_model
            d[strat] = (np.asarray(mlp_proba(m, b["X_val"])),
                        np.asarray(mlp_proba(m, b["X_test"])))
        for en in expert_names:
            m = experts[bid].get(en)
            if m is not None and getattr(m, "trained", False):
                d[en] = (np.asarray(m.predict_proba(b["X_val"])),
                         np.asarray(m.predict_proba(b["X_test"])))
            else:
                d[en] = (np.full(len(b["y_val"]), 0.5), np.full(len(b["y_test"]), 0.5))
        out[bid] = d
    return out


def save_probs_npz(path: str, probs: dict, banks) -> None:
    """Atomic checkpoint of one combo's captured probabilities + labels."""
    arrays = {}
    for bid, d in probs.items():
        for en, (pv, pt) in d.items():
            arrays[f"b{bid}__{en}__val"] = np.asarray(pv)
            arrays[f"b{bid}__{en}__test"] = np.asarray(pt)
    for b in banks:
        arrays[f"b{b['id']}__y_val"] = np.asarray(b["y_val"])
        arrays[f"b{b['id']}__y_test"] = np.asarray(b["y_test"])
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def load_probs_npz(path: str):
    """Inverse of save_probs_npz -> (probs, y_val_by_bank, y_test_by_bank)."""
    z = np.load(path)
    probs: Dict[int, Dict[str, list]] = {}
    y_val: Dict[int, np.ndarray] = {}
    y_test: Dict[int, np.ndarray] = {}
    prob_pat = re.compile(r"^b(\d+)__(.+)__(val|test)$")
    label_pat = re.compile(r"^b(\d+)__y_(val|test)$")
    for key in z.files:
        lm = label_pat.match(key)
        if lm:
            (y_val if lm.group(2) == "val" else y_test)[int(lm.group(1))] = z[key]
            continue
        pm = prob_pat.match(key)
        if pm:
            bid, en, fold = int(pm.group(1)), pm.group(2), pm.group(3)
            slot = probs.setdefault(bid, {}).setdefault(en, [None, None])
            slot[0 if fold == "val" else 1] = z[key]
    return ({bid: {en: (v[0], v[1]) for en, v in d.items()} for bid, d in probs.items()},
            y_val, y_test)


# ----------------------------------------------------------------------------------
# The triage function
# ----------------------------------------------------------------------------------

def _bank_scores(yv, val_p, te_p, dataset, alpha, seed, bid, cfg):
    """Per-bank layer 1+2: fold split, calibration, gate weights, conformal threshold.

    Returns s_conf/y_conf (the conformal half: the only validation scores that are
    out-of-sample w.r.t. the calibrators, used for the conformal threshold AND as the
    reference fold for the defer band and the recall-80 threshold), s_test, t (or None),
    split_ok, and the gate weights used. The calibration half is never used as a
    reference distribution: isotonic maps are in-sample there, which made reference
    quantiles optimistic (measured achieved test recall 0.72 vs a nominal 0.80).
    """
    rng = np.random.default_rng(
        combo_seed(dataset, alpha, seed, bid, cfg.split_seed_offset))
    cal_idx, conf_idx, split_ok = T.split_val_indices(yv, rng, cal_frac=cfg.cal_frac)

    cals = T.fit_calibrators([p[cal_idx] for p in val_p], yv[cal_idx],
                             min_fraud=cfg.min_fraud)
    cal_val = T.apply_calibrators(cals, val_p)
    if cfg.gate == "brier":
        w = T.brier_gate_weights([p[cal_idx] for p in cal_val], yv[cal_idx])
    else:
        w = None

    s_val = T.combine(cal_val, w)
    s_conf, y_conf = s_val[conf_idx], yv[conf_idx]
    t = T.conformal_fnr_threshold(s_conf, y_conf, cfg.target_fnr,
                                  min_fraud=cfg.min_fraud)
    s_test = T.combine(T.apply_calibrators(cals, te_p), w)
    return {"s_conf": s_conf, "y_conf": y_conf, "s_test": s_test, "t": t,
            "split_ok": split_ok, "weights": w}


def run_triage(y_val_by_bank: Dict[int, np.ndarray], y_test_by_bank: Dict[int, np.ndarray],
               probs: dict, dataset: str, alpha: float, seed: int,
               cfg: TriageConfig = TriageConfig(),
               expert_names: Optional[Sequence[str]] = None) -> list:
    """Apply calibration -> conformal -> cost rule -> defer to one (dataset, alpha, seed)
    combo. Returns one result-row dict per rho (the SPEC's new CSV columns).

    expert_names=None (default) uses every expert present in `probs`, in sorted order --
    the capture side is responsible for excluding oracle experts (the notebook captures
    only the 7 deployable ones). Pass an explicit list to restrict the ensemble; names
    absent from the capture raise rather than being silently dropped.

    Fold hygiene: each bank's validation fold is split into a calibration half (fits the
    isotonic maps and the gate weights) and a disjoint conformal half (supplies the
    threshold quantile, the defer band, and the recall-80 reference). The deployed flag
    threshold is tau = min(bayes_cut, t_bank-or-global).
    """
    first = probs[next(iter(probs))]
    if expert_names is None:
        names = sorted(first)
    else:
        missing = [n for n in expert_names if n not in first]
        if missing:
            raise ValueError(f"experts {missing} not present in captured probs "
                             f"(found {sorted(first)})")
        names = list(expert_names)
    bids = sorted(probs.keys())

    scored = {}
    for bid in bids:
        yv = np.asarray(y_val_by_bank[bid])
        val_p = [np.asarray(probs[bid][n][0], dtype=float) for n in names]
        te_p = [np.asarray(probs[bid][n][1], dtype=float) for n in names]
        scored[bid] = _bank_scores(yv, val_p, te_p, dataset, alpha, seed, bid, cfg)
        scored[bid]["y_test"] = np.asarray(y_test_by_bank[bid])

    thresholds = [scored[bid]["t"] for bid in bids]
    global_t = T.quantile_of_quantiles(thresholds)
    n_fallback = sum(1 for t in thresholds if t is None)
    s_conf_cat = np.concatenate([scored[bid]["s_conf"] for bid in bids])
    y_conf_cat = np.concatenate([scored[bid]["y_conf"] for bid in bids])
    n_banks_with_fraud = int(sum((scored[bid]["y_test"] == 1).any() for bid in bids))

    rows = []
    for rho in cfg.rhos:
        cost = T.CostMatrix(c_fn=float(rho), c_fp=1.0)
        taus = {bid: T.flag_threshold(cost, scored[bid]["t"] if scored[bid]["t"] is not None
                                      else global_t) for bid in bids}

        if cfg.defer_rule == "capacity":
            masks = T.capacity_defer_masks([scored[bid]["s_test"] for bid in bids],
                                           [taus[bid] for bid in bids],
                                           cfg.defer_budget)
            defer_by_bank = dict(zip(bids, masks))

        defer_rates, fprs, costs, checks, policy_misses, policy_w = [], [], [], [], [], []
        s_all, y_all, n_all = [], [], 0
        for bid in bids:
            bk = scored[bid]
            s, y, tau = bk["s_test"], bk["y_test"], taus[bid]
            if cfg.defer_rule == "chow":
                acts = T.chow_actions(s, cost, cfg.c_defer, tau=tau)
            elif cfg.defer_rule == "capacity":
                acts = np.where(s >= tau, T.FLAG, T.CLEAR).astype(object)
                acts[defer_by_bank[bid]] = T.DEFER
            else:  # band, sized on the held-out conformal half of the val fold
                band = T.band_for_defer_budget(bk["s_conf"], tau, cfg.defer_budget)
                acts = T.triage_actions(s, tau, band)

            defer_rates.append(float(np.mean(acts == T.DEFER)) if s.size else float("nan"))
            legit = y == 0
            fprs.append(float(np.mean((acts == T.FLAG)[legit])) if legit.any() else None)
            costs.append(T.expected_cost(y, acts, cost, cfg.c_defer,
                                         human_accuracy=cfg.human_accuracy))
            # audit ONLY at a genuine conformal threshold; auditing the rho-dependent
            # Bayes cut would report pseudo-coverage carrying no conformal claim
            audit_t = bk["t"] if bk["t"] is not None else global_t
            if audit_t is not None:
                checks.append(T.coverage_check(s, y, audit_t, cfg.target_fnr))
            policy_misses.append(T.policy_miss_rate(y, acts))
            policy_w.append(int((y == 1).sum()))
            s_all.append(s); y_all.append(y); n_all += s.size

        valid = [c["covers"] for c in checks if c["covers"] is not None]
        s_cat, y_cat = np.concatenate(s_all), np.concatenate(y_all)
        pool_tau = float(np.mean([taus[bid] for bid in bids]))
        rc = T.risk_coverage_curve(s_cat, y_cat, pool_tau, cost=cost)
        wl = T.workload_reduction_at_recall(s_conf_cat, y_conf_cat, s_cat, y_cat, 0.80)
        fair = T.deferral_fairness(defer_rates, fprs)
        pm = np.array(policy_misses, dtype=float)
        pw = np.array(policy_w, dtype=float)
        pol_mask = np.isfinite(pm) & (pw > 0)
        rows.append(dict(
            dataset=dataset, alpha=float(alpha), seed=int(seed), rho=float(rho),
            target_fnr=cfg.target_fnr, defer_budget=cfg.defer_budget,
            defer_rule=cfg.defer_rule, gate=cfg.gate,
            human_accuracy=cfg.human_accuracy,
            n_banks=len(bids),
            n_banks_with_fraud=n_banks_with_fraud,
            n_banks_audited=len(checks),
            n_banks_fallback_threshold=n_fallback,
            n_banks_split_ok=int(sum(scored[bid]["split_ok"] for bid in bids)),
            defer_frac=float(np.nanmean(defer_rates)),
            miss_rate_audit=(T.pooled_miss_rate(checks) if checks else float("nan")),
            miss_rate_policy=(float(np.average(pm[pol_mask], weights=pw[pol_mask]))
                              if pol_mask.any() else float("nan")),
            coverage_valid_frac=(float(np.mean(valid)) if valid else float("nan")),
            expected_cost=float(np.sum(costs)),
            expected_cost_per_txn=(float(np.sum(costs) / n_all) if n_all else float("nan")),
            aurc=rc["aurc"],
            auto_clear_at_recall80=wl["auto_clear_frac"],
            fpr_at_recall80=wl["fpr"],
            achieved_recall80=wl["achieved_recall"],
            defer_jain=fair["defer_jain"], fpr_gap=fair["fpr_gap"],
        ))
    return rows


# ----------------------------------------------------------------------------------
# Offline sweep over capture files
# ----------------------------------------------------------------------------------

NPZ_PATTERN = re.compile(r"probs_(?P<dataset>[a-z]+)_alpha(?P<alpha>[0-9.]+)_seed(?P<seed>\d+)\.npz$")


def run_from_npz(folder: str, out_csv: str, cfg: TriageConfig = TriageConfig()) -> "object":
    """Apply the triage layer to every capture file in a folder; write one results CSV.

    Returns the pandas DataFrame. Files that do not match the capture naming scheme are
    reported and skipped, never silently mixed in.
    """
    import pandas as pd

    rows, skipped = [], []
    for path in sorted(glob.glob(os.path.join(folder, "probs_*.npz"))):
        m = NPZ_PATTERN.search(os.path.basename(path))
        if not m:
            skipped.append(os.path.basename(path))
            continue
        probs, y_val, y_test = load_probs_npz(path)
        rows.extend(run_triage(y_val, y_test, probs, m["dataset"], float(m["alpha"]),
                               int(m["seed"]), cfg=cfg))
    if skipped:
        print(f"[triage] skipped {len(skipped)} non-matching file(s): {skipped}")
    if not rows:
        raise FileNotFoundError(f"no capture files matching probs_*.npz under {folder}")
    df = pd.DataFrame(rows)
    tmp = out_csv + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_csv)
    print(f"[triage] wrote {out_csv} ({len(df)} rows, "
          f"{df.dataset.nunique()} dataset(s), {df.seed.nunique()} seed(s))")
    return df


def main():
    ap = argparse.ArgumentParser(description="Run the triage layer over captured "
                                             "expert probabilities (probs_*.npz).")
    ap.add_argument("folder", help="folder containing probs_<dataset>_alpha<a>_seed<s>.npz")
    ap.add_argument("--out", default="triage_results.csv")
    ap.add_argument("--defer-rule", default="band", choices=["band", "chow", "capacity"])
    ap.add_argument("--gate", default="brier", choices=["uniform", "brier"])
    ap.add_argument("--target-fnr", type=float, default=0.10)
    ap.add_argument("--defer-budget", type=float, default=0.05)
    ap.add_argument("--human-accuracy", type=float, default=1.0)
    args = ap.parse_args()
    cfg = TriageConfig(target_fnr=args.target_fnr, defer_budget=args.defer_budget,
                       defer_rule=args.defer_rule, gate=args.gate,
                       human_accuracy=args.human_accuracy)
    run_from_npz(args.folder, args.out, cfg)


if __name__ == "__main__":
    main()
