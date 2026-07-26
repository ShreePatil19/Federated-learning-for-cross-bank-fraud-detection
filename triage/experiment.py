"""
Standalone triage-and-defer experiment: compact expert set + the decision layer.

Trains a 7-expert set per bank (4 federated MLP strategies: FedAvg, FedProx, FedNova,
PersFL + 3 sklearn tree experts) over a Dirichlet non-IID partition, captures per-bank
val/test probabilities in the same npz format the sweep notebook emits, and applies
triage.integration.run_triage. Useful for iterating on the decision layer without the
full GPU sweep; the paper numbers should come from the captured GROUP-A experts instead.

Reproducibility / resume design
-------------------------------
Every (dataset, alpha, seed) CELL is seeded from its own coordinates only, so a cell
reproduces the same numbers regardless of what ran before it (no global RNG stream to
desynchronise). Each finished cell checkpoints two files ATOMICALLY (tmp + os.replace):

  cells/triage_cell_<dataset>_a<alpha>_s<seed>.csv     the result rows
  cells/probs_<dataset>_alpha<alpha>_seed<seed>.npz    the captured probabilities

A re-run (optionally with --resume-dir pointing at a previous session's cells/) skips
whatever is already there; the final combine reads ONLY the requested grid, so stale cells
from other configurations can never contaminate the output.

FL parity with the corrected sweep notebook: local training continues from the injected
global weights via per-epoch partial_fit (never a reinitialising fit), FedProx applies its
proximal pull each epoch, FedNova's tau counts real optimizer steps at batch_size=128, and
the weight-template is allocated on dummy zeros -- no raw rows ever leave a bank.

Usage
-----
  python -m triage.experiment --dataset synthetic --quick     # ~1 min smoke test
  python -m triage.experiment --dataset ulb --seeds 42 0 1 2 3
  python -m triage.experiment --dataset ulb --resume-dir /kaggle/input/triage-partial
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

from . import core as T
from .integration import TriageConfig, load_probs_npz, run_triage, save_probs_npz

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------
# Config (kept in parity with the GROUP-A sweep where the concepts overlap)
# ----------------------------------------------------------------------------------

N_BANKS = 4
ALPHAS = [0.05, 0.1, 0.5]
SEEDS = [42, 0, 1, 2, 3]
FL_ROUNDS = 15
LOCAL_EPOCHS = 5
BATCH = 128
LR_INIT = 1e-3
FL_EXPERTS = ("fedavg", "fedprox", "fednova", "persfl")
TREE_EXPERTS = ("histgb", "rf", "extratrees")
FEDPROX_MU = 0.01
PERSFL_BLEND = 0.5
MAX_ROWS = 300_000
HEADLINE_RHO = 10
CELL_DIR = "cells"
CLASSES = np.array([0, 1])


def _afmt(alpha: float) -> str:
    """Canonical alpha string for filenames: float32/float64 representations of the same
    grid value map to the same name (0.05, not 0.05000000074505806)."""
    return format(float(alpha), "g")


def cell_seed(dataset: str, alpha: float, seed: int) -> int:
    """Deterministic per-cell seed. Same coordinates -> same numbers, whatever ran before."""
    return (seed * 100003 + int(round(float(alpha) * 1000)) * 97
            + sum(ord(c) for c in dataset)) % (2**31 - 1)


def cell_csv(out: str, dataset: str, alpha: float, seed: int) -> str:
    return os.path.join(out, CELL_DIR, f"triage_cell_{dataset}_a{_afmt(alpha)}_s{seed}.csv")


def cell_npz(out: str, dataset: str, alpha: float, seed: int) -> str:
    return os.path.join(out, CELL_DIR, f"probs_{dataset}_alpha{_afmt(alpha)}_seed{seed}.npz")


def _atomic_csv(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _cfg_matches(path: str, cfg: TriageConfig) -> bool:
    """A cached cell counts as done only if it was computed under the SAME triage config.
    Without this, resuming with a different --defer-rule/--gate/--human-accuracy would
    silently return rows computed under the old settings."""
    try:
        head = pd.read_csv(path)
    except Exception:
        return False
    if head.empty:
        return False
    want = {"defer_rule": cfg.defer_rule, "gate": cfg.gate,
            "human_accuracy": cfg.human_accuracy, "target_fnr": cfg.target_fnr,
            "defer_budget": cfg.defer_budget}
    for k, v in want.items():
        if k not in head.columns:
            return False
        if isinstance(v, str):
            if not (head[k] == v).all():
                return False
        elif not np.allclose(head[k].astype(float), float(v)):
            return False
    return sorted(head["rho"].astype(float).unique().tolist()) == sorted(
        float(r) for r in cfg.rhos)


# ----------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------

def _find(names):
    if os.path.isdir("/kaggle/input"):
        low = {n.lower() for n in names}
        for root, _, files in os.walk("/kaggle/input"):
            for f in files:
                if f in names or f.lower() in low:
                    return os.path.join(root, f)
    for n in names:
        if os.path.exists(n):
            return n
    return None


def _cap(X, y, rng):
    """Stratified subsample to MAX_ROWS: keep ALL fraud, subsample legit."""
    if len(y) <= MAX_ROWS:
        return X, y
    fraud = np.where(y == 1)[0]
    legit = np.where(y == 0)[0]
    keep = rng.choice(legit, size=min(max(MAX_ROWS - len(fraud), 0), len(legit)),
                      replace=False)
    idx = np.concatenate([fraud, keep])
    rng.shuffle(idx)
    return X[idx], y[idx]


# --- AML feature engineering (parity with the GROUP-A benchmark notebook) -----------
# The triage layer must see the SAME features the paper's experts were trained on. Blind
# pd.factorize of raw columns is banned here: SAML's Laundering_type deterministically
# encodes the label, and factorized account IDs / timestamps memorise entities across the
# random split. These are the benchmark's engineered signals with those columns excluded.
HIGH_RISK_COUNTRIES = {"mexico", "turkey", "morocco", "uae", "iran",
                       "nigeria", "russia", "venezuela", "north korea"}
IBM_HIGH_RISK_BANKS = {11, 22, 33, 44, 55, 66}
IBM_PAY_MAP = {"ACH": "Bank Transfer", "Wire": "Wire", "Cheque": "Cheque",
               "Credit Card": "Credit Card", "Debit Card": "Debit Card",
               "Cash": "Cash", "Bitcoin": "Bitcoin", "Reinvestment": "Reinvestment"}
SAML_FILES = ["SAML-D.csv"]
IBM_FILES = ["HI-Small_Trans.csv"]
ULB_FILES = ["creditcard.csv"]


def _preprocess_saml(df):
    from sklearn.preprocessing import LabelEncoder
    df.columns = [c.strip() for c in df.columns]
    y = df["Is_laundering"].astype(int).to_numpy()
    amt = df["Amount"].fillna(0).abs()
    amt_log = np.log1p(amt).to_numpy()
    amt_round = (amt % 1000 == 0).astype(int).to_numpy()
    amt_thresh = (amt < 10000).astype(int).to_numpy()
    t = pd.to_numeric(df["Time"], errors="coerce").fillna(0)
    hr_sin = np.sin(2 * np.pi * t / 86400).to_numpy()
    hr_cos = np.cos(2 * np.pi * t / 86400).to_numpy()
    pt_enc = LabelEncoder().fit_transform(df["Payment_type"].astype(str).fillna("Unknown"))
    sl_risk = df["Sender_bank_location"].astype(str).str.lower().apply(
        lambda x: int(any(h in x for h in HIGH_RISK_COUNTRIES))).to_numpy()
    rl_risk = df["Receiver_bank_location"].astype(str).str.lower().apply(
        lambda x: int(any(h in x for h in HIGH_RISK_COUNTRIES))).to_numpy()
    cross_border = (df["Sender_bank_location"].astype(str)
                    != df["Receiver_bank_location"].astype(str)).astype(int).to_numpy()
    curr_mm = (df["Payment_currency"].astype(str)
               != df["Received_currency"].astype(str)).astype(int).to_numpy()
    X = np.stack([amt_log, amt_round, amt_thresh, hr_sin, hr_cos, pt_enc.astype(float),
                  sl_risk, rl_risk, cross_border, curr_mm], axis=1).astype(np.float32)
    return X, y


def _preprocess_ibm(df):
    from sklearn.preprocessing import LabelEncoder
    df.columns = [c.strip() for c in df.columns]
    y = df["Is Laundering"].astype(int).to_numpy()
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    amt = df["Amount Paid"].fillna(0).abs()
    amt_log = np.log1p(amt).to_numpy().astype(np.float32)
    amt_round = (amt % 1000 == 0).astype(int).to_numpy().astype(np.float32)
    amt_thresh = (amt < 10000).astype(int).to_numpy().astype(np.float32)
    hours = ts.dt.hour.fillna(0).to_numpy()
    hr_sin = np.sin(2 * np.pi * hours / 24).astype(np.float32)
    hr_cos = np.cos(2 * np.pi * hours / 24).astype(np.float32)
    pay_fmt = df["Payment Format"].astype(str).fillna("Unknown")
    pay_mapped = pay_fmt.map(lambda x: IBM_PAY_MAP.get(x, x))
    pt_enc = LabelEncoder().fit_transform(pay_mapped).astype(np.float32)
    fb_int = df["From Bank"].fillna(0).astype(int)
    tb_int = df["To Bank"].fillna(0).astype(int)
    sl_risk = fb_int.isin(IBM_HIGH_RISK_BANKS).astype(int).to_numpy().astype(np.float32)
    rl_risk = tb_int.isin(IBM_HIGH_RISK_BANKS).astype(int).to_numpy().astype(np.float32)
    cross_border = (df["From Bank"].fillna(-1)
                    != df["To Bank"].fillna(-2)).astype(int).to_numpy().astype(np.float32)
    curr_mm = (df["Payment Currency"].astype(str)
               != df["Receiving Currency"].astype(str)).astype(int).to_numpy().astype(np.float32)
    X = np.stack([amt_log, amt_round, amt_thresh, hr_sin, hr_cos, pt_enc,
                  sl_risk, rl_risk, cross_border, curr_mm], axis=1).astype(np.float32)
    return X, y


def _preprocess_ulb(df):
    df.columns = [c.strip() for c in df.columns]
    y = df["Class"].astype(int).to_numpy()
    vcols = [c for c in df.columns if c.startswith("V")]
    Xpca = df[vcols].to_numpy(dtype=np.float32)
    amt_log = np.log1p(df["Amount"].fillna(0).abs()).to_numpy().astype(np.float32)
    X = np.column_stack([Xpca, amt_log]).astype(np.float32)
    return X, y


def load_dataset(name, rng):
    if name == "synthetic":
        n, d = 30000, 20
        y = (rng.random(n) < 0.02).astype(int)
        X = rng.normal(0, 1, (n, d))
        X[y == 1] += rng.normal(0.8, 0.3, (int(y.sum()), d)) * (rng.random(d) < 0.4)
        return X.astype(np.float32), y

    spec = {"ulb": (ULB_FILES, _preprocess_ulb),
            "saml": (SAML_FILES, _preprocess_saml),
            "ibm": (IBM_FILES, _preprocess_ibm)}
    if name not in spec:
        raise ValueError(name)
    files, preprocess = spec[name]
    p = _find(files)
    if not p:
        raise FileNotFoundError(f"{name}: none of {files} found under /kaggle/input")
    df = pd.read_csv(p, low_memory=False)
    X, y = preprocess(df)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return _cap(X, y, rng)


class Bank:
    def __init__(self, Xtr, ytr, Xva, yva, Xte, yte, bid):
        self.Xtr, self.ytr = Xtr, ytr
        self.Xva, self.yva = Xva, yva
        self.Xte, self.yte = Xte, yte
        self.bid = bid


def dirichlet_partition(X, y, k, alpha, rng):
    from sklearn.preprocessing import StandardScaler
    fraud = np.where(y == 1)[0]
    legit = np.where(y == 0)[0]
    rng.shuffle(fraud)
    rng.shuffle(legit)
    props = rng.dirichlet([alpha] * k)
    fsplit = np.split(fraud, (np.cumsum(props)[:-1] * len(fraud)).astype(int))
    lsplit = np.array_split(legit, k)
    banks = []
    for b in range(k):
        idx = np.concatenate([fsplit[b], lsplit[b]])
        rng.shuffle(idx)
        Xb, yb = X[idx], y[idx]
        n = len(idx)
        ntr, nva = int(0.70 * n), int(0.15 * n)
        sc = StandardScaler().fit(Xb[:ntr])                  # fit on train only, no leakage
        Xb = sc.transform(Xb).astype(np.float32)
        banks.append(Bank(Xb[:ntr], yb[:ntr], Xb[ntr:ntr + nva], yb[ntr:ntr + nva],
                          Xb[ntr + nva:], yb[ntr + nva:], b))
    return banks


# ----------------------------------------------------------------------------------
# Experts
# ----------------------------------------------------------------------------------

def make_tree(kind, seed):
    from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    if kind == "histgb":
        return HistGradientBoostingClassifier(max_iter=150, random_state=seed,
                                              class_weight="balanced")
    if kind == "rf":
        return RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1,
                                      class_weight="balanced")
    return ExtraTreesClassifier(n_estimators=150, random_state=seed, n_jobs=-1,
                                class_weight="balanced")


def _mlp(seed):
    from sklearn.neural_network import MLPClassifier
    # warm_start stays False ON PURPOSE: partial_fit never reinitialises once coefs_
    # exist regardless of this flag, while warm_start=True makes sklearn's partial_fit
    # input validation REJECT batches whose y has a single class -- which legitimately
    # happens to zero-fraud banks at alpha=0.05. Same configuration as the corrected
    # sweep notebook.
    return MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                         learning_rate_init=LR_INIT, max_iter=LOCAL_EPOCHS,
                         random_state=seed, batch_size=BATCH, warm_start=False, tol=1e-4)


def _get_w(m):
    return [c.copy() for c in m.coefs_] + [i.copy() for i in m.intercepts_]


def _set_w(m, w):
    n = len(m.coefs_)
    m.coefs_ = [x.copy() for x in w[:n]]
    m.intercepts_ = [x.copy() for x in w[n:]]


def _alloc(seed, n_feat):
    """Allocate an MLP's weight arrays on dummy zeros -- no bank data involved."""
    m = _mlp(seed)
    m.partial_fit(np.zeros((2, n_feat), dtype=np.float32), CLASSES, classes=CLASSES)
    return m


def _local_fit(lm, X, y, gw=None, mu=0.0):
    """Continue training from the model's CURRENT weights for LOCAL_EPOCHS epochs.

    partial_fit never reinitialises once coefs_ exist, so the global weights injected by
    _set_w are the true starting point (the reinitialising-fit bug class cannot occur).
    classes=CLASSES keeps single-class banks trainable. With mu > 0 the FedProx proximal
    pull is applied after each epoch (epoch-granularity approximation of the proximal
    objective, matching the corrected sweep notebook's local_fit)."""
    steps = max(1, int(np.ceil(len(X) / BATCH)))
    pull = min(LR_INIT * mu * steps, 0.5)
    for _ in range(LOCAL_EPOCHS):
        lm.partial_fit(X, y, classes=CLASSES)
        if mu > 0.0 and gw is not None:
            w = _get_w(lm)
            _set_w(lm, [wi - pull * (wi - gi) for wi, gi in zip(w, gw)])
    return lm


def train_fl(banks, strategy, seed):
    """Return {bank_id: fitted MLP} after FL_ROUNDS of the given strategy."""
    n_feat = banks[0].Xtr.shape[1]
    gw = _get_w(_alloc(seed, n_feat))
    bw = {b.bid: gw for b in banks}

    for _ in range(FL_ROUNDS):
        wl, sz, deltas, taus = [], [], [], []
        for b in banks:
            if len(b.Xtr) == 0:
                continue
            base = bw[b.bid] if strategy == "persfl" else gw
            lm = _alloc(seed, n_feat)
            _set_w(lm, base)
            _local_fit(lm, b.Xtr, b.ytr,
                       gw=gw if strategy == "fedprox" else None,
                       mu=FEDPROX_MU if strategy == "fedprox" else 0.0)
            w = _get_w(lm)
            if strategy == "persfl":
                bw[b.bid] = [PERSFL_BLEND * lw + (1 - PERSFL_BLEND) * g
                             for lw, g in zip(w, gw)]
            wl.append(w)
            sz.append(len(b.Xtr))
            deltas.append([lw - g for lw, g in zip(w, gw)])
            taus.append(LOCAL_EPOCHS * max(1, int(np.ceil(len(b.Xtr) / BATCH))))
        if not wl:
            break
        tot = sum(sz)
        if strategy == "fednova":
            tau_g = sum(t * s / tot for t, s in zip(taus, sz))
            gw = [gw[i] + tau_g * sum((s / tot) * (d[i] / max(t, 1))
                  for d, s, t in zip(deltas, sz, taus)) for i in range(len(gw))]
        else:
            gw = [sum((s / tot) * w[i] for w, s in zip(wl, sz)) for i in range(len(gw))]

    def build(w):
        m = _alloc(seed, n_feat)
        _set_w(m, w)
        return m

    if strategy == "persfl":
        return {b.bid: build(bw[b.bid]) for b in banks}
    g = build(gw)
    return {b.bid: g for b in banks}


def expert_probs(banks, seed):
    """{bank_id: {expert_name: (val_probs, test_probs)}} for all 7 experts."""
    fl = {s: train_fl(banks, s, seed) for s in FL_EXPERTS}
    out = {}
    for b in banks:
        d = {}
        for s in FL_EXPERTS:
            m = fl[s][b.bid]
            d[s] = (m.predict_proba(b.Xva)[:, 1], m.predict_proba(b.Xte)[:, 1])
        for kind in TREE_EXPERTS:
            if len(np.unique(b.ytr)) < 2:
                d[kind] = (np.full(len(b.yva), 0.5), np.full(len(b.yte), 0.5))
                continue
            m = make_tree(kind, seed).fit(b.Xtr, b.ytr)
            d[kind] = (m.predict_proba(b.Xva)[:, 1], m.predict_proba(b.Xte)[:, 1])
        out[b.bid] = d
    return out


# ----------------------------------------------------------------------------------
# Aggregation + figures
# ----------------------------------------------------------------------------------

def summarize(df):
    keys = ["dataset", "alpha", "rho"]
    num = [c for c in df.select_dtypes("number").columns if c not in keys + ["seed"]]
    out = df.groupby(keys)[num].mean().reset_index()
    out["n_seeds"] = df.groupby(keys)["seed"].nunique().values
    return out


def _figs(df, dataset, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    head = df[df.rho == HEADLINE_RHO].groupby("alpha")[
        ["expected_cost", "auto_clear_at_recall80"]].agg(["mean", "std"])
    a = head.index.values
    for col, ylab, fn in [
        ("expected_cost", "expected cost", f"cost_vs_alpha_{dataset}.png"),
        ("auto_clear_at_recall80", "auto-clear fraction @ recall 0.80",
         f"autoclear_vs_alpha_{dataset}.png"),
    ]:
        m = head[(col, "mean")].values
        sd = np.nan_to_num(head[(col, "std")].values)
        plt.figure(figsize=(5.5, 3.6))
        plt.errorbar(a, m, yerr=sd, fmt="o-", lw=2, capsize=4)
        plt.xlabel("Dirichlet alpha (higher = more IID)")
        plt.ylabel(ylab)
        plt.title(f"{dataset}: {ylab} vs heterogeneity (rho={HEADLINE_RHO})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out, fn), dpi=130)
        plt.close()


def _risk_coverage_fig(npz_path, dataset, alpha, seed, out,
                       cfg: TriageConfig = TriageConfig()):
    """Pooled risk-coverage curve recomputed from a cell's capture file, so the figure is
    a pure function of (dataset, alpha, seed) -- not of resume history."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .integration import _bank_scores

    probs, y_val, y_test = load_probs_npz(npz_path)
    names = sorted(next(iter(probs.values())))
    cost = T.CostMatrix(HEADLINE_RHO, 1)
    s_all, y_all, taus = [], [], []
    thresholds = {}
    scored = {}
    for bid in sorted(probs):
        yv = np.asarray(y_val[bid])
        val_p = [np.asarray(probs[bid][n][0], dtype=float) for n in names]
        te_p = [np.asarray(probs[bid][n][1], dtype=float) for n in names]
        bk = _bank_scores(yv, val_p, te_p, dataset, alpha, seed, bid, cfg)
        scored[bid] = bk
        thresholds[bid] = bk["t"]
    global_t = T.quantile_of_quantiles(list(thresholds.values()))
    for bid, bk in scored.items():
        t = thresholds[bid] if thresholds[bid] is not None else global_t
        taus.append(T.flag_threshold(cost, t))
        s_all.append(bk["s_test"])
        y_all.append(np.asarray(y_test[bid]))
    rc = T.risk_coverage_curve(np.concatenate(s_all), np.concatenate(y_all),
                               float(np.mean(taus)), cost=cost)
    plt.figure(figsize=(5.5, 3.6))
    plt.plot(rc["coverage"], rc["selective_risk"], lw=2)
    plt.xlabel("coverage (fraction auto-decided)")
    plt.ylabel(f"selective cost per txn (rho={HEADLINE_RHO})")
    plt.title(f"{dataset}: risk-coverage (alpha={_afmt(alpha)}, seed={seed}, "
              f"AURC={rc['aurc']:.3f})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"risk_coverage_{dataset}.png"), dpi=130)
    plt.close()


# ----------------------------------------------------------------------------------
# Sweep driver (seeded + resumable)
# ----------------------------------------------------------------------------------

def run_sweep(dataset, seeds, alphas, out=".", resume_dir=None, verbose=True,
              cfg: TriageConfig = TriageConfig()):
    import glob as _glob
    import shutil

    os.makedirs(os.path.join(out, CELL_DIR), exist_ok=True)

    # adopt checkpoints from a previous session (cell CSVs + capture npz)
    if resume_dir and os.path.isdir(resume_dir):
        for pat in ("triage_cell_*.csv", "probs_*.npz"):
            for src in _glob.glob(os.path.join(resume_dir, "**", pat), recursive=True):
                dst = os.path.join(out, CELL_DIR, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy(src, dst)

    grid = [(s, a) for s in seeds for a in alphas]
    todo, stale_cfg = [], 0
    for s, a in grid:
        csv_path = cell_csv(out, dataset, a, s)
        if os.path.exists(csv_path) and _cfg_matches(csv_path, cfg):
            continue
        if os.path.exists(csv_path):
            stale_cfg += 1
        todo.append((s, a))
    if verbose:
        note = f" ({stale_cfg} recomputed: config changed)" if stale_cfg else ""
        print(f"[{dataset}] {len(grid) - len(todo)} cell(s) cached, "
              f"{len(todo)} to compute (of {len(grid)}){note}")

    for seed in sorted({s for s, _ in todo}):
        X = y = None
        for alpha in [a for s, a in todo if s == seed]:
            npz_path = cell_npz(out, dataset, alpha, seed)
            if os.path.exists(npz_path):
                # expert probs are config-independent: reuse them, redo only decisions
                probs, y_val, y_test = load_probs_npz(npz_path)
            else:
                if X is None:
                    X, y = load_dataset(dataset,
                                        np.random.default_rng(cell_seed(dataset, -1, seed)))
                cs = cell_seed(dataset, alpha, seed)
                banks = dirichlet_partition(X, y, N_BANKS, alpha,
                                            np.random.default_rng(cs))
                probs = expert_probs(banks, cs)
                bank_meta = [{"id": b.bid, "y_val": b.yva, "y_test": b.yte} for b in banks]
                save_probs_npz(npz_path, probs, bank_meta)
                y_val = {b.bid: b.yva for b in banks}
                y_test = {b.bid: b.yte for b in banks}
            rows = run_triage(y_val, y_test, probs, dataset, alpha, seed, cfg=cfg)
            _atomic_csv(pd.DataFrame(rows), cell_csv(out, dataset, alpha, seed))
            if verbose:
                print(f"  [{dataset}] seed={seed} alpha={_afmt(alpha)} done")

    # combine ONLY the requested grid; stray cells are reported, never mixed in
    missing = [(s, a) for s, a in grid if not os.path.exists(cell_csv(out, dataset, a, s))]
    if missing:
        raise RuntimeError(f"cells missing after sweep: {missing}")
    df = pd.concat([pd.read_csv(cell_csv(out, dataset, a, s)) for s, a in grid],
                   ignore_index=True)
    all_cells = set(_glob.glob(os.path.join(out, CELL_DIR, f"triage_cell_{dataset}_*.csv")))
    used = {cell_csv(out, dataset, a, s) for s, a in grid}
    stray = sorted(os.path.basename(p) for p in all_cells - used)
    if stray and verbose:
        print(f"[{dataset}] NOTE: {len(stray)} cell file(s) outside the requested grid "
              f"were ignored: {stray}")

    _atomic_csv(df, os.path.join(out, f"triage_results_{dataset}.csv"))
    _atomic_csv(summarize(df), os.path.join(out, f"triage_summary_{dataset}.csv"))
    _figs(df, dataset, out)
    s0, a0 = seeds[0], alphas[0]
    fig_npz = cell_npz(out, dataset, a0, s0)
    if os.path.exists(fig_npz):
        _risk_coverage_fig(fig_npz, dataset, a0, s0, out, cfg)
    elif verbose:
        print(f"[{dataset}] risk-coverage figure skipped: {os.path.basename(fig_npz)} "
              f"not present (cell CSV was adopted without its capture file)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic",
                    choices=["synthetic", "ulb", "saml", "ibm"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    ap.add_argument("--out", default=".")
    ap.add_argument("--resume-dir", default=None,
                    help="folder with cells/ checkpoints from a previous session")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--defer-rule", default="band", choices=["band", "chow", "capacity"])
    ap.add_argument("--gate", default="brier", choices=["uniform", "brier"])
    ap.add_argument("--human-accuracy", type=float, default=1.0)
    args = ap.parse_args()

    seeds = args.seeds[:1] if args.quick else args.seeds
    alphas = args.alphas[:1] if args.quick else args.alphas
    cfg = TriageConfig(defer_rule=args.defer_rule, gate=args.gate,
                       human_accuracy=args.human_accuracy)
    df = run_sweep(args.dataset, seeds, alphas, args.out, args.resume_dir, cfg=cfg)

    s = summarize(df)
    print(f"\nwrote triage_results_{args.dataset}.csv ({len(df)} rows), summary, 3 figures")
    print(s[s.rho == HEADLINE_RHO][["dataset", "alpha", "n_seeds", "expected_cost",
          "miss_rate_policy", "coverage_valid_frac", "auto_clear_at_recall80",
          "aurc"]].round(4).to_string(index=False))
    return df


if __name__ == "__main__":
    main()
