#!/usr/bin/env python3
# ============================================================================
# GROUP-A alpha sweep on Kaggle — SAML, seeds [2, 3] (parallel-safe, resumable)
#
# Single-file version of the canonical merged notebook
#   notebooks/MOE_experiments/seed_runs/moe-fl-per-dataset-alpha-sweep-GROUP-A.ipynb
# (PR #17: triage probability capture + eval fixes) restricted to one dataset
# with PR #16's Fix E order-independent partitioning and session resume, plus a
# session time-budget guard sized to Kaggle's 12 h cap. Fix E makes every
# (seed, dataset, alpha) cell canonical regardless of run order, so seed chunks
# run independently — even in parallel on separate accounts.
#
# Run on Kaggle: create a new Script (or one-cell `%run`) kernel, paste this
# file, attach the input dataset below, Accelerator = GPU, then Save & Run All.
#   attach : berkanoztas/synthetic-transaction-monitoring-dataset-aml  (SAML-D.csv)
#   cost   : seeds [2,3] x ~3.2 h = ~6.4 h -> ONE session, guaranteed.
#   part 2/2 - run in parallel with sweep_SAML_part1.py (other account).
#   If you run this part LAST, attach part1 output and this session
#   emits the complete 5-seed package.
#
# Resuming/merging: attach any previous session outputs of THIS dataset as
# Dataset inputs and run unchanged — finished (seed, alpha) cells are skipped,
# and sibling seed-part outputs are merged into this session's final package
# (RESUME_DIR='auto' finds every mount).
# Full instructions + sanity checklist: scripts/kaggle/README.md
# ============================================================================

# ─────────── CONFIG — the only per-script block; body below is shared ───────────
ONLY_DATASET     = 'SAML'
SEEDS            = [2, 3]            # this script's seed chunk (canonical full list:
                                     # [42, 0, 1, 2, 3]; Fix E keeps chunks canonical)
SESSION_BUDGET_H = 10.5              # stop starting new cells after this many hours
RESUME_DIR       = 'auto'            # 'auto' | None | path | list of paths

# ==== SHARED PIPELINE — byte-identical across every sweep_*.py; regenerate all together (see check_shared_sync.sh) ====

import os, sys, subprocess

ON_KAGGLE = os.path.exists('/kaggle')
INPUT_DIR = os.environ.get('SWEEP_INPUT_DIR', '/kaggle/input')

if ON_KAGGLE:   # packages are preinstalled on Kaggle images; this is a cheap no-op top-up
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        'lightgbm', 'xgboost', 'imbalanced-learn', 'catboost'],
                       check=False, timeout=900)
    except Exception as _pip_err:
        print(f"pip top-up skipped ({_pip_err}) — relying on preinstalled packages")


# ------------------------ Imports + Config ------------------------

# ================================================================
# SEPARATE DATASET BENCHMARK (LEAKAGE-FREE + ALPHA SWEEP + TEMPORAL SPLIT)
# Each dataset (SAML-D, IBM AML, ULB) runs the SAME pipeline independently,
# across Dirichlet alpha sweep {0.05, 0.1, 0.5}.
#
# FL  (4): FedAvg | FedProx | FedNova | PersFL  (sklearn MLP via cuml.accel)
# ML  (3): XGBoost (cuda) + LightGBM (cpu) + CatBoost (gpu)
# MoE (4): Static | Performance | Confidence | Typology-Aware (oracle)
#
# Rigor fixes:
#  - No feature leakage (typ_enc, sl_enc, rl_enc, ds_src dropped)
#  - Temporal split (sort-by-time, train=early / val=mid / test=late)
#  - Per-model threshold tuned on validation (maximize F2)
#  - TypologyAwareGate routes only on fraud typologies (excludes Unknown/Normal)
# ================================================================

from imblearn.over_sampling import SMOTE

try:
    import cuml
    cuml.accel.install()
    print("cuml.accel installed — sklearn MLP on GPU")
except Exception as _cuml_err:
    if ON_KAGGLE:
        raise SystemExit(
            f"cuml unavailable ({_cuml_err}). Enable the GPU accelerator "
            "(notebook settings -> Accelerator -> GPU): a CPU run of this sweep "
            "does not finish inside one Kaggle session.")
    print(f"WARNING: cuml unavailable ({_cuml_err}) — CPU sklearn MLP (local test only)")

import os, gc, time, warnings, random
from collections import defaultdict

import numpy as np
import pandas as pd
try:
    import cupy as cp
except ImportError:
    cp = None   # CPU-only local runs; Kaggle GPU images ship cupy

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, confusion_matrix, fbeta_score
)

import xgboost as xgb
import lightgbm as lgb

# CatBoost optional — graceful fallback if install failed
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("WARNING: catboost not available — will skip CatBoost expert")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)
CURRENT_SEED = 42          # updated at start of each seed iteration

print("All imports OK")
print(f"  CatBoost available: {HAS_CATBOOST}")

T0  = time.time()
OUT = '/kaggle/working' if os.path.exists('/kaggle') else '.'
def elapsed(): return f"{(time.time()-T0)/60:.1f}m"
def flush():
    gc.collect()
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()

# ================================================================
# PER-DATASET CONFIG — full data, no row caps
# ================================================================
DATASETS = {
    'SAML': {'n_banks': 4, 'rows': None},
    'IBM':  {'n_banks': 4, 'rows': None},
    'ULB':  {'n_banks': 4, 'rows': None},
}

# Dirichlet alpha sweep: extreme non-IID → moderate non-IID
ALPHAS = [0.05, 0.1, 0.5]

# FL strategies to run
FL_STRATEGIES = ['fedavg', 'fedprox', 'fednova', 'persfl']

# ================================================================
# SHARED CONFIG
# ================================================================
FL_ROUNDS      = 20
LOCAL_EPOCHS   = 5
MLP_CAP        = 100_000
LR             = 0.001
FEDPROX_MU     = 0.01
TEST_FRAC      = 0.15
VAL_FRAC       = 0.15
MIN_FRAUD      = 5
THRESH_DEFAULT = 0.5                    # fallback if tuning fails
TUNE_THRESHOLD = True                   # tune per-model threshold on val (maximize F2)
THRESH_GRID    = np.arange(0.05, 0.95, 0.05)
AP_KS          = (50, 100, 200)

IBM_FILE = 'HI-Small_Trans.csv'
IBM_HIGH_RISK_BANKS = {11, 22, 33, 44, 55, 66}
IBM_PAY_MAP = {
    'ACH':          'Bank Transfer',
    'Wire':         'Wire',
    'Cheque':       'Cheque',
    'Credit Card':  'Credit Card',
    'Debit Card':   'Debit Card',
    'Cash':         'Cash',
    'Bitcoin':      'Bitcoin',
    'Reinvestment': 'Reinvestment',
}
HIGH_RISK_COUNTRIES = {'mexico', 'turkey', 'morocco', 'uae', 'iran',
                        'nigeria', 'russia', 'venezuela', 'north korea'}

# ================================================================
# GROUP-A ENHANCEMENT FLAGS  (added by Group-A patch)
# Toggle these to control gate logging, multi-seed sweep, centralised baseline.
# ================================================================
LOG_GATE_WEIGHTS         = True              # A1: log MoE gate weights to JSONL
RUN_CENTRALISED_BASELINE = True              # A2: pooled-data xgb/lgbm/cat baseline
SEEDS                    = list(SEEDS)       # A3: canonical multi-seed rerun (set in CONFIG at top)
PREV_SEEDS_DIR           = "/kaggle/input/moe-fl-seeds"  # A3: mount previous seed outputs here (or None)
RUN_COST_SENSITIVE       = True              # A4: post-hoc cost-sensitive eval

GATE_LOG_PATH      = f'{OUT}/gate_weights_log.jsonl'
CENTRAL_OUT_CSV    = f'{OUT}/a2_centralised_results.csv'
COST_OUT_CSV       = f'{OUT}/a4_cost_results.csv'
COST_FLIPS_CSV     = f'{OUT}/a4_cost_ranking_flips.csv'


# ================================================================
# SPLIT + RESUME + SESSION BUDGET   (PR #16 machinery on the merged PR #17 base)
# One dataset per script so the three datasets run in parallel on separate
# Kaggle accounts; Fix E keeps every (seed, dataset, alpha) cell canonical
# regardless of run order, which is what makes the split and resume sound.
# ================================================================
DATASETS = {k: v for k, v in DATASETS.items() if k == ONLY_DATASET}
assert DATASETS, f"ONLY_DATASET={ONLY_DATASET!r} is not one of SAML/IBM/ULB"

import shutil as _sh_r

# Measured cost per seed (gate-log timestamps of a real partial run, PR #16).
PER_SEED_H = {'SAML': 3.2, 'IBM': 4.0, 'ULB': 1.2}
_CELL_H = {k: v / max(len(ALPHAS), 1) for k, v in PER_SEED_H.items()}
_BUDGET_SAFETY = 1.3
_STOPPED_EARLY = False

def _budget_allows(ds_name):
    """True if one more (dataset, alpha, seed) cell fits in the session budget."""
    if not SESSION_BUDGET_H:
        return True
    left_h = SESSION_BUDGET_H - (time.time() - T0) / 3600
    return left_h >= _CELL_H.get(ds_name, 1.5) * _BUDGET_SAFETY

def _cell_rng_seed(seed, ds_name, alpha):
    """Fix E: partition seed derived only from the cell's own coordinates."""
    return (seed * 100003 + int(round(float(alpha) * 1000)) * 97
            + sum(ord(_ch) for _ch in ds_name)) % (2**31 - 1)

def _autodetect_resume():
    """Find previous-session outputs mounted under INPUT_DIR (ALL of them —
    sibling seed-part outputs can be attached together and are merged).
    A mount qualifies if it holds this dataset's per-cell benchmark CSVs.
    Mounts whose root has a triage/ dir are the REPO checkout, not a results
    mount — excluded so the pre-Fix-E preview results committed under
    results/ can never be resumed from."""
    if not os.path.isdir(INPUT_DIR):
        return []
    _pat = tuple(f"{d.lower()}_alpha" for d in DATASETS)
    hits = []
    for root, _dirs, files in os.walk(INPUT_DIR):
        rel = os.path.relpath(root, INPUT_DIR)
        mount = os.path.join(INPUT_DIR, rel.split(os.sep)[0]) if rel != '.' else INPUT_DIR
        if os.path.isdir(os.path.join(mount, 'triage')):
            continue
        n = sum(1 for f in files
                if f.startswith(_pat) and '_seed' in f and f.endswith('_benchmark.csv'))
        if n:
            hits.append(root)
    return sorted(hits)

# RESUME_DIR accepts 'auto' | None | one path | a list of paths.
if RESUME_DIR == 'auto':
    RESUME_DIRS = _autodetect_resume()
elif not RESUME_DIR:
    RESUME_DIRS = []
elif isinstance(RESUME_DIR, str):
    RESUME_DIRS = [RESUME_DIR]
else:
    RESUME_DIRS = list(RESUME_DIR)
print(f"RESUME_DIRS = {RESUME_DIRS or 'none (fresh run)'}")

def _resume_find(fname):
    """First attached mount holding fname, else None."""
    for _d in RESUME_DIRS:
        _p = os.path.join(_d, fname)
        if os.path.exists(_p):
            return _p
    return None

def _resume_has(tag, seed):
    return _resume_find(f"{tag}_seed{seed}_benchmark.csv") is not None

def _resume_load(tag, seed):
    for suf in ("benchmark", "fl_history"):
        p = _resume_find(f"{tag}_seed{seed}_{suf}.csv")
        if p:
            _sh_r.copy(p, os.path.join(OUT, f"{tag}_seed{seed}_{suf}.csv"))
            _sh_r.copy(p, os.path.join(OUT, f"{tag}_{suf}.csv"))
    # carry the triage probability capture forward so the final session's
    # triage layer covers cells computed in earlier sessions too
    _npz = f"probs_{tag}_seed{seed}.npz"
    _p = _resume_find(_npz)
    if _p:
        _sh_r.copy(_p, os.path.join(OUT, _npz))
    elif globals().get('CAPTURE_PROBS', True):
        print(f"    [triage] NOTE: {_npz} not in any resume mount — that cell has no capture")
    return pd.read_csv(os.path.join(OUT, f"{tag}_seed{seed}_benchmark.csv"))

def _aggregate_sibling_outputs():
    """Pull OTHER seeds' finished cells of THIS dataset from the attached
    mounts into OUT, so the last seed-part run with its sibling parts attached
    emits the complete multi-seed package (A2 comparison, A3, A4 and triage
    over every seed). Sibling seeds are discovered from per-cell filenames,
    which are dataset-prefixed, so another dataset's mount is never mixed in."""
    import re as _re_agg
    _ds = ONLY_DATASET
    _cell_pat = _re_agg.compile(
        _re_agg.escape(_ds.lower()) + r'_alpha[\d.]+_seed(\d+)_benchmark\.csv$')
    _sib_seeds = set()
    for _d2 in RESUME_DIRS:
        if not os.path.isdir(_d2):
            continue
        for _f in os.listdir(_d2):
            _m = _cell_pat.match(_f)
            if _m and int(_m.group(1)) not in SEEDS:
                _sib_seeds.add(int(_m.group(1)))
    if not _sib_seeds:
        return
    _carried = []
    for _s in sorted(_sib_seeds):
        for _a in ALPHAS:
            _t = f"{_ds.lower()}_alpha{_a}"
            for _fn in (f"{_t}_seed{_s}_benchmark.csv",
                        f"{_t}_seed{_s}_fl_history.csv",
                        f"probs_{_t}_seed{_s}.npz"):
                _p = _resume_find(_fn)
                if _p and not os.path.exists(os.path.join(OUT, _fn)):
                    _sh_r.copy(_p, os.path.join(OUT, _fn))
        _cmb = f"all_benchmarks_combined_seed{_s}.csv"
        _p = _resume_find(_cmb)
        if _p and not os.path.exists(os.path.join(OUT, _cmb)):
            try:  # only carry a combined CSV that is really THIS dataset's
                if set(pd.read_csv(_p, usecols=['dataset'])['dataset'].unique()) <= set(DATASETS):
                    _sh_r.copy(_p, os.path.join(OUT, _cmb))
                    _carried.append(_s)
                else:
                    print(f"[AGGREGATE] skipped {_cmb}: belongs to another dataset")
            except Exception as _agg_err:
                print(f"[AGGREGATE] skipped {_cmb}: {_agg_err}")
        elif _p is None:
            print(f"[AGGREGATE] sibling seed {_s} is incomplete in the mounts "
                  "(no combined CSV) — its finished cells were still carried; "
                  "finish via the full-seed script with everything attached")
    print(f"\n[AGGREGATE] carried sibling seed cells for seeds {sorted(_sib_seeds)}")
    import glob as _glob_agg
    _seed_csvs = sorted(_glob_agg.glob(f"{OUT}/all_benchmarks_combined_seed*.csv"))
    if _carried and _seed_csvs:
        pd.concat([pd.read_csv(_f) for _f in _seed_csvs], ignore_index=True).to_csv(
            f"{OUT}/all_benchmarks_combined.csv", index=False)
        print(f"[AGGREGATE] rebuilt all_benchmarks_combined.csv "
              f"from {len(_seed_csvs)} per-seed CSVs")

def _resume_report():
    todo = [(s, d, a) for s in SEEDS for d in DATASETS for a in ALPHAS
            if not _resume_has(f"{d.lower()}_alpha{a}", s)]
    total = len(SEEDS) * len(DATASETS) * len(ALPHAS)
    est_h = sum(_CELL_H.get(d, 1.5) for _s, d, _a in todo)
    print(f"[{ONLY_DATASET}] {total - len(todo)} cached, {len(todo)} to compute "
          f"(of {total})  ~{est_h:.1f}h")
    if RESUME_DIR and not os.path.isdir(RESUME_DIR):
        print(f"  WARNING: RESUME_DIR not found ({RESUME_DIR}) - nothing will be skipped")
    for s, d, a in todo:
        print(f"    TODO  seed={s:<3} {d:<5} alpha={a}")


# ------------------------ Data Loading + Preprocessing ------------------------

# ================================================================
# FILE FINDERS
# ================================================================
def find_saml():
    candidates = [
        f'{INPUT_DIR}/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml/SAML-D.csv',
        f'{INPUT_DIR}/synthetic-transaction-monitoring-dataset-aml/SAML-D.csv',
    ]
    for p in candidates:
        if os.path.exists(p): return p
    for root, _, files in os.walk(INPUT_DIR):
        for f in files:
            if f.endswith('.csv') and 'SAML' in f.upper():
                return os.path.join(root, f)
    return None

def find_ibm():
    for root, _, files in os.walk(INPUT_DIR):
        for f in files:
            if f == IBM_FILE:
                return os.path.join(root, f)
    return None

def find_ulb():
    candidates = [
        f'{INPUT_DIR}/creditcardfraud/creditcard.csv',
        f'{INPUT_DIR}/creditcard-fraud/creditcard.csv',
    ]
    for p in candidates:
        if os.path.exists(p): return p
    for root, _, files in os.walk(INPUT_DIR):
        for f in files:
            if f.lower() == 'creditcard.csv':
                return os.path.join(root, f)
    return None

# ================================================================
# SAML-D
# ================================================================
def load_saml():
    path = find_saml()
    if not path: raise FileNotFoundError("SAML-D not found")
    print(f"Loading SAML-D: {path}")
    df = pd.read_csv(path, nrows=DATASETS.get('SAML', {'rows': None})['rows'], low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    print(f"  {len(df):,} rows | fraud: {df['Is_laundering'].sum():,} ({df['Is_laundering'].mean()*100:.3f}%)")
    return df

def preprocess_saml(df):
    """
    Returns X, y, typ, src, t_col.
    Features (10): amt_log, amt_round, amt_thresh, hr_sin, hr_cos, pt_enc,
                   sl_risk, rl_risk, cross_border, curr_mm
    Removed:
      - typ_enc  (label leakage: NaN→Unknown for non-fraud)
      - sl_enc, rl_enc (memorization: raw country ordinal encoding)
      - ds_src  (constant per dataset)
    t_col: numeric Time column — used for TEMPORAL split only (not in X)
    """
    y    = df['Is_laundering'].astype(int).values
    typ  = df['Laundering_type'].astype(str).fillna('Unknown').values
    src  = np.full(len(df), 'saml', dtype=object)

    # time column for temporal ordering (ONLY used for train/val/test split)
    t_raw = pd.to_numeric(df['Time'], errors='coerce').fillna(0).values
    if np.ptp(t_raw) < 1e-6:
        t_col = np.arange(len(df), dtype=np.int64)    # fallback: row order
    else:
        t_col = t_raw.astype(np.float64)

    amt = df['Amount'].fillna(0).abs()
    amt_log   = np.log1p(amt).values
    amt_round = (amt % 1000 == 0).astype(int).values
    amt_thresh= (amt < 10000).astype(int).values

    t = pd.to_numeric(df['Time'], errors='coerce').fillna(0)
    hr_sin = np.sin(2 * np.pi * t / 86400).values
    hr_cos = np.cos(2 * np.pi * t / 86400).values

    le_p = LabelEncoder()
    pt_enc = le_p.fit_transform(df['Payment_type'].astype(str).fillna('Unknown'))

    sl_risk = df['Sender_bank_location'].astype(str).str.lower().apply(
        lambda x: int(any(h in x for h in HIGH_RISK_COUNTRIES))).values
    rl_risk = df['Receiver_bank_location'].astype(str).str.lower().apply(
        lambda x: int(any(h in x for h in HIGH_RISK_COUNTRIES))).values
    cross_border = (df['Sender_bank_location'].astype(str) !=
                    df['Receiver_bank_location'].astype(str)).astype(int).values
    curr_mm = (df['Payment_currency'].astype(str) !=
               df['Received_currency'].astype(str)).astype(int).values

    X = np.stack([
        amt_log, amt_round, amt_thresh,
        hr_sin, hr_cos, pt_enc.astype(float),
        sl_risk, rl_risk, cross_border, curr_mm,
    ], axis=1).astype(np.float32)

    print(f"  SAML-D features: {X.shape}  (no typ_enc, no sl/rl_enc; temporal split enabled)")
    return X, y, typ, src, t_col

# ================================================================
# IBM AML
# ================================================================
def load_ibm():
    path = find_ibm()
    if not path: raise FileNotFoundError(f"{IBM_FILE} not found")
    print(f"Loading IBM AML: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    print(f"  {len(df):,} rows | fraud: {df['Is Laundering'].sum():,} ({df['Is Laundering'].mean()*100:.3f}%)")
    return df

def preprocess_ibm(df):
    """
    Returns X, y, typ, src, t_col.
    Features (10): amt_log, amt_round, amt_thresh, hr_sin, hr_cos, pt_enc,
                   sl_risk, rl_risk, cross_border, curr_mm
    Removed: typ_enc (label leakage), sl_enc/rl_enc (bank ID memorization), ds_src (constant)
    t_col: nanoseconds-since-epoch from Timestamp — used for TEMPORAL split only
    """
    y   = df['Is Laundering'].astype(int).values
    src = np.full(len(df), 'ibm', dtype=object)

    pay_fmt = df['Payment Format'].astype(str).fillna('Unknown')
    typ = np.where(
        y == 1,
        ('IBM_' + pay_fmt).values,
        'IBM_Normal'
    )

    # Robust Timestamp → int64 nanoseconds (NaT becomes 0). Fall back to row index on failure.
    ts = pd.to_datetime(df['Timestamp'], errors='coerce')
    valid_mask = ts.notna().values
    if valid_mask.sum() < len(df) * 0.5:
        t_col = np.arange(len(df), dtype=np.int64)
    else:
        ts_int = ts.astype('int64', copy=False).values     # NaT → numpy int64 min
        t_col = np.where(valid_mask, ts_int, 0).astype(np.int64)

    amt = df['Amount Paid'].fillna(0).abs()
    amt_log   = np.log1p(amt).values.astype(np.float32)
    amt_round = (amt % 1000 == 0).astype(int).values.astype(np.float32)
    amt_thresh= (amt < 10000).astype(int).values.astype(np.float32)

    hours = ts.dt.hour.fillna(0).values
    hr_sin = np.sin(2 * np.pi * hours / 24).astype(np.float32)
    hr_cos = np.cos(2 * np.pi * hours / 24).astype(np.float32)

    pay_mapped = pay_fmt.map(lambda x: IBM_PAY_MAP.get(x, x))
    le_p = LabelEncoder()
    pt_enc = le_p.fit_transform(pay_mapped).astype(np.float32)

    fb_int = df['From Bank'].fillna(0).astype(int)
    tb_int = df['To Bank'].fillna(0).astype(int)
    sl_risk = fb_int.isin(IBM_HIGH_RISK_BANKS).astype(int).values.astype(np.float32)
    rl_risk = tb_int.isin(IBM_HIGH_RISK_BANKS).astype(int).values.astype(np.float32)

    cross_border = (df['From Bank'].fillna(-1) != df['To Bank'].fillna(-2)).astype(int).values.astype(np.float32)

    curr_mm = (df['Payment Currency'].astype(str) !=
               df['Receiving Currency'].astype(str)).astype(int).values.astype(np.float32)

    X = np.stack([
        amt_log, amt_round, amt_thresh,
        hr_sin, hr_cos, pt_enc,
        sl_risk, rl_risk, cross_border, curr_mm,
    ], axis=1).astype(np.float32)

    print(f"  IBM AML features: {X.shape}  (no typ_enc, no sl/rl_enc; temporal split enabled)")
    return X, y, typ, src, t_col

# ================================================================
# ULB CREDIT CARD FRAUD
# ================================================================
def load_ulb():
    path = find_ulb()
    if not path: raise FileNotFoundError("creditcard.csv not found")
    print(f"Loading ULB Credit Card: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    print(f"  {len(df):,} rows | fraud: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
    return df

def preprocess_ulb(df):
    """
    Returns X, y, typ, src, t_col.
    Features (29): V1-V28 (PCA) + Amount_log
    t_col: Time column (seconds since first transaction in dataset)
    """
    y   = df['Class'].astype(int).values
    src = np.full(len(df), 'ulb', dtype=object)

    typ = np.where(y == 1, 'CC_Fraud', 'CC_Normal')

    t_col = df['Time'].fillna(0).values.astype(np.float64)

    feature_cols = [c for c in df.columns if c.startswith('V')]
    X_pca = df[feature_cols].values.astype(np.float32)

    amt_log = np.log1p(df['Amount'].fillna(0).abs()).values.astype(np.float32)

    X = np.column_stack([X_pca, amt_log]).astype(np.float32)

    print(f"  ULB features: {X.shape}  (V1-V28 + Amount_log; temporal split enabled)")
    return X, y, typ, src, t_col


# ------------------------ Dirichlet Partition + Temporal Split ------------------------

# ================================================================
# GENERIC DIRICHLET PARTITION + TEMPORAL SPLIT
# - Dirichlet(alpha) allocates fraud across banks (heterogeneous typology)
# - Legit is split evenly across banks
# - Each bank's rows then sorted by TIME, split 70% train / 15% val / 15% test
#   (train=earliest, test=latest — prevents temporal leakage)
# No MIN_FRAUD enforcement — banks may legitimately receive 0 fraud at extreme alpha.
# ================================================================
def _allocate_fraud(n_fraud, n_banks, alpha):
    """
    Dirichlet fraud allocation with only rounding fix (no MIN_FRAUD floor).
    Guarantees fcnts.sum() == n_fraud exactly.
    """
    props = np.random.dirichlet([alpha] * n_banks)
    fcnts = (props * n_fraud).astype(int)

    # Fix rounding: make sum exactly n_fraud
    diff = n_fraud - fcnts.sum()
    if diff > 0:
        for _ in range(diff):
            fcnts[np.argmin(fcnts)] += 1
    elif diff < 0:
        for _ in range(-diff):
            fcnts[np.argmax(fcnts)] -= 1

    assert fcnts.sum() == n_fraud, f"fraud allocation mismatch: {fcnts.sum()} != {n_fraud}"
    return fcnts

def partition_dataset(X, y, typ, t_col, n_banks, alpha, ds_name):

    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]
    np.random.shuffle(fraud_idx)
    np.random.shuffle(legit_idx)

    fcnts = _allocate_fraud(len(fraud_idx), n_banks, alpha)
    lper = len(legit_idx) // n_banks

    banks = []
    fp = lp = 0
    for i in range(n_banks):
        ln = lper if i < n_banks - 1 else len(legit_idx) - lp
        idx = np.concatenate([fraud_idx[fp:fp+fcnts[i]], legit_idx[lp:lp+ln]])
        name = f'{ds_name}-Bank{i}'
        banks.append(_split(i, X[idx], y[idx], typ[idx], t_col[idx],
                            name, ds_name.lower()))
        fp += fcnts[i]
        lp += ln

    print(f"\nPartition [{ds_name}] — {n_banks} banks | Dirichlet alpha={alpha} | temporal split:")
    for b in banks:
        print(f"  {b['name']:20s}: {b['n_total']:>10,} txns | "
              f"{b['n_fraud']:>6,} fraud ({b['fraud_frac']*100:.3f}%) | "
              f"train={b['n_train_fraud']} val={b['n_val_fraud']} test={b['n_test_fraud']} fraud")

    # Info-only — banks are NOT skipped during evaluation anymore
    for b in banks:
        if b['n_test_fraud'] == 0:
            print(f"  NOTE: {b['name']} has 0 fraud in test — will report specificity/FPR instead of recall/AUPRC")

    return banks

def _split(bid, X, y, typ, t, name, source):
    """
    Temporal split: sort by time, then first 70% train / next 15% val / last 15% test.
    StandardScaler fit on train only, applied to val/test.
    """
    order = np.argsort(t, kind='stable')
    X, y, typ = X[order], y[order], typ[order]

    n = len(X)
    n_train = int(n * (1 - TEST_FRAC - VAL_FRAC))
    n_val   = int(n * VAL_FRAC)
    # ensure we always have at least 1 row in each split
    n_train = max(n_train, 1)
    n_val   = max(n_val, 1)
    if n_train + n_val >= n:
        n_train = max(n - 2, 1)
        n_val   = 1

    Xtr, ytr, ttr = X[:n_train],              y[:n_train],              typ[:n_train]
    Xvl, yvl, tvl = X[n_train:n_train+n_val], y[n_train:n_train+n_val], typ[n_train:n_train+n_val]
    Xte, yte, tte = X[n_train+n_val:],        y[n_train+n_val:],        typ[n_train+n_val:]

    sc  = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xvl = sc.transform(Xvl).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)

    return dict(
        id=bid, name=name, source=source,
        X_train=Xtr, y_train=ytr, typ_train=ttr,
        X_val=Xvl,   y_val=yvl,   typ_val=tvl,
        X_test=Xte,  y_test=yte,  typ_test=tte,
        n_fraud=int(y.sum()), n_total=len(y),
        fraud_frac=float(y.mean()),
        n_train_fraud=int(ytr.sum()),
        n_val_fraud=int(yvl.sum()),
        n_test_fraud=int(yte.sum()),
    )

def safe_smote(X, y):
    if y.sum() < 5 or len(X) < 20: return X, y
    try:
        k = min(5, int(y.sum()) - 1)
        Xs, ys = SMOTE(random_state=CURRENT_SEED, k_neighbors=k).fit_resample(X, y)
        return Xs.astype(np.float32), ys.astype(np.int64)
    except Exception:
        return X, y

def get_mlp_data(bank):
    Xtr, ytr = bank['X_train'], bank['y_train']
    if len(Xtr) > MLP_CAP:
        fi = np.where(ytr == 1)[0]
        li = np.where(ytr == 0)[0]
        nf = min(len(fi), MLP_CAP // 10)
        nl = min(len(li), MLP_CAP - nf)
        # deterministic per-(seed, bank) subsample: identical for every
        # strategy and round, independent of global RNG state
        rng = np.random.default_rng(CURRENT_SEED * 1000 + int(bank['id']))
        rng.shuffle(fi); rng.shuffle(li)
        idx = np.concatenate([fi[:nf], li[:nl]])
        rng.shuffle(idx)
        Xtr, ytr = Xtr[idx], ytr[idx]
    return safe_smote(Xtr, ytr)


# ------------------------ MLP + FL Algorithms ------------------------

# ================================================================
# MLP (cuml.accel)
# ================================================================
def make_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation='relu',
        solver='adam', learning_rate_init=LR,
        max_iter=LOCAL_EPOCHS, random_state=CURRENT_SEED,
        batch_size=128, warm_start=False, tol=1e-4)

def get_weights(m):
    return [c.copy() for c in m.coefs_] + [i.copy() for i in m.intercepts_]

def set_weights(m, w):
    n = len(m.coefs_)
    m.coefs_ = [x.copy() for x in w[:n]]
    m.intercepts_ = [x.copy() for x in w[n:]]

def init_mlp(X, y):
    if 0 not in y: X,y = np.vstack([X,X[0:1]]), np.append(y,0)
    if 1 not in y: X,y = np.vstack([X,X[0:1]]), np.append(y,1)
    m = make_mlp(); m.fit(X, y); return m

def clone_mlp(template, w):
    m = make_mlp()
    tiny = np.zeros((2, template.coefs_[0].shape[0]), dtype=np.float32)
    try: m.fit(tiny, np.array([0,1]))
    except: pass
    set_weights(m, w); return m

def mlp_proba(m, X):
    try:    return m.predict_proba(X)[:,1]
    except: return np.full(len(X), 0.5)

CLASSES = np.array([0, 1])

def local_fit(lm, Xtr, ytr, gw=None, mu=0.0):
    """Continue training lm from its CURRENT weights for LOCAL_EPOCHS.
    partial_fit never reinitializes when coefs_ already exist, so the
    global weights injected by clone_mlp are the true starting point.
    If mu > 0 and gw is given, apply the FedProx proximal pull after each
    epoch (epoch granularity approximation of Eq. 3 in the paper):
    the per step proximal gradient is lr*mu*(w - gw); accumulated over
    one epoch of S steps that is approximately S*lr*mu*(w - gw).
    NOTE: exact Eq. 3 requires a per-optimizer-step proximal term, which
    sklearn does not expose; this epoch-granularity pull is a documented
    approximation."""
    steps_per_epoch = max(1, int(np.ceil(len(Xtr) / 128)))
    pull = min(LR * mu * steps_per_epoch, 0.5)
    for _ in range(LOCAL_EPOCHS):
        lm.partial_fit(Xtr, ytr, classes=CLASSES)
        if mu > 0.0 and gw is not None:
            w = get_weights(lm)
            set_weights(lm, [wi - pull * (wi - gi) for wi, gi in zip(w, gw)])
    return lm

# ================================================================
# FL ALGORITHMS
# ================================================================
def fedavg_agg(gw, wlist, sizes):
    total = sum(sizes)
    return [sum((s/total)*wl[i] for wl,s in zip(wlist,sizes))
            for i in range(len(gw))]

def run_fedavg(banks, gw, tmpl):
    wl=[]; sz=[]
    for b in banks:
        Xtr,ytr=get_mlp_data(b)
        if len(Xtr) == 0:
            continue
        lm=clone_mlp(tmpl,gw); local_fit(lm,Xtr,ytr)
        wl.append(get_weights(lm)); sz.append(len(Xtr)); del lm; flush()
    return fedavg_agg(gw,wl,sz) if wl else gw

def run_fedprox(banks, gw, tmpl):
    wl=[]; sz=[]
    for b in banks:
        Xtr,ytr=get_mlp_data(b)
        if len(Xtr) == 0:
            continue
        lm=clone_mlp(tmpl,gw); local_fit(lm,Xtr,ytr,gw=gw,mu=FEDPROX_MU)
        wl.append(get_weights(lm)); sz.append(len(Xtr)); del lm; flush()
    return fedavg_agg(gw,wl,sz) if wl else gw

def run_fednova(banks, gw, tmpl):
    dl=[]; sz=[]; st=[]
    for b in banks:
        Xtr,ytr=get_mlp_data(b)
        if len(Xtr) == 0:
            continue
        lm=clone_mlp(tmpl,gw); local_fit(lm,Xtr,ytr)
        lw=get_weights(lm); delta=[l-g for l,g in zip(lw,gw)]
        tau = LOCAL_EPOCHS * max(1, int(np.ceil(len(Xtr) / 128)))
        dl.append(delta); sz.append(len(Xtr)); st.append(tau); del lm; flush()
    if not dl:
        return gw
    total=sum(sz); tau_g=sum(t*s/total for t,s in zip(st,sz))
    return [gw[i]+tau_g*sum((s/total)*(d[i]/max(t,1))
            for d,s,t in zip(dl,sz,st)) for i in range(len(gw))]

def run_persfl(banks, gw, tmpl, bw):
    BLEND=0.5; wl=[]; sz=[]
    for b in banks:
        bid=b['id']; Xtr,ytr=get_mlp_data(b)
        if len(Xtr) == 0:
            continue
        lm=clone_mlp(tmpl,bw.get(bid,gw)); local_fit(lm,Xtr,ytr)
        lw=get_weights(lm)
        bw[bid]=[BLEND*l+(1-BLEND)*g for l,g in zip(lw,gw)]
        wl.append(lw); sz.append(len(Xtr)); del lm; flush()
    return (fedavg_agg(gw,wl,sz) if wl else gw), bw

def run_fl(banks, strategy, n_feat):
    """FL training for a single strategy. Evaluates ALL banks at rounds 5/10/15/20."""
    print(f"  [{strategy.upper()}] {FL_ROUNDS} rounds", end='  ', flush=True)
    init_X = np.vstack([b['X_train'][:50] for b in banks])
    init_y = np.concatenate([b['y_train'][:50] for b in banks])
    tmpl = init_mlp(init_X, init_y)
    gw = get_weights(tmpl); bw = {}; hist = []

    for rnd in range(1, FL_ROUNDS+1):
        if strategy=='fedavg':   gw=run_fedavg(banks,gw,tmpl)
        elif strategy=='fedprox': gw=run_fedprox(banks,gw,tmpl)
        elif strategy=='fednova': gw=run_fednova(banks,gw,tmpl)
        elif strategy=='persfl':  gw,bw=run_persfl(banks,gw,tmpl,bw)

        if rnd%5==0 or rnd==FL_ROUNDS:
            print(f"{rnd}", end=' ', flush=True)
            gm=clone_mlp(tmpl,gw)
            for b in banks:
                yt=b['y_test']
                em=clone_mlp(tmpl,bw[b['id']]) if strategy=='persfl' and b['id'] in bw else gm
                probs=mlp_proba(em,b['X_test'])
                preds=(probs>=THRESH_DEFAULT).astype(int)
                # 0-fraud banks get F1=0, AUPRC=0 (not skipped)
                f1_val = f1_score(yt,preds,zero_division=0) if yt.sum()>0 else 0.
                auprc_val = average_precision_score(yt,probs) if yt.sum()>0 else 0.
                hist.append(dict(round=rnd, bank_id=b['id'], bank_name=b['name'],
                    strategy=strategy, source=b['source'],
                    f1=f1_val, auprc=auprc_val,
                    n_test_fraud=int(yt.sum())))
        flush()
    print()
    return clone_mlp(tmpl,gw), bw, hist


# ------------------------ Local Experts + MoE Gates + Metrics ------------------------

# ================================================================
# THRESHOLD TUNING UTIL
# For each model, tune the threshold on validation (maximize F2).
# F2 weighs recall higher — appropriate for fraud detection.
# Falls back to default if val has no positives.
# ================================================================
def tune_threshold(y_val, probs_val, default=THRESH_DEFAULT):
    if not TUNE_THRESHOLD or y_val.sum() == 0:
        return default
    best_t, best_f2 = default, -1.0
    for t in THRESH_GRID:
        preds = (probs_val >= t).astype(int)
        if preds.sum() == 0:
            continue
        f2 = fbeta_score(y_val, preds, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2, best_t = f2, float(t)
    return best_t

# ================================================================
# LOCAL EXPERTS
# ================================================================
class XGBExpert:
    name='xgb'
    def __init__(self,pw):
        self.m=xgb.XGBClassifier(n_estimators=200,max_depth=6,
            scale_pos_weight=pw,eval_metric='aucpr',
            subsample=0.8,colsample_bytree=0.8,device='cuda',
            random_state=CURRENT_SEED,verbosity=0); self.trained=False
    def fit(self,X,y): self.m.fit(X,y); self.trained=True
    def predict_proba(self,X):
        return self.m.predict_proba(X)[:,1] if self.trained else np.full(len(X),0.5)

class LGBMExpert:
    name='lgbm'
    def __init__(self,pw):
        self.m=lgb.LGBMClassifier(n_estimators=200,max_depth=6,
            scale_pos_weight=pw,subsample=0.8,colsample_bytree=0.8,
            device='cpu',random_state=CURRENT_SEED,verbose=-1); self.trained=False
    def fit(self,X,y): self.m.fit(X,y); self.trained=True
    def predict_proba(self,X):
        return self.m.predict_proba(X)[:,1] if self.trained else np.full(len(X),0.5)

class CatBoostExpert:
    name='catboost'
    def __init__(self,pw):
        # Try GPU, fall back to CPU
        try:
            self.m = CatBoostClassifier(iterations=200, depth=6,
                scale_pos_weight=pw, task_type='GPU', devices='0',
                eval_metric='PRAUC', verbose=0, random_seed=CURRENT_SEED,
                subsample=0.8, bagging_temperature=0.5,
                bootstrap_type='Bernoulli')
        except Exception:
            self.m = CatBoostClassifier(iterations=200, depth=6,
                scale_pos_weight=pw, eval_metric='PRAUC', verbose=0,
                random_seed=CURRENT_SEED)
        self.trained=False
    def fit(self,X,y):
        try:
            self.m.fit(X, y)
        except Exception:
            # GPU CatBoost occasionally fails — retry CPU
            self.m = CatBoostClassifier(iterations=200, depth=6,
                scale_pos_weight=float(((y==0).sum()/max((y==1).sum(),1))),
                eval_metric='PRAUC', verbose=0, random_seed=CURRENT_SEED)
            self.m.fit(X, y)
        self.trained=True
    def predict_proba(self,X):
        return self.m.predict_proba(X)[:,1] if self.trained else np.full(len(X),0.5)

EXPERT_CLASSES = [XGBExpert, LGBMExpert]
if HAS_CATBOOST:
    EXPERT_CLASSES.append(CatBoostExpert)
EXPERT_NAMES = [c.name for c in EXPERT_CLASSES]

def train_experts(banks):
    experts={}
    for b in banks:
        bid=b['id']; experts[bid]={}
        Xtr,ytr=safe_smote(b['X_train'],b['y_train'])
        n0=(ytr==0).sum(); n1=max((ytr==1).sum(),1); pw=float(n0/n1)
        # skip bank entirely if no training data at all
        if len(Xtr) == 0 or ytr.sum() == 0:
            for Cls in EXPERT_CLASSES:
                experts[bid][Cls.name] = None
            print(f"  {b['name']:20s} SKIPPED: no fraud in training set")
            continue
        for Cls in EXPERT_CLASSES:
            try:
                m=Cls(pw); m.fit(Xtr,ytr)
                p=m.predict_proba(b['X_test']); pd_=(p>=THRESH_DEFAULT).astype(int)
                if b['y_test'].sum() > 0:
                    f1=f1_score(b['y_test'],pd_,zero_division=0)
                    ap=average_precision_score(b['y_test'],p)
                    print(f"  {b['name']:20s} {m.name.upper():8s}: F1={f1:.4f}  AUPRC={ap:.4f}")
                else:
                    # no fraud in test — report specificity + FPR
                    tn = int(((b['y_test']==0) & (pd_==0)).sum())
                    fp = int(((b['y_test']==0) & (pd_==1)).sum())
                    spec = tn / max(tn + fp, 1)
                    fpr = fp / max(tn + fp, 1)
                    print(f"  {b['name']:20s} {m.name.upper():8s}: (no test fraud) "
                          f"Specificity={spec:.4f} FPR={fpr:.4f} FP={fp}")
                experts[bid][m.name]=m
            except Exception as e:
                print(f"  {b['name']:20s} {Cls.name.upper():8s}: FAILED ({e})")
                experts[bid][Cls.name]=None
    return experts

# ================================================================
# MoE GATES
# ================================================================
class StaticGate:
    name='moe_static'
    def get_weights(self,n,avail,**kw):
        w=np.where(avail,1.,0.); return w/w.sum() if w.sum()>0 else w
    def update(self,*a,**k): pass

class PerformanceGate:
    name='moe_performance'
    def __init__(self): self._w={}
    def update(self,bid,vf1s,avail,**kw):
        s=np.where(avail,np.clip(vf1s,0,None),0.)
        self._w[bid]=s/s.sum() if s.sum()>0 else np.ones(len(s))/len(s)
    def get_weights(self,n,avail,bid=None,**kw):
        return self._w.get(bid,np.ones(n)/n)

class ConfidenceGate:
    """Weights experts by mean prediction confidence on validation."""
    name='moe_confidence'
    def __init__(self): self._w={}
    def update(self,bid,val_probs_list,avail,**kw):
        confs=[]
        for i,p in enumerate(val_probs_list):
            if avail[i] and p is not None and len(p)>0:
                confs.append(float(np.mean(np.abs(p - 0.5) * 2)))
            else:
                confs.append(0.)
        s=np.array(confs)
        self._w[bid]=s/s.sum() if s.sum()>0 else np.ones(len(s))/len(s)
    def get_weights(self,n,avail,bid=None,**kw):
        return self._w.get(bid,np.ones(n)/n)

class TypologyAwareGate:
    """Oracle gate — routes by fraud typology. Upper bound; not deployable."""
    name='moe_typology_aware'
    def __init__(self):
        self._tbl=defaultdict(lambda:defaultdict(dict)); self._fb={}
    def update(self,bid,typ_f1s,avail,val_f1s=None,**kw):
        for ei,(tf,av) in enumerate(zip(typ_f1s,avail)):
            if av: self._tbl[bid][ei]=tf
        if val_f1s is not None:
            s=np.where(avail,np.clip(val_f1s,0,None),0.)
            self._fb[bid]=s/s.sum() if s.sum()>0 else np.ones(len(s))/len(s)
    def get_weights(self,n,avail,bid=None,txn_typ=None,**kw):
        if txn_typ is None or bid not in self._tbl \
           or txn_typ in ('Unknown', 'IBM_Normal', 'CC_Normal'):
            return self._fb.get(bid,np.ones(n)/n)
        w=np.zeros(n)
        for ei in range(n):
            if avail[ei] and ei in self._tbl[bid]:
                w[ei]=self._tbl[bid][ei].get(txn_typ,0.)
        return w/w.sum() if w.sum()>0 else self._fb.get(bid,np.ones(n)/n)

def _typ_f1(y_val, probs, typ_val, thresh):
    preds=(probs>=thresh).astype(int); out={}
    for t in np.unique(typ_val):
        if t in ('Unknown', 'IBM_Normal', 'CC_Normal'):
            continue
        mask = typ_val == t
        if mask.sum()<2 or y_val[mask].sum()==0: continue
        out[t]=f1_score(y_val[mask],preds[mask],zero_division=0)
    return out

# ================================================================
# METRICS
# ================================================================
TYP_W={
    'Cycle':2.5,'Bipartite':2.5,'Stacked Bipartite':2.5,
    'Structuring':2.5,'Smurfing':2.0,'Scatter-Gather':2.0,
    'Gather-Scatter':2.0,'Fan_Out':2.0,'Fan_In':2.0,
    'Layered_Fan_Out':2.0,'Layered_Fan_In':2.0,
    'Deposit-Send':2.0,'Over-Invoicing':2.0,
    'IBM_Wire':2.5,'IBM_ACH':2.0,'IBM_Bitcoin':2.5,
    'IBM_Cheque':1.5,'IBM_Cash':1.5,'IBM_Credit Card':2.0,
    'IBM_Reinvestment':2.0,
    'CC_Fraud':2.0,
}

def compute_metrics(y_true, probs, typ=None, thresh=THRESH_DEFAULT):
    """
    Returns a metrics dict. For banks with 0 fraud in test, fraud-specific metrics
    (f1, precision, recall, auprc, mcc, f2, typ_*) are 0 — but specificity / FPR
    are still computed on the negatives. The bank is NOT skipped.
    """
    preds = (probs >= thresh).astype(int)
    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())

    # Negative-side metrics (defined regardless of fraud presence)
    tn = int(((y_true == 0) & (preds == 0)).sum())
    fp = int(((y_true == 0) & (preds == 1)).sum())
    specificity = tn / max(tn + fp, 1)
    fpr         = fp / max(tn + fp, 1)

    if n_pos == 0:
        # No fraud in test — fraud-specific metrics are zero / undefined
        apk = {f'ap_at_{k}': 0. for k in AP_KS}
        return {
            'f1':0., 'precision':0., 'recall':0., 'auprc':0.,
            'mcc':0., 'f2':0., **apk,
            'typ_coverage':0., 'typ_wf1':0.,
            'specificity':float(specificity), 'fpr':float(fpr),
            'false_positives':fp, 'n_test_fraud':0,
            'threshold':thresh,
        }

    # Fraud-aware metrics (standard path)
    f1   = f1_score(y_true, preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    rec  = recall_score(y_true, preds, zero_division=0)
    try:    auprc = average_precision_score(y_true, probs)
    except: auprc = float(y_true.mean())
    mcc = matthews_corrcoef(y_true, preds) if preds.sum() > 0 else 0.
    b2 = 4; d = b2*prec + rec
    f2 = (1 + b2) * prec * rec / d if d > 0 else 0.

    sidx = np.argsort(probs)[::-1]
    apk = {f'ap_at_{k}': float(y_true[sidx[:min(k, len(y_true))]].sum() /
                                max(min(k, len(y_true)), 1)) for k in AP_KS}

    typ_cov = 0.; typ_wf1 = 0.
    if typ is not None:
        laund = [t for t in np.unique(typ) if t not in ('Unknown','IBM_Normal','CC_Normal')]
        if laund:
            typ_cov = sum(1 for t in laund
                if ((typ == t) & (y_true == 1) & (preds == 1)).sum() > 0) / len(laund)
        ws = wt = 0.
        for t in np.unique(typ):
            if t in ('Unknown','IBM_Normal','CC_Normal'):
                continue
            mask = typ == t
            if mask.sum() < 2 or y_true[mask].sum() == 0: continue
            f = f1_score(y_true[mask], preds[mask], zero_division=0)
            w = TYP_W.get(t, 1.5)
            ws += w * f; wt += w
        typ_wf1 = ws / max(wt, 1e-8)

    return {
        'f1':f1, 'precision':prec, 'recall':rec, 'auprc':auprc,
        'mcc':mcc, 'f2':f2, **apk,
        'typ_coverage':typ_cov, 'typ_wf1':typ_wf1,
        'specificity':float(specificity), 'fpr':float(fpr),
        'false_positives':fp, 'n_test_fraud':n_pos,
        'threshold':thresh,
    }

def _em():
    # Default-empty metrics — kept for backward compat but no longer used for 0-fraud banks
    return {k:0. for k in ['f1','precision','recall','auprc','mcc','f2',
                            'ap_at_50','ap_at_100','ap_at_200',
                            'typ_coverage','typ_wf1',
                            'specificity','fpr','false_positives','n_test_fraud',
                            'threshold']}

def agg(ml):
    if not ml: return {}
    return {k:round(float(np.mean([m[k] for m in ml])),4) for k in ml[0]}

def fairness(f1s,lf=None):
    r=dict(client_equity=round(float(np.std(f1s)),4),
           worst_bank_f1=round(float(min(f1s)),4),
           best_bank_f1=round(float(max(f1s)),4))
    if lf and len(lf)==len(f1s):
        r['collab_gain']=round(float(np.mean([a-b for a,b in zip(f1s,lf)])),4)
    return r


# ------------------------ Evaluation + Plots ------------------------

# ================================================================
# A1 - gate weight logger (writes one JSON line per (gate, bank, [typology]))
# ================================================================
import json as _json
from datetime import datetime as _dt

def _log_gate_weights(gate_name, dataset, alpha, bid, expert_names,
                      weights, n_samples, n_fraud, typology=None):
    if not globals().get('LOG_GATE_WEIGHTS', False):
        return
    rec = {
        'ts': _dt.utcnow().isoformat(timespec='seconds'),
        'gate': gate_name, 'dataset': dataset, 'alpha': float(alpha),
        'bank_id': int(bid),
        'seed': int(globals().get('CURRENT_SEED', -1)),
        'experts': list(expert_names),
        'weights': [round(float(w), 5) for w in weights],
        'n_samples': int(n_samples),
        'n_fraud': int(n_fraud),
        'typology': str(typology) if typology is not None else None,
    }
    with open(GATE_LOG_PATH, 'a', encoding='utf-8') as _fh:
        _fh.write(_json.dumps(rec) + '\n')

# ================================================================
# EVALUATION
# Every bank is evaluated (not skipped). For 0-fraud test banks, fraud metrics
# are 0 but specificity / FPR / false_positives are still reported in the CSV.
# ================================================================
def evaluate_all(banks, fl_results, experts, dataset='?', alpha=0.0):
    rows = []
    n_fl = len(fl_results)
    n_ml = len(EXPERT_NAMES)
    n_exp = n_fl + n_ml

    # Local-only F1 baseline per bank (for collab_gain). The best local expert is
    # chosen on the VALIDATION fold; only the chosen expert's test F1 is reported.
    # Taking the max of the experts' TEST F1 (previous behaviour) is test-set model
    # selection: it inflates the local baseline and biases collab_gain downward.
    loc_f1 = {}
    for b in banks:
        bid = b['id']
        yt = b['y_test']
        best, best_val = 0., -1.
        for en in EXPERT_NAMES:
            m = experts[bid].get(en)
            if m and m.trained and yt.sum() > 0:
                pv = m.predict_proba(b['X_val'])
                t_tune = tune_threshold(b['y_val'], pv)
                vf1 = f1_score(b['y_val'], (pv >= t_tune).astype(int), zero_division=0)
                if vf1 > best_val:
                    best_val = vf1
                    p = m.predict_proba(b['X_test'])
                    best = f1_score(yt, (p >= t_tune).astype(int), zero_division=0)
        loc_f1[bid] = best

    def _row(strat, mtype, bm, lf=None, n_eval=None, n_with_fraud=None, total_fraud=None):
        a = agg(bm)
        fa = fairness([m['f1'] for m in bm], lf)
        extra = {}
        if n_eval is not None:
            extra['n_eval_banks'] = n_eval
        if n_with_fraud is not None:
            extra['n_banks_with_fraud'] = n_with_fraud
        if total_fraud is not None:
            extra['total_test_fraud'] = total_fraud
        return {'strategy': strat, 'model_type': mtype, **a, **fa, **extra}

    total_fraud = sum(int(b['y_test'].sum()) for b in banks)

    # ── FL models ── (evaluate ALL banks)
    for strat, (fl_model, bw, _) in fl_results.items():
        bm = []; lf = []
        for b in banks:
            yt = b['y_test']; tte = b.get('typ_test')
            if strat == 'persfl' and b['id'] in bw:
                pv = mlp_proba(bw[b['id']], b['X_val'])
                pt = mlp_proba(bw[b['id']], b['X_test'])
            else:
                pv = mlp_proba(fl_model, b['X_val'])
                pt = mlp_proba(fl_model, b['X_test'])
            t_tune = tune_threshold(b['y_val'], pv)
            bm.append(compute_metrics(yt, pt, tte, thresh=t_tune))
            lf.append(loc_f1.get(b['id'], 0))
        n_with_fraud = sum(1 for b in banks if b['y_test'].sum() > 0)
        rows.append(_row(strat, 'fl', bm, lf, len(banks), n_with_fraud, total_fraud))

    # ── Local experts ── (evaluate ALL banks)
    for en in EXPERT_NAMES:
        bm = []
        for b in banks:
            yt = b['y_test']; tte = b.get('typ_test')
            m = experts[b['id']].get(en)
            if m and m.trained:
                pv = m.predict_proba(b['X_val'])
                pt = m.predict_proba(b['X_test'])
                t_tune = tune_threshold(b['y_val'], pv)
            else:
                pt = np.full(len(yt), 0.5)
                t_tune = THRESH_DEFAULT
            bm.append(compute_metrics(yt, pt, tte, thresh=t_tune))
        n_with_fraud = sum(1 for b in banks if b['y_test'].sum() > 0)
        rows.append(_row(en, 'local_expert', bm, None, len(banks), n_with_fraud, total_fraud))

    # ── MoE Gates ── (evaluate ALL banks)
    gates = [StaticGate(), PerformanceGate(), ConfidenceGate(), TypologyAwareGate()]
    for gate in gates:
        bm = []; lf = []
        for b in banks:
            bid = b['id']; yt = b['y_test']
            tte = b.get('typ_test', np.array(['Unknown'] * len(yt)))
            tvl = b.get('typ_val', np.array(['Unknown'] * len(b['y_val'])))

            # Collect val/test predictions per expert (FL + ML experts)
            vps = []; tps = []
            vf1s = []; avl = []; tyf = []

            for strat, (fl_model, bw, _) in fl_results.items():
                if strat == 'persfl' and bid in bw:
                    pv = mlp_proba(bw[bid], b['X_val'])
                    pt_probs = mlp_proba(bw[bid], b['X_test'])
                else:
                    pv = mlp_proba(fl_model, b['X_val'])
                    pt_probs = mlp_proba(fl_model, b['X_test'])
                t_tune = tune_threshold(b['y_val'], pv)
                vps.append(pv); tps.append(pt_probs)
                vf1s.append(fbeta_score(b['y_val'], (pv >= t_tune).astype(int), beta=2, zero_division=0)
                            if b['y_val'].sum() > 0 else 0.)
                tyf.append(_typ_f1(b['y_val'], pv, tvl, t_tune) if b['y_val'].sum() > 0 else {})
                avl.append(True)

            for en in EXPERT_NAMES:
                m = experts[bid].get(en)
                if m and m.trained:
                    pv = m.predict_proba(b['X_val'])
                    pt_probs = m.predict_proba(b['X_test'])
                    t_tune = tune_threshold(b['y_val'], pv)
                    vps.append(pv); tps.append(pt_probs)
                    vf1s.append(fbeta_score(b['y_val'], (pv >= t_tune).astype(int), beta=2, zero_division=0)
                                if b['y_val'].sum() > 0 else 0.)
                    tyf.append(_typ_f1(b['y_val'], pv, tvl, t_tune) if b['y_val'].sum() > 0 else {})
                    avl.append(True)
                else:
                    vps.append(np.full(len(b['y_val']), 0.5))
                    tps.append(np.full(len(yt), 0.5))
                    vf1s.append(0.); tyf.append({})
                    avl.append(False)

            avl = np.array(avl); vf1s = np.array(vf1s)
            # A1: build expert-name list (FL experts first, then ML)
            _expert_names = list(fl_results.keys()) + list(EXPERT_NAMES)

            # Gate-specific update + test combination
            if isinstance(gate, TypologyAwareGate):
                gate.update(bid, tyf, avl, val_f1s=vf1s)
                ap = np.zeros(len(yt))
                for ut in np.unique(tte):
                    mask = tte == ut
                    w = gate.get_weights(n_exp, avl, bid=bid, txn_typ=ut)
                    _log_gate_weights(gate.name, dataset, alpha, bid,
                                      _expert_names, w, int(mask.sum()),
                                      int(yt[mask].sum()), typology=str(ut))
                    for i, pt_probs in enumerate(tps):
                        ap[mask] += w[i] * pt_probs[mask]
            elif isinstance(gate, PerformanceGate):
                gate.update(bid, vf1s, avl)
                w = gate.get_weights(n_exp, avl, bid=bid)
                _log_gate_weights(gate.name, dataset, alpha, bid,
                                  _expert_names, w, len(yt), int(yt.sum()))
                ap = sum(w[i] * tps[i] for i in range(n_exp))
            elif isinstance(gate, ConfidenceGate):
                gate.update(bid, vps, avl)
                w = gate.get_weights(n_exp, avl, bid=bid)
                _log_gate_weights(gate.name, dataset, alpha, bid,
                                  _expert_names, w, len(yt), int(yt.sum()))
                ap = sum(w[i] * tps[i] for i in range(n_exp))
            else:
                w = gate.get_weights(n_exp, avl)
                _log_gate_weights(gate.name, dataset, alpha, bid,
                                  _expert_names, w, len(yt), int(yt.sum()))
                ap = sum(w[i] * tps[i] for i in range(n_exp))

            # Tune threshold for the MoE-combined probability on val
            if isinstance(gate, TypologyAwareGate):
                ap_val = np.zeros(len(b['y_val']))
                for ut in np.unique(tvl):
                    mask = tvl == ut
                    w_val = gate.get_weights(n_exp, avl, bid=bid, txn_typ=ut)
                    for i, pv_probs in enumerate(vps):
                        ap_val[mask] += w_val[i] * pv_probs[mask]
            else:
                if isinstance(gate, PerformanceGate) or isinstance(gate, ConfidenceGate):
                    w_val = gate.get_weights(n_exp, avl, bid=bid)
                else:
                    w_val = gate.get_weights(n_exp, avl)
                ap_val = sum(w_val[i] * vps[i] for i in range(n_exp))

            t_gate = tune_threshold(b['y_val'], ap_val)
            bm.append(compute_metrics(yt, ap, tte, thresh=t_gate))
            lf.append(loc_f1.get(bid, 0))

        n_with_fraud = sum(1 for b in banks if b['y_test'].sum() > 0)
        rows.append(_row(gate.name, 'moe', bm, lf, len(banks), n_with_fraud, total_fraud))
    return rows

# ================================================================
# PLOTS
# ================================================================
COLORS = {
    'fedavg':'#4F46A0','fedprox':'#0F6E56','fednova':'#854F0B','persfl':'#185FA5',
    'xgb':'#A32D2D','lgbm':'#D85A30','catboost':'#F2A154',
    'moe_static':'#AAAAAA','moe_performance':'#1D9E75',
    'moe_confidence':'#E04B8A','moe_typology_aware':'#9333EA',
}

def plot_results(df_bm, fl_hist, tag):
    strats = sorted(df_bm['strategy'].unique())
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'{tag} — Temporal Split, Dirichlet\n'
                 'MLP FL (4) + XGB/LGBM/CatBoost + MoE (4 gates) [leakage-free + tuned thresholds]',
                 fontsize=10, fontweight='bold')

    ax = axes[0]; x = np.arange(len(strats)); w = 0.22
    for mi, (m, lbl, c) in enumerate([('auprc','AUPRC','#2E4057'),
                                       ('f2','F2','#048A81'),
                                       ('mcc','MCC','#54C6EB')]):
        vals = [float(df_bm[df_bm['strategy']==s][m].mean()) for s in strats]
        ax.bar(x+(mi-1)*w, vals, w, label=lbl, color=c, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('moe_','MoE-') for s in strats], fontsize=7, rotation=45, ha='right')
    ax.set_ylim(0,1); ax.legend(fontsize=8, frameon=False)
    ax.set_title('Primary Metrics (avg across ALL banks)', fontsize=9, fontweight='bold')
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', alpha=0.25)

    ax = axes[1]
    for si, s in enumerate(strats):
        c = COLORS.get(s, '#999')
        ap = float(df_bm[df_bm['strategy']==s]['auprc'].mean())
        rc = float(df_bm[df_bm['strategy']==s]['recall'].mean())
        ax.bar(si-0.18, ap, 0.35, color=c, alpha=0.85)
        ax.bar(si+0.18, rc, 0.35, color=c, alpha=0.4, edgecolor=c, linewidth=1.5)
    ax.set_xticks(range(len(strats)))
    ax.set_xticklabels([s.replace('moe_','MoE-') for s in strats], fontsize=7, rotation=45, ha='right')
    ax.set_ylim(0,1); ax.set_title('AUPRC (solid) vs Recall (hollow)', fontsize=9, fontweight='bold')
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', alpha=0.25)

    ax = axes[2]
    dfh = pd.DataFrame(fl_hist)
    if not dfh.empty:
        for strat in ['fedavg','fedprox','fednova','persfl']:
            sub = dfh[dfh['strategy']==strat]
            if sub.empty: continue
            g = sub.groupby('round')['auprc'].mean().reset_index()
            ax.plot(g['round'], g['auprc'], label=strat.upper(),
                    color=COLORS.get(strat,'#999'), lw=2, marker='o', ms=4)
    ax.set_xlabel('FL Round'); ax.set_ylim(0,1)
    ax.set_title('FL Learning Curves (avg AUPRC)', fontsize=9, fontweight='bold')
    ax.legend(fontsize=8, frameon=False)
    ax.spines[['top','right']].set_visible(False); ax.grid(alpha=0.25)

    plt.tight_layout()
    p = f'{OUT}/{tag.lower()}_benchmark_results.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved: {p}")

def print_summary(df_bm, label):
    print(f"\n{'='*70}")
    print(f"{label} BENCHMARK — Temporal Split, Dirichlet (4 Banks)")
    print("All banks included. Fraud metrics=0 for banks with no test fraud.")
    print("="*70)
    cols = ['strategy','auprc','mcc','f2','f1','recall','specificity','fpr',
            'ap_at_100','typ_coverage','typ_wf1',
            'client_equity','worst_bank_f1','collab_gain',
            'threshold','n_eval_banks','n_banks_with_fraud','total_test_fraud']
    avail = [c for c in cols if c in df_bm.columns]
    print(df_bm[avail].sort_values('auprc', ascending=False).to_string(index=False))

    print(f"\nBEST (AUPRC):")
    best = df_bm.loc[df_bm['auprc'].idxmax()]
    print(f"  {best['strategy']:25s}  AUPRC={best['auprc']:.4f}  "
          f"MCC={best['mcc']:.4f}  F2={best['f2']:.4f}  "
          f"banks_with_fraud={best.get('n_banks_with_fraud','?')}/{best.get('n_eval_banks','?')}")


# ------------------------ Triage probability capture (7b) ------------------------

# ================================================================
# Cell 7b: Triage probability capture (wraps evaluate_all; no retraining)
# ================================================================
CAPTURE_PROBS = True

def _triage_capture(banks, fl_results, experts, dataset, alpha, seed):
    arrays = {}
    for b in banks:
        bid = b['id']
        for strat, (fl_model, bw, _h) in fl_results.items():
            m = bw[bid] if (strat == 'persfl' and bid in bw) else fl_model
            arrays[f'b{bid}__{strat}__val'] = np.asarray(mlp_proba(m, b['X_val']))
            arrays[f'b{bid}__{strat}__test'] = np.asarray(mlp_proba(m, b['X_test']))
        for en in EXPERT_NAMES:
            m = experts[bid].get(en)
            if m is not None and getattr(m, 'trained', False):
                pv = np.asarray(m.predict_proba(b['X_val']))
                pt = np.asarray(m.predict_proba(b['X_test']))
            else:
                pv = np.full(len(b['y_val']), 0.5)
                pt = np.full(len(b['y_test']), 0.5)
            arrays[f'b{bid}__{en}__val'] = pv
            arrays[f'b{bid}__{en}__test'] = pt
        arrays[f'b{bid}__y_val'] = np.asarray(b['y_val'])
        arrays[f'b{bid}__y_test'] = np.asarray(b['y_test'])
    path = (f"{OUT}/probs_{str(dataset).lower()}"
            f"_alpha{format(float(alpha), 'g')}_seed{int(seed)}.npz")
    with open(path + '.tmp', 'wb') as f:          # atomic: a killed session never
        np.savez_compressed(f, **arrays)          # leaves a half-written capture
    os.replace(path + '.tmp', path)
    print(f"    [triage] captured expert probs -> {os.path.basename(path)}")

if '_evaluate_all_pre_capture' not in globals():   # idempotent under cell re-runs
    _evaluate_all_pre_capture = evaluate_all

def evaluate_all(banks, fl_results, experts, dataset='?', alpha=0.0):
    if CAPTURE_PROBS:
        try:
            _triage_capture(banks, fl_results, experts, dataset, alpha, CURRENT_SEED)
        except Exception as e:
            print(f"    [triage] capture failed (benchmark continues): {e}")
    return _evaluate_all_pre_capture(banks, fl_results, experts,
                                     dataset=dataset, alpha=alpha)


# ------------------------ MAIN: seeds x alphas ------------------------

# ================================================================
# MAIN: multi-seed wrapper (A3) - if SEEDS=[42] runs once like the original.
# ================================================================
import os, random as _random

if LOG_GATE_WEIGHTS and os.path.exists(GATE_LOG_PATH):
    os.remove(GATE_LOG_PATH)
    print(f"Cleared old {GATE_LOG_PATH}")
if LOG_GATE_WEIGHTS and RESUME_DIRS:
    with open(GATE_LOG_PATH, 'a', encoding='utf-8') as _out_log:
        for _d in RESUME_DIRS:
            _prev_log = os.path.join(_d, os.path.basename(GATE_LOG_PATH))
            if os.path.exists(_prev_log):
                with open(_prev_log, 'r', encoding='utf-8') as _in_log:
                    _out_log.write(_in_log.read())
                print(f"Seeded gate log from: {_prev_log}")

_master_results = []
_resume_report()

for _SEED in SEEDS:
    print(f"\n{'#'*70}\n###  SEED = {_SEED}\n{'#'*70}")
    _random.seed(_SEED); np.random.seed(_SEED)
    CURRENT_SEED = _SEED  # propagate to all model constructors
    try:
        import torch as _torch
        _torch.manual_seed(_SEED)
        if _torch.cuda.is_available(): _torch.cuda.manual_seed_all(_SEED)
    except Exception:
        pass

    # ================================================================
    # MAIN: Run each dataset × each alpha through the SAME pipeline
    # 3 datasets × 3 alphas = 9 benchmark combos
    # Each combo: 4 FL + 3 ML experts + 4 MoE gates = 11 methods compared
    # ================================================================
    all_results = {}

    for ds_name, cfg in DATASETS.items():
        print(f"\n{'#'*60}")
        print(f"DATASET: {ds_name}")
        print(f"{'#'*60}")

        # RESUME: skip the expensive load if every alpha for this (dataset, seed) is cached
        if all(_resume_has(f"{ds_name.lower()}_alpha{_a}", _SEED) for _a in ALPHAS):
            for _a in ALPHAS:
                _t = f"{ds_name.lower()}_alpha{_a}"
                all_results[_t] = _resume_load(_t, _SEED)
            print(f"  [RESUME] all alphas cached for {ds_name} seed{_SEED} - skipping load")
            continue

        # ── Load & Preprocess ONCE per dataset (expensive) ──
        if ds_name == 'SAML':
            df = load_saml()
            X, y, typ, src, t_col = preprocess_saml(df)
        elif ds_name == 'IBM':
            df = load_ibm()
            X, y, typ, src, t_col = preprocess_ibm(df)
        elif ds_name == 'ULB':
            df = load_ulb()
            X, y, typ, src, t_col = preprocess_ulb(df)
        del df; flush()
        print(f"Preprocessing done | {elapsed()}")

        # ── Alpha Sweep ──
        for alpha in ALPHAS:
            tag = f"{ds_name.lower()}_alpha{alpha}"

            # RESUME: reuse this cell if a previous session already computed it
            if _resume_has(tag, _SEED):
                all_results[tag] = _resume_load(tag, _SEED)
                print(f"  [RESUME] {tag} seed{_SEED} cached - skipping")
                continue

            # SESSION BUDGET: never start a cell that cannot finish inside the cap
            if not _budget_allows(ds_name):
                _STOPPED_EARLY = True
                print(f"\n[BUDGET] {elapsed()} elapsed -- the next cell "
                      f"(~{_CELL_H.get(ds_name, 1.5):.1f}h) does not fit inside "
                      f"SESSION_BUDGET_H={SESSION_BUDGET_H}h. Stopping cleanly; "
                      "completed cells are on disk and the next session resumes from them.")
                break
            print(f"\n{'='*60}")
            print(f"  {ds_name} | alpha={alpha} | {cfg['n_banks']} banks")
            print(f"{'='*60}")

            # Partition (Dirichlet on fraud, even legit, temporal split inside each bank)
            # Fix E (PR #16): the partition depends only on (seed, dataset, alpha) — never
            # on what ran before it — so per-dataset and per-session runs stay canonical.
            _pseed = _cell_rng_seed(_SEED, ds_name, alpha)
            _random.seed(_pseed); np.random.seed(_pseed)
            banks = partition_dataset(X, y, typ, t_col, cfg['n_banks'], alpha, ds_name)

            # FL Training (all 4 strategies)
            fl_results = {}; all_hist = []
            print(f"\nFL training:")
            for strat in FL_STRATEGIES:
                fl_model, bw, hist = run_fl(banks, strat, X.shape[1])
                if strat == 'persfl':
                    init_X = np.vstack([b['X_train'][:50] for b in banks])
                    init_y = np.concatenate([b['y_train'][:50] for b in banks])
                    tm = init_mlp(init_X, init_y)
                    bw_m = {bid: clone_mlp(tm, w) for bid, w in bw.items()}
                    fl_results[strat] = (fl_model, bw_m, hist)
                else:
                    fl_results[strat] = (fl_model, {}, hist)
                for h in hist: h['alpha'] = alpha
                all_hist.extend(hist)
                flush()

            # Local Experts (XGB + LGBM + CatBoost)
            print(f"\nLocal experts (XGB cuda + LGBM cpu"
                  f"{' + CatBoost gpu' if HAS_CATBOOST else ''})...")
            experts = train_experts(banks)
            flush()

            # Evaluate + MoE Gates (4 gates: Static, Performance, Confidence, TypologyAware)
            print(f"\nEvaluating all models + MoE gates...")
            bm_rows = evaluate_all(banks, fl_results, experts, dataset=ds_name, alpha=alpha)
            for r in bm_rows:
                r['dataset'] = ds_name
                r['alpha'] = alpha

            # Save Outputs
            df_bm = pd.DataFrame(bm_rows)
            df_bm['seed'] = _SEED
            df_bm.to_csv(f'{OUT}/{tag}_benchmark.csv', index=False)
            df_bm.to_csv(f'{OUT}/{tag}_seed{_SEED}_benchmark.csv', index=False)
            pd.DataFrame(all_hist).to_csv(f'{OUT}/{tag}_fl_history.csv', index=False)
            pd.DataFrame(all_hist).to_csv(f'{OUT}/{tag}_seed{_SEED}_fl_history.csv', index=False)

            # Summary + Plot
            print_summary(df_bm, f"{ds_name} (alpha={alpha})")
            plot_results(df_bm, all_hist, f"{ds_name}_alpha{alpha}")

            all_results[tag] = df_bm

            del banks, fl_results, experts, all_hist
            flush()
            print(f"\n{ds_name} alpha={alpha} done | {elapsed()}")

        del X, y, typ, src, t_col
        flush()
        if _STOPPED_EARLY:
            print(f"\n{ds_name} — stopped at the session budget | {elapsed()}")
            break
        print(f"\n{ds_name} — all alphas complete | {elapsed()}")

    # ================================================================
    # CONSOLIDATED RESULTS (all 9 combos)
    # ================================================================
    _n_expected = len(DATASETS) * len(ALPHAS)
    if all_results and len(all_results) < _n_expected:
        print(f"\nSeed {_SEED} incomplete ({len(all_results)}/{_n_expected} cells) -- "
              "per-cell CSVs are saved; the per-seed combined CSV is written on resume.")
    if all_results and len(all_results) == _n_expected:
        combined = pd.concat(all_results.values(), ignore_index=True)
        combined['seed'] = _SEED
        combined.to_csv(f'{OUT}/all_benchmarks_combined_seed{_SEED}.csv', index=False)  # aggregator
        _master_results.append(combined)
        # A2/A4 input: all seeds accumulated so far. Writing only the current seed here
        # made 'combined' silently equal to the LAST completed seed, so every downstream
        # analysis (centralised comparison, cost sweep) was single-seed without saying so.
        pd.concat(_master_results, ignore_index=True).to_csv(
            f'{OUT}/all_benchmarks_combined.csv', index=False)  # A2/A4
        print(f"\n{'='*70}")
        print("CONSOLIDATED — ALL DATASETS × ALL ALPHAS")
        print(f"{'='*70}")
        piv = combined.pivot_table(
            index=['dataset','alpha'],
            columns='strategy',
            values='auprc',
            aggfunc='mean'
        ).round(4)
        print("\nAUPRC pivot (dataset × alpha × strategy):")
        print(piv.to_string())

        # Also a MCC pivot (more robust to imbalance)
        piv_mcc = combined.pivot_table(
            index=['dataset','alpha'],
            columns='strategy',
            values='mcc',
            aggfunc='mean'
        ).round(4)
        print("\nMCC pivot (dataset × alpha × strategy):")
        print(piv_mcc.to_string())

    if _STOPPED_EARLY:
        break

    print(f"\n{'='*60}")
    print("ALL RUNS COMPLETE")
    print(f"{'='*60}")
    print(f"Total runtime: {elapsed()}")
    print(f"\nOutputs:")
    for f in sorted(os.listdir(OUT)):
        if f.endswith(('.csv', '.png')):
            print(f"  {f}")

# merge sibling seed-part outputs attached to this session (no-op otherwise);
# A3 below re-globs the per-seed CSVs, so carried seeds enter the statistics
_aggregate_sibling_outputs()

# ================================================================
# A3 - if multi-seed, write the master concatenated CSV
# ================================================================
if len(SEEDS) > 1 and _master_results:
    _master = pd.concat(_master_results, ignore_index=True)
    _master.to_csv(f'{OUT}/all_benchmarks_multiseed.csv', index=False)
    print(f"\nMulti-seed master CSV: {len(_master)} rows -> all_benchmarks_multiseed.csv")
    _mu = (_master.groupby(['strategy','dataset','alpha'])['auprc']
                  .agg(['mean','std','count']).round(4))
    print("\nPer-cell AUPRC mean/std/n_seeds:")
    print(_mu.to_string())


# ------------------------ A2: Centralised pooled baseline ------------------------

# ================================================================
# A2 - CENTRALISED POOLED BASELINE
# ================================================================
if _STOPPED_EARLY and RUN_CENTRALISED_BASELINE:
    print("Session budget hit -> A2 centralised baseline skipped this session "
          "(it reruns in the final resume session).")
    RUN_CENTRALISED_BASELINE = False
if RUN_CENTRALISED_BASELINE:
    print(f"\n{'='*70}\nA2: CENTRALISED POOLED BASELINE\n{'='*70}")
    _central_rows = []

    def _central_temporal_split(X, y, t_col):
        order = np.argsort(t_col, kind='stable')
        X, y = X[order], y[order]
        n = len(X)
        n_train = int(n * (1 - TEST_FRAC - VAL_FRAC))
        n_val   = int(n * VAL_FRAC)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[:n_train]).astype(np.float32)
        Xvl = sc.transform(X[n_train:n_train+n_val]).astype(np.float32)
        Xte = sc.transform(X[n_train+n_val:]).astype(np.float32)
        ytr = y[:n_train]; yvl = y[n_train:n_train+n_val]; yte = y[n_train+n_val:]
        return Xtr, ytr, Xvl, yvl, Xte, yte

    def _central_metrics(y_true, probs, t):
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel()
        return {
            'auprc': float(average_precision_score(y_true, probs)),
            'f1':    float(f1_score(y_true, preds, zero_division=0)),
            'f2':    float(fbeta_score(y_true, preds, beta=2, zero_division=0)),
            'precision': float(precision_score(y_true, preds, zero_division=0)),
            'recall': float(recall_score(y_true, preds, zero_division=0)),
            'mcc': float(matthews_corrcoef(y_true, preds)) if len(set(preds))>1 else 0.0,
            'specificity': tn/(tn+fp) if (tn+fp) else 0.,
            'fpr': fp/(tn+fp) if (tn+fp) else 0.,
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'n_test_fraud': int(y_true.sum()),
            'threshold': float(t),
        }

    for _ds in list(DATASETS.keys()):
        print(f"\n[{_ds}] loading + pooling ...")
        try:
            if _ds == 'SAML':
                _df = load_saml(); _X, _y, _typ, _src, _t = preprocess_saml(_df)
            elif _ds == 'IBM':
                _df = load_ibm();  _X, _y, _typ, _src, _t = preprocess_ibm(_df)
            else:
                _df = load_ulb();  _X, _y, _typ, _src, _t = preprocess_ulb(_df)
            del _df; flush()
        except Exception as _e:
            print(f"  failed to load {_ds}: {_e}")
            continue

        _Xtr, _ytr, _Xvl, _yvl, _Xte, _yte = _central_temporal_split(_X, _y, _t)
        print(f"  pooled splits: train={_Xtr.shape}, val={_Xvl.shape}, test={_Xte.shape}")
        print(f"  fraud rates: tr={_ytr.mean():.4f} vl={_yvl.mean():.4f} te={_yte.mean():.4f}")

        _spw = float((_ytr==0).sum() / max((_ytr==1).sum(), 1))

        _models = [
            ('xgb_central',     lambda: xgb.XGBClassifier(n_estimators=400, max_depth=6,
                scale_pos_weight=_spw, eval_metric='aucpr', tree_method='hist',
                device='cuda', random_state=CURRENT_SEED, verbosity=0)),
            ('lgbm_central',    lambda: lgb.LGBMClassifier(n_estimators=500, num_leaves=63,
                scale_pos_weight=_spw, learning_rate=0.05, device='cpu',
                random_state=CURRENT_SEED, verbose=-1)),
        ]
        if HAS_CATBOOST:
            _models.append(('catboost_central', lambda: CatBoostClassifier(iterations=500,
                depth=6, scale_pos_weight=_spw, eval_metric='PRAUC',
                task_type='GPU', devices='0',
                verbose=False, random_seed=CURRENT_SEED)))

        for _name, _ctor in _models:
            try:
                _t0 = time.time()
                _m = _ctor(); _m.fit(_Xtr, _ytr)
                _pv = _m.predict_proba(_Xvl)[:,1]
                _pt = _m.predict_proba(_Xte)[:,1]
                _t_tune = tune_threshold(_yvl, _pv)
                _row = _central_metrics(_yte, _pt, _t_tune)
                _row.update({'strategy': _name, 'model_type': 'central_pooled',
                             'dataset': _ds, 'alpha': 'centralised',
                             'train_secs': round(time.time()-_t0, 1)})
                print(f"  {_name:<20s}  AUPRC={_row['auprc']:.4f}  "
                      f"F2={_row['f2']:.3f}  recall={_row['recall']:.3f}  "
                      f"FP={_row['false_positives']:>5d}  ({_row['train_secs']:.1f}s)")
                _central_rows.append(_row)
            except Exception as _e:
                print(f"  {_name}: FAILED - {_e}")
        del _X, _y, _typ, _src, _t, _Xtr, _Xvl, _Xte; flush()

    if _central_rows:
        _cdf = pd.DataFrame(_central_rows)
        _cdf.to_csv(CENTRAL_OUT_CSV, index=False)
        print(f"\nWrote {CENTRAL_OUT_CSV}  ({len(_cdf)} rows)")

        try:
            _local = pd.read_csv(f'{OUT}/all_benchmarks_combined.csv')
            # combined now holds ALL seeds; collapse to seed-means so the
            # best-local lookup is not a max over seeds
            _local = _local.groupby(
                [c for c in ('dataset', 'alpha', 'strategy') if c in _local.columns],
                as_index=False).mean(numeric_only=True)
            print(f"\n{'='*70}\nCENTRALISED vs BEST-LOCAL AUPRC\n{'='*70}")
            for _d in _cdf['dataset'].unique():
                _lb = (_local[(_local['dataset']==_d) &
                              (_local['strategy'].isin(['xgb','lgbm','catboost']))]
                       .groupby('strategy')['auprc'].max())
                _cb = _cdf[_cdf['dataset']==_d].set_index('strategy')['auprc']
                print(f"\n  {_d}:")
                for _mn in ['xgb','lgbm','catboost']:
                    _lo = _lb.get(_mn, float('nan'))
                    _ce = _cb.get(f'{_mn}_central', float('nan'))
                    if not (np.isnan(_lo) or np.isnan(_ce)):
                        _lift = (_ce - _lo)/max(_lo,1e-9)*100
                        _arrow = 'UP  ' if _ce > _lo else 'DOWN'
                        print(f"    {_mn:<10s}  local={_lo:.4f}  central={_ce:.4f}  {_arrow} {_lift:+.1f}%")
        except Exception as _e:
            print(f"comparison skipped: {_e}")
    else:
        print("no centralised rows produced")
else:
    print("RUN_CENTRALISED_BASELINE=False -> skipped")


# ------------------------ A1: Gate-weight diagnostic ------------------------

# ================================================================
# A1 - GATE-WEIGHT DIAGNOSTIC
# ================================================================
def _expert_family(name):
    n = name.lower()
    if any(k in n for k in ('xgb','lgbm','lgb','cat')): return 'ML'
    if any(k in n for k in ('fed','mlp','persfl')):     return 'FL'
    if 'gnn' in n: return 'GNN'
    return 'OTHER'

if LOG_GATE_WEIGHTS and os.path.exists(GATE_LOG_PATH):
    print(f"Reading {GATE_LOG_PATH} ...")
    _recs = []
    with open(GATE_LOG_PATH, 'r', encoding='utf-8') as _fh:
        for _ln in _fh:
            _ln = _ln.strip()
            if _ln: _recs.append(_json.loads(_ln))
    print(f"  loaded {len(_recs):,} log records")

    _long = []
    for _r in _recs:
        for _e, _w in zip(_r['experts'], _r['weights']):
            _long.append({'gate': _r['gate'], 'dataset': _r['dataset'],
                          'alpha': _r['alpha'], 'bank_id': _r['bank_id'],
                          'expert': _e, 'weight': _w,
                          'typology': _r.get('typology'),
                          'n_samples': _r['n_samples'], 'n_fraud': _r['n_fraud']})
    _gdf = pd.DataFrame(_long)
    print(f"  exploded to {len(_gdf):,} (record, expert) rows")

    _means = _gdf.groupby(['gate','dataset','alpha','expert'])['weight'].mean().reset_index()
    _gates = sorted(_means['gate'].unique())
    _dsets = sorted(_means['dataset'].unique())
    fig, axes = plt.subplots(len(_gates), len(_dsets),
                              figsize=(4.5*len(_dsets), 3.0*len(_gates)),
                              squeeze=False)
    for _i, _g in enumerate(_gates):
        for _j, _d in enumerate(_dsets):
            _ax = axes[_i][_j]
            _sub = _means[(_means['gate']==_g) & (_means['dataset']==_d)]
            if _sub.empty:
                _ax.set_axis_off(); continue
            _piv = _sub.pivot_table(index='expert', columns='alpha',
                                     values='weight', aggfunc='mean').fillna(0)
            _piv.plot.bar(ax=_ax, width=0.8)
            _ax.set_title(f"{_g} - {_d}", fontsize=9)
            _ax.set_ylabel('mean weight'); _ax.set_ylim(0,1)
            _ax.tick_params(axis='x', labelsize=7, rotation=45)
            _ax.legend(title='alpha', fontsize=7)
    plt.tight_layout()
    plt.savefig(f'{OUT}/chart_a1_gate_mean_weights.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote chart_a1_gate_mean_weights.png")

    _ents = []
    for (_gn, _ds, _al, _bid, _typ), _sub in _gdf.groupby(['gate','dataset','alpha','bank_id','typology'], dropna=False):
        _w = _sub['weight'].values
        _s = _w.sum()
        if _s <= 0: continue
        _p = _w/_s; _p = _p[_p>0]
        _h = float(-(_p * np.log2(_p)).sum())
        _ents.append({'gate':_gn,'dataset':_ds,'alpha':_al,'entropy':_h})
    _edf = pd.DataFrame(_ents)
    if not _edf.empty:
        fig, axes = plt.subplots(1, len(_gates), figsize=(4*len(_gates), 3.5), squeeze=False)
        for _j, _g in enumerate(_gates):
            _ax = axes[0][_j]
            for _d in _dsets:
                _v = _edf[(_edf['gate']==_g)&(_edf['dataset']==_d)]['entropy']
                if len(_v)==0: continue
                _ax.hist(_v, bins=20, alpha=0.55, label=_d)
            _ax.set_title(f"{_g}\nrouting entropy (bits)", fontsize=10)
            _ax.set_xlabel('H(weights)'); _ax.set_ylabel('count')
            _ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f'{OUT}/chart_a1_gate_entropy.png', dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f"  wrote chart_a1_gate_entropy.png")

    _typ_sub = _gdf.dropna(subset=['typology'])
    if not _typ_sub.empty:
        _piv = _typ_sub.groupby(['typology','expert'])['weight'].mean().unstack(fill_value=0)
        fig, _ax = plt.subplots(figsize=(max(6, 0.5*len(_piv.columns)+4),
                                          max(4, 0.3*len(_piv.index)+2)))
        _im = _ax.imshow(_piv.values, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        _ax.set_xticks(range(len(_piv.columns)))
        _ax.set_xticklabels(_piv.columns, rotation=45, ha='right', fontsize=8)
        _ax.set_yticks(range(len(_piv.index)))
        _ax.set_yticklabels(_piv.index, fontsize=8)
        _ax.set_xlabel('expert'); _ax.set_ylabel('typology')
        _ax.set_title('Gate weight per (typology, expert) - TypologyAwareGate')
        plt.colorbar(_im, ax=_ax, label='mean weight')
        plt.tight_layout()
        plt.savefig(f'{OUT}/chart_a1_gate_typology_heatmap.png', dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f"  wrote chart_a1_gate_typology_heatmap.png")

    _rows = []
    for (_g, _d, _a), _sub in _gdf.groupby(['gate','dataset','alpha']):
        _expmeans = _sub.groupby('expert')['weight'].mean()
        _top = _expmeans.idxmax(); _topw = float(_expmeans.max())
        _fam = _sub.copy(); _fam['family'] = _fam['expert'].map(_expert_family)
        _famshare = _fam.groupby('family')['weight'].mean().to_dict()
        _rows.append({'gate':_g,'dataset':_d,'alpha':_a,
                      'top_expert':_top,'top_expert_mean_weight':round(_topw,3),
                      'ml_share':round(_famshare.get('ML',0.),3),
                      'fl_share':round(_famshare.get('FL',0.),3),
                      'collapsed_to_one_expert':_topw>0.85})
    _summ = pd.DataFrame(_rows).sort_values(['dataset','alpha','gate'])
    _summ.to_csv(f'{OUT}/a1_gate_summary.csv', index=False)
    print(f"  wrote a1_gate_summary.csv ({len(_summ)} rows)")
    print("\nA1 summary:")
    print(_summ.to_string(index=False))

    _ncoll = int(_summ['collapsed_to_one_expert'].sum())
    print(f"\nVERDICT: {_ncoll}/{len(_summ)} cells have one expert >85% mass.")
    if _ncoll >= 0.5*len(_summ):
        print("  -> MoE largely collapsed; 'matches ML' may just be 'always picks ML'.")
    else:
        print("  -> Healthy routing; MoE is genuinely blending experts.")
else:
    print("LOG_GATE_WEIGHTS=False or no log file -> A1 skipped")


# ------------------------ A4: Cost-sensitive evaluation ------------------------

# ================================================================
# A4 - COST-SENSITIVE EVALUATION
# ================================================================
if RUN_COST_SENSITIVE:
    _COMB = f'{OUT}/all_benchmarks_combined.csv'
    if not os.path.exists(_COMB):
        print(f"missing {_COMB} -> A4 skipped")
    else:
        _RATIOS = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]
        _COST_FP = 1.0
        _bdf = pd.read_csv(_COMB)
        # combined now holds ALL seeds; collapse to seed-means so cost sums and
        # argmin/argmax are per-combo statistics, not multiplied across seeds
        _bdf = _bdf.groupby(
            [c for c in ('dataset', 'alpha', 'strategy') if c in _bdf.columns],
            as_index=False).mean(numeric_only=True)
        _bdf['strategy'] = _bdf['strategy'].str.lower()

        def _derive(row):
            _nf = float(row.get('n_test_fraud', 0) or 0)
            if _nf == 0 and row.get('total_test_fraud'):
                _nf = float(row['total_test_fraud']) / float(row.get('n_eval_banks', 1) or 1)
            _r = float(row['recall'])
            return _r*_nf, (1-_r)*_nf, float(row['false_positives'])

        _rows = []
        for _, _r in _bdf.iterrows():
            _tp, _fn, _fp = _derive(_r)
            for _ratio in _RATIOS:
                _loss = _fn*_ratio*_COST_FP + _fp*_COST_FP
                _rows.append({'strategy':_r['strategy'],'dataset':_r['dataset'],
                              'alpha':_r['alpha'],'fn_fp_ratio':_ratio,
                              'TP':round(_tp,1),'FN':round(_fn,1),'FP':int(_fp),
                              'expected_loss':round(_loss,1),'auprc':_r['auprc']})
        _cost = pd.DataFrame(_rows)
        _cost.to_csv(COST_OUT_CSV, index=False)
        print(f"wrote {COST_OUT_CSV}  ({len(_cost)} rows)")

        _flips = []
        for (_d, _a), _sub in _cost.groupby(['dataset','alpha']):
            _ab = _sub.loc[_sub['auprc'].idxmax(), 'strategy']
            for _ratio, _rsub in _sub.groupby('fn_fp_ratio'):
                _cb = _rsub.loc[_rsub['expected_loss'].idxmin(), 'strategy']
                _abl = _rsub[_rsub['strategy']==_ab]['expected_loss'].iloc[0]
                _cbl = _rsub['expected_loss'].min()
                _flips.append({'dataset':_d,'alpha':_a,'fn_fp_ratio':_ratio,
                               'auprc_best':_ab,'cost_best':_cb,
                               'flip':_ab!=_cb,
                               'money_left':round(_abl-_cbl,2)})
        _fdf = pd.DataFrame(_flips)
        _fdf.to_csv(COST_FLIPS_CSV, index=False)
        _nf = int(_fdf['flip'].sum())
        print(f"wrote {COST_FLIPS_CSV}  -- {_nf}/{len(_fdf)} cells flip")

        _dsets = sorted(_cost['dataset'].unique())
        fig, axes = plt.subplots(1, len(_dsets), figsize=(6*len(_dsets), 4.5), squeeze=False)
        for _j, _d in enumerate(_dsets):
            _ax = axes[0][_j]
            _avg = (_cost[_cost['dataset']==_d]
                    .groupby(['strategy','fn_fp_ratio'])['expected_loss']
                    .mean().reset_index())
            _top = (_avg[_avg['fn_fp_ratio']==100].nsmallest(6,'expected_loss')['strategy'].tolist())
            for _m in _top:
                _msub = _avg[_avg['strategy']==_m].sort_values('fn_fp_ratio')
                _ax.plot(_msub['fn_fp_ratio'], _msub['expected_loss'],
                         marker='o', label=_m, lw=1.5)
            _ax.set_xscale('log'); _ax.set_yscale('log')
            _ax.set_xlabel('FN/FP cost ratio')
            _ax.set_ylabel('expected loss (log scale)')
            _ax.set_title(f'{_d} - cost curves (top 6)')
            _ax.legend(fontsize=7); _ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{OUT}/chart_a4_cost_curves.png', dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f"wrote chart_a4_cost_curves.png")

        _fr = _nf/len(_fdf) if len(_fdf) else 0
        print(f"\nVERDICT: {_fr:.0%} of cells flip method when judged by cost.")
        if _fr > 0.3:
            print("  AUPRC ranking is misleading at realistic FN/FP ratios.")
            print("  Report cost-sensitive results in your paper alongside AUPRC.")
        elif _fr > 0:
            print("  Some flips - report both AUPRC and cost-sensitive ranking.")
        else:
            print("  No flips - AUPRC ranking matches cost-optimal.")

        if _nf:
            print("\nExample flips:")
            for _, _r in _fdf[_fdf['flip']].head(10).iterrows():
                print(f"  {_r['dataset']:<5s} a={_r['alpha']:<5} ratio={_r['fn_fp_ratio']:>4d}  "
                      f"auprc-best={_r['auprc_best']:<22s} cost-best={_r['cost_best']:<22s}  "
                      f"saves {_r['money_left']:.1f} cost-units")
else:
    print("RUN_COST_SENSITIVE=False -> A4 skipped")


# ------------------------ A3: Multi-seed statistics ------------------------

try:
    # ================================================================
    # A3 - LOAD ALL SEED CSVs (current session + any previous sessions)
    # ================================================================
    import glob, pandas as _pd_loader, os as _os_loader

    # Collect per-seed CSVs: current session output + any mounted dataset
    _seed_csvs = []

    # 1. Current session
    for _f in sorted(glob.glob(f'{OUT}/all_benchmarks_combined_seed*.csv')):
        _seed_csvs.append(_f)

    # 2. Previous sessions mounted as Kaggle dataset (PREV_SEEDS_DIR in config)
    if PREV_SEEDS_DIR and _os_loader.path.exists(PREV_SEEDS_DIR):
        prev_files = sorted(glob.glob(f'{PREV_SEEDS_DIR}/all_benchmarks_combined_seed*.csv'))
        _seed_csvs.extend(prev_files)
        print(f"Mounted dataset at {PREV_SEEDS_DIR}: {len(prev_files)} seed CSVs found")

    if _seed_csvs:
        _dfs = []
        seen_seeds = set()
        for _f in _seed_csvs:
            _seed_val = int(_f.split('_seed')[-1].replace('.csv', ''))
            if _seed_val in seen_seeds:
                continue
            seen_seeds.add(_seed_val)
            _df = _pd_loader.read_csv(_f)
            _df['seed'] = _seed_val
            _dfs.append(_df)
            print(f"  Loaded {_os_loader.path.basename(_f)}  ({len(_df)} rows, seed={_seed_val})")
        _master_all = _pd_loader.concat(_dfs, ignore_index=True)
        _master_all.to_csv(f'{OUT}/all_benchmarks_multiseed.csv', index=False)
        n_seeds_loaded = _master_all['seed'].nunique()
        print(f"Master CSV: {len(_master_all)} rows, {n_seeds_loaded} seeds -> all_benchmarks_multiseed.csv")
    else:
        print("No per-seed CSVs found - using existing combined CSV if present")

    # ================================================================
    # A3 - MULTI-SEED STATISTICAL ANALYSIS (runs fully on Kaggle)
    # ================================================================
    import os, itertools
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy import stats

    OUT_STAT  = f'{OUT}/a3_statistics_report.txt'
    OUT_SUM   = f'{OUT}/a3_statistics_summary.csv'
    OUT_WIL   = f'{OUT}/a3_wilcoxon.csv'
    OUT_CHART = f'{OUT}/chart_a3_mean_std.png'

    # ── Load data ────────────────────────────────────────────────────────────────
    ms_path = f'{OUT}/all_benchmarks_multiseed.csv'
    ss_path = f'{OUT}/all_benchmarks_combined.csv'

    if os.path.exists(ms_path):
        df_a3 = pd.read_csv(ms_path)
        print(f"Loaded multi-seed CSV: {len(df_a3)} rows, {df_a3['seed'].nunique()} seeds")
    elif os.path.exists(ss_path):
        df_a3 = pd.read_csv(ss_path)
        if 'seed' not in df_a3.columns:
            df_a3['seed'] = 42
        print("WARNING: multi-seed CSV not found - falling back to single seed. Stats will be low-power.")
    else:
        raise FileNotFoundError("No benchmark CSV found. Run the main sweep first.")

    N_SEEDS_A3  = df_a3['seed'].nunique()
    METHODS_A3  = sorted(df_a3['strategy'].unique())
    DS_LIST_A3  = sorted(df_a3['dataset'].unique())
    ALPHA_LIST_A3 = sorted(df_a3['alpha'].unique())
    CONDITIONS_A3 = [(ds, a) for ds in DS_LIST_A3 for a in ALPHA_LIST_A3]

    ML_S   = {'xgb', 'lgbm', 'catboost'}
    MOE_S  = {'moe_static', 'moe_performance', 'moe_confidence', 'moe_typology_aware'}
    FL_S   = {'fedavg', 'fedprox', 'fednova', 'persfl'}
    MLAB   = {
        'fedavg':'FedAvg','fedprox':'FedProx','fednova':'FedNova','persfl':'PersFL',
        'xgb':'XGBoost','lgbm':'LightGBM','catboost':'CatBoost',
        'moe_static':'MoE-Static','moe_performance':'MoE-Perf',
        'moe_confidence':'MoE-Conf','moe_typology_aware':'MoE-TypAware',
    }

    lines = []
    def pr(s=''):
        print(s)
        lines.append(str(s))

    pr('=' * 78)
    pr('A3 - Multi-Seed Statistical Tests')
    pr(f"Seeds: {sorted(df_a3['seed'].unique())}   n_conditions={len(CONDITIONS_A3)}   n_seeds={N_SEEDS_A3}")
    pr(f"Effective paired blocks for Wilcoxon: {len(CONDITIONS_A3) * N_SEEDS_A3}")
    pr('=' * 78)

    # ── Helper: build paired vectors across all seed x condition ─────────────────
    def paired_vectors(m1, m2, metric='auprc'):
        v1, v2 = [], []
        for ds, a in CONDITIONS_A3:
            for s in df_a3['seed'].unique():
                mask1 = (df_a3.strategy==m1)&(df_a3.dataset==ds)&(df_a3.alpha.round(3)==round(a,3))&(df_a3.seed==s)
                mask2 = (df_a3.strategy==m2)&(df_a3.dataset==ds)&(df_a3.alpha.round(3)==round(a,3))&(df_a3.seed==s)
                r1 = df_a3[mask1]; r2 = df_a3[mask2]
                if len(r1)==1 and len(r2)==1:
                    v1.append(float(r1[metric].values[0]))
                    v2.append(float(r2[metric].values[0]))
        return np.array(v1), np.array(v2)

    def wilcoxon_safe(x, y):
        diff = x - y
        if len(diff) < 5 or np.all(diff == 0): return float('nan'), 1.0
        try:
            w, p = stats.wilcoxon(x, y, zero_method='wilcox')
            return float(w), float(p)
        except Exception:
            return float('nan'), float('nan')

    def cliff_delta(x, y):
        n1, n2 = len(x), len(y)
        if n1 == 0 or n2 == 0: return 0.0
        cnt = sum(1 if xi > yj else (-1 if xi < yj else 0) for xi in x for yj in y)
        return cnt / (n1 * n2)

    def effect_label(d):
        d = abs(d)
        if d >= 0.474: return 'large'
        if d >= 0.330: return 'medium'
        if d >= 0.147: return 'small'
        return 'negligible'

    def family_vecs(fset, best=False, include_oracle=True):
        methods = [m for m in METHODS_A3 if m in fset
                   and (include_oracle or m != 'moe_typology_aware')]
        out = []
        for ds, a in CONDITIONS_A3:
            for s in df_a3['seed'].unique():
                vals = []
                for m in methods:
                    r = df_a3[(df_a3.strategy==m)&(df_a3.dataset==ds)&
                              (df_a3.alpha.round(3)==round(a,3))&(df_a3.seed==s)]
                    if len(r)==1: vals.append(float(r['auprc'].values[0]))
                if vals: out.append(max(vals) if best else np.mean(vals))
        return np.array(out)

    # ── 1. Per-cell summary ───────────────────────────────────────────────────────
    pr(); pr('-'*78); pr('1) Per-cell AUPRC mean +/- std across seeds'); pr('-'*78)
    summ_a3 = (df_a3.groupby(['strategy','dataset','alpha'])['auprc']
                 .agg(['mean','std','count']).reset_index()
                 .rename(columns={'mean':'auprc_mean','std':'auprc_std','count':'n_seeds'}))
    summ_a3['auprc_std'] = summ_a3['auprc_std'].fillna(0)
    summ_a3.to_csv(OUT_SUM, index=False)
    pr(f"Wrote {os.path.basename(OUT_SUM)}  ({len(summ_a3)} rows)")
    piv = summ_a3.pivot_table(index=['dataset','alpha'], columns='strategy',
                              values='auprc_mean').round(4)
    pr("\nAUPRC mean pivot (dataset x alpha x strategy):")
    pr(piv.to_string())

    piv_std = summ_a3.pivot_table(index=['dataset','alpha'], columns='strategy',
                                   values='auprc_std').round(4)
    pr("\nAUPRC std pivot:")
    pr(piv_std.to_string())

    # ── 2. Friedman rank test ─────────────────────────────────────────────────────
    pr(); pr('-'*78); pr('2) Friedman rank test'); pr('-'*78)

    def friedman_block(methods_list, label):
        blocks = []
        for ds, a in CONDITIONS_A3:
            for s in df_a3['seed'].unique():
                row = []
                for m in methods_list:
                    r = df_a3[(df_a3.strategy==m)&(df_a3.dataset==ds)&
                              (df_a3.alpha.round(3)==round(a,3))&(df_a3.seed==s)]
                    row.append(float(r['auprc'].values[0]) if len(r)==1 else np.nan)
                if not any(np.isnan(row)):
                    blocks.append(row)
        blocks = np.array(blocks)
        n_blocks, k = blocks.shape
        stat, p = stats.friedmanchisquare(*[blocks[:,i] for i in range(k)])
        ranks = np.mean(np.apply_along_axis(lambda r: stats.rankdata(-r), 1, blocks), axis=0)
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        pr(f"  [{label}]  n_blocks={n_blocks}  k={k}")
        pr(f"    chi2={stat:.4f}  df={k-1}  p={p:.6f}  {sig}")
        pr("    Average ranks (1=best):")
        for m, r in sorted(zip(methods_list, ranks), key=lambda x: x[1]):
            pr(f"      {MLAB.get(m,m):<22s}  {r:.3f}")

    friedman_block(METHODS_A3, 'All 11 methods')
    friedman_block([m for m in METHODS_A3 if m != 'moe_typology_aware'], '10 methods (no oracle)')

    # ── 3. Wilcoxon: family comparisons ──────────────────────────────────────────
    pr(); pr('-'*78); pr('3) Wilcoxon signed-rank -- family comparisons'); pr('-'*78)

    wil_rows = []
    comparisons = [
        ('MoE_mean(no_oracle) vs FL_mean',  family_vecs(MOE_S, False, False), family_vecs(FL_S)),
        ('MoE_mean(no_oracle) vs ML_mean',  family_vecs(MOE_S, False, False), family_vecs(ML_S)),
        ('MoE_mean(all) vs FL_mean',        family_vecs(MOE_S, False, True),  family_vecs(FL_S)),
        ('MoE_mean(all) vs ML_mean',        family_vecs(MOE_S, False, True),  family_vecs(ML_S)),
        ('MoE_best(no_oracle) vs FL_best',  family_vecs(MOE_S, True,  False), family_vecs(FL_S, True)),
        ('MoE_best(no_oracle) vs ML_best',  family_vecs(MOE_S, True,  False), family_vecs(ML_S, True)),
    ]
    for label, x, y in comparisons:
        if len(x) == 0 or len(x) != len(y): continue
        w, p = wilcoxon_safe(x, y)
        cd = cliff_delta(x, y)
        wins = int((x>y).sum()); losses = int((x<y).sum())
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        pr(f"  [{label}]  n={len(x)}  W={w}  p={p:.4f} {sig}  delta={cd:+.2f} ({effect_label(cd)})  +/-={wins}/{losses}")
        wil_rows.append({'test':'family','comparison':label,'n':len(x),'W':w,'p':p,
                         'cliff_delta':round(cd,4),'effect':effect_label(cd)})

    # ── 4. Wilcoxon: each method vs FedAvg ───────────────────────────────────────
    pr(); pr('-'*78); pr('4) Wilcoxon signed-rank -- each method vs FedAvg'); pr('-'*78)

    for m in sorted(METHODS_A3):
        if m == 'fedavg': continue
        x, y = paired_vectors(m, 'fedavg')
        if len(x) < 5: continue
        w, p = wilcoxon_safe(x, y)
        cd = cliff_delta(x, y)
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        pr(f"  [{MLAB.get(m,m):<22s} vs FedAvg]  n={len(x)}  W={w}  p={p:.4f} {sig}  delta={cd:+.2f} ({effect_label(cd)})")
        wil_rows.append({'test':'vs_fedavg','comparison':f'{m} vs fedavg','n':len(x),
                         'W':w,'p':p,'cliff_delta':round(cd,4),'effect':effect_label(cd)})

    pd.DataFrame(wil_rows).to_csv(OUT_WIL, index=False)
    pr(f"\nWrote {os.path.basename(OUT_WIL)}")

    # ── 5. Kruskal-Wallis ─────────────────────────────────────────────────────────
    pr(); pr('-'*78); pr('5) Kruskal-Wallis -- FL vs ML vs MoE'); pr('-'*78)

    for include_oracle, lbl in [(False,'excl. oracle'), (True,'incl. oracle')]:
        groups = {}
        for fname, fset in [('FL',FL_S),('ML',ML_S),('MoE',MOE_S)]:
            ms = [m for m in METHODS_A3 if m in fset and (include_oracle or m!='moe_typology_aware')]
            groups[fname] = df_a3[df_a3.strategy.isin(ms)]['auprc'].values
        h, p = stats.kruskal(*groups.values())
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        pr(f"  [{lbl}]  H={h:.4f}  df=2  p={p:.4f} {sig}")

    # ── 6. Spearman: AUPRC vs alpha ───────────────────────────────────────────────
    pr(); pr('-'*78); pr('6) Spearman: AUPRC vs alpha per dataset'); pr('-'*78)

    for ds in DS_LIST_A3:
        sub = df_a3[df_a3.dataset == ds]
        rho, p = stats.spearmanr(sub['alpha'], sub['auprc'])
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        pr(f"  {ds:<6s}  rho={rho:+.3f}  p={p:.2e} {sig}  (n={len(sub)})")

    # ── 7. Bootstrap 95% CIs using seed variance ─────────────────────────────────
    pr(); pr('-'*78); pr('7) Bootstrap 95% CIs using seed variance (true CIs)'); pr('-'*78)

    rng_a3 = np.random.default_rng(0)
    N_BOOT = 10_000
    for ds in DS_LIST_A3:
        sub = df_a3[df_a3.dataset == ds]
        means = sub[sub.strategy != 'moe_typology_aware'].groupby('strategy')['auprc'].mean()
        best_m = means.idxmax()
        vals = sub[sub.strategy == best_m]['auprc'].values
        boot = rng_a3.choice(vals, size=(N_BOOT, len(vals)), replace=True).mean(axis=1)
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        pr(f"  {ds:<6s}  best={MLAB.get(best_m,best_m):<18s}  mean={vals.mean():.4f}"
           f"  95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]  (n_obs={len(vals)})")

    # ── 8. Cohen's d / Hedges' g vs FedAvg ───────────────────────────────────────
    pr(); pr('-'*78); pr("8) Cohen's d / Hedges' g vs FedAvg"); pr('-'*78)

    for m in sorted(METHODS_A3):
        if m == 'fedavg': continue
        x, y = paired_vectors(m, 'fedavg')
        if len(x) < 2: continue
        diff = x - y
        n = len(diff)
        d = diff.mean() / (diff.std(ddof=1) + 1e-12)
        g = d * (1 - 3 / (4*(n-1) - 1))
        mag = 'large' if abs(d)>=0.8 else 'medium' if abs(d)>=0.5 else 'small' if abs(d)>=0.2 else 'negligible'
        pr(f"  {MLAB.get(m,m):<22s}  d={d:+.3f}  g={g:+.3f}  ({mag})")

    # ── 9. Mean +/- std bar chart ─────────────────────────────────────────────────
    pr(); pr('-'*78); pr('9) Mean +/- std AUPRC chart'); pr('-'*78)

    COLORS_A3 = {'fl':'#3b82f6', 'ml':'#ef4444', 'moe':'#8b5cf6'}
    order_a3 = ['fedavg','fedprox','fednova','persfl','xgb','lgbm','catboost',
                'moe_static','moe_performance','moe_confidence','moe_typology_aware']

    fig, axes = plt.subplots(1, len(DS_LIST_A3), figsize=(16, 5))
    if len(DS_LIST_A3) == 1: axes = [axes]

    for ax, ds in zip(axes, DS_LIST_A3):
        sub = (summ_a3[summ_a3.dataset == ds]
               .groupby('strategy')
               .agg(auprc_mean=('auprc_mean','mean'), auprc_std=('auprc_std','mean'))
               .reset_index())
        sub = (sub.set_index('strategy')
                  .reindex([m for m in order_a3 if m in sub.strategy.values])
                  .reset_index())
        cols = [COLORS_A3['moe'] if m in MOE_S else COLORS_A3['ml'] if m in ML_S else COLORS_A3['fl']
                for m in sub.strategy]
        ax.bar(range(len(sub)), sub.auprc_mean, yerr=sub.auprc_std,
               color=cols, alpha=0.82, capsize=4, error_kw={'linewidth':1.2})
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([MLAB.get(m,m) for m in sub.strategy],
                           rotation=45, ha='right', fontsize=7)
        ax.set_title(f'{ds}', fontweight='bold')
        ax.set_ylabel('AUPRC (mean +/- std)')
        ax.set_ylim(0, min(1.0, sub.auprc_mean.max() * 1.35 + 0.01))
        ax.spines[['top','right']].set_visible(False)

    legend_patches = [Patch(color=COLORS_A3['fl'],label='FL'),
                      Patch(color=COLORS_A3['ml'],label='ML'),
                      Patch(color=COLORS_A3['moe'],label='MoE')]
    fig.legend(handles=legend_patches, loc='upper right', fontsize=8)
    fig.suptitle(f'AUPRC Mean +/- Std across {N_SEEDS_A3} Seeds', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_CHART, dpi=150, bbox_inches='tight')
    plt.close()
    pr(f"Chart saved: {os.path.basename(OUT_CHART)}")

    # ── Write report ──────────────────────────────────────────────────────────────
    pr(); pr('='*78); pr('FILES WRITTEN'); pr('='*78)
    for f_ in [OUT_STAT, OUT_SUM, OUT_WIL, OUT_CHART]:
        if os.path.exists(f_):
            pr(f"  {os.path.basename(f_)}  ({os.path.getsize(f_):,} bytes)")

    with open(OUT_STAT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print(f"Written: {os.path.basename(OUT_STAT)}")
except FileNotFoundError as _a3_err:
    print(f"A3 skipped: {_a3_err}")


# ------------------------ Output listing ------------------------

# ================================================================
# Final summary of all output files
# ================================================================
print(f"\n{'='*70}\nOUTPUTS IN {OUT}\n{'='*70}")
for _f in sorted(os.listdir(OUT)):
    if _f.endswith(('.csv','.png','.jsonl')):
        _sz = os.path.getsize(os.path.join(OUT,_f))
        print(f"  {_sz:>10,} B   {_f}")
print(f"\nTotal runtime: {elapsed()}")


# ------------------------ Triage decision layer ------------------------

# ================================================================
# Cell 13: Triage decision layer over the captured probabilities
# ================================================================
import sys
_here = os.path.dirname(os.path.abspath(globals().get('__file__', '.')))
for _cand in (os.path.join(_here, '..', '..'), '../../..', '..', '.',
              '/kaggle/input/federated-learning-for-cross-bank-fraud-detection'):
    if os.path.isdir(os.path.join(_cand, 'triage')):
        sys.path.insert(0, os.path.abspath(_cand))
        break
try:
    from triage.integration import run_from_npz
    _tri = run_from_npz(OUT, f'{OUT}/triage_results_all.csv')
    print(_tri[_tri.rho == 10].groupby(['dataset', 'alpha'])[
        ['expected_cost_per_txn', 'miss_rate_policy', 'coverage_valid_frac',
         'defer_frac', 'auto_clear_at_recall80']].mean().round(4).to_string())
except ImportError:
    print('triage package not importable here - download probs_*.npz and run:')
    print('  python -m triage.integration <folder> --out triage_results.csv')
except FileNotFoundError as e:
    print(f'no capture files found ({e}) - was CAPTURE_PROBS enabled in Cell 7b?')


# ================================================================
# Session verdict + resume instructions
# ================================================================
print(f"\n{'='*70}")
if _STOPPED_EARLY:
    _todo_left = [(s, d, a) for s in SEEDS for d in DATASETS for a in ALPHAS
                  if not os.path.exists(os.path.join(OUT, f"{d.lower()}_alpha{a}_seed{s}_benchmark.csv"))]
    print("PARTIAL RUN — stopped at the session budget.")
    print(f"Remaining cells ({len(_todo_left)}):")
    for _s, _d, _a in _todo_left:
        print(f"  seed={_s:<3} {_d:<5} alpha={_a}")
    print("\nTo finish: Save Version, then attach this session's OUTPUT as a Dataset")
    print("input of a new session and run this same script again, unchanged.")
    print("(RESUME_DIR='auto' finds the mount and skips every finished cell.)")
else:
    print(f"COMPLETE — all {len(SEEDS)} seed(s) finished inside the session budget.")
print(f"Total: {elapsed()}")
