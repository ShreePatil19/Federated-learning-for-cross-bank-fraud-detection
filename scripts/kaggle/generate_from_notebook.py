#!/usr/bin/env python3
"""Generate scripts/kaggle/sweep_{ULB,SAML,IBM}.py from the canonical merged
GROUP-A notebook (PR #17) + PR #16's Fix E / resume machinery + a session
time-budget guard. Every patch is an exact-match string replacement that
asserts it matched exactly once, so any upstream notebook drift fails loudly
instead of producing a silently-wrong script."""
import json
import os
import py_compile
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
NB = os.path.join(REPO, 'notebooks/MOE_experiments/seed_runs/moe-fl-per-dataset-alpha-sweep-GROUP-A.ipynb')
OUTDIR = HERE

nb = json.load(open(NB))
def cell(i):
    return ''.join(nb['cells'][i]['source'])

def patch(src, old, new, count=1, label=''):
    n = src.count(old)
    assert n == count, f"patch {label!r}: expected {count} match(es), found {n}"
    return src.replace(old, new)

# ---------------------------------------------------------------- cell 3: imports+config
c3 = cell(3)
c3 = patch(c3, """import cuml
cuml.accel.install()
print("cuml.accel installed — sklearn MLP on GPU")""",
"""try:
    import cuml
    cuml.accel.install()
    print("cuml.accel installed — sklearn MLP on GPU")
except Exception as _cuml_err:
    if ON_KAGGLE:
        raise SystemExit(
            f"cuml unavailable ({_cuml_err}). Enable the GPU accelerator "
            "(notebook settings -> Accelerator -> GPU): a CPU run of this sweep "
            "does not finish inside one Kaggle session.")
    print(f"WARNING: cuml unavailable ({_cuml_err}) — CPU sklearn MLP (local test only)")""",
label='cuml guard')

c3 = patch(c3, "import cupy as cp",
"""try:
    import cupy as cp
except ImportError:
    cp = None   # CPU-only local runs; Kaggle GPU images ship cupy""",
label='cupy guard')

c3 = patch(c3, """def flush():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()""",
"""def flush():
    gc.collect()
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()""",
label='flush guard')

c3 = patch(c3,
"SEEDS                    = [42, 0, 1, 2, 3]  # A3: canonical multi-seed rerun",
"SEEDS                    = list(SEEDS)       # A3: canonical multi-seed rerun (set in CONFIG at top)",
label='seeds from config')

SPLIT_BLOCK = '''

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

def _resume_has(tag, seed):
    if not RESUME_DIR or not os.path.isdir(RESUME_DIR):
        return False
    return os.path.exists(os.path.join(RESUME_DIR, f"{tag}_seed{seed}_benchmark.csv"))

def _resume_load(tag, seed):
    for suf in ("benchmark", "fl_history"):
        p = os.path.join(RESUME_DIR, f"{tag}_seed{seed}_{suf}.csv")
        if os.path.exists(p):
            _sh_r.copy(p, os.path.join(OUT, f"{tag}_seed{seed}_{suf}.csv"))
            _sh_r.copy(p, os.path.join(OUT, f"{tag}_{suf}.csv"))
    # carry the triage probability capture forward so the final session's
    # triage layer covers cells computed in earlier sessions too
    _npz = f"probs_{tag}_seed{seed}.npz"
    _p = os.path.join(RESUME_DIR, _npz)
    if os.path.exists(_p):
        _sh_r.copy(_p, os.path.join(OUT, _npz))
    elif globals().get('CAPTURE_PROBS', True):
        print(f"    [triage] NOTE: {_npz} not in RESUME_DIR — that cell has no capture")
    return pd.read_csv(os.path.join(OUT, f"{tag}_seed{seed}_benchmark.csv"))

def _autodetect_resume():
    """Find a previous session's output mounted under INPUT_DIR.
    A mount qualifies if it holds this dataset's per-cell benchmark CSVs.
    Mounts whose root has a triage/ dir are the REPO checkout, not a results
    mount — excluded so the pre-Fix-E preview results committed under
    results/ can never be resumed from."""
    if not os.path.isdir(INPUT_DIR):
        return None
    _pat = tuple(f"{d.lower()}_alpha" for d in DATASETS)
    hits = {}
    for root, _dirs, files in os.walk(INPUT_DIR):
        rel = os.path.relpath(root, INPUT_DIR)
        mount = os.path.join(INPUT_DIR, rel.split(os.sep)[0]) if rel != '.' else INPUT_DIR
        if os.path.isdir(os.path.join(mount, 'triage')):
            continue
        n = sum(1 for f in files
                if f.startswith(_pat) and '_seed' in f and f.endswith('_benchmark.csv'))
        if n:
            hits[root] = n
    if not hits:
        return None
    best = max(hits, key=hits.get)
    if len(hits) > 1:
        print(f"NOTE: several mounts look resumable ({sorted(hits)}) — using {best}")
    return best

if RESUME_DIR == 'auto':
    RESUME_DIR = _autodetect_resume()
print(f"RESUME_DIR = {RESUME_DIR or 'None (fresh run)'}")

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
'''
c3 = c3.rstrip() + '\n' + SPLIT_BLOCK

# ---------------------------------------------------------------- cell 5: loaders/finders
c5 = cell(5)
c5 = patch(c5, "nrows=DATASETS['SAML']['rows']",
           "nrows=DATASETS.get('SAML', {'rows': None})['rows']", label='saml rows cap')
c5 = patch(c5, "os.walk('/kaggle/input')", "os.walk(INPUT_DIR)", count=3, label='finder walks')
c5 = patch(c5, "        '/kaggle/input/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml/SAML-D.csv',\n        '/kaggle/input/synthetic-transaction-monitoring-dataset-aml/SAML-D.csv',",
           "        f'{INPUT_DIR}/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml/SAML-D.csv',\n        f'{INPUT_DIR}/synthetic-transaction-monitoring-dataset-aml/SAML-D.csv',",
           label='saml candidates')
c5 = patch(c5, "        '/kaggle/input/creditcardfraud/creditcard.csv',\n        '/kaggle/input/creditcard-fraud/creditcard.csv',",
           "        f'{INPUT_DIR}/creditcardfraud/creditcard.csv',\n        f'{INPUT_DIR}/creditcard-fraud/creditcard.csv',",
           label='ulb candidates')

# ---------------------------------------------------------------- cell 17: main loop
c17 = cell(17)
c17 = patch(c17, """if LOG_GATE_WEIGHTS and os.path.exists(GATE_LOG_PATH):
    os.remove(GATE_LOG_PATH)
    print(f"Cleared old {GATE_LOG_PATH}")""",
"""if LOG_GATE_WEIGHTS and os.path.exists(GATE_LOG_PATH):
    os.remove(GATE_LOG_PATH)
    print(f"Cleared old {GATE_LOG_PATH}")
if LOG_GATE_WEIGHTS and RESUME_DIR:
    _prev_log = os.path.join(RESUME_DIR, os.path.basename(GATE_LOG_PATH))
    if os.path.exists(_prev_log):
        _sh_r.copy(_prev_log, GATE_LOG_PATH)
        print(f"Seeded gate log from previous session: {_prev_log}")""",
label='gate log carry')

c17 = patch(c17, "_master_results = []",
"""_master_results = []
_resume_report()""", label='resume report call')

c17 = patch(c17, """        print(f"DATASET: {ds_name}")
        print(f"{'#'*60}")

        # ── Load & Preprocess ONCE per dataset (expensive) ──""",
"""        print(f"DATASET: {ds_name}")
        print(f"{'#'*60}")

        # RESUME: skip the expensive load if every alpha for this (dataset, seed) is cached
        if all(_resume_has(f"{ds_name.lower()}_alpha{_a}", _SEED) for _a in ALPHAS):
            for _a in ALPHAS:
                _t = f"{ds_name.lower()}_alpha{_a}"
                all_results[_t] = _resume_load(_t, _SEED)
            print(f"  [RESUME] all alphas cached for {ds_name} seed{_SEED} - skipping load")
            continue

        # ── Load & Preprocess ONCE per dataset (expensive) ──""",
label='seed-level resume skip')

c17 = patch(c17, """        for alpha in ALPHAS:
            tag = f"{ds_name.lower()}_alpha{alpha}"
            print(f"\\n{'='*60}")""",
"""        for alpha in ALPHAS:
            tag = f"{ds_name.lower()}_alpha{alpha}"

            # RESUME: reuse this cell if a previous session already computed it
            if _resume_has(tag, _SEED):
                all_results[tag] = _resume_load(tag, _SEED)
                print(f"  [RESUME] {tag} seed{_SEED} cached - skipping")
                continue

            # SESSION BUDGET: never start a cell that cannot finish inside the cap
            if not _budget_allows(ds_name):
                _STOPPED_EARLY = True
                print(f"\\n[BUDGET] {elapsed()} elapsed -- the next cell "
                      f"(~{_CELL_H.get(ds_name, 1.5):.1f}h) does not fit inside "
                      f"SESSION_BUDGET_H={SESSION_BUDGET_H}h. Stopping cleanly; "
                      "completed cells are on disk and the next session resumes from them.")
                break
            print(f"\\n{'='*60}")""",
label='cell-level resume + budget')

c17 = patch(c17, """            # Partition (Dirichlet on fraud, even legit, temporal split inside each bank)
            banks = partition_dataset(""",
"""            # Partition (Dirichlet on fraud, even legit, temporal split inside each bank)
            # Fix E (PR #16): the partition depends only on (seed, dataset, alpha) — never
            # on what ran before it — so per-dataset and per-session runs stay canonical.
            _pseed = _cell_rng_seed(_SEED, ds_name, alpha)
            _random.seed(_pseed); np.random.seed(_pseed)
            banks = partition_dataset(""",
label='fix E')

c17 = patch(c17, """        del X, y, typ, src, t_col
        flush()
        print(f"\\n{ds_name} — all alphas complete | {elapsed()}")""",
"""        del X, y, typ, src, t_col
        flush()
        if _STOPPED_EARLY:
            print(f"\\n{ds_name} — stopped at the session budget | {elapsed()}")
            break
        print(f"\\n{ds_name} — all alphas complete | {elapsed()}")""",
label='dataset-loop early break')

c17 = patch(c17, """    if all_results:
        combined = pd.concat(all_results.values(), ignore_index=True)""",
"""    _n_expected = len(DATASETS) * len(ALPHAS)
    if all_results and len(all_results) < _n_expected:
        print(f"\\nSeed {_SEED} incomplete ({len(all_results)}/{_n_expected} cells) -- "
              "per-cell CSVs are saved; the per-seed combined CSV is written on resume.")
    if all_results and len(all_results) == _n_expected:
        combined = pd.concat(all_results.values(), ignore_index=True)""",
label='complete-seed consolidation guard')

c17 = patch(c17, """    print(f"\\n{'='*60}")
    print("ALL RUNS COMPLETE")""",
"""    if _STOPPED_EARLY:
        break

    print(f"\\n{'='*60}")
    print("ALL RUNS COMPLETE")""",
label='seed-loop early break')

# ---------------------------------------------------------------- cell 19: A2 baseline
c19 = cell(19)
c19 = patch(c19, "    for _ds in ['SAML', 'IBM', 'ULB']:",
            "    for _ds in list(DATASETS.keys()):", label='A2 dataset list')
c19 = patch(c19, """if RUN_CENTRALISED_BASELINE:
    print(f"\\n{'='*70}\\nA2: CENTRALISED POOLED BASELINE\\n{'='*70}")""",
"""if _STOPPED_EARLY and RUN_CENTRALISED_BASELINE:
    print("Session budget hit -> A2 centralised baseline skipped this session "
          "(it reruns in the final resume session).")
    RUN_CENTRALISED_BASELINE = False
if RUN_CENTRALISED_BASELINE:
    print(f"\\n{'='*70}\\nA2: CENTRALISED POOLED BASELINE\\n{'='*70}")""",
label='A2 early-stop skip')

# ---------------------------------------------------------------- cell 29: triage
c29 = cell(29)
c29 = patch(c29, """for _cand in ('../../..', '..', '.',
              '/kaggle/input/federated-learning-for-cross-bank-fraud-detection'):""",
"""_here = os.path.dirname(os.path.abspath(globals().get('__file__', '.')))
for _cand in (os.path.join(_here, '..', '..'), '../../..', '..', '.',
              '/kaggle/input/federated-learning-for-cross-bank-fraud-detection'):""",
label='triage path candidates')
c29 = patch(c29, "    display(_tri[_tri.rho == 10].groupby(['dataset', 'alpha'])[",
            "    print(_tri[_tri.rho == 10].groupby(['dataset', 'alpha'])[",
            label='display->print')
c29 = patch(c29, "         'defer_frac', 'auto_clear_at_recall80']].mean().round(4))",
            "         'defer_frac', 'auto_clear_at_recall80']].mean().round(4).to_string())",
            label='to_string')

# ---------------------------------------------------------------- cell 25: A3 stats
# In the notebook, hitting cell 25 with no benchmark CSV deserves a hard error.
# In the linear script the sweep always ran first, so "no CSV" only means the
# session budget stopped before any cell finished — degrade to a skip so the
# triage cell and the resume epilogue still run.
import textwrap
c25 = 'try:\n' + textwrap.indent(cell(25).rstrip(), '    ') + '''
except FileNotFoundError as _a3_err:
    print(f"A3 skipped: {_a3_err}")'''

EPILOGUE = '''
# ================================================================
# Session verdict + resume instructions
# ================================================================
print(f"\\n{'='*70}")
if _STOPPED_EARLY:
    _todo_left = [(s, d, a) for s in SEEDS for d in DATASETS for a in ALPHAS
                  if not os.path.exists(os.path.join(OUT, f"{d.lower()}_alpha{a}_seed{s}_benchmark.csv"))]
    print("PARTIAL RUN — stopped at the session budget.")
    print(f"Remaining cells ({len(_todo_left)}):")
    for _s, _d, _a in _todo_left:
        print(f"  seed={_s:<3} {_d:<5} alpha={_a}")
    print("\\nTo finish: Save Version, then attach this session's OUTPUT as a Dataset")
    print("input of a new session and run this same script again, unchanged.")
    print("(RESUME_DIR='auto' finds the mount and skips every finished cell.)")
else:
    print(f"COMPLETE — all {len(SEEDS)} seed(s) finished inside the session budget.")
print(f"Total: {elapsed()}")
'''

PROLOGUE = '''
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
'''

MD_TITLES = {3: 'Imports + Config', 5: 'Data Loading + Preprocessing',
             7: 'Dirichlet Partition + Temporal Split', 9: 'MLP + FL Algorithms',
             11: 'Local Experts + MoE Gates + Metrics', 13: 'Evaluation + Plots',
             15: 'Triage probability capture (7b)', 17: 'MAIN: seeds x alphas',
             19: 'A2: Centralised pooled baseline', 21: 'A1: Gate-weight diagnostic',
             23: 'A4: Cost-sensitive evaluation', 25: 'A3: Multi-seed statistics',
             27: 'Output listing', 29: 'Triage decision layer'}

bodies = {3: c3, 5: c5, 17: c17, 19: c19, 25: c25, 29: c29}
SHARED_MARK = ('# ==== SHARED PIPELINE — byte-identical across sweep_ULB/SAML/IBM.py; '
               'edit all three together (see check_shared_sync.sh) ====')
shared = [SHARED_MARK, PROLOGUE]
for i in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]:
    src = bodies.get(i, cell(i))
    shared.append(f"\n# {'-'*24} {MD_TITLES[i]} {'-'*24}\n")
    shared.append(src.rstrip() + '\n')
shared.append(EPILOGUE)
SHARED_BODY = '\n'.join(shared)

DS_META = {
    'ULB':  dict(hours='~1.2 h per seed -> all 5 seeds fit in ONE session (~6 h)',
                 sessions='1',
                 attach='mlg-ulb/creditcardfraud  (creditcard.csv)'),
    'SAML': dict(hours='~3.2 h per seed -> 5 seeds is ~16 h = TWO sessions',
                 sessions='2 (the budget guard stops after ~3 seeds; rerun to finish)',
                 attach='berkanoztas/synthetic-transaction-monitoring-dataset-aml  (SAML-D.csv)'),
    'IBM':  dict(hours='~4.0 h per seed -> 5 seeds is ~20 h = TWO sessions',
                 sessions='2 (the budget guard stops after ~2 seeds; rerun to finish)',
                 attach='ealtman2019/ibm-transactions-for-anti-money-laundering-aml  (HI-Small_Trans.csv)'),
}

HEADER = '''#!/usr/bin/env python3
# ============================================================================
# GROUP-A alpha sweep on Kaggle — {ds} only (parallel-safe, resumable)
#
# Single-file version of the canonical merged notebook
#   notebooks/MOE_experiments/seed_runs/moe-fl-per-dataset-alpha-sweep-GROUP-A.ipynb
# (PR #17: triage probability capture + eval fixes) restricted to one dataset
# with PR #16's Fix E order-independent partitioning and session resume, plus a
# session time-budget guard sized to Kaggle's 12 h cap.
#
# Run on Kaggle: create a new Script (or one-cell `%run`) kernel, paste this
# file, attach the input dataset below, Accelerator = GPU, then Save & Run All.
#   attach : {attach}
#   cost   : {hours}
#   sessions needed: {sessions}
#
# If a session stops at the budget: attach that session's output as a Dataset
# input of a fresh session and run this same script again, unchanged —
# finished (seed, alpha) cells are detected and skipped (RESUME_DIR='auto').
# Full instructions + sanity checklist: scripts/kaggle/README.md
# ============================================================================

# ─────────── CONFIG — the only per-script block; body below is shared ───────────
ONLY_DATASET     = '{ds}'
SEEDS            = [42, 0, 1, 2, 3]  # canonical rerun seeds; the budget guard
                                     # decides how many fit in this session
SESSION_BUDGET_H = 10.5              # stop starting new cells after this many hours
RESUME_DIR       = 'auto'            # 'auto' | None | '/kaggle/input/<mount>'

'''

os.makedirs(OUTDIR, exist_ok=True)
for ds, meta in DS_META.items():
    path = os.path.join(OUTDIR, f'sweep_{ds}.py')
    with open(path, 'w') as f:
        f.write(HEADER.format(ds=ds, **meta) + SHARED_BODY)
    py_compile.compile(path, doraise=True)
    print(f'wrote + compiled {path}  ({os.path.getsize(path):,} B)')
print('OK')
