"""Tests for the FL warm-start fix in the GROUP-A sweep notebook.

Exercises the ACTUAL notebook code: loads the notebook JSON, extracts the
code cells containing the constants block, safe_smote/get_mlp_data, and
make_mlp/FL algorithms (located by anchor, not cell index), stubs the GPU
stack, and execs them into one namespace.

Run against the unpatched notebook (tests 1-3 must FAIL, demonstrating the
bug) by pointing FL_FIX_NOTEBOOK at an unpatched copy:

    FL_FIX_NOTEBOOK=/path/to/unpatched.ipynb pytest tests/test_fl_fix.py

Run without the env var to test the canonical (patched) notebook: all five
tests must pass.
"""
import json
import os
import sys
import types

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_NB = os.path.join(
    REPO, 'notebooks', 'MOE_experiments', 'seed_runs',
    'moe-fl-per-dataset-alpha-sweep-GROUP-A.ipynb')
NB_PATH = os.environ.get('FL_FIX_NOTEBOOK', DEFAULT_NB)

ANCHORS = ['FEDPROX_MU     =', 'def safe_smote', 'def make_mlp']


def _install_stubs():
    sys.modules['cuml'] = types.SimpleNamespace(
        accel=types.SimpleNamespace(install=lambda: None))
    sys.modules['cupy'] = types.SimpleNamespace(
        get_default_memory_pool=lambda: types.SimpleNamespace(
            free_all_blocks=lambda: None))
    for name in ('xgboost', 'lightgbm'):
        sys.modules[name] = types.ModuleType(name)
    mpl = types.ModuleType('matplotlib')
    mpl.use = lambda *a, **k: None
    plt = types.ModuleType('matplotlib.pyplot')
    patches = types.ModuleType('matplotlib.patches')
    patches.Patch = object
    mpl.pyplot = plt
    mpl.patches = patches
    sys.modules['matplotlib'] = mpl
    sys.modules['matplotlib.pyplot'] = plt
    sys.modules['matplotlib.patches'] = patches


def _load_namespace():
    _install_stubs()
    nb = json.load(open(NB_PATH))
    code_cells = [''.join(c['source']) for c in nb['cells']
                  if c['cell_type'] == 'code']
    picked = []
    for anchor in ANCHORS:
        for src in code_cells:
            if anchor in src and src not in picked:
                picked.append(src)
                break
        else:
            raise AssertionError(f'anchor not found in any code cell: {anchor}')
    ns = {}
    for src in picked:
        exec(compile(src, NB_PATH, 'exec'), ns)
    return ns


@pytest.fixture(scope='module')
def ns():
    return _load_namespace()


def make_banks(ns, sizes=(2000, 1400), pos_frac=0.05, n_feat=10, seed=7):
    """Synthetic banks matching the schema the FL functions expect."""
    rng = np.random.default_rng(seed)
    banks = []
    for i, n in enumerate(sizes):
        X = rng.normal(size=(n, n_feat)).astype(np.float32)
        y = (rng.random(n) < pos_frac).astype(np.int64)
        y[:5] = 1  # guarantee some positives
        X[y == 1] += 1.5  # give the classifier signal
        banks.append(dict(
            id=i, name=f'T-Bank{i}', source='test',
            X_train=X, y_train=y,
            X_val=X[:50], y_val=y[:50],
            X_test=X[:100], y_test=y[:100]))
    return banks


def train_local(ns, lm, X, y, **kw):
    """Use the notebook's local training procedure: local_fit if present
    (patched), else plain fit (unpatched)."""
    if 'local_fit' in ns:
        return ns['local_fit'](lm, X, y, **kw)
    lm.fit(X, y)
    return lm


def max_abs_diff(w1, w2):
    return max(float(np.max(np.abs(a - b))) for a, b in zip(w1, w2))


def make_template(ns, banks):
    init_X = np.vstack([b['X_train'][:50] for b in banks])
    init_y = np.concatenate([b['y_train'][:50] for b in banks])
    return ns['init_mlp'](init_X, init_y)


def test_continuation_from_global(ns):
    """Discriminates: local training must continue from the injected global
    weights, not silently reinitialize from random_state."""
    banks = make_banks(ns)
    tmpl = make_template(ns, banks)
    ref = ns['make_mlp']()
    ref.fit(banks[1]['X_train'], banks[1]['y_train'])
    gw = ns['get_weights'](ref)

    D_X, D_y = banks[0]['X_train'], banks[0]['y_train']
    lm1 = ns['clone_mlp'](tmpl, gw)
    train_local(ns, lm1, D_X, D_y)
    w_from_global = ns['get_weights'](lm1)

    lm2 = ns['make_mlp']()
    train_local(ns, lm2, D_X, D_y)
    w_from_scratch = ns['get_weights'](lm2)

    assert max_abs_diff(w_from_global, w_from_scratch) > 1e-6, (
        'training from injected global weights produced the same model as '
        'training from scratch: fit() is discarding the global weights')


def test_rounds_change_global(ns):
    """Discriminates: the global model must evolve across FL rounds instead
    of being frozen at the round-1 average."""
    banks = make_banks(ns)
    tmpl = make_template(ns, banks)
    gw0 = ns['get_weights'](tmpl)

    gw1 = ns['run_fedavg'](banks, gw0, tmpl)
    gw = gw1
    for _ in range(2):
        gw = ns['run_fedavg'](banks, gw, tmpl)
    gw3 = gw

    diff = sum(float(np.linalg.norm(a - b)) for a, b in zip(gw1, gw3))
    assert diff > 1e-6, (
        'global weights identical between round 1 and round 3: '
        'no federation is happening')


def test_strategies_diverge(ns):
    """Discriminates: FedAvg, FedProx and FedNova must be genuinely
    different algorithms, not arithmetic variants of the same fixed models.

    Note: on the unpatched code the post-hoc FedProx pull and FedNova's tau
    arithmetic still produce tiny numeric differences between the globals,
    so 'globals differ' alone does not discriminate. The genuine-federation
    assertion is the last one: local training must actually respond to the
    strategy's global weights (unpatched fit() ignores them, so the local
    models are identical whatever the global is)."""
    banks = make_banks(ns)  # unequal sizes -> fednova must differ
    tmpl = make_template(ns, banks)
    gw0 = ns['get_weights'](tmpl)

    old_mu = ns['FEDPROX_MU']
    ns['FEDPROX_MU'] = 0.1  # amplify the proximal term
    try:
        gw_avg, gw_prox, gw_nova = gw0, gw0, gw0
        for _ in range(3):
            gw_avg = ns['run_fedavg'](banks, gw_avg, tmpl)
            gw_prox = ns['run_fedprox'](banks, gw_prox, tmpl)
            gw_nova = ns['run_fednova'](banks, gw_nova, tmpl)
    finally:
        ns['FEDPROX_MU'] = old_mu

    d_prox = sum(float(np.linalg.norm(a - b)) for a, b in zip(gw_avg, gw_prox))
    d_nova = sum(float(np.linalg.norm(a - b)) for a, b in zip(gw_avg, gw_nova))
    assert d_prox > 1e-6, 'FedProx global identical to FedAvg after 3 rounds'
    assert d_nova > 1e-6, 'FedNova global identical to FedAvg after 3 rounds'

    # Genuine divergence: the same bank trained from the FedAvg global vs
    # the FedProx global must yield different local models.
    D_X, D_y = ns['get_mlp_data'](banks[0])
    lm_a = ns['clone_mlp'](tmpl, gw_avg)
    train_local(ns, lm_a, D_X, D_y)
    lm_p = ns['clone_mlp'](tmpl, gw_prox)
    train_local(ns, lm_p, D_X, D_y)
    assert max_abs_diff(ns['get_weights'](lm_a),
                        ns['get_weights'](lm_p)) > 1e-6, (
        'local training produced identical models from different global '
        'weights: local fit() is ignoring the injected globals, so the '
        'strategy differences are arithmetic artifacts, not federation')


def test_single_class_bank_no_crash(ns):
    """Discriminates: a bank whose local data has a single class (legitimate
    at extreme alpha) must not crash local training."""
    banks = make_banks(ns)
    zero_X = np.random.default_rng(3).normal(size=(300, 10)).astype(np.float32)
    banks.append(dict(
        id=2, name='T-Bank2-legit-only', source='test',
        X_train=zero_X, y_train=np.zeros(300, dtype=np.int64),
        X_val=zero_X[:20], y_val=np.zeros(20, dtype=np.int64),
        X_test=zero_X[:20], y_test=np.zeros(20, dtype=np.int64)))
    tmpl = make_template(ns, banks)
    gw0 = ns['get_weights'](tmpl)
    gw1 = ns['run_fedavg'](banks, gw0, tmpl)  # must not raise
    assert len(gw1) == len(gw0)


def test_subsample_deterministic(ns):
    """Discriminates: the MLP_CAP subsample must be identical on every call
    (per seed and bank), not dependent on global RNG state."""
    banks = make_banks(ns, sizes=(2000,))
    old_cap = ns['MLP_CAP']
    ns['MLP_CAP'] = 500  # force the subsample path
    try:
        X1, y1 = ns['get_mlp_data'](banks[0])
        X2, y2 = ns['get_mlp_data'](banks[0])
    finally:
        ns['MLP_CAP'] = old_cap
    assert np.array_equal(y1, y2) and np.array_equal(X1, X2), (
        'get_mlp_data returned different subsamples on identical calls: '
        'results depend on global RNG state / strategy execution order')
