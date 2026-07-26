# Triage decision layer: auto-clear / flag / defer-to-human

Inference-time decision layer over the benchmark's trained experts. Instead of ranking
models by AUPRC, it turns the per-bank expert probabilities into an **action** — auto-clear,
flag-for-SAR, or defer to a human investigator — chosen by calibrated risk under asymmetric
costs, with a per-bank conformal guarantee on the fraud miss-rate. No expert is ever
retrained: everything is predict-and-decide.

```
raw expert probs p_e  --calibrate-->  p~_e  --gate weights-->  s(x)   calibrated risk
s(x) + conformal fold --conformal-->  t     miss-rate guarantee, computed locally per bank
cost matrix           --Bayes-->      cut = 1/(1+rho)
tau = min(cut, t)     --defer rule--> action in {clear, flag, defer}
```

## Quick start

Standalone (compact experts, no GPU, ~1 min smoke test):

```bash
python -m triage.experiment --dataset synthetic --quick
python -m triage.experiment --dataset ulb --seeds 42 0 1 2 3        # needs creditcard.csv
python -m triage.experiment --dataset ulb --resume-dir /kaggle/input/prev-session
```

On the paper's real experts: run the GROUP-A sweep notebook — its "Triage probability
capture" cell saves `probs_<dataset>_alpha<a>_seed<s>.npz` per combo (atomic, resumable,
no retraining) — then:

```bash
python -m triage.integration /path/to/npz-folder --out triage_results.csv \
       --defer-rule band --gate brier
```

or use the notebook's final "Triage decision layer" cell, which does the same in place.

## Fold hygiene (why the layer is shaped like this)

Each rule below prevents a failure mode that was **measured**, not hypothesised, in the
prototype this package replaces:

1. **Calibration and conformal never share a fold.** Each bank's validation fold is split
   in half: isotonic calibrators (and gate weights) fit on one half, the conformal
   quantile comes from the other. Fitting both on the same fold broke the guarantee
   anti-conservatively — test miss-rate 0.14–0.24 against a 0.10 target with 7 experts;
   the split restores it. (`tests/test_triage.py::test_double_dip_regression`)
2. **The conformal threshold is part of the policy**: the deployed flag threshold is
   `tau = min(bayes_cut, t)`. Computing `t` but acting on the raw Bayes cut (the prototype
   behaviour) reports a guarantee about a threshold nobody uses; at rho=1 the deployed
   policy cleared 3.5x the promised fraud miss budget.
3. **The defer band and the recall-80 threshold come from the conformal half** of the
   validation fold, then apply to test. Sizing them on the scored batch makes
   "defer <= budget" tautological; taking them from the full validation fold is also
   wrong, because the calibration half is in-sample for the isotonic maps — measured
   effect: nominal recall-0.80 thresholds achieved only ~0.72 on test when derived from
   the full fold, ~0.79 from the conformal half. Realized test rates are reported as
   outcomes (`defer_frac`, `achieved_recall80`).
4. **Scarce-fraud banks fall back, loudly.** Below `MIN_FRAUD_FOR_FIT` validation fraud
   cases (or when `floor(target_fnr * (n1+1)) < 1`) a bank gets `t = None` and uses the
   cross-bank fallback; it never emits a flag-everything `-inf` threshold, and calibrators
   degrade to identity rather than memorising 3 positives as isotonic steps. Rows carry
   `n_banks_fallback_threshold` and `n_banks_split_ok` so degradation is visible.
5. **The cross-bank fallback is a heuristic and says so.** `quantile_of_quantiles` (median
   of per-bank thresholds, conservative `method='lower'`) carries **no** finite-sample
   guarantee — a certified one-shot federated conformal estimator needs the
   jointly-corrected order statistics of Humbert et al. (2023), under cross-bank
   exchangeability that Dirichlet non-IID deliberately breaks. Report its coverage
   empirically per bank (`coverage_valid_frac`); do not cite it as guaranteed.
6. **Test labels never tune anything.** `workload_reduction_at_recall` picks its
   threshold on held-out validation fraud and reports `achieved_recall80` on test (the
   prototype selected it on test fraud — an oracle number). Same for the risk-coverage
   curve, which is cost-weighted (`c_fn`/`c_fp` per item): plain 0/1 error at a
   cost-skewed cut is dominated by false positives the cost model prices as cheap
   (AURC ~0.9 artifacts).
7. **Coverage is audited only where a conformal claim exists.** When a bank has neither
   its own threshold nor a federated fallback, it is excluded from `miss_rate_audit` /
   `coverage_valid_frac` (`n_banks_audited` says how many were audited) instead of being
   audited against the rho-dependent Bayes cut, which would report pseudo-coverage.

## Decision rules (`--defer-rule`)

- `band` — defer inside `|s - tau| < band`, band sized on validation to a per-bank budget.
- `chow` — Chow (1970) rule: defer exactly when the human is the cheapest option,
  `min(c_fn * s, c_fp * (1-s)) > c_defer`. Asymmetric region, uses the defer cost in the
  decision itself.
- `capacity` — one federation-level review budget: pool `|s - tau|` across banks and defer
  the globally most ambiguous fraction. With per-bank budgets every bank defers exactly the
  budget, which makes the Jain fairness index vacuous by construction; a shared queue makes
  "which bank's customers get deferred" a real, reportable quantity.

`--gate brier` weights experts by inverse Brier score of their calibrated probabilities
(computed on the calibration half) — the probability-quality metric the Bayes rule actually
consumes. `--human-accuracy a` models an imperfect investigator: deferred cases resolve
correctly with probability `a`, adding `(1-a) * c_fn` / `(1-a) * c_fp` expected residual
cost per deferred fraud/legit case.

## Output columns

One row per (dataset, alpha, seed, rho): `miss_rate_policy` (fraud the deployed policy
auto-cleared — the number the guarantee is about), `miss_rate_audit` (miss-rate at the
conformal threshold), `coverage_valid_frac`, `expected_cost`, `expected_cost_per_txn`,
`defer_frac`, `aurc` (cost-weighted), `auto_clear_at_recall80` / `fpr_at_recall80` /
`achieved_recall80`, `defer_jain`, `fpr_gap`, plus provenance (`defer_rule`, `gate`,
`human_accuracy`, `n_banks_audited`, `n_banks_fallback_threshold`, `n_banks_split_ok`).

`run_triage` uses every expert present in the capture by default (the capture side is
responsible for excluding oracle experts); pass `expert_names` to restrict the ensemble —
unknown names raise instead of being silently dropped.

## Files

| File | What |
|------|------|
| `core.py` | pure decision-layer functions (calibration, conformal, cost, defer, metrics) |
| `integration.py` | capture/npz IO, `run_triage`, `run_from_npz` CLI |
| `experiment.py` | standalone compact-expert sweep (seeded, resumable, atomic checkpoints) |
| `../tests/test_triage.py` | unit + regression tests for every hygiene rule above |

Excluded on purpose: the `moe_typology_aware` gate — its routing key is derived from the
label on IBM/ULB, so it is a test-time oracle and no deployable triage stack may consume it.
