# Revised training — changes log (2026-07)

Clean re-run package for the cross-bank fraud MoE-FL benchmark. Start training from
the notebook in this folder, not the old GROUP-A copies.

## Files

- `moe-fl-alpha-sweep-GROUP-A_FIXED.ipynb` — the corrected notebook (Kaggle, GPU).

## What was wrong

The FL numbers looked "fake" because FedAvg / FedProx / FedNova were byte-identical on
ULB (every metric, every seed) and nearly identical elsewhere. Root cause, in Cell 9:

```python
def make_mlp():
    return MLPClassifier(..., warm_start=False, ...)   # BUG
...
lm = clone_mlp(tmpl, gw)   # set local weights = global model gw
lm.fit(Xtr, ytr)           # warm_start=False -> fit() ERASES gw, reinits from the seed
```

With `warm_start=False`, sklearn's `fit()` discards the global weights and reinitialises
from `random_state`. So the global model was never the starting point for local training.
Consequences:

1. No federation across rounds — each bank retrained from the same fixed init every round.
2. FedAvg / FedProx / FedNova reduced to trivial arithmetic on fixed local weights and
   converged to the same fixed point.
3. On ULB the four banks are equally sized, so FedNova normalisation reduces exactly to
   FedAvg and FedProx (mu=0.01) converges to the same point -> identical numbers.

## The fix

1. `make_mlp()`: `warm_start=False` -> `warm_start=True`. Now `lm.fit()` continues from the
   global weights `gw`, so local training actually refines the shared model and the FL
   algorithms diverge meaningfully.
2. `SEEDS = [42, 0, 1, 2, 3]` (was `[0]`) so the full 5-seed sweep runs.

Nothing else in the pipeline was changed. The tree experts (XGB/LGBM/CatBoost) and the MoE
gates were not touched.

## How to run (Kaggle, GPU)

1. Upload `moe-fl-alpha-sweep-GROUP-A_FIXED.ipynb` to Kaggle.
2. Attach datasets:
   - ULB: `creditcardfraud` (creditcard.csv)
   - SAML: `berkanoztas/synthetic-transaction-monitoring-dataset-aml` (SAML-D.csv)
   - IBM: `HI-Small_Trans.csv`
3. Accelerator: GPU (uses cuml.accel, XGBoost CUDA, CatBoost GPU).
4. Run all. Full 5-seed sweep is ~5x a single-seed run.

## Sanity checks after the run

- On ULB at alpha=0.5, FedAvg / FedProx / FedNova should be CLOSE but NOT byte-identical.
  If still identical, the fix did not take effect.
- Per-seed AUPRC should vary across seeds (the whole point of a 5-seed sweep).
- Regenerate the paper's per-seed table (Table VI, "ULB AUPRC by seed") from this run's
  CSVs, not by hand — the old table's per-seed columns did not match the data.

## Not fixed here (separate, non-code issues)

- Paper Table VI per-seed values were mismatched (means correct, per-seed columns wrong).
  Regenerate from the new run.
- The old `5 parallel Outputs for group A/` folder was an un-seeded duplicate. Ignore it;
  use only fresh output from this notebook.
- Known limitations already in the paper: StandardScaler.fit on full data (minor leakage),
  SCAFFOLD 60-round not re-run, centralised CatBoost single-seed only.
