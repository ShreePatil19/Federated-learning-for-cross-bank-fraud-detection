# ULB-only 5-seed preview run (2026-07-27)

Complete Kaggle output of the **ULB-only variant** of the canonical GROUP-A notebook,
plus the triage decision-layer results computed from the captured probabilities.

## Provenance

- **Code**: `notebooks/MOE_experiments/seed_runs/moe-fl-per-dataset-alpha-sweep-GROUP-A.ipynb`
  at commit `e353272` (PR #17 head — includes the merged warm-start fix from PR #14 and
  PR #17's three evaluation fixes + probability capture), with exactly three edits:
  1. Cell 3 `DATASETS`: `SAML` and `IBM` entries commented out (ULB only).
  2. Cell 19 (A2 centralised baseline): hard-coded `['SAML', 'IBM', 'ULB']` loop changed
     to `for _ds in list(DATASETS)` so the baseline follows the config.
  3. Title cell: provenance note added.
- **Run**: Kaggle, GPU, 2026-07-27. `SEEDS = [42, 0, 1, 2, 3]`, `ALPHAS = [0.05, 0.1, 0.5]`,
  4 banks, full ULB (`mlg-ulb/creditcardfraud`, 284,807 rows / 492 frauds). All 15
  (seed, alpha) combos completed in a single session.
- **`triage_results_all.csv`**: generated locally from the `probs_ulb_*.npz` captures via
  `python -m triage.integration <this folder> --out triage_results_all.csv` (triage package
  at the same commit).

## ⚠️ Not canonical — do not merge into paper seed tables

Datasets consume the seed's global RNG stream **sequentially** (SAML → IBM → ULB) in the
canonical notebook; running ULB alone therefore draws different bank partitions than the
full 3-dataset run under the same seed labels. Runs of the ULB-only variant are
reproducible among themselves. Use this folder for triage validation, method development,
and qualitative preview; the paper's canonical tables require the full 3-dataset rerun.

## Headline numbers

- FL strategies now **differentiate** (ULB α=0.5 AUPRC: FedNova .560, FedProx .558,
  FedAvg .554, PersFL .550) and global-model AUPRC rises over rounds — the warm-start
  fix is active. The pre-fix runs produced byte-identical FL rows (e.g. 0.699±0.116 × 3).
- Collaboration gain is heterogeneity-dependent: mean collab_gain +0.04..+0.09 at
  α ≤ 0.1, ≈ 0 at α = 0.5.
- Friedman over 15 blocks: p = 1.6e-5 (best avg rank: MoE-Perf). Wilcoxon families:
  MoE vs local ML experts *** (15/0 wins); MoE vs FL backbone n.s. — the flat horse-race
  that motivates the triage layer.
- Triage (rho = c_fn/c_fp): auto-clears **99.89%** of transactions at recall-80,
  defers ~4.5% to human, policy miss rate ~9.2%, expected cost stable across rho 1–50.
  Caveat: conformal coverage valid for only 35–57% of banks — ULB's per-bank calibration
  fraud counts are tiny, so many banks fall back to threshold mode (frame ULB as the
  small-data case; SAML/IBM carry the triage headline).
- Known issue: `lgbm_central` in `a2_centralised_results.csv` is degenerate
  (AUPRC 0.015, near-constant alarms) — exclude or debug before citing A2.
- Seed std (±0.14–0.23) exceeds strategy gaps; compare strategies via rank tests only.

## File inventory

| Pattern | Contents |
|---|---|
| `ulb_alpha{a}_seed{s}_benchmark.csv` | per-seed benchmark (11 strategies × metrics) |
| `ulb_alpha{a}_benchmark.csv` / `_results.png` | last-seed benchmark + chart |
| `ulb_alpha{a}_seed{s}_fl_history.csv` | per-round global/local AUPRC & F1 |
| `all_benchmarks_combined*.csv`, `all_benchmarks_multiseed.csv` | pooled results |
| `a1_*` / `chart_a1_*` | gate-weight logging summary + charts |
| `a2_centralised_results.csv` | pooled-data upper bound (xgb .764 / catboost .753) |
| `a3_*` | multi-seed statistics (Friedman, Wilcoxon, report) |
| `a4_*` | cost-sensitive ranking analysis |
| `probs_ulb_alpha{a}_seed{s}.npz` | captured expert/FL probabilities (triage input). **Not in git** (`*.npz` is git-ignored): keep them from the Kaggle output zip of this run, or regenerate by re-running the notebook variant described above |
| `gate_weights_log.jsonl` | raw A1 gate-weight records |
| `triage_results_all.csv` | triage layer output (90 rows: 3α × 5 seeds × 6 rho) |
