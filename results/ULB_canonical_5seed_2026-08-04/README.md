# ULB canonical 5-seed run (2026-08-04)

Complete Kaggle output of `scripts/kaggle/sweep_ULB.ipynb` (PR #19 notebook twin,
merged pipeline: PR #17 triage capture + eval fixes, PR #16 Fix E order-independent
partitioning, session budget guard). **CONFIG cell unedited** — `ONLY_DATASET='ULB'`,
`SEEDS=[42,0,1,2,3]`, `ALPHAS=[0.05,0.1,0.5]`, 4 banks, full ULB
(`mlg-ulb/creditcardfraud`, 284,807 rows / 492 frauds).

## Provenance

- **Code**: `scripts/kaggle/sweep_ULB.ipynb` at `main` (PR #19 merge `8c475f2`;
  shared body unchanged at `2a7c571`), imported via Kaggle File → Import Notebook,
  GPU, fresh run (no resume mounts). All 15 (seed, alpha) cells completed in one
  session on 2026-08-04.
- Every partition is derived only from `(seed, dataset, alpha)` (Fix E), so each
  cell is reproducible in isolation by re-running this notebook unchanged.

## ✅ Canonical — supersedes `ULB_only_5seed_2026-07-27` for paper tables

The 2026-07-27 folder was a preview run of the PR #17 sequential notebook with a
hand-edited `DATASETS` list (non-canonical partition draws; see its README warning).
This run is the released-code path. Use THIS folder for the paper's ULB seed tables;
keep the old folder for triage validation history only.

Cross-check vs the preview: pooled means shift by a uniform +0.03..+0.06 (within
seed noise, cell std ±0.09–0.16) and every qualitative conclusion replicates —
same family ranking, same flat MoE-vs-best-FL horse race, same per-bank zeros.
Two independent partition mechanisms, same story: good robustness evidence.

## Headline numbers (pooled 5-seed AUPRC mean ± std)

| strategy | α=0.05 | α=0.10 | α=0.50 |
|---|---|---|---|
| MoE gates (all four, near-identical) | 0.326 ± 0.114 | 0.453 ± 0.093 | 0.612 ± 0.151 |
| FedAvg / FedProx / FedNova | 0.313–0.319 | 0.423–0.449 | 0.604–0.611 |
| PersFL | 0.327 ± 0.116 | 0.436 ± 0.108 | 0.590 ± 0.122 |
| XGB / LGBM / CatBoost (local) | 0.318–0.324 | 0.422–0.438 | 0.565–0.574 |
| Centralised (A2): xgb 0.764 · catboost 0.753 | | | |

- **The old Table II numbers do not reproduce** (FL 0.699 / MoE-Static 0.691 ± 0.122 /
  PersFL 0.696 came from the pre-warm-start-fix pipeline that emitted byte-identical
  FL rows). Canonical α=0.5 FL sits at 0.59–0.61, gates at 0.612.
- **Seed-42 "beats centralised by +0.089" is dead**: moe_static seed 42 α=0.5 is
  0.572 vs catboost_central 0.753 (−0.181). Best single cell anywhere is +0.058 over
  central — seed noise, do not cherry-pick.
- **MoE dual-convention Wilcoxon (15 blocks)**: gates vs FL family MEAN +0.0115
  (10/15, p=0.030) and vs ML family mean +0.0234 (14/15, p=1.2e-4); gates vs the
  per-cell BEST FL backbone −0.0003 (8/15, p=0.85) — the flat horse race that
  motivates the triage layer, now on the canonical path. Gates beat the best local
  tree +0.0163 (14/15, p=0.002): on ULB the ensemble effect is real.
- All four gates are numerically near-identical with near-uniform weights
  (top expert mean weight 0.143–0.180 vs uniform 0.143): report gates as weighted
  ensembles, not routers.
- **No global collapse** (0/60 FL rows with F1 = 0), but per-bank failure is
  universal at low α: 55/55 strategy-cells have `worst_bank_f1 = 0` at α ≤ 0.1
  (36/55 at α = 0.5) — the per-bank story motivating capacity-aware deferral.
- **Known issue (4th dataset-run confirmation)**: `lgbm_central` is degenerate
  again (AUPRC 0.0153 vs xgb 0.764). Exclude from A2 tables until the central-LGBM
  path is fixed.
- Test split is temporal; the ULB test window holds 52 frauds — per-bank calibration
  counts are tiny, so conformal triage on ULB stays the small-data case (headline
  triage numbers belong to SAML/IBM).

## File inventory

Same layout as the other sweep packages: per-cell
`ulb_alpha{a}_seed{s}_benchmark.csv` / `_fl_history.csv`, per-alpha last-seed
benchmark + chart, `all_benchmarks_combined*.csv`, `all_benchmarks_multiseed.csv`,
A1 gate summary + charts + `gate_weights_log.jsonl`, A2 centralised, A3 5-seed
statistics (native pooled run — single session), A4 cost analysis.

`probs_ulb_alpha{a}_seed{s}.npz` (15 files, triage input) are **not in git**
(`*.npz` ignored): they live in the Kaggle output zip of this run
(`results.zip`, 2026-08-04). Run the decision layer locally via
`python -m triage.integration <folder-with-npz> --out triage_results.csv`.
