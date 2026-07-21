# Kaggle rerun checklist (patched GROUP-A sweep)

Prepared on branch `fix/fl-warm-start`. A human runs this on Kaggle; nothing
here runs locally.

## 1. Notebook

- Upload the **patched** `notebooks/MOE_experiments/seed_runs/moe-fl-per-dataset-alpha-sweep-GROUP-A.ipynb`
  (must include Fixes A-D: `local_fit`/partial_fit warm start + real FedProx,
  FedNova tau from real steps, per-(seed,bank) deterministic subsample,
  `SEEDS = [42, 0, 1, 2, 3]`).
- Confirm in the toggles cell before launch:
  - `SEEDS = [42, 0, 1, 2, 3]`
  - `RUN_CENTRALISED_BASELINE = True` (closes the single-seed-baseline
    limitation (4) of the paper)
- Do NOT run the deprecated `moe-fl-seed*.ipynb` copies (see
  `seed_runs/DEPRECATED.md`).

## 2. Kaggle inputs to attach

| Dataset | Kaggle source | File the notebook reads |
|---|---|---|
| ULB | `creditcardfraud` | `creditcard.csv` |
| SAML | berkanoztas synthetic transaction monitoring (AML) | `SAML-D.csv` |
| IBM | IBM transactions for AML | `HI-Small_Trans.csv` |

GPU accelerator on (cuml path), internet off as usual.

## 3. In-flight sanity signals (check the logs while it runs)

Abort early and investigate if any of these fail:

1. **Rounds evolve.** The round 5/10/15/20 eval metrics must change across
   rounds. Frozen metrics from round 1 onward = the warm-start bug is still
   present in the uploaded copy.
2. **Strategies diverge.** FedAvg and FedProx must differ beyond the third
   decimal on at least SAML or IBM.
3. **Seeds diverge.** Per-seed outputs must differ across the five seeds. Five
   identical "seed" outputs = the unseeded-duplicate failure mode; the
   `scripts/regen_table6.py` integrity check will also refuse such a folder.
4. Single-class banks at alpha=0.05 must train without a `ValueError`
   (handled by `partial_fit(..., classes=CLASSES)`).

## 4. Post-rerun steps

1. Download the output folder (per-seed
   `all_benchmarks_combined_seed{SEED}.csv` files).
2. Regenerate Table 6: `python scripts/regen_table6.py <output_folder>`.
   The script refuses identical-across-seeds folders and prints a
   recomputed-vs-paper diff.
3. Regenerate the FL and MoE rows of Tables II, III, IV from the same CSVs by
   script (extend `regen_table6.py` into `regen_tables.py`; same
   population-std convention, no value typed by hand).
4. Fill in `CLAIMS_DELTA.md`: mark every FL-ordering-dependent paper claim
   KEPT or CHANGED. Changed claims get flagged to Chloe, never silently
   edited in the paper.
