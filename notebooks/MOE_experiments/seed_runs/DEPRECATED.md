# DEPRECATED notebooks in this directory

**Canonical notebook: `moe-fl-per-dataset-alpha-sweep-GROUP-A.ipynb`** with
`SEEDS = [42, 0, 1, 2, 3]` (all five seeds run inside the one notebook).

The per-seed copies below are **superseded** and must not be run or patched:

- `moe-fl-seed0.ipynb`
- `moe-fl-seed1.ipynb`
- `moe-fl-seed2.ipynb`
- `moe-fl-seed3.ipynb`
- `moe-fl-seed42.ipynb`

They are stale copies of the same code and still contain the FL warm-start bug
(`warm_start=False` + `fit()` discards the injected global weights, so no
federation happens). The bug is fixed only in GROUP-A (branch
`fix/fl-warm-start`, Fixes A-D). Any rerun must use the patched GROUP-A
notebook; results produced by these copies must not enter `results/` or the
paper tables.

`moe-fl-a3-aggregator.ipynb` / `a3_multi_seed_runner.py` are aggregation
utilities, not training notebooks; they are unaffected by the warm-start bug
but should only be pointed at outputs of the patched GROUP-A notebook.
