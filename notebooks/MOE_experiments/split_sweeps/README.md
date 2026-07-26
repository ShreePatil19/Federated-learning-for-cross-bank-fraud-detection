# Split sweeps: one dataset per notebook (parallel + resumable)

These are `notebooks/MOE_experiments/seed_runs/moe-fl-per-dataset-alpha-sweep-GROUP-A.ipynb`
(Yike's Fixes A-D, PR #14) restricted to a single dataset each, so the three can run
concurrently on separate Kaggle accounts and survive a session timeout.

| Notebook | Attach | Cost per seed | 5 seeds |
|---|---|---|---|
| `sweep_IBM.ipynb` | `HI-Small_Trans.csv` | ~4.0h | ~20h |
| `sweep_SAML.ipynb` | `SAML-D.csv` | ~3.2h | ~16h |
| `sweep_ULB.ipynb` | `creditcard.csv` | ~1.2h | ~6h |

Timings are measured, not estimated: they come from the `ts` fields in `gate_weights_log.jsonl`
of an actual partial run. A single notebook covering all three datasets is ~42h of compute
against Kaggle's 12h session cap, which is why the earlier full run died partway.

## Why the split needs Fix E

Upstream seeds the global RNG once per seed and then partitions SAML, then IBM, then ULB in
sequence:

```python
for _SEED in SEEDS:
    _random.seed(_SEED); np.random.seed(_SEED)
    for ds_name in DATASETS:        # SAML -> IBM -> ULB
        ...
        banks = partition_dataset(...)   # uses global np.random.shuffle
```

`partition_dataset` draws from the global RNG, so IBM's and ULB's Dirichlet partitions depend
on how much of the RNG stream SAML's processing already consumed. Running one dataset on its
own would therefore produce a different partition than the same cell inside the full
three-dataset sequence, and the results would not be comparable.

**Fix E** derives the partition seed from the cell's own coordinates only:

```python
_pseed = (_SEED * 100003 + int(round(float(alpha) * 1000)) * 97
          + sum(ord(_ch) for _ch in ds_name)) % (2**31 - 1)
_random.seed(_pseed); np.random.seed(_pseed)
banks = partition_dataset(...)
```

Every `(seed, dataset, alpha)` cell now partitions identically regardless of what ran before
it. That is what makes both the split and the resume sound, and it is the same idea as
upstream's Fix C (deterministic per-(seed, bank) MLP subsample) applied to the partition.

`CURRENT_SEED` is left untouched, so model `random_state` and the Fix C subsample keep using
the run seed exactly as upstream intended.

**Consequence:** these numbers do not match a sequential three-dataset run cell for cell. Run
all three notebooks under this scheme so the sweep is internally consistent. Upstream's
sequential notebook has the same order dependence and would benefit from Fix E as well.

## Resume

Each notebook prints its plan before spending GPU time:

```
[IBM] 0 cached, 15 to compute (of 15)  ~20.0h
    TODO  seed=42  IBM   alpha=0.05
```

If a session ends early, nothing is lost. Upstream already writes per-cell checkpoints
(`{tag}_seed{N}_benchmark.csv`), so:

1. Download the notebook's output and upload it as a Kaggle Dataset.
2. Add it to the notebook and set `RESUME_DIR` to its mount path.
3. Run All. Finished `(dataset, alpha, seed)` cells are skipped, including the expensive
   dataset load when every alpha for that seed is already cached.

## Suggested launch order

Start IBM first since it is the critical path at ~20h (two sessions), then SAML, then ULB.
ULB finishes in a single ~6h session, which frees that account first.
