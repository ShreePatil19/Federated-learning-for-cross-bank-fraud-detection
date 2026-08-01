# Kaggle sweep scripts — one dataset per script, one 12 h session at a time

Single-file, directly runnable versions of the canonical GROUP-A notebook
([`notebooks/MOE_experiments/seed_runs/moe-fl-per-dataset-alpha-sweep-GROUP-A.ipynb`](../../notebooks/MOE_experiments/seed_runs/moe-fl-per-dataset-alpha-sweep-GROUP-A.ipynb),
i.e. the merged PR #17 pipeline: triage probability capture + evaluation fixes),
with PR #16's split machinery applied on top:

- **one dataset per script, plus per-seed part scripts** — datasets and seed
  chunks run in parallel on separate Kaggle accounts (each account has
  ~30 GPU-hours/week); every `_partN.py` is guaranteed to finish in ONE session;
- **Fix E order-independent partitioning** — every `(seed, dataset, alpha)` cell
  partitions identically no matter what ran before it, which is what makes the
  split and the resume canonical (verified byte-identical: seed 42 run alone
  equals seed 42 run after seed 0, benchmark CSVs and `probs_*.npz` both);
- **session resume** — finished cells are skipped when a previous session's
  output is attached as an input;
- **session time-budget guard** (new here) — the script stops *starting* new
  cells once `SESSION_BUDGET_H` (default 10.5 h) is reached, so Kaggle's 12 h
  cap never kills a session mid-cell and nothing is lost.

| Script | Seeds | Measured cost | Fits one session? |
|---|---|---|---|
| `sweep_ULB.py` | 42,0,1,2,3 | ~1.2 h/seed → ~6 h | **yes** |
| `sweep_SAML.py` | 42,0,1,2,3 | ~16 h | no — 2 sessions via resume, or use the parts |
| `sweep_SAML_part1.py` | 42,0,1 | ~9.6 h | **yes** |
| `sweep_SAML_part2.py` | 2,3 | ~6.4 h | **yes** |
| `sweep_IBM.py` | 42,0,1,2,3 | ~20 h | no — 2–3 sessions via resume, or use the parts |
| `sweep_IBM_part1.py` | 42,0 | ~8 h | **yes** |
| `sweep_IBM_part2.py` | 1,2 | ~8 h | **yes** |
| `sweep_IBM_part3.py` | 3 | ~4 h | **yes** |

Attach per dataset: ULB → `mlg-ulb/creditcardfraud` (`creditcard.csv`);
SAML → `berkanoztas/synthetic-transaction-monitoring-dataset-aml` (`SAML-D.csv`);
IBM → `ealtman2019/ibm-transactions-for-anti-money-laundering-aml` (`HI-Small_Trans.csv`).

Costs are measured (gate-log timestamps of a real run, PR #16), not estimates.
Fix E makes every `(seed, dataset, alpha)` cell canonical regardless of run
order, so the seed chunks are fully independent — run them **in parallel on
separate accounts**. Whichever part runs **last** should attach the sibling
parts' outputs: it merges them and emits the complete all-seed package.

## Recommended plans

- **ULB** — run `sweep_ULB.py`, done in one ~6 h session.
- **SAML** — `part1` and `part2` in parallel on two accounts. Then either
  (a) attach part1's output when you start part2 *after* part1 finished —
  part2 emits the full package; or (b) if they ran simultaneously, run
  `sweep_SAML.py` afterwards with both outputs attached — everything resumes
  instantly and it finishes with A2 + full-seed A3/A4/triage in ~1 h.
- **IBM** — `part1` ∥ `part2` first (~8 h each), then `part3` (~4 h) with both
  outputs attached: part3 computes seed 3 and emits the full package.
  Wall clock ≈ 12 h instead of ~20 h serial.
- No parts, one account: just run `sweep_SAML.py` / `sweep_IBM.py` repeatedly —
  the budget guard splits the work and each rerun (with the previous output
  attached) continues where it stopped.

## Running a script

1. Kaggle → **Create → New Notebook**, delete the starter cell, paste the whole
   script into a single cell (a Script-type kernel works too).
2. **Add Input** → the dataset for that script. **Accelerator = GPU** (the script
   aborts immediately with a clear message if GPU is off — a CPU run cannot
   finish in one session). Internet can stay off.
3. **Save Version → Save & Run All (Commit)**.

## Continuing / merging sessions

1. In a fresh session: **Add Input → Your Work → the previous session(s)**
   (their Outputs mount under `/kaggle/input/`). Uploading an output folder as
   a Dataset works the same. Multiple mounts are fine — attach every sibling
   part you have.
2. Run the script, unchanged. `RESUME_DIR = 'auto'` finds all mounts and
   prints the plan before spending GPU time:

   ```
   [IBM] 6 cached, 9 to compute (of 15)  ~12.0h
       TODO  seed=1   IBM   alpha=0.05
       ...
   ```

   Finished cells of THIS script's seeds are skipped (including the expensive
   dataset load when a whole seed is cached); finished cells of OTHER seeds
   found in the mounts are merged in at the end (`[AGGREGATE]` lines). The
   `probs_*.npz` triage captures and the gate-weight log are carried forward
   both ways, so the **final session's output is the complete package** — A1
   gate diagnostic, A2 centralised baseline, A3 multi-seed statistics, A4 cost
   curves and the triage layer computed over every seed. (A2 is skipped in
   budget-stopped sessions on purpose; it reruns in the final session.)

Auto-detection ignores any mount whose root contains a `triage/` directory —
that is the repo checkout itself, whose committed `results/` must never be
mistaken for a resumable previous session.

## In-flight sanity signals (from `RERUN_CHECKLIST.md` — abort early if violated)

1. **Rounds evolve** — round 5/10/15/20 eval metrics change across rounds.
   Frozen metrics = the warm-start bug is back.
2. **Strategies diverge** — FedAvg vs FedProx differ beyond the third decimal
   on at least SAML or IBM.
3. **Seeds diverge** — per-seed outputs differ across seeds
   (`scripts/regen_table6.py` refuses identical-across-seeds folders).
4. **Single-class banks train** without `ValueError` at alpha = 0.05
   (`partial_fit(..., classes=CLASSES)` handles them).

## Canonicality

Fix E derives the partition seed from the cell's own coordinates only:

```python
_pseed = (_SEED * 100003 + int(round(float(alpha) * 1000)) * 97
          + sum(ord(_ch) for _ch in ds_name)) % (2**31 - 1)
```

Consequences, same as the `split_sweeps/` notebooks:

- Numbers do **not** match a pre-Fix-E sequential run cell for cell. In
  particular `results/ULB_only_5seed_2026-07-27/` (preview run of the merged
  PR #17 notebook, which has no Fix E) is not cell-for-cell comparable. Run all
  three datasets under this scheme so the sweep is internally consistent.
- Partitions match `notebooks/MOE_experiments/split_sweeps/*.ipynb` (same
  formula), but those notebooks predate PR #17 — they lack the triage capture
  and the evaluation fixes. Prefer these scripts.

## Editing the scripts

Each script is `CONFIG` (per-script, a few lines) + a **shared body that is
byte-identical across every `sweep_*.py`**. Never patch one file alone:

- regenerate them all from the canonical notebook:
  `python3 scripts/kaggle/generate_from_notebook.py`
  (every patch it applies is an exact-match assertion, so upstream notebook
  drift fails the build instead of producing silently-wrong scripts);
- verify: `bash scripts/kaggle/check_shared_sync.sh`.

For a CPU-only local smoke run (no cuml/cupy/GPU), the scripts degrade
gracefully; point `SWEEP_INPUT_DIR` at a folder that mimics `/kaggle/input`.
