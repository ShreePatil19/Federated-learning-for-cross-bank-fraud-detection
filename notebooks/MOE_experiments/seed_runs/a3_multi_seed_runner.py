"""
A3 - Multi-seed runner.

What this answers: "Are the differences I see real, or seed noise?"
With one seed per cell, you cannot tell. Re-running each cell with N seeds
turns your n=9 paired blocks into n=9*N pseudo-independent observations and
gives you proper statistical power.

USAGE
=====
This script is a TEMPLATE that drives your existing notebook code from a
loop. You need to expose ONE callable from your notebook that runs a single
(dataset, alpha, seed) experiment and returns a dict of metrics per method.

  STEP 1: in your notebook, refactor the per-cell experiment into a function:

      # paste this near the top of your notebook
      def run_one_experiment(dataset: str, alpha: float, seed: int) -> list[dict]:
          '''Run all 11 methods for one (dataset, alpha, seed) and return rows.

          Each returned dict must contain keys:
            strategy, model_type, auprc, f1, f2, precision, recall, mcc,
            specificity, fpr, false_positives, n_test_fraud, threshold,
            dataset, alpha, seed
          '''
          import random, numpy as np
          random.seed(seed); np.random.seed(seed)
          try:
              import torch
              torch.manual_seed(seed)
              if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
          except ImportError:
              pass

          # ---- paste your existing per-cell pipeline here, using the seed
          # ---- to reseed every random source (data partition, MLP init,
          # ---- SMOTE, train/test split, etc.). Return a list of metric dicts.
          rows = []
          # rows = ... your existing evaluate_all() return, with `seed` added
          return rows

  STEP 2: export it (so this script can import it). Add at the END of the
  notebook a cell:

      import pickle, sys
      sys.modules['exp_entry'] = type(sys)('exp_entry')
      sys.modules['exp_entry'].run_one_experiment = run_one_experiment
      # also dump to a pickle the runner can unpickle
      with open(r'D:\\Masters\\Sem 1\\Neural Network and Fuzzy Logic\\PROJECT\\moe-fl-per-dataset-alpha-sweep-results\\exp_entry.pkl', 'wb') as fh:
          pickle.dump(run_one_experiment, fh)

  STEP 3: run this script:

      py -3 a3_multi_seed_runner.py --seeds 0 1 2 3 4 \\
          --datasets ULB SAML IBM --alphas 0.05 0.1 0.5

  STEP 4: it writes a3_multi_seed_results.csv (long format) and runs paired
  Wilcoxon on the per-seed AUPRCs (now n = 5*9 = 45 paired observations).

If you don't want to refactor: see ALTERNATIVE at the bottom of this file -
a thin shell that just re-executes your notebook N times via `papermill` with
different seed parameters.
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import argparse
import importlib.util

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ENTRY_PKL = os.path.join(HERE, "exp_entry.pkl")
OUT_CSV = os.path.join(HERE, "a3_multi_seed_results.csv")
SUMMARY_CSV = os.path.join(HERE, "a3_multi_seed_summary.csv")
WILCOXON_CSV = os.path.join(HERE, "a3_multi_seed_wilcoxon.csv")


def load_runner(custom_module: str | None):
    """Load the run_one_experiment callable.
    Order of preference:
      1. --module flag pointing at a .py file with `run_one_experiment`
      2. exp_entry.pkl in the results folder (dumped from notebook)
    """
    if custom_module:
        spec = importlib.util.spec_from_file_location("exp_entry_user", custom_module)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "run_one_experiment"):
            sys.exit(f"{custom_module} has no `run_one_experiment` function.")
        return mod.run_one_experiment

    if os.path.exists(ENTRY_PKL):
        with open(ENTRY_PKL, "rb") as fh:
            return pickle.load(fh)

    sys.exit(
        f"Need either --module path/to/runner.py with run_one_experiment(...)\n"
        f"or {ENTRY_PKL} dumped from your notebook (see top-of-file STEP 2)."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--datasets", nargs="+", default=["ULB", "SAML", "IBM"])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.5])
    p.add_argument("--module", type=str, default=None,
                   help="path to .py file exposing run_one_experiment(...)")
    p.add_argument("--resume", action="store_true",
                   help="skip (dataset,alpha,seed) cells already in OUT_CSV")
    args = p.parse_args()

    runner = load_runner(args.module)

    done = set()
    if args.resume and os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        for _, r in prev[["dataset", "alpha", "seed"]].drop_duplicates().iterrows():
            done.add((r["dataset"], float(r["alpha"]), int(r["seed"])))
        print(f"resume: {len(done)} (dataset,alpha,seed) cells already done")

    rows = []
    total = len(args.seeds) * len(args.datasets) * len(args.alphas)
    i = 0
    t0_all = time.time()
    for ds in args.datasets:
        for a in args.alphas:
            for s in args.seeds:
                i += 1
                key = (ds, float(a), int(s))
                if key in done:
                    print(f"[{i}/{total}] {ds} a={a} seed={s}  (skip - already done)")
                    continue
                print(f"[{i}/{total}] {ds} a={a} seed={s}  ...", flush=True)
                t0 = time.time()
                try:
                    cell_rows = runner(dataset=ds, alpha=a, seed=s)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    continue
                for r in cell_rows:
                    r.setdefault("dataset", ds)
                    r.setdefault("alpha", a)
                    r.setdefault("seed", s)
                rows.extend(cell_rows)
                print(f"  ok ({time.time() - t0:.1f}s, {len(cell_rows)} rows)")

                # Incremental write so a crash mid-sweep doesn't lose progress
                df = pd.DataFrame(rows)
                if os.path.exists(OUT_CSV) and args.resume:
                    df = pd.concat([pd.read_csv(OUT_CSV), df], ignore_index=True)
                    df = df.drop_duplicates(subset=["strategy", "dataset", "alpha", "seed"], keep="last")
                df.to_csv(OUT_CSV, index=False)

    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    df = pd.read_csv(OUT_CSV)
    print(f"Loaded {len(df)} total rows -> {os.path.basename(OUT_CSV)}")

    # ----- Per-method summary across seeds -----
    summ = (df.groupby(["strategy", "dataset", "alpha"])["auprc"]
              .agg(["mean", "std", "count"])
              .reset_index()
              .rename(columns={"mean": "auprc_mean", "std": "auprc_std", "count": "n_seeds"}))
    summ.to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote {os.path.basename(SUMMARY_CSV)}  ({len(summ)} rows)")

    # ----- Multi-seed paired Wilcoxon: each method vs fedavg -----
    print("\nPer-seed paired Wilcoxon (each method vs fedavg, all conditions pooled)")
    base = df[df["strategy"] == "fedavg"]
    if base.empty:
        print("  (no fedavg rows - skipping)")
        return
    base_idx = base.set_index(["dataset", "alpha", "seed"])["auprc"]
    out_rows = []
    for m in sorted(df["strategy"].unique()):
        if m == "fedavg":
            continue
        sub = df[df["strategy"] == m].set_index(["dataset", "alpha", "seed"])["auprc"]
        common = base_idx.index.intersection(sub.index)
        if len(common) < 5:
            continue
        x = sub.loc[common].values
        y = base_idx.loc[common].values
        try:
            w, p = stats.wilcoxon(x, y, zero_method="wilcox")
        except Exception:
            w, p = float("nan"), float("nan")
        diff_mean = float((x - y).mean())
        out_rows.append({
            "method": m,
            "vs": "fedavg",
            "n_pairs": len(common),
            "mean_diff_auprc": round(diff_mean, 5),
            "wilcoxon_W": round(float(w), 2) if not np.isnan(w) else None,
            "p_value": p,
            "sig_at_0.05": (p < 0.05) if not np.isnan(p) else None,
        })
        print(f"  {m:<22s}  n={len(common):>3d}  mean_diff={diff_mean:+.4f}  p={p:.4g}")

    pd.DataFrame(out_rows).to_csv(WILCOXON_CSV, index=False)
    print(f"\nWrote {os.path.basename(WILCOXON_CSV)}")


if __name__ == "__main__":
    main()


# =====================================================================
# ALTERNATIVE: papermill-style notebook re-execution (no refactor needed)
# =====================================================================
# If refactoring run_one_experiment is too much work, install papermill and
# parameterise your notebook with a top cell tagged "parameters":
#
#     # parameters
#     SEED = 0
#     DATASETS = ["ULB", "SAML", "IBM"]
#     ALPHAS = [0.05, 0.1, 0.5]
#
# Then run:
#     pip install papermill
#     for s in 0 1 2 3 4; do
#       papermill moe-fl-per-dataset-alpha-sweep.ipynb \
#                 runs/moe_seed${s}.ipynb -p SEED $s
#     done
#
# Each run writes its own all_benchmarks_combined.csv suffixed with the seed.
# Then concatenate and pass to this script's stats functions.
# =====================================================================
