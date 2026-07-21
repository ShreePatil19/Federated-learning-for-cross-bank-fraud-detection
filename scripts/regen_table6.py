#!/usr/bin/env python3
"""Regenerate the paper's Table 6 (per-seed AUPRC, ULB, alpha=0.5) from the
genuine per-seed Kaggle output CSVs.

Input: a directory containing the per-seed combined result CSVs written by the
GROUP-A sweep notebook (`all_benchmarks_combined_seed{SEED}.csv`; falls back to
any *.csv carrying the required columns). The CSVs are NOT in this repo — ask
Chloe for the genuine Kaggle output folder ("Seed-Moe-sweep-result").

Output: per-strategy per-seed AUPRC as a markdown table and as LaTeX tabular
rows, with mean +/- population std (ddof=0, matching the paper's Table II).
The .tex table is replaced wholesale from this output; no value is ever typed
by hand.

Integrity checks enforced:
  (a) refuses to emit if the seed values are identical for every strategy —
      that is the unseeded duplicate folder, not the genuine one;
  (b) prints the recomputed mean +/- std next to the values currently in the
      paper (PAPER_VALUES below) so a human can diff them.

Usage:
    python scripts/regen_table6.py /path/to/Seed-Moe-sweep-result
    python scripts/regen_table6.py /path/to/csvs --dataset ULB --alpha 0.5
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

REQUIRED_COLS = {'strategy', 'dataset', 'alpha', 'auprc', 'seed'}

# Values currently printed in the paper for ULB alpha=0.5 (mean, population
# std). Only the FedAvg fingerprint is known from the audit; fill in the rest
# from the paper before relying on the diff column. None -> printed as "n/a".
PAPER_VALUES = {
    'fedavg': (0.699, 0.116),
    'fedprox': None,
    'fednova': None,
    'persfl': None,
    'xgb': None,
    'lgbm': None,
    'catboost': None,
    'moe_static': None,
    'moe_performance': None,
    'moe_confidence': None,
    'moe_typology_aware': None,
}

DISPLAY = {
    'fedavg': 'FedAvg', 'fedprox': 'FedProx', 'fednova': 'FedNova',
    'persfl': 'PersFL', 'xgb': 'XGBoost', 'lgbm': 'LightGBM',
    'catboost': 'CatBoost', 'moe_static': 'MoE-Static',
    'moe_performance': 'MoE-Performance', 'moe_confidence': 'MoE-Confidence',
    'moe_typology_aware': 'MoE-Typology',
}
STRATEGY_ORDER = list(DISPLAY)


def load_runs(csv_dir):
    paths = sorted(glob.glob(os.path.join(csv_dir, 'all_benchmarks_combined_seed*.csv')))
    if not paths:
        paths = sorted(glob.glob(os.path.join(csv_dir, '*.csv')))
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if REQUIRED_COLS.issubset(df.columns):
            frames.append(df[list(REQUIRED_COLS)])
    if not frames:
        sys.exit(f'ERROR: no CSV in {csv_dir!r} has the required columns '
                 f'{sorted(REQUIRED_COLS)}')
    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=['strategy', 'dataset', 'alpha', 'seed'])
    if len(df) < before:
        print(f'note: dropped {before - len(df)} duplicate '
              f'(strategy, dataset, alpha, seed) rows', file=sys.stderr)
    return df


def check_genuine(pivot):
    """Refuse the unseeded duplicate folder: there, every strategy has one
    identical value across all seeds."""
    all_frozen = bool(
        (pivot.round(12).nunique(axis=1) <= 1).all()) and pivot.shape[1] > 1
    if all_frozen:
        sys.exit(
            'ERROR: every strategy has identical AUPRC across all seeds. '
            'This is the unseeded duplicate output folder, not the genuine '
            '"Seed-Moe-sweep-result" one. Refusing to emit a table from it.')


def fmt(v):
    return 'n/a' if pd.isna(v) else f'{v:.3f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('csv_dir', help='directory of per-seed combined CSVs')
    ap.add_argument('--dataset', default='ULB')
    ap.add_argument('--alpha', type=float, default=0.5)
    args = ap.parse_args()

    df = load_runs(args.csv_dir)
    df = df[(df['dataset'].astype(str).str.upper() == args.dataset.upper())
            & (np.isclose(df['alpha'].astype(float), args.alpha))]
    if df.empty:
        sys.exit(f'ERROR: no rows for dataset={args.dataset} alpha={args.alpha}')

    pivot = df.pivot_table(index='strategy', columns='seed', values='auprc')
    order = [s for s in STRATEGY_ORDER if s in pivot.index] + \
            [s for s in pivot.index if s not in STRATEGY_ORDER]
    pivot = pivot.loc[order]
    seeds = list(pivot.columns)
    if len(seeds) != 5:
        print(f'warning: expected 5 seeds, found {len(seeds)}: {seeds}',
              file=sys.stderr)

    check_genuine(pivot)

    mean = pivot.mean(axis=1)
    std = pivot.std(axis=1, ddof=0)  # population std, as in the paper

    # ---- integrity diff vs paper ----------------------------------------
    print(f'\n== Recomputed vs paper ({args.dataset}, alpha={args.alpha}, '
          f'population std) ==')
    for s in pivot.index:
        paper = PAPER_VALUES.get(s)
        paper_s = f'{paper[0]:.3f} +/- {paper[1]:.3f}' if paper else 'n/a'
        flag = ''
        if paper:
            flag = ('  <-- MATCH' if abs(paper[0] - mean[s]) < 5e-4
                    and abs(paper[1] - std[s]) < 5e-4 else '  <-- DIFFERS')
        print(f'  {DISPLAY.get(s, s):<16} recomputed {mean[s]:.3f} +/- '
              f'{std[s]:.3f}   paper {paper_s}{flag}')

    # ---- markdown -------------------------------------------------------
    print(f'\n== Markdown ==\n')
    head = ['Strategy'] + [f'seed {int(c)}' for c in seeds] + ['mean ± std']
    print('| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    for s in pivot.index:
        cells = [DISPLAY.get(s, s)] + [fmt(pivot.loc[s, c]) for c in seeds]
        cells.append(f'{mean[s]:.3f} ± {std[s]:.3f}')
        print('| ' + ' | '.join(cells) + ' |')

    # ---- LaTeX ----------------------------------------------------------
    print(f'\n== LaTeX tabular rows ==\n')
    for s in pivot.index:
        cells = [DISPLAY.get(s, s)] + [fmt(pivot.loc[s, c]) for c in seeds]
        cells.append(f'{mean[s]:.3f} $\\pm$ {std[s]:.3f}')
        print(' & '.join(cells) + r' \\')


if __name__ == '__main__':
    main()
