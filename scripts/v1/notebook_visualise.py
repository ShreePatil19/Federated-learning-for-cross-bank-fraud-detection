"""
═══════════════════════════════════════════════════════════════════
FL Benchmark — NOTEBOOK: VISUALISE
Merges master_LR.csv, master_MLP.csv, master_TabNet.csv,
master_ResNet.csv and generates the full results matrix plot.

Run this AFTER all 4 model notebooks have finished.
Upload all 4 master CSVs as a Kaggle dataset or place in outputs/
═══════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Config ────────────────────────────────────────────────────────
OUTPUT_ROOT = "outputs"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Update these paths to wherever you saved your 4 master CSVs
MASTER_FILES = {
    "LR"     : "/kaggle/input/fl-results/master_LR.csv",
    "MLP"    : "/kaggle/input/fl-results/master_MLP.csv",
    "TabNet" : "/kaggle/input/fl-results/master_TabNet.csv",
    "ResNet" : "/kaggle/input/fl-results/master_ResNet.csv",
}

ALL_FL      = ["FedAvg", "FedProx", "SCAFFOLD", "FedNova", "PersFL"]
ALL_MODELS  = ["LR", "MLP", "TabNet", "ResNet"]
ALL_PRIVACY = ["NoDP", "DP", "Sparsification"]

# ── Merge ─────────────────────────────────────────────────────────
dfs = []
for model, path in MASTER_FILES.items():
    if os.path.exists(path):
        dfs.append(pd.read_csv(path))
        print(f"  Loaded {model}: {len(dfs[-1])} rows")
    else:
        print(f"  WARNING: {path} not found — skipping {model}")

if not dfs:
    raise FileNotFoundError("No master CSV files found. Check MASTER_FILES paths.")

df = pd.concat(dfs, ignore_index=True)
df.to_csv(os.path.join(OUTPUT_ROOT, "all_results_master.csv"), index=False)
print(f"\nMerged {len(df)} total rows → all_results_master.csv")
print(df[["fl_algorithm","ml_model","privacy_mode","best_auc","best_f1"]].to_string(index=False))

# ── Plot ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--"
})

fig = plt.figure(figsize=(22, 15))
fig.suptitle(
    "FL Benchmark Results Matrix — 60 Combinations\n"
    "ULB Credit Card Fraud · 5 Banks · 60 Rounds · 4 Models",
    fontsize=13, fontweight="bold", y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── 1. AUC Heatmap ────────────────────────────────────────────────
ax1       = fig.add_subplot(gs[0, :2])
priv_best = df.groupby(["fl_algorithm", "ml_model"])["best_auc"].max().reset_index()
pivot     = priv_best.pivot(index="fl_algorithm", columns="ml_model", values="best_auc")
pivot     = pivot.reindex(index=ALL_FL, columns=ALL_MODELS)
im        = ax1.imshow(pivot.values, cmap="RdYlGn", vmin=0.88, vmax=0.99, aspect="auto")
ax1.set_xticks(range(len(ALL_MODELS)));  ax1.set_xticklabels(ALL_MODELS, fontsize=9)
ax1.set_yticks(range(len(ALL_FL)));      ax1.set_yticklabels(ALL_FL, fontsize=9)
ax1.set_title("Best AUC Heatmap (FL Algorithm × ML Model)", fontweight="bold")
plt.colorbar(im, ax=ax1, shrink=0.8)
for i in range(len(ALL_FL)):
    for j in range(len(ALL_MODELS)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax1.text(j, i, f"{val:.4f}", ha="center", va="center",
                     fontsize=7.5, fontweight="bold",
                     color="white" if val > 0.975 else "black")

# ── 2. Fairness σ_AUC ─────────────────────────────────────────────
ax2      = fig.add_subplot(gs[0, 2])
fair_df  = df.groupby("fl_algorithm")["sigma_auc"].mean().reindex(ALL_FL)
colors_f = ["#E24B4A" if v > 0.01 else "#EF9F27" if v > 0.001 else "#1D9E75"
            for v in fair_df.values]
ax2.barh(fair_df.index, fair_df.values, color=colors_f)
ax2.set_title("Avg σ_AUC by FL Algorithm\n(lower = fairer)", fontweight="bold")
ax2.set_xlabel("σ_AUC")
for i, v in enumerate(fair_df.values):
    ax2.text(v + 0.0002, i, f"{v:.4f}", va="center", fontsize=8)

# ── 3. Privacy Mode AUC ───────────────────────────────────────────
ax3    = fig.add_subplot(gs[1, 0])
pv_df  = df.groupby("privacy_mode")["best_auc"].mean().reindex(ALL_PRIVACY)
p_cols = ["#378ADD", "#1D9E75", "#BA7517"]
ax3.bar(pv_df.index, pv_df.values, color=p_cols, width=0.5)
ax3.set_ylim(0.88, 0.99)
ax3.set_title("Avg Best AUC by Privacy Mode", fontweight="bold")
ax3.set_ylabel("AUC")
for i, v in enumerate(pv_df.values):
    if not np.isnan(v):
        ax3.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")

# ── 4. KS Statistic ───────────────────────────────────────────────
ax4   = fig.add_subplot(gs[1, 1])
ks_df = df.groupby("fl_algorithm")["ks_stat"].mean().reindex(ALL_FL)
ax4.bar(ks_df.index, ks_df.values,
        color=["#D4537E","#378ADD","#1D9E75","#BA7517","#7F77DD"], width=0.6)
ax4.axhline(0.40, color="#E24B4A", ls="--", lw=1.5, label="Regulatory threshold")
ax4.set_title("Avg KS Statistic by FL Algorithm", fontweight="bold")
ax4.set_ylabel("KS Statistic")
ax4.legend(fontsize=7)
for i, v in enumerate(ks_df.values):
    if not np.isnan(v):
        ax4.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

# ── 5. Rounds to 95% AUC ──────────────────────────────────────────
ax5    = fig.add_subplot(gs[1, 2])
r95_df = df.groupby("fl_algorithm")["rounds_to_95pct"].mean().reindex(ALL_FL)
ax5.bar(r95_df.index, r95_df.values,
        color=["#D4537E","#378ADD","#1D9E75","#BA7517","#7F77DD"], width=0.6)
ax5.set_title("Avg Rounds to 95% Peak AUC\n(lower = faster convergence)", fontweight="bold")
ax5.set_ylabel("Rounds")
for i, v in enumerate(r95_df.values):
    if not np.isnan(v):
        ax5.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")

plt.savefig(os.path.join(OUTPUT_ROOT, "results_matrix.png"),
            dpi=180, bbox_inches="tight", facecolor="white")
plt.savefig(os.path.join(OUTPUT_ROOT, "results_matrix.pdf"),
            bbox_inches="tight", facecolor="white")
print(f"\nSaved: {OUTPUT_ROOT}/results_matrix.png")
print(f"Saved: {OUTPUT_ROOT}/results_matrix.pdf")
plt.show()
