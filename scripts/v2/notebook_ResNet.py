"""
═══════════════════════════════════════════════════════════════════
FL Benchmark — NOTEBOOK: ResNet
Runs: ResNet × 5 FL algos × 3 privacy = 15 combinations × 60 rounds
Paste notebook_base.py and notebook_fl_algorithms.py above this cell
═══════════════════════════════════════════════════════════════════
"""

MODEL      = "ResNet"
MASTER_CSV = os.path.join(OUTPUT_ROOT, f"master_{MODEL}.csv")

DATA_PATH = "/kaggle/input/creditcardfraud/creditcard.csv"

# ── Run ───────────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = load_data(DATA_PATH)

combos = [(fl, MODEL, pr) for fl in ALL_FL for pr in ALL_PRIVACY]
total  = len(combos)

print(f"\nRunning {total} combinations for model: {MODEL}")
print(f"Results → {MASTER_CSV}\n")

results = []
for idx, (fl, ml, pr) in enumerate(combos, 1):
    r = run_combination_safe(fl, ml, pr, X_tr, X_te, y_tr, y_te,
                             idx, total, MASTER_CSV)
    if r:
        results.append(r)

print(f"\n{'='*60}")
print(f"DONE — {MODEL} | {len(results)}/{total} completed")
print(f"{'='*60}")

# FIX 5: Guard against FileNotFoundError when all combos fail
if os.path.exists(MASTER_CSV) and os.path.getsize(MASTER_CSV) > 0:
    print(pd.read_csv(MASTER_CSV).to_string(index=False))
else:
    print(f"⚠  No results written to {MASTER_CSV} — all combinations failed.")
    print("   Check CUDA compatibility and re-run.")
