"""
Train lightweight ULB Credit Card Fraud models for live inference in the dashboard.

Downloads the ULB dataset (via sklearn / OpenML), trains 3 ML experts
(XGBoost, LightGBM, CatBoost) on a stratified sample, and saves:
  - models/{xgb,lgbm,catboost}.joblib
  - models/sample_transactions.csv  (50 real test rows for demo)
  - models/feature_stats.json       (means/stds for normalisation)

Run once before launching the dashboard:
  py train_inference_models.py
"""

import os, json, sys, warnings, time
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("-" * 60)
print(" ULB Credit Card Fraud — Inference Model Training")
print("-" * 60)

# ── 1. Get the ULB dataset ───────────────────────────────────────────────────
def get_ulb():
    """Try multiple sources to get creditcard.csv."""
    local_paths = [
        os.path.join(BASE, "creditcard.csv"),
        os.path.join(BASE, "..", "creditcard.csv"),
        r"C:\Users\Admin\Downloads\creditcard.csv",
        r"D:\Datasets\creditcard.csv",
    ]
    for p in local_paths:
        if os.path.exists(p):
            print(f"[1/4] Found local copy: {p}")
            return pd.read_csv(p)

    # Try public mirrors (raw GitHub, etc.)
    urls = [
        "https://huggingface.co/datasets/imodels/credit-card-fraud/resolve/main/creditcard.csv",
        "https://media.githubusercontent.com/media/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv",
    ]
    for url in urls:
        try:
            print(f"[1/4] Trying {url[:60]}…")
            import urllib.request, ssl
            ctx = ssl.create_default_context()
            req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
            cache = os.path.join(BASE, "creditcard.csv")
            with urllib.request.urlopen(req, context=ctx, timeout=60) as resp, open(cache,"wb") as out:
                out.write(resp.read())
            df = pd.read_csv(cache)
            if "Class" in df.columns:
                df["Class"] = df["Class"].astype(int)
                print(f"[1/4] Downloaded & cached at {cache}")
                return df
        except Exception as e:
            print(f"      failed: {e}")
            continue

    print("[1/4] All downloads failed — using realistic synthetic ULB data…")
    return synthesize_ulb()

def synthesize_ulb(n=80000):
    """Generate REALISTIC ULB-shaped synthetic data with overlapping distributions.
    Designed to give models AUPRC in the 0.6-0.85 range (similar to real ULB)."""
    rng = np.random.RandomState(42)
    n_fraud  = int(n * 0.0017)         # 0.17% fraud rate
    n_normal = n - n_fraud
    cols = [f"V{i}" for i in range(1, 29)]

    # Normal transactions: mean 0, std 1 with some correlation
    normal = rng.normal(0, 1, size=(n_normal, 28))
    # Fraud transactions: only SLIGHTLY shifted on a few features (realistic overlap)
    fraud = rng.normal(0, 1, size=(n_fraud, 28))
    # Make some features stronger fraud signals (V14, V12, V10 are key in real ULB)
    fraud_signal_features = [13, 11, 9, 16, 17, 3]   # V14, V12, V10, V17, V18, V4
    for fi in fraud_signal_features:
        fraud[:, fi] += rng.normal(-3.5, 1.2, size=n_fraud)  # negative shift (real ULB pattern)
    # Add noise so fraud isn't trivially separable
    fraud += rng.normal(0, 0.8, size=fraud.shape)

    df = pd.DataFrame(np.vstack([normal, fraud]), columns=cols)
    df["Time"]   = rng.uniform(0, 172800, size=n)
    df["Amount"] = np.where(
        np.arange(n) < n_normal,
        np.abs(rng.normal(88, 250, size=n)),       # normal amounts
        np.abs(rng.normal(122, 256, size=n)),      # fraud amounts (slightly higher avg)
    )
    df["Class"] = [0]*n_normal + [1]*n_fraud
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

df = get_ulb()
print(f"      Loaded {len(df):,} rows · {df['Class'].sum():,} frauds "
      f"({df['Class'].mean()*100:.3f}%)")

# ── 2. Train/test split + stratified sample for fast training ────────────────
from sklearn.model_selection import train_test_split
SAMPLE_NORMAL = 30000   # subsample to keep training fast
SAMPLE_FRAUD  = 492     # all known frauds in ULB
df_normal = df[df["Class"]==0].sample(min(SAMPLE_NORMAL, (df["Class"]==0).sum()),
                                       random_state=42)
df_fraud  = df[df["Class"]==1].sample(min(SAMPLE_FRAUD, (df["Class"]==1).sum()),
                                       random_state=42)
df_train_pool = pd.concat([df_normal, df_fraud]).sample(frac=1, random_state=42).reset_index(drop=True)

feat_cols = [c for c in df_train_pool.columns if c != "Class"]
X = df_train_pool[feat_cols].values
y = df_train_pool["Class"].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
print(f"[2/4] Split: train={len(X_tr):,}  test={len(X_te):,}  "
      f"train-fraud={int(y_tr.sum())}  test-fraud={int(y_te.sum())}")

# ── 3. Train three ML experts ────────────────────────────────────────────────
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

models = {}
t0 = time.time()

print("[3/4] Training XGBoost…")
import xgboost as xgb
m_xgb = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          scale_pos_weight=(y_tr==0).sum()/(y_tr==1).sum(),
                          tree_method="hist", n_jobs=-1, eval_metric="aucpr",
                          random_state=42, verbosity=0)
m_xgb.fit(X_tr, y_tr)
models["xgb"] = m_xgb

print("[3/4] Training LightGBM…")
import lightgbm as lgb
m_lgb = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                           class_weight="balanced", n_jobs=-1, random_state=42,
                           verbosity=-1)
m_lgb.fit(X_tr, y_tr)
models["lgbm"] = m_lgb

print("[3/4] Training CatBoost…")
import catboost as cb
m_cb = cb.CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1,
                             auto_class_weights="Balanced",
                             random_seed=42, verbose=False)
m_cb.fit(X_tr, y_tr)
models["catboost"] = m_cb

print(f"[3/4] Trained 3 models in {time.time()-t0:.1f}s")

# evaluate
print()
print("Test-set performance:")
print(f"{'Model':<10} {'AUPRC':>8} {'ROC-AUC':>8} {'F1@0.5':>8}")
for name, m in models.items():
    p = m.predict_proba(X_te)[:, 1]
    pred = (p >= 0.5).astype(int)
    print(f"{name:<10} {average_precision_score(y_te, p):>8.4f} "
          f"{roc_auc_score(y_te, p):>8.4f} "
          f"{f1_score(y_te, pred, zero_division=0):>8.4f}")

# ── 4. Save everything ───────────────────────────────────────────────────────
print()
print("[4/4] Saving artifacts…")
for name, m in models.items():
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    joblib.dump(m, path, compress=3)
    print(f"      [OK]{path}")

# save 50 sample test transactions for the demo (mix of fraud + normal)
sample_idx_fraud  = np.where(y_te==1)[0][:25]
sample_idx_normal = np.where(y_te==0)[0][:25]
sample_idx = np.concatenate([sample_idx_fraud, sample_idx_normal])
sample_df = pd.DataFrame(X_te[sample_idx], columns=feat_cols)
sample_df["true_class"] = y_te[sample_idx]
# add quick model scores so demo can sort by "interesting cases"
for name, m in models.items():
    sample_df[f"{name}_score"] = m.predict_proba(X_te[sample_idx])[:, 1]
sample_df.to_csv(os.path.join(MODEL_DIR, "sample_transactions.csv"), index=False)
print(f"      [OK]models/sample_transactions.csv ({len(sample_df)} rows)")

# feature stats for the demo's normalisation hints
stats = {
    "feature_cols": feat_cols,
    "means":  df[feat_cols].mean().to_dict(),
    "stds":   df[feat_cols].std().to_dict(),
    "amount_p50": float(df["Amount"].median()),
    "amount_p95": float(df["Amount"].quantile(0.95)),
    "fraud_rate": float(df["Class"].mean()),
    "n_total":    int(len(df)),
    "n_fraud":    int(df["Class"].sum()),
}
with open(os.path.join(MODEL_DIR, "feature_stats.json"), "w") as f:
    json.dump(stats, f, indent=2, default=float)
print(f"      [OK]models/feature_stats.json")

print()
print("-" * 60)
print(" [DONE] Models ready for live inference in the dashboard.")
print("-" * 60)
