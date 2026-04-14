"""
data_utils.py
=============
Dataset loading, preprocessing, and non-IID partitioning
for the multi-bank federated fraud detection pipeline.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_creditcard_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and preprocess the Kaggle Credit Card Fraud dataset.

    Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    Features: V1–V28 (PCA-transformed), Amount, Time
    Label:    Class (0 = legitimate, 1 = fraud)

    Preprocessing applied:
      • Drop 'Time' (not informative across sessions)
      • Log-scale 'Amount' (heavy right skew)
      • StandardScaler on all features

    Args:
        path: Path to creditcard.csv

    Returns:
        X (n_samples, n_features), y (n_samples,), feature_names
    """
    df = pd.read_csv(path)

    # Basic validation
    required_cols = {"Class", "Amount", "Time"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Drop Time — not meaningful across different sessions
    df = df.drop(columns=["Time"])

    # Log-transform Amount (right-skewed)
    df["Amount"] = np.log1p(df["Amount"])

    y = df["Class"].values
    X_raw = df.drop(columns=["Class"]).values
    feature_names = df.drop(columns=["Class"]).columns.tolist()

    # Normalise
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    return X, y, feature_names


def load_synthetic_dataset(
    n_samples: int = 8000,
    n_features: int = 28,
    fraud_rate: float = 0.12,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate a synthetic fraud dataset when real data is unavailable.
    Mirrors the structure of the Kaggle Credit Card Fraud dataset.

    Uses sklearn's make_classification with realistic class imbalance.
    """
    from sklearn.datasets import make_classification

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    Xl, _ = make_classification(
        n_samples=n_legit, n_features=n_features,
        n_informative=18, n_redundant=6, n_classes=2,
        weights=[0.98, 0.02], flip_y=0.005, random_state=seed,
    )
    Xf, _ = make_classification(
        n_samples=n_fraud, n_features=n_features,
        n_informative=18, n_redundant=6, n_classes=2,
        weights=[0.05, 0.95], flip_y=0.005, random_state=seed + 7,
    )

    X = np.vstack([Xl, Xf])
    y = np.concatenate([np.zeros(n_legit, int), np.ones(n_fraud, int)])

    perm = np.random.default_rng(seed).permutation(len(y))
    X, y = X[perm], y[perm]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    feature_names = [f"V{i+1}" for i in range(n_features - 1)] + ["Amount"]
    return X, y, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def train_test_split_stratified(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split preserving fraud rate in both sets."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


# ─────────────────────────────────────────────────────────────────────────────
# NON-IID PARTITIONING
# ─────────────────────────────────────────────────────────────────────────────

def dirichlet_partition(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Non-IID data partition using Dirichlet sampling.

    In real federated banking, each institution has a different fraud
    profile — retail banks see card fraud, corporate banks see invoice
    fraud, tax authorities see evasion schemes. This partition simulates
    that heterogeneity.

    The Dirichlet distribution with parameter α controls skewness:
      α → ∞   :  IID (all clients see same class distribution)
      α = 1.0 :  moderately heterogeneous
      α = 0.5 :  strongly non-IID  ← default
      α → 0   :  each client sees only one class (extreme)

    Args:
        X:         Feature matrix
        y:         Labels
        n_clients: Number of federated clients (banks)
        alpha:     Dirichlet concentration parameter
        seed:      Random seed

    Returns:
        List of (X_client, y_client) tuples, one per client
    """
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(n_clients)]

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)

        proportions = rng.dirichlet(np.ones(n_clients) * alpha)
        sizes = (proportions * len(cls_idx)).astype(int)
        sizes[-1] = len(cls_idx) - sizes[:-1].sum()  # fix rounding

        start = 0
        for c, size in enumerate(sizes):
            client_indices[c].extend(cls_idx[start : start + size])
            start += size

    result = []
    for indices in client_indices:
        idx = np.array(indices)
        rng.shuffle(idx)
        result.append((X[idx], y[idx]))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# EDA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_eda_stats(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
) -> dict:
    """
    Compute basic EDA statistics for the dataset.

    Returns a dict with:
      - fraud_rate: proportion of fraudulent samples
      - class_counts: {0: n_legit, 1: n_fraud}
      - top_features: feature names ranked by variance
      - feature_stats: mean/std/min/max per feature
    """
    fraud_rate = float(y.mean())
    class_counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}

    variances = X.var(axis=0)
    top_idx = np.argsort(variances)[::-1][:top_n]
    top_features = [feature_names[i] for i in top_idx]

    feature_stats = {
        feature_names[i]: {
            "mean": float(X[:, i].mean()),
            "std":  float(X[:, i].std()),
            "min":  float(X[:, i].min()),
            "max":  float(X[:, i].max()),
        }
        for i in top_idx
    }

    return {
        "n_samples":     len(y),
        "n_features":    X.shape[1],
        "fraud_rate":    fraud_rate,
        "class_counts":  class_counts,
        "top_features":  top_features,
        "feature_stats": feature_stats,
    }


def print_eda_report(stats: dict, client_data: Optional[List] = None) -> None:
    """Pretty-print EDA summary to stdout."""
    print("\n" + "═" * 60)
    print("  DATASET OVERVIEW")
    print("═" * 60)
    print(f"  Samples     : {stats['n_samples']:,}")
    print(f"  Features    : {stats['n_features']}")
    print(f"  Fraud rate  : {stats['fraud_rate']*100:.2f}%")
    print(f"  Legitimate  : {stats['class_counts'].get(0, 0):,}")
    print(f"  Fraudulent  : {stats['class_counts'].get(1, 0):,}")

    print(f"\n  Top Features by Variance:")
    print(f"  {'Feature':<14} {'Mean':>8} {'Std':>8}")
    print("  " + "-" * 34)
    for name in stats["top_features"][:6]:
        s = stats["feature_stats"][name]
        print(f"  {name:<14} {s['mean']:>8.3f} {s['std']:>8.3f}")

    if client_data:
        print(f"\n  Client (Bank) Distribution:")
        print(f"  {'Client':<18} {'Samples':>8} {'Fraud':>8} {'Rate':>8}")
        print("  " + "-" * 46)
        for i, (cx, cy) in enumerate(client_data):
            nf = int((cy == 1).sum())
            print(f"  Bank {i+1:<13} {len(cy):>8,} {nf:>8,} {cy.mean()*100:>7.1f}%")
