"""
main.py
=======
Entry point for the Federated Fraud Detection pipeline.

Run with Kaggle dataset:
    python main.py --data data/creditcard.csv

Run with synthetic data (no download needed):
    python main.py --synthetic

Full options:
    python main.py --help
"""

import argparse
import json
import os
import sys
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_utils import (
    load_creditcard_dataset,
    load_synthetic_dataset,
    train_test_split_stratified,
    dirichlet_partition,
    compute_eda_stats,
    print_eda_report,
)
from federated_learning import FederatedClient, FederatedServer


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Federated setup
    "num_banks":        6,
    "num_rounds":       10,
    "local_epochs":     15,
    "batch_size":       64,
    # Model
    "hidden_dims":      [64, 32],
    "learning_rate":    0.005,
    # Non-IID
    "dirichlet_alpha":  0.5,
    # Security
    "clip_norm":        2.0,
    "dp_sigma":         0.003,
    "n_byzantine":      1,
    # Fairness
    "fairness_alpha":   0.3,
    # Misc
    "seed":             42,
    "test_size":        0.2,
}

BANK_NAMES = [
    "RetailBank-A",
    "PrivateBank-B",
    "CorporateBank-C",
    "TaxAuthority-D",
    "CreditUnion-E",
    "InvestBank-F",
]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run(config: dict, data_path: str = None, synthetic: bool = False) -> dict:

    np.random.seed(config["seed"])

    print("\n" + "█" * 60)
    print("  FEDERATED FRAUD DETECTION SYSTEM")
    print("  Multi-Bank · Non-IID · Privacy-Preserving")
    print("█" * 60)

    # ── Load Data ─────────────────────────────────────────────
    print("\n[1/5]  Loading dataset...")

    if synthetic or data_path is None:
        print("       Using synthetic data (run with --data to use Kaggle CSV)")
        X, y, feature_names = load_synthetic_dataset(seed=config["seed"])
    else:
        print(f"       Loading: {data_path}")
        X, y, feature_names = load_creditcard_dataset(data_path)

    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X, y, test_size=config["test_size"], seed=config["seed"]
    )

    # ── EDA ───────────────────────────────────────────────────
    print("\n[2/5]  Running EDA...")
    bank_data = dirichlet_partition(
        X_train, y_train,
        n_clients=config["num_banks"],
        alpha=config["dirichlet_alpha"],
        seed=config["seed"],
    )
    stats = compute_eda_stats(X_train, y_train, feature_names)
    print_eda_report(stats, client_data=bank_data)

    # ── Compute class weight ───────────────────────────────────
    n_legit = int((y_train == 0).sum())
    n_fraud = int((y_train == 1).sum())
    class_weight = n_legit / max(n_fraud, 1)

    input_dim = X_train.shape[1]

    # ── Initialise clients ────────────────────────────────────
    print("\n[3/5]  Initialising federated clients (banks)...")
    clients = [
        FederatedClient(
            client_id=BANK_NAMES[i],
            X=bx,
            y=by,
            input_dim=input_dim,
            hidden_dims=config["hidden_dims"],
            learning_rate=config["learning_rate"],
            class_weight=class_weight,
        )
        for i, (bx, by) in enumerate(bank_data)
    ]

    # ── Initialise server ─────────────────────────────────────
    server = FederatedServer(
        input_dim=input_dim,
        hidden_dims=config["hidden_dims"],
        learning_rate=config["learning_rate"],
        class_weight=class_weight,
        clip_norm=config["clip_norm"],
        dp_sigma=config["dp_sigma"],
        n_byzantine=config["n_byzantine"],
        fairness_alpha=config["fairness_alpha"],
    )

    # ── FL Training Loop ──────────────────────────────────────
    print(f"\n[4/5]  Federated training — "
          f"{config['num_banks']} banks × "
          f"{config['num_rounds']} rounds × "
          f"{config['local_epochs']} local epochs")
    print("─" * 60)

    for rnd in range(1, config["num_rounds"] + 1):
        print(f"\n  Round {rnd:2d}/{config['num_rounds']}")

        global_weights = server.get_global_weights()
        c_weights, c_sizes, c_f1s, c_ids = [], [], [], []

        for client in clients:
            # Simulate Byzantine attack on last client in final 3 rounds
            is_byzantine = (
                client.client_id == BANK_NAMES[-1]
                and rnd > config["num_rounds"] - 3
            )

            weights, norm, f1 = client.train(
                global_weights=global_weights,
                epochs=config["local_epochs"],
                batch_size=config["batch_size"],
                clip_norm=config["clip_norm"],
                poisoned=is_byzantine,
            )

            tag = " ⚠ Byzantine" if is_byzantine else ""
            print(
                f"    {client.client_id:<18}  "
                f"n={client.n_samples:>5,}  "
                f"fraud={client.fraud_rate*100:>5.1f}%  "
                f"F1={f1:.3f}  ‖Δw‖={norm:.2f}{tag}"
            )

            c_weights.append(weights)
            c_sizes.append(client.n_samples)
            c_f1s.append(f1)
            c_ids.append(client.client_id)

        # Aggregate
        excluded, agg_weights = server.aggregate(c_weights, c_sizes, c_f1s, c_ids)
        if excluded:
            print(f"    Krum excluded : {[BANK_NAMES[i] for i in excluded]}")

        # Evaluate
        metrics = server.evaluate(X_test, y_test, rnd)
        print(
            f"  → Global  Acc={metrics['accuracy']:.4f}  "
            f"Prec={metrics['precision']:.4f}  "
            f"Rec={metrics['recall']:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"AUC={metrics['auc']:.4f}"
        )

    # ── Final Report ──────────────────────────────────────────
    print("\n[5/5]  Final evaluation")
    print("═" * 60)

    final = server.history[-1]
    preds = server.global_model.predict(X_test)

    print(f"\n  Accuracy  : {final['accuracy']:.4f}")
    print(f"  Precision : {final['precision']:.4f}")
    print(f"  Recall    : {final['recall']:.4f}")
    print(f"  F1 Score  : {final['f1']:.4f}")
    print(f"  AUC-ROC   : {final['auc']:.4f}")

    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, preds)
    print(f"               Pred Legit  Pred Fraud")
    print(f"  Actual Legit  {cm[0,0]:>8,}  {cm[0,1]:>8,}")
    print(f"  Actual Fraud  {cm[1,0]:>8,}  {cm[1,1]:>8,}")

    print(f"\n  Per-Class Report:")
    print(classification_report(
        y_test, preds,
        target_names=["Legitimate", "Fraudulent"],
        zero_division=0,
    ))

    # ── Save Results ──────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    results = {
        "config":  config,
        "history": server.history,
        "final":   final,
    }
    out_path = "results/fl_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved → {out_path}")
    print("\n  Done.\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Federated Learning for Multi-Bank Fraud Detection"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to creditcard.csv (Kaggle dataset)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of real dataset",
    )
    parser.add_argument("--banks",    type=int, default=6,   help="Number of banks")
    parser.add_argument("--rounds",   type=int, default=10,  help="FL rounds")
    parser.add_argument("--epochs",   type=int, default=15,  help="Local epochs per round")
    parser.add_argument("--alpha",    type=float, default=0.5, help="Dirichlet alpha (non-IID degree)")
    parser.add_argument("--dp-sigma", type=float, default=0.003, help="DP noise sigma")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = DEFAULT_CONFIG.copy()
    config["num_banks"]       = args.banks
    config["num_rounds"]      = args.rounds
    config["local_epochs"]    = args.epochs
    config["dirichlet_alpha"] = args.alpha
    config["dp_sigma"]        = args.dp_sigma

    run(config, data_path=args.data, synthetic=args.synthetic)
