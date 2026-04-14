# 🏦 Federated Fraud Detection

> **Privacy-preserving fraud detection across multiple banks using Federated Learning — no raw transaction data ever leaves a client.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=flat-square&logo=numpy)](https://numpy.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Overview

Traditional fraud detection requires pooling transaction data in a central server — which is legally and ethically problematic across separate financial institutions. This project implements a **Federated Learning (FL)** system where:

- Each bank trains a fraud detection model **locally on its own data**
- Only **model weights** are shared with a central aggregation server
- Raw transactions **never leave the originating institution**
- The global model improves across communication rounds via **FedAvg**

The system is hardened with three security mechanisms:

| Mechanism | Purpose |
|-----------|---------|
| **Gradient Clipping** | Limits update magnitude — prevents any single client from dominating |
| **Krum Filter** | Detects and excludes Byzantine (faulty/adversarial) clients |
| **Differential Privacy** | Adds calibrated Gaussian noise — protects individual records |

And a **Fairness-Weighted FedAvg** aggregation that prevents large banks from drowning out smaller clients with rare fraud patterns.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  GLOBAL SERVER                       │
│   • FedAvg aggregation                               │
│   • Krum Byzantine filter                            │
│   • Differential privacy noise                       │
│   • Global model evaluation                          │
└───────────┬────────────────────┬────────────────────┘
            │  weights only      │  weights only
    ┌───────▼──────┐    ┌────────▼─────┐    ...
    │ RetailBank-A │    │ PrivateBank-B│
    │ local MLP    │    │ local MLP    │
    │ local data   │    │ local data   │
    └──────────────┘    └─────────────┘
         (non-IID Dirichlet partition — each bank sees a different fraud rate)
```

**Model:** MLP `Input → Dense(64, ReLU) → Dense(32, ReLU) → Sigmoid(1)`  
**Loss:** Class-weighted binary cross-entropy (handles heavy fraud imbalance)  
**Threshold:** 0.35 (recall-optimised for fraud detection)

---

## Dataset

This project uses the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud):

- **284,807** transactions from European cardholders (Sept 2013)
- **492 fraudulent** transactions (0.172% — heavily imbalanced)
- **28 PCA-transformed features** (V1–V28) + Amount + Time
- Binary label: `Class` (0 = legitimate, 1 = fraud)

Download `creditcard.csv` from Kaggle and place it in `data/`.

> **No Kaggle account?** Run with `--synthetic` for auto-generated data.

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/yourusername/federated-fraud-detection.git
cd federated-fraud-detection
pip install -r requirements.txt
```

### 2. Add the dataset

```bash
# Download creditcard.csv from Kaggle, then:
mv ~/Downloads/creditcard.csv data/
```

### 3. Run the pipeline

```bash
# With Kaggle data
python main.py --data data/creditcard.csv

# With synthetic data (no download needed)
python main.py --synthetic

# Custom configuration
python main.py --data data/creditcard.csv \
               --banks 6   \
               --rounds 10 \
               --epochs 15 \
               --alpha 0.5
```

---

## Project Structure

```
federated-fraud-detection/
│
├── main.py                        # Entry point — run the full pipeline
├── requirements.txt
├── LICENSE
│
├── src/
│   ├── federated_learning.py      # Core FL engine (FraudDetectorMLP, FederatedClient, FederatedServer)
│   └── data_utils.py              # Data loading, EDA, Dirichlet partitioning
│
├── notebooks/
│   ├── exploration.ipynb              # EDA + step-by-step walkthrough
│   ├── lr_60epoch_check.ipynb         # LR 60-epoch benchmark
│   ├── resnet_60epoch_check.ipynb     # ResNet 60-epoch benchmark
│   ├── FL_algorithms/                 # One notebook per FL algorithm
│   │   ├── fedavg_logistic_regression.ipynb
│   │   ├── fedavg_xgboost_leaf_embeddings.ipynb
│   │   ├── fedprox_neural_network.ipynb
│   │   ├── personalized_fl_peravg.ipynb
│   │   ├── fednova_resnet_tabular_v1.ipynb
│   │   ├── fednova_resnet_tabular_v2.ipynb
│   │   ├── moon_contrastive_xgboost.ipynb
│   │   └── scaffold_tabnet.ipynb
│   └── MOE_experiments/               # Mixture of Experts experiment notebooks
│       ├── moe_2_trial.ipynb
│       ├── moe_3_train_test.ipynb
│       ├── moe_3_lgbm_target.ipynb
│       └── moe_ds_merged_final.ipynb  # ← FINAL canonical MOE results
│
├── scripts/
│   ├── v1/                            # Original training scripts
│   └── v2/                            # Updated training scripts
│
├── results/
│   ├── FL_LR/outputs/                 # Logistic Regression FL results (CSV + JSON)
│   └── FL_MLP/outputs/                # MLP FL results (CSV + JSON)
│
├── moe_experiments/
│   ├── v1/                            # Initial MOE GPU experiments
│   ├── v2_check/                      # Validation run
│   ├── v3_merged/                     # DS merged results ← FINAL research results
│   ├── v4_moe2_trial/                 # MOE trial 2 outputs
│   ├── v5_moe3_train_test/            # MOE 3 train/test outputs
│   └── v6_moe3_lgbm_target/           # MOE 3 LGBM target outputs
│
├── reports/
│   ├── dashboards/                # Interactive HTML dashboards
│   ├── figures/                   # Result plots and charts
│   └── documents/                 # PDF reports and DOCX write-ups
│
└── data/
    └── creditcard.csv             # ← place Kaggle dataset here (gitignored)
```

---

## Key Concepts

### Non-IID Data Partitioning

In reality, different banks see very different fraud profiles — a retail bank sees card-present fraud, a corporate bank sees invoice fraud, a tax authority sees evasion patterns. We simulate this using **Dirichlet sampling** with parameter α:

```
α = 0.5 (default)  →  strongly non-IID  (realistic)
α = 1.0            →  moderately heterogeneous
α → ∞              →  IID (all banks see the same distribution)
```

### FedAvg with Fairness Weighting

Standard FedAvg weights clients by dataset size — large banks dominate. Our variant blends in an inverse-F1 term:

```
w_i = (1 - α) × (n_i / N)  +  α × (1/F1_i / Σ 1/F1_j)

α = 0.3  →  30% fairness weight, 70% size weight
```

This ensures smaller banks with rare fraud patterns still contribute meaningfully.

### Byzantine Fault Tolerance (Krum)

We simulate one adversarial bank (label-flipped data) in the final training rounds. The **Krum filter** scores each client by its distance to all other clients and excludes the most divergent one per round.

### Differential Privacy

After aggregation, Gaussian noise σ is added to the global weights:

```
global_weights += N(0, σ²)
```

This provides (ε, δ)-differential privacy guarantees, preventing reverse-engineering of individual transactions from the published model.

---

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--banks` | `6` | Number of federated clients |
| `--rounds` | `10` | Global communication rounds |
| `--epochs` | `15` | Local training epochs per round |
| `--alpha` | `0.5` | Dirichlet α — controls non-IID degree |
| `--dp-sigma` | `0.003` | DP noise standard deviation |

---

## Results

Sample results on the Kaggle Credit Card Fraud dataset (6 banks, 10 rounds, 15 local epochs):

| Metric | Score |
|--------|-------|
| Accuracy | ~0.91 |
| Precision (Fraud) | ~0.78 |
| Recall (Fraud) | ~0.82 |
| F1 (Fraud) | ~0.80 |
| AUC-ROC | ~0.94 |

> Results vary by run due to DP noise and random partitioning. Re-run with `--synthetic` for fully reproducible benchmarks.

---

## Implementation Notes

- **Pure NumPy** — no PyTorch or TensorFlow dependency. The MLP is implemented from scratch to make the FL mechanics fully transparent.
- **FL-compatible interface** — any model implementing `get_weights()` / `set_weights()` can be dropped into this framework.
- **Modular** — swap in a different aggregation strategy, model, or security mechanism by editing `src/federated_learning.py`.

---

## References

- McMahan et al. (2017) — *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg)
- Blanchard et al. (2017) — *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent* (Krum)
- Dwork & Roth (2014) — *The Algorithmic Foundations of Differential Privacy*
- Li et al. (2020) — *Fair Resource Allocation in Federated Learning*

---

## License

MIT — see [LICENSE](LICENSE) for details.
