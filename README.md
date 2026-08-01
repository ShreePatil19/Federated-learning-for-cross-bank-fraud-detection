# Federated Learning for Cross-Bank Fraud Detection

Privacy-preserving fraud detection across multiple banks using Federated Learning, Mixture-of-Experts, and Differential Privacy. No raw transaction data ever leaves a client.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

UTS Masters of AI, Neural Networks and Fuzzy Logic (49275), Group 17.2, Autumn 2026.

## What This Project Does

Traditional fraud detection requires pooling transaction data in a central server, which is legally and ethically problematic across separate financial institutions. This project benchmarks **Federated Learning** strategies where:

- Each bank trains a fraud detection model **locally on its own data**
- Only **model weights** are shared with a central aggregation server
- Raw transactions **never leave the originating institution**
- A **Mixture-of-Experts (MoE)** ensemble combines per-bank expert models with FL-trained global models

We evaluate **6 FL algorithms** (FedAvg, FedProx, FedNova, SCAFFOLD, Per-FedAvg, FedAvgM) across **5 datasets** with **3 non-IID heterogeneity levels** (alpha = 0.05, 0.1, 0.5), alongside centralized ML baselines (XGBoost, LightGBM, CatBoost, Random Forest, Logistic Regression).

## Datasets

| Dataset | Source | Transactions | Fraud Rate |
|---------|--------|-------------|------------|
| ULB Credit Card | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | 284,807 | 0.17% |
| PaySim | [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) | 6.3M | 0.13% |
| IEEE-CIS | [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection) | 590,540 | 3.5% |
| IBM AML (Hi/Li, S/M/L) | [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) | varies | varies |
| SAML | Synthetic AML | varies | varies |

All datasets are gitignored. Download from the linked sources and place in `data/`.

## Project Structure

```
.
├── src/                          # Core FL engine
│   ├── federated_learning.py     #   FraudDetectorMLP, FederatedClient, FederatedServer
│   └── data_utils.py             #   Data loading, EDA, Dirichlet partitioning
│
├── triage/                       # Decision layer: auto-clear / flag / defer-to-human
│   ├── core.py                   #   calibration, federated conformal, cost rule, defer
│   ├── integration.py            #   capture sweep probabilities -> npz -> triage results
│   └── experiment.py             #   standalone compact-expert sweep (resumable)
│
├── notebooks/
│   ├── FL_algorithms/            # One notebook per FL algorithm (ULB dataset)
│   │   ├── fedavg_logistic_regression.ipynb
│   │   ├── fedprox_neural_network.ipynb
│   │   ├── fednova_resnet_tabular_v1.ipynb
│   │   ├── scaffold_tabnet.ipynb
│   │   └── ...
│   ├── MOE_experiments/          # Mixture-of-Experts experiment notebooks
│   │   ├── moe_ds_merged_final.ipynb     # <- canonical MoE results
│   │   ├── moe_ds_separate.ipynb
│   │   └── seed_runs/                    # Multi-seed reproducibility runs
│   ├── exploration.ipynb
│   ├── lr_60epoch_check.ipynb
│   └── resnet_60epoch_check.ipynb
│
├── IBM/                          # IBM AML dataset experiments (LiYike0225)
│   ├── IBM_HI_Large&Li_Large/
│   ├── IBM_Hi_Medium/
│   └── IBM_Hi_small&Li_small/
│
├── Paysim_dataset_training/      # PaySim dataset experiments (LiYike0225)
│   ├── basic nn + FedAvg/
│   └── paysim-moe-benchmark-alpha-sweep.ipynb
│
├── IEEE-CIS-MoE/                 # IEEE-CIS MoE benchmark (Taasnuba)
│   ├── plots/
│   └── results/
│
├── ieee-cis-fraud-detection/     # IEEE-CIS FL experiments (Taasnuba)
│   ├── fl_fraud_detection.ipynb
│   └── plot_*.png
│
├── banksim/                      # BankSim experiments (nordonezc)
│   └── fraud_detection.ipynb
│
├── SAML/                         # Synthetic AML FL experiments (nordonezc)
│   ├── fedavg_p.ipynb
│   ├── fedprox.ipynb
│   ├── fednova.ipynb
│   ├── scaffold.ipynb
│   └── persfl.ipynb
│
├── scripts/                      # Training scripts
│   ├── kaggle/                   #   GROUP-A sweep as 12h-session Kaggle scripts (one per dataset)
│   ├── v1/                       #   Original
│   ├── v2/                       #   Updated
│   └── v3/                       #   Report generation
│
├── results/                      # FL benchmark results (CSV + JSON summaries)
│   ├── FL_LR/
│   └── FL_MLP/
│
├── moe_experiments/              # MoE GPU experiment outputs (plots)
│
├── reports/
│   ├── dashboards/               # Interactive HTML dashboards
│   │   └── moe_fl_streamlit/     # Streamlit demo app
│   ├── documents/                # PDF research reports
│   └── figures/
│
├── main.py                       # Entry point for ULB FL pipeline
├── requirements.txt
└── LICENSE
```

## Quickstart

```bash
git clone https://github.com/ShreePatil19/Federated-learning-for-cross-bank-fraud-detection.git
cd Federated-learning-for-cross-bank-fraud-detection
pip install -r requirements.txt

# Run the core FL pipeline (ULB Credit Card dataset)
python main.py --data data/creditcard.csv

# Or use synthetic data (no download needed)
python main.py --synthetic
```

For the Streamlit demo dashboard, see [reports/dashboards/moe_fl_streamlit/README.md](reports/dashboards/moe_fl_streamlit/README.md).

## Reproducing the GROUP-A Sweep on Kaggle

The canonical multi-seed sweep does not fit Kaggle's 12 h session cap in one go
(~42 h of GPU compute). [`scripts/kaggle/`](scripts/kaggle/README.md) ships
directly runnable scripts — one per dataset plus per-seed part scripts that each
finish inside ONE session — with order-independent partitioning (Fix E), automatic
session resume/merging, and a time-budget guard. Paste a script into a kernel,
attach its dataset, enable GPU, run; parts can run in parallel on separate
accounts, and the last part run with its siblings attached emits the full
multi-seed package. Sanity checklist: [`RERUN_CHECKLIST.md`](RERUN_CHECKLIST.md).

## Triage Decision Layer

On top of the scored experts, `triage/` turns per-bank probabilities into an **action** —
auto-clear / flag-for-SAR / defer-to-human — via per-expert calibration, a per-bank
conformal miss-rate guarantee, a cost-optimal flag threshold, and a budgeted defer rule.
The GROUP-A sweep notebook captures expert probabilities per combo (no retraining); the
decision layer runs over the captures. See [triage/README.md](triage/README.md).

```bash
python -m triage.experiment --dataset synthetic --quick   # 1-minute smoke test
```

## FL Algorithms Benchmarked

| Algorithm | Key Idea |
|-----------|----------|
| **FedAvg** | Weighted averaging of client model updates |
| **FedProx** | Adds proximal term to handle heterogeneity |
| **FedNova** | Normalizes updates by local steps |
| **SCAFFOLD** | Variance reduction via control variates |
| **Per-FedAvg** | Personalized FL with local fine-tuning |
| **FedAvgM** | FedAvg with server-side momentum |

Each algorithm is tested with three privacy/robustness configurations: No DP, Differential Privacy, and Gradient Sparsification.

## Key Concepts

### Non-IID Data Partitioning

Different banks see different fraud profiles. We simulate this using **Dirichlet sampling** with parameter alpha:

- alpha = 0.05: strongly non-IID (realistic, high heterogeneity)
- alpha = 0.1: moderately non-IID
- alpha = 0.5: mildly non-IID

### Mixture of Experts (MoE)

The MoE ensemble trains per-bank expert models (XGBoost, LightGBM, CatBoost) alongside FL-trained global models, then combines their predictions via a learned gating network. This captures both bank-specific fraud patterns and cross-bank generalizations.

### Security Mechanisms

| Mechanism | Purpose |
|-----------|---------|
| **Gradient Clipping** | Limits update magnitude, prevents any single client from dominating |
| **Krum Filter** | Detects and excludes Byzantine (adversarial) clients |
| **Differential Privacy** | Adds calibrated Gaussian noise, protects individual records |

## Contributors

This is a group project. Each member contributed specific components:

| Contributor | GitHub | Contributions |
|-------------|--------|---------------|
| **Yike Li** | [@LiYike0225](https://github.com/LiYike0225) | IBM AML dataset experiments (Hi/Li, S/M/L variants), PaySim dataset training, basic NN + FedAvg implementation, model benchmarking |
| **Nicolas Ordonez** | [@nordonezc](https://github.com/nordonezc) | BankSim ML models and FL experiments, SCAFFOLD comparison, SAML (5 FL + 3 ML) experiments, IEEE fraud detection initial work |
| **Taasnuba** | [@Taasnuba](https://github.com/Taasnuba) | IEEE-CIS fraud detection FL experiments (FedAvg, FedProx, FedNova, PersFL with MLP), IEEE-CIS MoE benchmark |
| **Shreeshailya Patil** | [@ShreePatil19](https://github.com/ShreePatil19) | Project structure, MoE experiment notebooks, Streamlit demo dashboard, multi-seed reproducibility runs, research reports |

## License

MIT. See [LICENSE](LICENSE) for details.
