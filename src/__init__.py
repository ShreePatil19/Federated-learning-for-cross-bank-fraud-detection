# src/__init__.py
from .federated_learning import FraudDetectorMLP, FederatedClient, FederatedServer
from .data_utils import (
    load_creditcard_dataset,
    load_synthetic_dataset,
    dirichlet_partition,
    compute_eda_stats,
)
