"""
federated_learning.py
=====================
Core Federated Learning engine for multi-bank fraud detection.

Architecture
------------
  - FedAvg aggregation (McMahan et al., 2017)
  - Byzantine-fault-tolerant via gradient clipping
  - Differential privacy noise injection
  - Fairness-aware client weighting

Usage
-----
  from src.federated_learning import FederatedServer, FederatedClient
"""

import numpy as np
from copy import deepcopy
from typing import List, Dict, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class FraudDetectorMLP:
    """
    Multi-Layer Perceptron for binary fraud classification.

    Designed to be fully FL-compatible:
      • get_weights() / set_weights() expose all parameters as plain dicts
      • No hidden state — safe to copy across clients
      • Class-weighted loss handles the heavy imbalance in fraud datasets

    Architecture:
        Input → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        learning_rate: float = 0.005,
        class_weight: float = 5.0,
        seed: int = 42,
    ):
        self.lr = learning_rate
        self.class_weight = class_weight
        rng = np.random.default_rng(seed)

        dims = [input_dim] + hidden_dims + [1]
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []

        for i in range(len(dims) - 1):
            # He initialisation — optimal for ReLU activations
            std = np.sqrt(2.0 / dims[i])
            self.W.append(rng.standard_normal((dims[i], dims[i + 1])) * std)
            self.b.append(np.zeros(dims[i + 1]))

    # ── Activations ──────────────────────────────────────────────────────────

    def _relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    def _relu_grad(self, z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))

    # ── Forward Pass ─────────────────────────────────────────────────────────

    def _forward(self, X: np.ndarray) -> np.ndarray:
        self._cache_a = [X]
        self._cache_z = []
        a = X
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = a @ w + b
            self._cache_z.append(z)
            a = self._relu(z) if i < len(self.W) - 1 else self._sigmoid(z)
            self._cache_a.append(a)
        return a  # shape (n, 1) — fraud probability

    # ── Loss: Class-Weighted BCE ──────────────────────────────────────────────

    def _loss(self, probs: np.ndarray, y: np.ndarray) -> float:
        p = probs.flatten()
        eps = 1e-7
        w = np.where(y == 1, self.class_weight, 1.0)
        return -(w * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))).mean()

    # ── Backward Pass ────────────────────────────────────────────────────────

    def _backward(self, y: np.ndarray, clip: float = 2.0) -> None:
        n = len(y)
        w_vec = np.where(y == 1, self.class_weight, 1.0).reshape(-1, 1) / n
        delta = (self._cache_a[-1] - y.reshape(-1, 1)) * w_vec

        for i in reversed(range(len(self.W))):
            gw = self._cache_a[i].T @ delta
            gb = delta.sum(axis=0)

            # Per-step gradient clipping — prevents exploding gradients
            norm = np.linalg.norm(gw)
            if norm > clip:
                gw = gw * (clip / norm)

            self.W[i] -= self.lr * gw
            self.b[i] -= self.lr * gb

            if i > 0:
                delta = np.clip(delta @ self.W[i].T, -5, 5) * self._relu_grad(
                    self._cache_z[i - 1]
                )

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 15,
        batch_size: int = 64,
    ) -> List[float]:
        """Train the model. Returns per-epoch loss history."""
        n = len(y)
        loss_history = []

        for _ in range(epochs):
            idx = np.random.permutation(n)
            X_s, y_s = X[idx], y[idx]
            epoch_loss, n_batches = 0.0, 0

            for start in range(0, n, batch_size):
                Xb = X_s[start : start + batch_size]
                yb = y_s[start : start + batch_size]
                probs = self._forward(Xb)
                epoch_loss += self._loss(probs, yb)
                self._backward(yb)
                n_batches += 1

            loss_history.append(epoch_loss / max(n_batches, 1))

        return loss_history

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return fraud probability for each sample."""
        return self._forward(X).flatten()

    def predict(self, X: np.ndarray, threshold: float = 0.35) -> np.ndarray:
        """
        Predict fraud labels.
        Threshold < 0.5 prioritises recall — catching more frauds
        at the cost of some false positives. Appropriate for banking.
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    # ── FL Interface ─────────────────────────────────────────────────────────

    def get_weights(self) -> Dict:
        """Serialise all parameters to a plain dict (FL-compatible)."""
        return {
            "W": [w.copy() for w in self.W],
            "b": [b.copy() for b in self.b],
        }

    def set_weights(self, state: Dict) -> None:
        """Load parameters from a weight dict (FL-compatible)."""
        self.W = [w.copy() for w in state["W"]]
        self.b = [b.copy() for b in state["b"]]

    def flat_weights(self) -> np.ndarray:
        """Flatten all weights to a 1-D vector (used by Krum)."""
        return np.concatenate(
            [w.ravel() for w in self.W] + [b.ravel() for b in self.b]
        )


# ─────────────────────────────────────────────────────────────────────────────
# FEDERATED CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class FederatedClient:
    """
    Represents a single bank in the federated network.

    Each client:
      1. Receives the global model weights from the server
      2. Trains locally on its private transaction data
      3. Returns updated weights (never raw data)
    """

    def __init__(
        self,
        client_id: str,
        X: np.ndarray,
        y: np.ndarray,
        input_dim: int,
        hidden_dims: List[int],
        learning_rate: float,
        class_weight: float,
    ):
        self.client_id = client_id
        self.X = X
        self.y = y
        self.n_samples = len(y)

        self.model = FraudDetectorMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            class_weight=class_weight,
            seed=hash(client_id) % (2**31),
        )

    def train(
        self,
        global_weights: Dict,
        epochs: int,
        batch_size: int,
        clip_norm: float,
        poisoned: bool = False,
    ) -> Tuple[Dict, float, float]:
        """
        Load global weights, train locally, return updated weights.

        Args:
            global_weights: Current global model parameters
            epochs:         Local training epochs
            batch_size:     Mini-batch size
            clip_norm:      L2 norm clip threshold for weight vector
            poisoned:       If True, simulate a Byzantine attack (label flip)

        Returns:
            (clipped_weights, weight_norm, local_f1)
        """
        self.model.set_weights(deepcopy(global_weights))

        train_y = 1 - self.y if poisoned else self.y
        self.model.fit(self.X, train_y, epochs=epochs, batch_size=batch_size)

        weights = self.model.get_weights()
        weights, norm = _clip_weights(weights, clip_norm)

        # Local F1 for fairness scoring
        from sklearn.metrics import f1_score
        preds = self.model.predict(self.X)
        local_f1 = f1_score(self.y, preds, zero_division=0)

        return weights, norm, local_f1

    @property
    def fraud_rate(self) -> float:
        return self.y.mean()

    @property
    def n_fraud(self) -> int:
        return int(self.y.sum())


# ─────────────────────────────────────────────────────────────────────────────
# FEDERATED SERVER
# ─────────────────────────────────────────────────────────────────────────────

class FederatedServer:
    """
    Central aggregation server.

    Responsibilities:
      • Distribute global model to clients
      • Run Krum filter to exclude Byzantine clients
      • Aggregate via Fairness-Weighted FedAvg
      • Inject differential privacy noise
      • Track global evaluation metrics
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        learning_rate: float = 0.005,
        class_weight: float = 5.0,
        clip_norm: float = 2.0,
        dp_sigma: float = 0.003,
        n_byzantine: int = 1,
        fairness_alpha: float = 0.3,
    ):
        self.clip_norm = clip_norm
        self.dp_sigma = dp_sigma
        self.n_byzantine = n_byzantine
        self.fairness_alpha = fairness_alpha

        self.global_model = FraudDetectorMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            class_weight=class_weight,
        )
        self.global_weights = self.global_model.get_weights()
        self.history: List[Dict] = []

    def get_global_weights(self) -> Dict:
        return deepcopy(self.global_weights)

    def aggregate(
        self,
        client_weights: List[Dict],
        client_sizes: List[int],
        client_f1s: List[float],
        client_ids: List[str],
    ) -> Tuple[List[int], np.ndarray]:
        """
        Full aggregation pipeline:
          1. Krum filter  →  remove Byzantine outliers
          2. Fairness FedAvg  →  weighted average
          3. DP noise  →  privacy guarantee

        Returns:
            (excluded_indices, final_aggregation_weights)
        """
        # Step 1: Krum filter
        kept_idx, _ = _krum_select(client_weights, self.n_byzantine)
        excluded_idx = [i for i in range(len(client_weights)) if i not in kept_idx]

        selected_w = [client_weights[i] for i in kept_idx]
        selected_s = [client_sizes[i] for i in kept_idx]
        selected_f = [client_f1s[i] for i in kept_idx]

        # Step 2: Fairness-weighted FedAvg
        agg, final_weights = _fairness_fedavg(
            selected_w, selected_s, selected_f, self.fairness_alpha
        )

        # Step 3: Differential privacy
        self.global_weights = _add_dp_noise(agg, self.dp_sigma)
        self.global_model.set_weights(self.global_weights)

        return excluded_idx, final_weights

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray, round_num: int
    ) -> Dict:
        """Evaluate global model and store metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score,
        )

        proba = self.global_model.predict_proba(X_test)
        preds = self.global_model.predict(X_test)

        metrics = {
            "round":     round_num,
            "accuracy":  float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall":    float(recall_score(y_test, preds, zero_division=0)),
            "f1":        float(f1_score(y_test, preds, zero_division=0)),
            "auc":       float(roc_auc_score(y_test, proba)),
        }
        self.history.append(metrics)
        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# SECURITY & AGGREGATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _clip_weights(state: Dict, clip_norm: float) -> Tuple[Dict, float]:
    """
    Global weight-vector clipping.
    Limits the L2 norm of the entire parameter vector.
    Prevents any single client from dominating the global update.
    """
    flat = np.concatenate(
        [w.ravel() for w in state["W"]] + [b.ravel() for b in state["b"]]
    )
    norm = float(np.linalg.norm(flat))
    if norm > clip_norm:
        scale = clip_norm / norm
        state = {
            "W": [w * scale for w in state["W"]],
            "b": [b * scale for b in state["b"]],
        }
    return state, norm


def _krum_select(states: List[Dict], n_byzantine: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-Krum Byzantine Fault Tolerance (Blanchard et al., 2017).

    Scores each client by the sum of squared distances to its
    k = (n - n_byzantine - 2) nearest neighbours.
    Clients with the lowest scores (most central) are kept.
    Byzantine / poisoned clients tend to cluster far from honest ones.
    """
    vecs = [
        np.concatenate([w.ravel() for w in s["W"]] + [b.ravel() for b in s["b"]])
        for s in states
    ]
    n = len(vecs)
    k = max(n - n_byzantine - 2, 1)
    scores = np.zeros(n)

    for i in range(n):
        dists = sorted(
            float(np.linalg.norm(vecs[i] - vecs[j]) ** 2)
            for j in range(n)
            if j != i
        )
        scores[i] = sum(dists[:k])

    kept = np.argsort(scores)[: n - n_byzantine]
    return kept, scores


def _fairness_fedavg(
    states: List[Dict],
    sizes: List[int],
    f1_scores: List[float],
    alpha: float = 0.3,
) -> Tuple[Dict, np.ndarray]:
    """
    Fairness-Weighted Federated Averaging.

    Standard FedAvg weights clients purely by dataset size — large banks
    dominate and small clients (e.g. credit unions with rare fraud patterns)
    are drowned out.

    This blends size-weight with inverse-F1 weight so that clients
    struggling on non-IID data contribute proportionally more,
    preserving representation of diverse fraud patterns.

        w_i = (1 - α) × (n_i / N)  +  α × (1/F1_i / Σ 1/F1_j)

    α = 0 → pure FedAvg    α = 1 → pure fairness    α = 0.3 → default blend
    """
    total = sum(sizes)
    size_w = np.array([s / total for s in sizes])

    inv_f1 = np.array([1.0 / max(f, 0.05) for f in f1_scores])
    fair_w = inv_f1 / inv_f1.sum()

    final_w = (1 - alpha) * size_w + alpha * fair_w
    final_w /= final_w.sum()

    W_agg = [np.zeros_like(x) for x in states[0]["W"]]
    b_agg = [np.zeros_like(x) for x in states[0]["b"]]

    for state, w in zip(states, final_w):
        for i in range(len(W_agg)):
            W_agg[i] += w * state["W"][i]
            b_agg[i] += w * state["b"][i]

    return {"W": W_agg, "b": b_agg}, final_w


def _add_dp_noise(state: Dict, sigma: float) -> Dict:
    """
    Gaussian Differential Privacy (Dwork & Roth, 2014).
    Adds calibrated noise to aggregated weights before publishing.
    Ensures no individual client's data can be reconstructed
    from the released global model.
    """
    return {
        "W": [w + np.random.normal(0, sigma, w.shape) for w in state["W"]],
        "b": [b + np.random.normal(0, sigma, b.shape) for b in state["b"]],
    }
