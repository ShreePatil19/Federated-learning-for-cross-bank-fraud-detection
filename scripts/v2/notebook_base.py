"""
═══════════════════════════════════════════════════════════════════
FL Benchmarking — BASE (shared across all model notebooks)
Dataset  : ULB Credit Card Fraud (creditcard.csv)
Contains : Config, HP, data loading, all model classes, helpers,
           privacy mechanisms, local training, metrics, checkpointing
═══════════════════════════════════════════════════════════════════
"""

# ── HYPERPARAMETERS ────────────────────────────────────────────────
HP = {
    "N_BANKS"        : 5,
    "FL_ROUNDS"      : 60,
    "LOCAL_EPOCHS"   : 3,
    "FINETUNE_EPOCHS": 5,
    "LOCAL_STEPS"    : 25,
    "BATCH_SIZE"     : 128,
    "LR"             : 0.001,
    "FINETUNE_LR"    : 0.0003,
    "FEDPROX_MU"     : 0.01,
    "FEDNOVA_RHO"    : 0.9,
    "DIRICHLET_ALPHA": 0.4,
    "DP_CLIP_NORM"   : 1.0,
    "DP_NOISE_MULT"  : 1.1,
    "DP_EPSILON"     : 4.0,
    "TOPK_RATIO"     : 0.01,
    "RANDOM_STATE"   : 42,
    "SAVE_CKPT_EVERY": 10,
    "PRINT_EVERY"    : 1,
}

import os, json, time, copy, warnings, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve,
                             f1_score, precision_score, recall_score)
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

# FIX 1: Surface real CUDA errors immediately instead of getting
# async/misleading stack traces
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

OUTPUT_ROOT = "outputs"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

ALL_FL      = ["FedAvg", "FedProx", "SCAFFOLD", "FedNova", "PersFL"]
ALL_PRIVACY = ["NoDP", "DP", "Sparsification"]


# FIX 1 (cont): Validate CUDA arch with an actual kernel call before
# committing to it — catches cudaErrorNoKernelImageForDevice at startup
# instead of mid-run.
def _get_device():
    if not torch.cuda.is_available():
        print("No CUDA device found — using CPU")
        return torch.device("cpu")
    try:
        _t = torch.zeros(1).cuda()
        _t = _t + _t   # forces a real kernel dispatch
        del _t
        torch.cuda.empty_cache()
        return torch.device("cuda")
    except RuntimeError as e:
        print(f"⚠  CUDA kernel validation failed ({e})")
        print("   Falling back to CPU for all runs.")
        return torch.device("cpu")

DEVICE = _get_device()

if torch.cuda.is_available():
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Device : {DEVICE}")
print(f"Output : {OUTPUT_ROOT}/")


# ══════════════════════════════════════════════════════════════════
#  DATA LOADING & SPLITTING
# ══════════════════════════════════════════════════════════════════

def load_data(path="creditcard.csv"):
    print("\nLoading dataset...")
    df  = pd.read_csv(path)
    X   = df.drop("Class", axis=1).values.astype(np.float32)
    y   = df["Class"].values
    sc  = StandardScaler()
    X   = sc.fit_transform(X)
    print(f"  Total records : {len(df):,}")
    print(f"  Fraud rate    : {y.mean()*100:.2f}%")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=HP["RANDOM_STATE"]
    )
    return X_tr, X_te, y_tr, y_te


def non_iid_split(X, y, n=5, seed=42):
    rng        = np.random.default_rng(seed)
    fi         = np.where(y == 1)[0]
    li         = np.where(y == 0)[0]
    splits     = np.round(
        rng.dirichlet(np.ones(n) * HP["DIRICHLET_ALPHA"]) * len(fi)
    ).astype(int)
    splits[-1] = len(fi) - splits[:-1].sum()
    banks, f0  = [], 0
    lpp        = len(li) // n
    for i in range(n):
        fe  = f0 + splits[i]
        idx = np.concatenate([fi[f0:fe], li[i*lpp:(i+1)*lpp]])
        rng.shuffle(idx)
        banks.append((X[idx], y[idx]))
        f0 = fe
    print(f"\nNon-IID split (Dirichlet α={HP['DIRICHLET_ALPHA']}):")
    for i, (xb, yb) in enumerate(banks):
        print(f"  Bank {i+1}: {len(xb):>6,} records | fraud {yb.mean()*100:.2f}%")
    return banks


def smote_bank(X, y):
    """SMOTE capped at 10k rows to prevent memory explosion."""
    try:
        k        = min(3, int(y.sum()) - 1)
        sm       = SMOTE(random_state=HP["RANDOM_STATE"], k_neighbors=k)
        X_r, y_r = sm.fit_resample(X, y)
        if len(X_r) > 10000:
            idx      = np.random.choice(len(X_r), 10000, replace=False)
            X_r, y_r = X_r[idx], y_r[idx]
        return X_r, y_r
    except Exception:
        return X, y


def make_loader(X, y):
    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.float32)),
        batch_size=HP["BATCH_SIZE"], shuffle=True
    )


# ══════════════════════════════════════════════════════════════════
#  ML MODELS
# ══════════════════════════════════════════════════════════════════

class LRWrapper:
    def __init__(self, n_features):
        self.model = LogisticRegression(
            max_iter=1000, random_state=HP["RANDOM_STATE"],
            class_weight="balanced", solver="lbfgs", warm_start=True
        )
        self._fitted    = False
        self.n_features = n_features

    def fit(self, X, y):
        self.model.fit(X, y)
        self._fitted = True

    def predict_proba(self, X):
        if not self._fitted:
            return np.full(len(X), 0.5)
        return self.model.predict_proba(X)[:, 1]

    def get_params(self):
        if not self._fitted:
            return {"coef": np.zeros((1, self.n_features)), "intercept": np.zeros(1)}
        return {"coef": self.model.coef_.copy(), "intercept": self.model.intercept_.copy()}

    def set_params(self, params):
        if not self._fitted:
            X_d = np.random.randn(10, self.n_features)
            y_d = np.array([0]*9 + [1])
            self.model.fit(X_d, y_d)
            self._fitted = True
        self.model.coef_      = params["coef"].copy()
        self.model.intercept_ = params["intercept"].copy()

    def copy(self):
        new = LRWrapper(self.n_features)
        if self._fitted:
            new.set_params(self.get_params())
            new._fitted = True
        return new


class MLP(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),  nn.ReLU(),
            nn.Linear(32, 1),   nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).squeeze()


class TabNet(nn.Module):
    def __init__(self, in_dim, n_steps=3, n_dim=64):
        super().__init__()
        self.n_steps   = n_steps
        self.step_attn = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, in_dim), nn.Softmax(dim=1))
            for _ in range(n_steps)
        ])
        self.step_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, n_dim), nn.LayerNorm(n_dim), nn.ReLU(),
                nn.Linear(n_dim, n_dim),  nn.LayerNorm(n_dim), nn.ReLU()
            )
            for _ in range(n_steps)
        ])
        self.final = nn.Sequential(nn.Linear(n_dim, 1), nn.Sigmoid())

    def forward(self, x):
        h = torch.zeros(x.size(0), self.step_fc[0][-2].normalized_shape[0], device=x.device)
        for i in range(self.n_steps):
            h = h + self.step_fc[i](x * self.step_attn[i](x))
        return self.final(h / self.n_steps).squeeze()


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim*2), nn.LayerNorm(dim*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim*2, dim), nn.LayerNorm(dim)
        )
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.block(x))


class ResNet(nn.Module):
    def __init__(self, in_dim, hidden=64, n_blocks=2):
        super().__init__()
        self.proj   = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.head   = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(),
                                    nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, x): return self.head(self.blocks(self.proj(x))).squeeze()


def build_model(model_name, n_features):
    if model_name == "LR":
        return LRWrapper(n_features)
    elif model_name == "MLP":
        return MLP(n_features).to(DEVICE)
    elif model_name == "TabNet":
        return TabNet(n_features).to(DEVICE)
    elif model_name == "ResNet":
        return ResNet(n_features).to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ══════════════════════════════════════════════════════════════════
#  PRIVACY MECHANISMS
# ══════════════════════════════════════════════════════════════════

def apply_privacy(model, privacy_mode, n_samples=1):
    if privacy_mode == "NoDP":
        return
    elif privacy_mode == "DP":
        total_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5
        clip_coef  = HP["DP_CLIP_NORM"] / max(total_norm, HP["DP_CLIP_NORM"])
        noise_std  = HP["DP_NOISE_MULT"] * HP["DP_CLIP_NORM"] / max(n_samples, 1)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
                p.grad.data.add_(torch.randn_like(p.grad.data) * noise_std)
    elif privacy_mode == "Sparsification":
        for p in model.parameters():
            if p.grad is not None:
                grad_flat = p.grad.data.abs().flatten()
                k         = max(1, int(len(grad_flat) * HP["TOPK_RATIO"]))
                threshold = torch.topk(grad_flat, k).values[-1]
                mask      = p.grad.data.abs() >= threshold
                p.grad.data.mul_(mask.float())


# ══════════════════════════════════════════════════════════════════
#  LOCAL TRAINING
# ══════════════════════════════════════════════════════════════════

def local_train_nn(model, loader, epochs, lr, privacy_mode,
                   global_model=None, mu=0.0,
                   c_local=None, c_global=None,
                   use_steps=False, n_steps=25):
    model.train()
    optimizer  = optim.Adam(model.parameters(), lr=lr)
    criterion  = nn.BCELoss()
    dev        = next(model.parameters()).device
    aux        = {}

    if use_steps:
        global_state = copy.deepcopy(model.state_dict())
        data_iter    = iter(loader)
        for step in range(n_steps):
            try:
                X_b, y_b = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                X_b, y_b = next(data_iter)
            X_b, y_b = X_b.to(dev), y_b.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            apply_privacy(model, privacy_mode, n_samples=len(X_b))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        tau       = n_steps
        local_st  = model.state_dict()
        norm_grad = {}
        for key in local_st:
            if local_st[key].dtype == torch.float32:
                norm_grad[key] = (local_st[key] - global_state[key].to(dev)) / tau
            else:
                norm_grad[key] = local_st[key].clone()
        del global_state
        aux = {"norm_grad": norm_grad, "tau": tau}

    else:
        c_l        = [cv.to(dev) for cv in c_local]  if c_local  else None
        c_g        = [cv.to(dev) for cv in c_global] if c_global else None
        step_count = 0

        # FIX 4: Capture initial weights BEFORE training so we can compute
        # the real SCAFFOLD control variate delta (w0 - w_t).
        # The original code had: torch.zeros_like(cl) / (step_count * lr)
        # which is always 0, making SCAFFOLD equivalent to plain FedAvg.
        w0 = {k: v.clone() for k, v in model.state_dict().items()} if c_l is not None else None

        for _ in range(epochs):
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(dev), y_b.to(dev)
                optimizer.zero_grad()
                pred = model(X_b)
                loss = criterion(pred, y_b)
                if global_model is not None and mu > 0:
                    prox = sum(
                        ((p - pg.detach())**2).sum()
                        for p, pg in zip(model.parameters(), global_model.parameters())
                    )
                    loss = loss + (mu / 2) * prox
                loss.backward()
                if c_l is not None and c_g is not None:
                    for param, cl, cg in zip(model.parameters(), c_l, c_g):
                        if param.grad is not None:
                            param.grad.data.add_(cg - cl)
                apply_privacy(model, privacy_mode, n_samples=len(X_b))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                step_count += 1

        if c_l is not None:
            # FIX 4 (cont): Correct SCAFFOLD formula:
            # c_i+ = c_i - c  +  (1 / (step_count * lr)) * (w0 - w_t)
            param_names = [n for n, _ in model.named_parameters()]
            w_t         = model.state_dict()
            new_c = [
                cl - cg + (1.0 / max(step_count * lr, 1e-12)) * (
                    w0[k].to(dev) - w_t[k].to(dev)
                )
                for cl, cg, k in zip(c_l, c_g, param_names)
            ]
            aux = {"new_c_local": [c.cpu() for c in new_c]}

    return model, aux


def local_train_lr(lr_wrapper, X, y):
    if len(np.unique(y)) < 2:
        return lr_wrapper
    X_r, y_r = smote_bank(X, y)
    if len(np.unique(y_r)) < 2:
        return lr_wrapper
    lr_wrapper.fit(X_r, y_r)
    return lr_wrapper


# ══════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════

# FIX 3: Use the model's actual device instead of the global DEVICE.
# If a combo falls back to CPU while DEVICE is still "cuda", the old
# code would send tensors to CUDA while the model lives on CPU → crash.
def get_probs_nn(model, X):
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        return model(
            torch.tensor(X, dtype=torch.float32).to(dev)
        ).cpu().numpy()


def compute_metrics(y_true, y_prob, k_pct=0.005):
    pred        = (y_prob >= 0.5).astype(int)
    auc         = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    f1          = f1_score(y_true, pred, zero_division=0)
    prec        = precision_score(y_true, pred, zero_division=0)
    rec         = recall_score(y_true, pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks          = float(np.max(tpr - fpr))
    idx         = np.where(fpr <= 0.01)[0]
    r1fpr       = float(tpr[idx[-1]]) if len(idx) else 0.0
    K           = max(1, int(len(y_true) * k_pct))
    pk          = float(y_true[np.argsort(y_prob)[::-1][:K]].mean())
    return dict(auc=round(auc,6), f1=round(f1,6), precision=round(prec,6),
                recall=round(rec,6), ks_stat=round(ks,6),
                recall_1fpr=round(r1fpr,6), prec_at_k=round(pk,6))


def fairness_metrics(bank_aucs):
    aucs = [a for a in bank_aucs if a is not None]
    if len(aucs) < 2:
        return 0.0, 1.0
    sigma = float(np.std(aucs))
    jfi   = sum(aucs)**2 / (len(aucs) * sum(a**2 for a in aucs))
    return round(sigma, 6), round(jfi, 6)


# ══════════════════════════════════════════════════════════════════
#  CHECKPOINT & LOGGING
# ══════════════════════════════════════════════════════════════════

def combo_dir(fl, ml, priv, output_root=OUTPUT_ROOT):
    name = f"{fl}_{ml}_{priv}"
    path = os.path.join(output_root, name)
    os.makedirs(path, exist_ok=True)
    return path, name


def save_checkpoint(path, name, rnd, model, metrics, is_lr=False):
    if rnd % HP["SAVE_CKPT_EVERY"] != 0:
        return
    fname = os.path.join(path, f"checkpoint_{name}_round{rnd:03d}.pt")
    if is_lr:
        torch.save({"round": rnd, "metrics": metrics,
                    "model_params": model.get_params()}, fname)
    else:
        torch.save({"round": rnd, "metrics": metrics,
                    "model_state": model.state_dict()}, fname)


def find_latest_checkpoint(path, name):
    ckpts = sorted([
        f for f in os.listdir(path)
        if f.startswith(f"checkpoint_{name}_round") and f.endswith(".pt")
    ])
    if not ckpts:
        return 0, None
    latest = os.path.join(path, ckpts[-1])
    ckpt   = torch.load(latest, map_location="cpu", weights_only=False)
    rnd    = ckpt["round"]
    print(f"  Resuming from checkpoint: round {rnd}")
    return rnd, ckpt


def append_csv_row(csv_path, row_dict, write_header=False):
    pd.DataFrame([row_dict]).to_csv(
        csv_path, mode="a",
        header=write_header or not os.path.exists(csv_path),
        index=False
    )


def save_summary(path, name, best_round, best_auc, best_f1, final_metrics, bank_aucs_final):
    summary = {
        "combination"      : name,
        "best_round"       : best_round,
        "best_auc"         : best_auc,
        "best_f1"          : best_f1,
        "final_metrics"    : final_metrics,
        "bank_aucs_final"  : bank_aucs_final,
    }
    with open(os.path.join(path, f"summary_{name}.json"), "w") as f:
        json.dump(summary, f, indent=2)


# FIX 2: Guard against an empty file left by a previously crashed run
# (os.path.exists would return True but the file has 0 bytes, causing
# the header to be skipped and pandas to write a headerless CSV).
def append_master(row, master_csv):
    os.makedirs(os.path.dirname(os.path.abspath(master_csv)), exist_ok=True)
    write_header = not os.path.exists(master_csv) or os.path.getsize(master_csv) == 0
    pd.DataFrame([row]).to_csv(master_csv, mode="a", header=write_header, index=False)
