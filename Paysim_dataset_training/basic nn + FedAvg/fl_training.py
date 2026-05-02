import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (f1_score, precision_score, 
                             recall_score, accuracy_score, 
                             roc_auc_score)
import joblib

# ── 1. Load saved data ────────────────────────────────────────
print("Loading data...")
# 用原始训练数据，不用SMOTE版本
X_train_res = np.load('X_train_original.npy')
y_train_res = np.load('y_train_original.npy')
X_test_scaled = np.load('X_test_scaled.npy')
y_test = np.load('y_test.npy')

# ── 2. Model architecture ─────────────────────────────────────
def build_model(input_dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
    )
    return model

# ── 3. FL helper functions ────────────────────────────────────
def get_weights(model):
    return [layer.numpy() for layer in model.weights]

def set_weights(model, weights):
    for layer, w in zip(model.weights, weights):
        layer.assign(w)

def fedavg(client_weight_lists, client_sizes):
    total = sum(client_sizes)
    avg_weights = []
    for layer_idx in range(len(client_weight_lists[0])):
        weighted = sum(
            (n / total) * cw[layer_idx]
            for cw, n in zip(client_weight_lists, client_sizes)
        )
        avg_weights.append(weighted)
    return avg_weights

def split_into_clients(X, y, n_clients=5):
    indices = np.random.permutation(len(y))
    chunks = np.array_split(indices, n_clients)
    return [(X[c], y[c]) for c in chunks]

# ── 4. Setup ──────────────────────────────────────────────────
INPUT_DIM = X_train_res.shape[1]
NUM_ROUNDS = 10
LOCAL_EPOCHS = 1   # increased from 3
N_CLIENTS = 5

print(f"\nSplitting into {N_CLIENTS} clients...")
clients = split_into_clients(X_train_res, y_train_res, n_clients=N_CLIENTS)
for i, (cx, cy) in enumerate(clients):
    print(f"  Client {i+1}: {len(cy)} samples, fraud rate: {cy.mean():.4%}")

# ── FIXED: Load pretrained model properly ─────────────────────
print("\nLoading pretrained model weights...")
base_model = tf.keras.models.load_model('paysim_model.keras')
global_weights = get_weights(base_model)
print("Pretrained weights loaded successfully!")

# Quick sanity check — first round eval before any FL training
eval_model = build_model(INPUT_DIM)
set_weights(eval_model, global_weights)
y_pred_prob = eval_model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int)
print(f"Sanity check (pretrained): Recall={recall_score(y_test, y_pred, zero_division=0):.4f} | "
      f"AUC={roc_auc_score(y_test, y_pred_prob):.4f}")

# ── 5. FL Training Loop ───────────────────────────────────────
round_results = []

for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\n--- Round {round_num}/{NUM_ROUNDS} ---")
    
    client_weights = []
    client_sizes = []
    
    for i, (client_X, client_y) in enumerate(clients):
        local_model = build_model(INPUT_DIM)
        set_weights(local_model, global_weights)  # load global weights
        
        local_model.fit(
            client_X, client_y,
            epochs=LOCAL_EPOCHS,
            batch_size=2048,
            class_weight={0: 1.0, 1: 100.0},  # 告诉模型欺诈样本更重要
            verbose=0
        )
        
        client_weights.append(get_weights(local_model))
        client_sizes.append(len(client_y))
        print(f"  Client {i+1} trained.")
    
    # FedAvg aggregation
    global_weights = fedavg(client_weights, client_sizes)
    
    # Evaluate global model
    eval_model = build_model(INPUT_DIM)
    set_weights(eval_model, global_weights)
    
    y_pred_prob = eval_model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_pred_prob)
    
    round_results.append({
        'round': round_num,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc
    })
    
    print(f"  → Accuracy: {acc:.4f} | Precision: {prec:.4f} | "
          f"Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

# ── 6. Save final global model ────────────────────────────────
final_model = build_model(INPUT_DIM)
set_weights(final_model, global_weights)
final_model.save('paysim_fl_global_model.keras')
print("\nFL global model saved!")

# ── 7. Results table ──────────────────────────────────────────
results_df = pd.DataFrame(round_results)
print("\n=== FL Results per Round ===")
print(results_df.to_string(index=False))

# ── 8. Plots ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(results_df['round'], results_df['f1'], 
             marker='o', label='F1', color='steelblue')
axes[0].plot(results_df['round'], results_df['recall'], 
             marker='s', label='Recall', color='tomato')
axes[0].plot(results_df['round'], results_df['precision'], 
             marker='^', label='Precision', color='green')
axes[0].axhline(y=0.95, color='tomato', linestyle='--', 
                alpha=0.5, label='Baseline Recall (0.95)')
axes[0].axhline(y=0.34, color='steelblue', linestyle='--', 
                alpha=0.5, label='Baseline F1 (0.34)')
axes[0].set_title('FL Global Model — Metrics per Round')
axes[0].set_xlabel('Round')
axes[0].set_ylabel('Score')
axes[0].legend()
axes[0].set_xticks(results_df['round'])

axes[1].plot(results_df['round'], results_df['auc'], 
             marker='o', color='purple', label='AUC-ROC')
axes[1].axhline(y=0.9989, color='purple', linestyle='--', 
                alpha=0.5, label='Baseline AUC (0.9989)')
axes[1].set_title('FL Global Model — AUC per Round')
axes[1].set_xlabel('Round')
axes[1].set_ylabel('AUC')
axes[1].legend()
axes[1].set_xticks(results_df['round'])

plt.tight_layout()
plt.savefig('fl_results.png', dpi=150)
plt.show()

# ── 9. Final comparison ───────────────────────────────────────
final = round_results[-1]
print("\n=== Centralised Model (Baseline) ===")
print(f"Recall: 0.9500 | Precision: 0.2100 | F1: 0.3400 | AUC: 0.9989")
print("\n=== FL Global Model (Final Round) ===")
print(f"Recall: {final['recall']:.4f} | Precision: {final['precision']:.4f} "
      f"| F1: {final['f1']:.4f} | AUC: {final['auc']:.4f}")
