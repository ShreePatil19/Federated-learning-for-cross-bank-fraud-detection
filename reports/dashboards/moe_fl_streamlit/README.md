# MoE-FL Fraud Detection Dashboard

Interactive Streamlit dashboard for the Masters AI project showing
**Mixture-of-Experts + Federated Learning** for multi-bank fraud detection.

## Files

| File | Purpose |
|------|---------|
| `dashboard.py` | The Streamlit app (9 tabs) |
| `train_inference_models.py` | Trains 3 ML experts on real ULB data, saves to `models/` |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | App theme + headless config |
| `architecture_diagram.png` | System diagram (sidebar) |
| `models/*.joblib` | Trained XGBoost/LightGBM/CatBoost (~550 KB total) |
| `models/sample_transactions.csv` | 50 real test transactions for the Live Inference tab |

## Local run

```bash
cd 06_reports
pip install -r requirements.txt
py train_inference_models.py     # only needed once
py -m streamlit run dashboard.py
```

Open `http://localhost:8501`.

---

## Deployment Options

### 1. Streamlit Community Cloud (recommended — free, public URL)

1. Push the `06_reports/` folder to a public GitHub repo
2. Go to **https://share.streamlit.io** → sign in with GitHub
3. Click **New app**, pick the repo, set **Main file path** = `06_reports/dashboard.py`
4. Hit **Deploy**

Notes:
- The 143 MB `creditcard.csv` is in `.gitignore` — `train_inference_models.py` re-downloads it at first boot
- The trained models in `models/*.joblib` are ~550 KB total — commit them so first load is instant
- Free tier: 1 GB RAM, sleeps after 7 days idle

### 2. Hugging Face Spaces (alternative — free, supports more RAM)

1. Create a new Space at **https://huggingface.co/new-space**
2. Pick **Streamlit** as the SDK
3. Upload `dashboard.py`, `requirements.txt`, `models/`, all CSVs
4. Done — auto-deploys

Notes:
- HF Spaces gives 16 GB RAM on the free CPU tier
- Public by default; can be made private under settings

### 3. ngrok Tunnel (fastest — public URL in 30 seconds)

For an in-person demo where you want to share the local app:

```bash
# Terminal 1
py -m streamlit run dashboard.py

# Terminal 2 (after installing ngrok from https://ngrok.com/download)
ngrok http 8501
```

ngrok prints a `https://xxxx.ngrok-free.app` URL — share it with anyone.

### 4. Local Network (zero config — same WiFi)

Already running. Streamlit prints:
```
Network URL: http://192.168.1.105:8501
```
Anyone on the same WiFi can open that URL.

---

## Demo Flow (15 minutes)

1. **Overview** — hero metrics + top-5 chart (1 min)
2. **🚀 Live Inference** — pick a fraud transaction, hit Run, show real model outputs (3 min)
3. **Strategy Comparison** — heatmap proves MoE family wins (2 min)
4. **Statistical Tests** — Friedman / Wilcoxon / Cohen's d tables (3 min)
5. **Cost Analysis** — show the 75% rank-flip insight (2 min)
6. **What's New** — AFL vs MoE-FL comparison table (2 min)
7. **Wrap up** — point at the architecture diagram in sidebar (2 min)

---

## Performance Numbers (real ULB)

| Model | AUPRC | ROC-AUC | F1@0.5 |
|-------|-------|---------|--------|
| XGBoost | 0.91 | 0.98 | 0.88 |
| LightGBM | 0.92 | 0.98 | 0.89 |
| CatBoost | 0.90 | 0.98 | 0.89 |

Matches published ULB Credit Card benchmarks.
