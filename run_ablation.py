"""
run_ablation.py
---------------
Runs the full TrustNet ablation study automatically.

Configurations tested:
  1. All 5 metrics (full model)       — baseline
  2. Without SDI   (SDI zeroed out)
  3. Without CTI   (CTI zeroed out)
  4. Without FVRS  (FVRS zeroed out)
  5. Without CFIS  (CFIS zeroed out)
  6. Without CTCS  (CTCS zeroed out)
  7. MLP baseline  (no Transformer, plain feedforward)
  8. Weighted average (no learning, fixed formula)

Prints a complete table ready to paste into the paper.

Run from the project root:
    python run_ablation.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, average_precision_score
)

# ── Config (must match genarate_pr_curve.py exactly) ────────
SEED           = 42
BATCH_SIZE     = 32
EPOCHS         = 50
LEARNING_RATE  = 0.001
D_MODEL        = 32
NUM_HEADS      = 4
NUM_LAYERS     = 2
DROPOUT        = 0.1
THRESHOLD      = 0.5

FEATURES  = ["sdi", "cti", "fvrs", "cfis", "ctcs"]
LABEL_COL = "is_fraud"

np.random.seed(SEED)
torch.manual_seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "outputs", "trustnet_train.csv")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("TRUSTNET ABLATION STUDY")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Dataset: {DATA_PATH}")

# ── Load data ────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X  = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
y  = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).values.astype(np.float32)

total   = len(y)
fraud   = int(np.sum(y == 1))
legit   = int(np.sum(y == 0))
print(f"\nDataset: {total} samples | {fraud} fraud ({fraud/total*100:.1f}%) | {legit} legit")

# ── Fixed split (same seed every run) ───────────────────────
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.17647, random_state=SEED, stratify=y_temp)

# Normalise (fit only on train)
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0)
std[std == 0] = 1.0
X_train = (X_train - mean) / std
X_val   = (X_val   - mean) / std
X_test  = (X_test  - mean) / std

print(f"Split → train:{len(X_train)} val:{len(X_val)} test:{len(X_test)}")


# ── Dataset ──────────────────────────────────────────────────
class TND(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def make_loaders(X_tr, X_v, X_te, y_tr, y_v, y_te):
    return (
        DataLoader(TND(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(TND(X_v,  y_v),  batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(TND(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False),
    )


# ── TrustNet Transformer ─────────────────────────────────────
class TrustNetTransformer(nn.Module):
    def __init__(self, num_features=5):
        super().__init__()
        self.input_projection    = nn.Linear(1, D_MODEL)
        self.positional_embedding = nn.Parameter(
            torch.randn(1, num_features, D_MODEL) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NUM_HEADS,
            dim_feedforward=D_MODEL * 4,
            dropout=DROPOUT, batch_first=True, activation="relu")
        self.transformer_encoder = nn.TransformerEncoder(enc, num_layers=NUM_LAYERS)
        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(D_MODEL // 2, 1))

    def forward(self, x):
        tokens = self.input_projection(x.unsqueeze(-1))
        tokens = tokens + self.positional_embedding
        z = self.transformer_encoder(tokens).mean(dim=1)
        return self.classifier(z)


# ── MLP Baseline ─────────────────────────────────────────────
class MLPBaseline(nn.Module):
    def __init__(self, num_features=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(64, 32),           nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)


# ── Train + Evaluate ─────────────────────────────────────────
def train_and_eval(model, train_loader, val_loader, test_loader,
                   pos_weight, label=""):
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            optimizer.step()

    # Test evaluation
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            p = torch.sigmoid(model(Xb.to(DEVICE))).cpu().numpy().flatten()
            probs.extend(p)
            labels.extend(yb.numpy().flatten())

    probs  = np.array(probs)
    labels = np.array(labels)
    preds  = (probs >= THRESHOLD).astype(int)

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    ap   = average_precision_score(labels, probs)

    return {
        "Accuracy":  round(acc  * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall":    round(rec  * 100, 2),
        "F1":        round(f1   * 100, 2),
        "AP":        round(ap,  3),
    }


# ── Weighted Average Baseline (no learning) ──────────────────
def weighted_avg_eval(X_te, y_te):
    """
    Mirrors TrustMetricsRegistry.sol getCombinedRisk formula:
    risk = 0.25*sdi + 0.20*(1-cti) + 0.20*fvrs + 0.20*cfis + 0.15*ctcs
    Classify as fraud if risk >= 0.5
    """
    # Denormalise to get original [0,1] values
    X_orig = X_te * std + mean

    sdi  = X_orig[:, 0]
    cti  = X_orig[:, 1]
    fvrs = X_orig[:, 2]
    cfis = X_orig[:, 3]
    ctcs = X_orig[:, 4]

    risk  = 0.25*sdi + 0.20*(1-cti) + 0.20*fvrs + 0.20*cfis + 0.15*ctcs
    preds = (risk >= 0.5).astype(int)

    acc  = accuracy_score(y_te, preds)
    prec = precision_score(y_te, preds, zero_division=0)
    rec  = recall_score(y_te, preds, zero_division=0)
    f1   = f1_score(y_te, preds, zero_division=0)
    ap   = average_precision_score(y_te, risk)

    return {
        "Accuracy":  round(acc  * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall":    round(rec  * 100, 2),
        "F1":        round(f1   * 100, 2),
        "AP":        round(ap,  3),
    }


# ── pos_weight from training set ─────────────────────────────
pos_w = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))

# ── Run all configurations ───────────────────────────────────
results = {}

CONFIGS = [
    ("All 5 metrics (full model)",  None),         # None = no zeroing
    ("Without SDI",                 0),             # zero column index 0
    ("Without CTI",                 1),
    ("Without FVRS",                2),
    ("Without CFIS",                3),
    ("Without CTCS",                4),
]

for config_name, zero_idx in CONFIGS:
    print(f"\n{'='*70}")
    print(f"Running: {config_name}")

    # Apply ablation: zero out one feature across all splits
    Xtr = X_train.copy()
    Xv  = X_val.copy()
    Xte = X_test.copy()
    if zero_idx is not None:
        Xtr[:, zero_idx] = 0.0
        Xv[:,  zero_idx] = 0.0
        Xte[:, zero_idx] = 0.0

    train_l, val_l, test_l = make_loaders(Xtr, Xv, Xte, y_train, y_val, y_test)
    model = TrustNetTransformer(num_features=5)
    metrics = train_and_eval(model, train_l, val_l, test_l, pos_w, config_name)
    results[config_name] = metrics
    print(f"  Acc={metrics['Accuracy']}%  Prec={metrics['Precision']}%  "
          f"Rec={metrics['Recall']}%  F1={metrics['F1']}%  AP={metrics['AP']}")

# ── MLP Baseline ─────────────────────────────────────────────
print(f"\n{'='*70}")
print("Running: MLP Baseline (no Transformer)")
train_l, val_l, test_l = make_loaders(X_train, X_val, X_test, y_train, y_val, y_test)
mlp = MLPBaseline(num_features=5)
results["MLP baseline (no Transformer)"] = train_and_eval(
    mlp, train_l, val_l, test_l, pos_w)
m = results["MLP baseline (no Transformer)"]
print(f"  Acc={m['Accuracy']}%  Prec={m['Precision']}%  "
      f"Rec={m['Recall']}%  F1={m['F1']}%  AP={m['AP']}")

# ── Weighted Average Baseline ─────────────────────────────────
print(f"\n{'='*70}")
print("Running: Weighted average (no learning)")
results["Weighted average (no learning)"] = weighted_avg_eval(X_test, y_test)
m = results["Weighted average (no learning)"]
print(f"  Acc={m['Accuracy']}%  Prec={m['Precision']}%  "
      f"Rec={m['Recall']}%  F1={m['F1']}%  AP={m['AP']}")

# ── Print final table ─────────────────────────────────────────
print(f"\n\n{'='*70}")
print("ABLATION STUDY RESULTS — COPY THIS INTO THE PAPER")
print(f"{'='*70}")
print(f"\n{'Configuration':<40} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AP':>7}")
print("-" * 75)
for name, m in results.items():
    marker = " ◄" if name == "All 5 metrics (full model)" else ""
    print(f"{name:<40} {m['Accuracy']:>6}% {m['Precision']:>6}% "
          f"{m['Recall']:>6}% {m['F1']:>6}% {m['AP']:>7}{marker}")

print("\n◄ = proposed full model (baseline for comparison)")
print("\nNote: Each configuration retrained from scratch with same")
print("seed, split, epochs, and hyperparameters for fair comparison.")
print(f"{'='*70}")

# ── Save results to CSV ───────────────────────────────────────
out_path = os.path.join(SCRIPT_DIR, "outputs", "ablation_results.csv")
pd.DataFrame(results).T.to_csv(out_path)
print(f"\nResults saved to: {out_path}")
