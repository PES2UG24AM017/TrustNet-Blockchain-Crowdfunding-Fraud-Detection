"""
TrustNet Transformer
------------------------
The final fusion model in the TrustNet pipeline. Takes the 5
trust metric scores (SDI, CTI, FVRS, CFIS, CTCS) for a campaign
and outputs a single fraud probability using a Transformer
encoder with multi-head self-attention.

Why a Transformer instead of a plain MLP:
  Self-attention lets the model learn WHICH metrics matter most
  for a given campaign and how they interact — e.g. high SDI
  combined with high CFIS should be treated very differently
  than high SDI alone.

Architecture:
  5 scalar scores → each projected to a d_model-dim token
      → + learnable positional embedding
      → TransformerEncoder (multi-head self-attention, N layers)
      → mean-pool across the 5 tokens
      → classification head → Sigmoid → fraud probability

Class imbalance handling:
  Labels come from output_combine_csv.py's percentile-based
  threshold, which targets ~20-25% fraud (not a 50/50 split).
  Training naively on an imbalanced set lets the model "cheat"
  by always predicting legit and still scoring ~75-80% accuracy
  while catching zero fraud. This script computes a pos_weight
  from the actual class balance and passes it into the loss
  function so fraud cases are weighted more heavily during
  training, and reports precision/recall/F1 (not just accuracy)
  so that trivial collapse is visible immediately if it happens.

Reads:
    outputs/trustnet_dataset.csv
    (produced by dataset_builder/output_combine_csv.py)

Usage:
    python trustnet_transformer.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use("Agg")   # headless-safe, no display needed to save PNGs
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEED           = 42
BATCH_SIZE     = 32
EPOCHS_FIRST_RUN  = 150   # full training when no prior model exists
EPOCHS_WARM_START = 40    # fewer epochs needed when continuing from
                           # existing weights (already knows old campaigns)
LEARNING_RATE  = 0.001
D_MODEL        = 32
NUM_HEADS      = 4
NUM_LAYERS     = 2
DROPOUT        = 0.1
TRAIN_SPLIT    = 0.8

DECISION_THRESHOLD = 0.5

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR  = os.path.join(SCRIPT_DIR, "..", "outputs")
MASTER_PATH  = os.path.join(OUTPUTS_DIR, "trustnet_dataset_master.csv")
LATEST_PATH  = os.path.join(OUTPUTS_DIR, "trustnet_dataset.csv")
MODEL_PATH   = os.path.join(OUTPUTS_DIR, "trustnet_transformer.pt")
PLOTS_DIR    = os.path.join(OUTPUTS_DIR, "plots")

# Prefer the accumulated master dataset (has "memory" of all past
# campaigns). Fall back to the latest single-batch file only if no
# master exists yet (i.e. this is truly the first run ever).
DATA_PATH = MASTER_PATH if os.path.exists(MASTER_PATH) else LATEST_PATH

LABEL_COL   = "fraud_label"
ID_COL      = "campaign_id"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────
# STEP 1 — DATASET
# ─────────────────────────────────────────────
class TrustNetDataset(Dataset):
    """
    Loads trustnet_dataset.csv and exposes:
      X → (num_samples, num_features)  the trust metric scores
      y → (num_samples, 1)             fraud label
    """
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        # Every column except campaign_id and fraud_label is a
        # feature — combined_risk_score (added by output_combine_csv)
        # is deliberately EXCLUDED, since it's just a linear
        # combination of the other 5 features used to derive the
        # label itself. Including it would let the model trivially
        # learn "just threshold this one column" instead of
        # learning genuine interactions between the 5 raw metrics.
        exclude_cols = (ID_COL, LABEL_COL, 'combined_risk_score')
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        self.feature_names = feature_cols

        if LABEL_COL not in df.columns:
            raise ValueError(
                f"'{LABEL_COL}' column not found in {csv_path}. "
                f"Run dataset_builder/output_combine_csv.py first."
            )

        X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        y = pd.to_numeric(df[LABEL_COL], errors='coerce').fillna(0.0)

        self.X = torch.tensor(X.values.astype(np.float32), dtype=torch.float32)
        self.y = torch.tensor(y.values.astype(np.float32), dtype=torch.float32).unsqueeze(1)

        self.fraud_count = int(y.sum())
        self.legit_count = int(len(y) - y.sum())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# STEP 2 — MODEL
# ─────────────────────────────────────────────
class TrustNetTransformer(nn.Module):
    def __init__(self, num_features, d_model=D_MODEL,
                 num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                 dropout=DROPOUT):
        super(TrustNetTransformer, self).__init__()

        self.num_features = num_features
        self.d_model = d_model

        self.input_projection = nn.Linear(1, d_model)
        self.positional_embedding = nn.Parameter(
            torch.randn(1, num_features, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # NOTE: outputs a raw logit (no Sigmoid here). BCEWithLogitsLoss
        # applies sigmoid internally in a numerically stable way and
        # is what lets us pass pos_weight for class imbalance. Sigmoid
        # is applied manually at inference time instead (see predict_proba).
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        tokens = self.input_projection(x)
        tokens = tokens + self.positional_embedding
        attended = self.transformer_encoder(tokens)
        pooled = attended.mean(dim=1)
        return self.classifier(pooled)   # raw logit

    def predict_proba(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


# ─────────────────────────────────────────────
# STEP 3 — TRAINING LOOP
# ─────────────────────────────────────────────
def train_model(model, train_loader, test_loader, device, pos_weight, epochs):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    print("\n[4/7] Training TrustNet Transformer...", flush=True)
    print(f"    Epochs: {epochs} | Batch size: {BATCH_SIZE} | "
          f"LR: {LEARNING_RATE} | d_model: {D_MODEL} | "
          f"heads: {NUM_HEADS} | layers: {NUM_LAYERS}", flush=True)
    print(f"    pos_weight (fraud class): {pos_weight.item():.3f} "
          f"(upweights the minority fraud class in the loss)", flush=True)
    print("-" * 60, flush=True)

    best_test_loss = float('inf')
    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item() * X_batch.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        total_test_loss = 0.0
        correct = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                total_test_loss += loss.item() * X_batch.size(0)

                probs = torch.sigmoid(logits)
                predicted = (probs >= DECISION_THRESHOLD).float()
                correct += (predicted == y_batch).sum().item()

        avg_test_loss = total_test_loss / len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)
        scheduler.step(avg_test_loss)

        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_acc)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss

        if epoch % 10 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Test Loss: {avg_test_loss:.4f} | "
                  f"Test Acc: {test_acc*100:.2f}% | "
                  f"LR: {current_lr:.6f}", flush=True)

    print("-" * 60, flush=True)
    print(f"    Final Test Accuracy: {history['test_acc'][-1]*100:.2f}%", flush=True)
    print(f"    Best Test Loss: {best_test_loss:.4f}", flush=True)
    return history


# ─────────────────────────────────────────────
# PLOTTING — training curve, confusion matrix, PR curve
# ─────────────────────────────────────────────
def plot_training_curve(history, save_path):
    """Train/test loss per epoch, with test accuracy on a secondary axis."""
    epochs_range = range(1, len(history['train_loss']) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs_range, history['train_loss'], label='Train Loss', color='tab:blue')
    ax1.plot(epochs_range, history['test_loss'], label='Test Loss', color='tab:orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    ax2 = ax1.twinx()
    ax2.plot(epochs_range, history['test_acc'], label='Test Accuracy',
              color='tab:green', linestyle='--', alpha=0.6)
    ax2.set_ylabel('Test Accuracy')
    ax2.set_ylim(0, 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax1.set_title('TrustNet Transformer — Training Curve')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"    Training curve saved to: {save_path}", flush=True)


def plot_confusion_matrix(tn, fp, fn, tp, save_path):
    matrix = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Predicted Legit', 'Predicted Fraud'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Actual Legit', 'Actual Fraud'])
    ax.set_title('TrustNet Transformer — Confusion Matrix')

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=14,
                     color='white' if matrix[i, j] > matrix.max() / 2 else 'black')

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"    Confusion matrix saved to: {save_path}", flush=True)


def plot_precision_recall_curve(all_labels, all_probs, save_path):
    """
    Sweeps decision thresholds manually (rather than pulling in
    sklearn.metrics) to stay consistent with the manual precision/
    recall calculation already used in evaluate_model.
    """
    thresholds = np.linspace(0.0, 1.0, 101)
    precisions, recalls = [], []

    for t in thresholds:
        preds = (all_probs >= t).astype(float)
        tp = ((preds == 1) & (all_labels == 1)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recalls, precisions, color='tab:green')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('TrustNet Transformer — Precision-Recall Curve')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"    Precision-recall curve saved to: {save_path}", flush=True)


# ─────────────────────────────────────────────
# STEP 4 — DETAILED EVALUATION
# ─────────────────────────────────────────────
def evaluate_model(model, test_loader, device):
    print("\n[5/7] Detailed evaluation on test set...", flush=True)

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            probs = torch.sigmoid(model(X_batch))
            predicted = (probs >= DECISION_THRESHOLD).float()
            all_preds.extend(predicted.cpu().numpy().flatten())
            all_labels.extend(y_batch.numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy  = (tp + tn) / len(all_labels)

    print(f"    Confusion Matrix:", flush=True)
    print(f"                    Predicted Legit   Predicted Fraud", flush=True)
    print(f"    Actual Legit         {tn:6d}            {fp:6d}", flush=True)
    print(f"    Actual Fraud         {fn:6d}            {tp:6d}", flush=True)
    print(f"\n    Accuracy : {accuracy*100:.2f}%", flush=True)
    print(f"    Precision: {precision*100:.2f}%", flush=True)
    print(f"    Recall   : {recall*100:.2f}%", flush=True)
    print(f"    F1 Score : {f1*100:.2f}%", flush=True)

    if tp == 0 and fp == 0:
        print(f"\n    WARNING: model never predicted fraud for any "
              f"test sample — it may have collapsed to the majority "
              f"class. Check pos_weight, learning rate, or whether "
              f"the 5 features actually separate the two classes.",
              flush=True)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_confusion_matrix(tn, fp, fn, tp,
                           os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plot_precision_recall_curve(all_labels, all_probs,
                                 os.path.join(PLOTS_DIR, "precision_recall_curve.png"))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60, flush=True)
    print("  TRUSTNET TRANSFORMER — TRAINING", flush=True)
    print("=" * 60, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[1/7] Using device: {device}", flush=True)

    print(f"\n[2/7] Loading dataset from: {DATA_PATH}", flush=True)
    if not os.path.exists(DATA_PATH):
        print(f"    ERROR: {DATA_PATH} not found.", flush=True)
        print(f"    Run dataset_builder/output_combine_csv.py first.", flush=True)
        return

    full_dataset = TrustNetDataset(DATA_PATH)
    print(f"    Loaded {len(full_dataset)} campaigns.", flush=True)
    print(f"    Features detected: {full_dataset.feature_names}", flush=True)
    print(f"    Class balance: {full_dataset.fraud_count} fraud / "
          f"{full_dataset.legit_count} legit "
          f"({full_dataset.fraud_count/len(full_dataset)*100:.1f}% fraud)",
          flush=True)

    n_train = int(len(full_dataset) * TRAIN_SPLIT)
    n_test  = len(full_dataset) - n_train
    train_dataset, test_dataset = random_split(
        full_dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )
    print(f"    Train: {n_train} | Test: {n_test}", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Compute pos_weight from the FULL dataset's class balance, so
    # the minority fraud class is upweighted in the loss function
    # proportionally to how rare it actually is.
    if full_dataset.fraud_count == 0:
        print("\n    WARNING: 0 fraud samples in dataset — cannot "
              "compute meaningful pos_weight. Defaulting to 1.0.",
              flush=True)
        pos_weight = torch.tensor([1.0])
    else:
        pos_weight = torch.tensor(
            [full_dataset.legit_count / full_dataset.fraud_count]
        )

    print("\n[3/7] Building TrustNet Transformer...", flush=True)
    model = TrustNetTransformer(
        num_features=len(full_dataset.feature_names)
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Total parameters: {total_params:,}", flush=True)

    # ── Warm-start from existing weights if available ──
    # This is what gives TrustNet "memory" between retrains: instead
    # of always starting from random weights, we continue training
    # on top of whatever it already learned. Combined with the
    # master dataset (which keeps old campaigns in the training set
    # every time), this avoids catastrophic forgetting — old
    # campaigns aren't just remembered because they're in the data,
    # the model also isn't reset and forced to relearn them from zero.
    warm_started = False
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            prior_features = checkpoint.get('feature_names', [])

            if prior_features == full_dataset.feature_names:
                model.load_state_dict(checkpoint['model_state_dict'])
                warm_started = True
                print(f"    Warm-starting from existing model at: "
                      f"{MODEL_PATH}", flush=True)
                print(f"    (feature set unchanged — safe to continue "
                      f"training on top of prior weights)", flush=True)
            else:
                print(f"    Existing model found but its feature set "
                      f"differs from the current dataset's — "
                      f"training a fresh model instead.", flush=True)
                print(f"      Prior : {prior_features}", flush=True)
                print(f"      Now   : {full_dataset.feature_names}", flush=True)
        except Exception as e:
            print(f"    Could not load existing model ({e}) — "
                  f"training a fresh model instead.", flush=True)
    else:
        print(f"    No existing model found — training from scratch.",
              flush=True)

    epochs = EPOCHS_WARM_START if warm_started else EPOCHS_FIRST_RUN
    print(f"    Mode: {'WARM START' if warm_started else 'FRESH TRAINING'} "
          f"→ {epochs} epochs", flush=True)

    history = train_model(model, train_loader, test_loader, device, pos_weight, epochs)
    evaluate_model(model, test_loader, device)

    print("\n[6/7] Generating diagrams...", flush=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_training_curve(history, os.path.join(PLOTS_DIR, "training_curve.png"))

    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_names': full_dataset.feature_names,
        'd_model': D_MODEL,
        'num_heads': NUM_HEADS,
        'num_layers': NUM_LAYERS,
        'decision_threshold': DECISION_THRESHOLD,
    }, MODEL_PATH)

    print(f"\n[7/7] Model saved to: {MODEL_PATH}", flush=True)
    print("\n" + "=" * 60, flush=True)
    print("  TRAINING COMPLETE", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()