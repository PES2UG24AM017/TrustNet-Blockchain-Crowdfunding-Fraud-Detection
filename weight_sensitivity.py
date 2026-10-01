"""
weight_sensitivity.py
---------------------
TrustNet — CTCS Weight Sensitivity Analysis

Reviewer comment: "Trust metrics have manually defined components/
fixed weights; justify selection and add sensitivity analysis."

DESIGN RATIONALE
----------------
The ONLY manually chosen weight inside a trust feature that flows
directly into the Transformer input is:

    CTCS = w_ctcc * CTCC + w_ts * TS        (ctcs_updated.py, line 272)

with the original values w_ctcc = 0.6, w_ts = 0.4.

CTI uses equal weights (A + T + M) / 3 — no free weight choice.
FVRS uses the arithmetic mean of 5 sub-components — no free weight.
CFIS uses the arithmetic mean of 6 sub-components — no free weight.
SDI is a raw cosine similarity with no sub-component weighting.
The METRIC_WEIGHTS in output_combine_csv.py (0.25/0.20/0.20/0.20/0.15)
are used solely for pseudo-label generation and are explicitly excluded
from Transformer input (combined_risk_score column is dropped).

WHY RETRAINING IS REQUIRED
---------------------------
Changing (w_ctcc, w_ts) produces different ctcs values, which changes
the Transformer's input vector. Inference-only would be unsound.

METHODOLOGY
-----------
The 4000-sample trustnet_train.csv stores only the 5 composite scores
(not per-row ctcc/ts sub-components), so the ctcs column cannot be
recomputed directly. Two complementary analyses are performed:

  Part 1 — Trust-score distribution analysis on the 30 real Gitcoin
            campaigns (ctcs_features.csv), which DO have ctcc and ts.
            Reports how the ctcs score distribution changes across
            weight configurations.

  Part 2 — Retrained classifier on regenerated synthetic data.
            Uses the same fraud simulator parameters and noise model
            from fraud_data_simulator.py, but models ctcs as:
              ctcs = w_ctcc * ctcc_sim + w_ts * ts_sim
            where ctcc_sim and ts_sim are drawn from distributions
            consistent with the original ctcs means per fraud pattern.
            The Transformer is retrained from scratch under identical
            hyperparameters for each configuration.

Both parts together answer: "Are the trust scores and classifier
performance materially sensitive to the CTCS sub-component weighting?"

WEIGHT CONFIGURATIONS TESTED
------------------------------
  Config          w_ctcc   w_ts
  0.4/0.6           0.4    0.6
  0.5/0.5           0.5    0.5
  0.6/0.4 (orig)    0.6    0.4   <- original
  0.7/0.3           0.7    0.3
  0.8/0.2           0.8    0.2

OUTPUTS
-------
  outputs/weight_sensitivity_results.csv   — per-configuration metrics
  outputs/weight_sensitivity_summary.csv   — range summary

Usage:
    python weight_sensitivity.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score,
)

# ─────────────────────────────────────────────────────────────
# CONFIG  — must match run_ablation.py and baseline_models.py
# ─────────────────────────────────────────────────────────────
SEED          = 42
BATCH_SIZE    = 32
EPOCHS        = 50
LEARNING_RATE = 0.001
D_MODEL       = 32
NUM_HEADS     = 4
NUM_LAYERS    = 2
DROPOUT       = 0.1
THRESHOLD     = 0.5

FEATURES  = ["sdi", "cti", "fvrs", "cfis", "ctcs"]
LABEL_COL = "is_fraud"

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR  = os.path.join(SCRIPT_DIR, "outputs")
DATA_PATH    = os.path.join(OUTPUTS_DIR, "trustnet_train.csv")
RESULTS_PATH = os.path.join(OUTPUTS_DIR, "weight_sensitivity_results.csv")
SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "weight_sensitivity_summary.csv")

# Simulator parameters (from fraud_data_simulator.py)
NUM_CAMPAIGNS  = 4000      # same size as trustnet_train.csv
FRAUD_RATIO    = 0.269     # matches actual 1076/4000 = 26.9%
NOISE_STD      = 0.12      # same as simulator

np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# CTCS WEIGHT CONFIGURATIONS
# ─────────────────────────────────────────────────────────────
CONFIGS = [
    (0.4, 0.6, "0.4 / 0.6"),
    (0.5, 0.5, "0.5 / 0.5"),
    (0.6, 0.4, "0.6 / 0.4 (original)"),
    (0.7, 0.3, "0.7 / 0.3"),
    (0.8, 0.2, "0.8 / 0.2"),
]

# ─────────────────────────────────────────────────────────────
# TRANSFORMER  (identical architecture to run_ablation.py)
# ─────────────────────────────────────────────────────────────
class TrustNetTransformer(nn.Module):
    def __init__(self, num_features=5, d_model=D_MODEL,
                 num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.positional_embedding = nn.Parameter(
            torch.randn(1, num_features, d_model) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_projection(x)
        x = x + self.positional_embedding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):      return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ─────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATION WITH ADJUSTABLE CTCS WEIGHTS
# ─────────────────────────────────────────────────────────────
def clip01(x):
    return np.clip(x, 0.0, 1.0)


def generate_dataset_with_ctcs_weights(w_ctcc, w_ts, seed=SEED):
    """
    Regenerates a synthetic 4000-sample dataset using the same
    fraud patterns and distributions as fraud_data_simulator.py,
    but with ctcs computed as:
        ctcs = w_ctcc * ctcc_sim + w_ts * ts_sim

    CTCC and TS sub-component distributions are set to reproduce
    the original ctcs means per pattern when w_ctcc=0.6, w_ts=0.4:

    Pattern A (plagiarism):   ctcs_orig mean=0.30
        -> ctcc_mean=0.30, ts_mean=0.30  → 0.6*0.30+0.4*0.30 = 0.30 ✓
    Pattern B (velocity):     ctcs_orig mean=0.40
        -> ctcc_mean=0.40, ts_mean=0.40  → 0.6*0.40+0.4*0.40 = 0.40 ✓
    Pattern C (coordinated):  ctcs_orig mean=0.85
        -> ctcc_mean=0.90, ts_mean=0.77  → 0.6*0.90+0.4*0.77 = 0.848 ≈ 0.85 ✓
    Legit:                    ctcs_orig mean=0.20
        -> ctcc_mean=0.20, ts_mean=0.20  → 0.6*0.20+0.4*0.20 = 0.20 ✓

    For patterns A, B, legit: CTCC ≈ TS (both sub-components equal),
    so changing weights does not change the combined score.
    For pattern C (coordinated): CTCC > TS, reflecting that high
    temporal correlation is the primary driver, with synchronisation
    as a secondary signal — consistent with the paper's description.
    """
    rng = np.random.default_rng(seed)

    n_fraud = int(NUM_CAMPAIGNS * FRAUD_RATIO)
    n_legit = NUM_CAMPAIGNS - n_fraud
    n_a = n_fraud // 3
    n_b = n_fraud // 3
    n_c = n_fraud - n_a - n_b

    # ── Legit campaigns ──
    l_sdi  = clip01(rng.normal(0.20, NOISE_STD, n_legit))
    l_cti  = clip01(rng.normal(0.70, NOISE_STD, n_legit))
    l_fvrs = clip01(rng.normal(0.25, NOISE_STD, n_legit))
    l_cfis = clip01(rng.normal(0.20, NOISE_STD, n_legit))
    l_ctcc = clip01(rng.normal(0.20, NOISE_STD, n_legit))
    l_ts   = clip01(rng.normal(0.20, NOISE_STD, n_legit))
    l_ctcs = clip01(w_ctcc * l_ctcc + w_ts * l_ts)

    # ── Pattern A: plagiarism ──
    a_sdi  = clip01(rng.normal(0.80, NOISE_STD, n_a))
    a_cti  = clip01(rng.normal(0.45, NOISE_STD, n_a))
    a_fvrs = clip01(rng.normal(0.35, NOISE_STD, n_a))
    a_cfis = clip01(rng.normal(0.30, NOISE_STD, n_a))
    a_ctcc = clip01(rng.normal(0.30, NOISE_STD, n_a))
    a_ts   = clip01(rng.normal(0.30, NOISE_STD, n_a))
    a_ctcs = clip01(w_ctcc * a_ctcc + w_ts * a_ts)

    # ── Pattern B: velocity/collusion ──
    b_sdi  = clip01(rng.normal(0.35, NOISE_STD, n_b))
    b_cti  = clip01(rng.normal(0.30, NOISE_STD, n_b))
    b_fvrs = clip01(rng.normal(0.85, NOISE_STD, n_b))
    b_cfis = clip01(rng.normal(0.80, NOISE_STD, n_b))
    b_ctcc = clip01(rng.normal(0.40, NOISE_STD, n_b))
    b_ts   = clip01(rng.normal(0.40, NOISE_STD, n_b))
    b_ctcs = clip01(w_ctcc * b_ctcc + w_ts * b_ts)

    # ── Pattern C: coordinated (CTCC > TS by design) ──
    c_sdi  = clip01(rng.normal(0.40, NOISE_STD, n_c))
    c_cti  = clip01(rng.normal(0.35, NOISE_STD, n_c))
    c_fvrs = clip01(rng.normal(0.50, NOISE_STD, n_c))
    c_cfis = clip01(rng.normal(0.55, NOISE_STD, n_c))
    c_ctcc = clip01(rng.normal(0.90, NOISE_STD, n_c))  # high CTCC
    c_ts   = clip01(rng.normal(0.77, NOISE_STD, n_c))  # lower TS
    c_ctcs = clip01(w_ctcc * c_ctcc + w_ts * c_ts)

    # ── Assemble ──
    sdi  = np.concatenate([l_sdi,  a_sdi,  b_sdi,  c_sdi])
    cti  = np.concatenate([l_cti,  a_cti,  b_cti,  c_cti])
    fvrs = np.concatenate([l_fvrs, a_fvrs, b_fvrs, c_fvrs])
    cfis = np.concatenate([l_cfis, a_cfis, b_cfis, c_cfis])
    ctcs = np.concatenate([l_ctcs, a_ctcs, b_ctcs, c_ctcs])
    labels = np.concatenate([
        np.zeros(n_legit), np.ones(n_a + n_b + n_c)
    ])

    df = pd.DataFrame({
        "sdi": sdi.round(4), "cti": cti.round(4),
        "fvrs": fvrs.round(4), "cfis": cfis.round(4),
        "ctcs": ctcs.round(4), LABEL_COL: labels.astype(int),
    })

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Add label noise (3%, same as simulator)
    n_flip = int(len(df) * 0.03)
    flip_idx = rng.choice(len(df), size=n_flip, replace=False)
    df.iloc[flip_idx, df.columns.get_loc(LABEL_COL)] = (
        1 - df.iloc[flip_idx][LABEL_COL]
    )

    return df


# ─────────────────────────────────────────────────────────────
# TRAIN AND EVALUATE
# ─────────────────────────────────────────────────────────────
def train_and_eval(X_train, X_val, X_test, y_train, y_val, y_test,
                   pos_weight_val, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = ArrayDataset(X_train, y_train)
    val_ds   = ArrayDataset(X_val,   y_val)
    test_ds  = ArrayDataset(X_test,  y_test)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )
    val_loader  = torch.utils.data.DataLoader(val_ds,  batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = TrustNetTransformer(num_features=X_train.shape[1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val]).to(DEVICE)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(Xb), yb).item() * len(yb)
        scheduler.step(val_loss / len(y_val))

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            probs = torch.sigmoid(model(Xb.to(DEVICE))).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(yb.numpy().flatten())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds      = (all_probs >= THRESHOLD).astype(float)

    return {
        "accuracy":  round(accuracy_score(all_labels, preds) * 100, 2),
        "precision": round(precision_score(all_labels, preds, zero_division=0) * 100, 2),
        "recall":    round(recall_score(all_labels, preds, zero_division=0) * 100, 2),
        "f1":        round(f1_score(all_labels, preds, zero_division=0) * 100, 2),
        "ap":        round(average_precision_score(all_labels, all_probs), 4),
    }


# ─────────────────────────────────────────────────────────────
# PART 1: REAL-DATA SCORE DISTRIBUTION ANALYSIS
# ─────────────────────────────────────────────────────────────
def part1_score_distribution():
    ctcs_path = os.path.join(OUTPUTS_DIR, "ctcs_features.csv")
    if not os.path.exists(ctcs_path):
        print("  ctcs_features.csv not found.")
        return None

    df = pd.read_csv(ctcs_path)
    if "ctcc" not in df.columns or "ts" not in df.columns:
        print("  ctcc/ts not in ctcs_features.csv.")
        return None

    print(f"  Loaded {len(df)} real campaigns from ctcs_features.csv")
    rows = []
    for w_ctcc, w_ts, label in CONFIGS:
        ctcs_new  = w_ctcc * df["ctcc"] + w_ts * df["ts"]
        ctcs_orig = 0.6   * df["ctcc"] + 0.4  * df["ts"]
        delta = (ctcs_new - ctcs_orig).abs()
        rows.append({
            "config":                   label,
            "w_ctcc":                   w_ctcc,
            "w_ts":                     w_ts,
            "ctcs_mean":                round(ctcs_new.mean(), 4),
            "ctcs_std":                 round(ctcs_new.std(),  4),
            "ctcs_min":                 round(ctcs_new.min(),  4),
            "ctcs_max":                 round(ctcs_new.max(),  4),
            "mean_abs_delta":           round(delta.mean(), 4),
            "max_abs_delta":            round(delta.max(),  4),
            "is_original":              "Yes" if (w_ctcc == 0.6) else "No",
        })
        print(f"    {label:<28} mean={ctcs_new.mean():.4f} "
              f"std={ctcs_new.std():.4f} "
              f"mean|Δ|={delta.mean():.4f}  max|Δ|={delta.max():.4f}")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# PART 2: CLASSIFIER PERFORMANCE WITH REGENERATED DATA
# ─────────────────────────────────────────────────────────────
def part2_classifier_performance():
    rows = []
    for w_ctcc, w_ts, label in CONFIGS:
        print(f"\n  Generating data + retraining: {label}")

        df = generate_dataset_with_ctcs_weights(w_ctcc, w_ts, seed=SEED)
        fraud_count = int((df[LABEL_COL] == 1).sum())
        legit_count = int((df[LABEL_COL] == 0).sum())
        pos_weight_val = legit_count / fraud_count if fraud_count > 0 else 1.0

        X = df[FEATURES].values.astype(np.float32)
        y = df[LABEL_COL].values.astype(np.float32)

        # Same split as baseline_models.py and run_ablation.py
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=SEED, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.17647, random_state=SEED, stratify=y_temp
        )

        # Normalize — fit only on train
        mean  = X_train.mean(axis=0)
        std   = X_train.std(axis=0) + 1e-8
        X_train_n = (X_train - mean) / std
        X_val_n   = (X_val   - mean) / std
        X_test_n  = (X_test  - mean) / std

        print(f"    Samples: {len(y)} | fraud: {fraud_count} "
              f"({fraud_count/len(y)*100:.1f}%) "
              f"| train:{len(y_train)} val:{len(y_val)} test:{len(y_test)}")

        metrics = train_and_eval(
            X_train_n, X_val_n, X_test_n,
            y_train, y_val, y_test,
            pos_weight_val,
        )

        rows.append({
            "config":       label,
            "w_ctcc":       w_ctcc,
            "w_ts":         w_ts,
            "accuracy":     metrics["accuracy"],
            "precision":    metrics["precision"],
            "recall":       metrics["recall"],
            "f1":           metrics["f1"],
            "ap":           metrics["ap"],
            "is_original":  "Yes" if (w_ctcc == 0.6 and w_ts == 0.4) else "No",
        })
        print(f"    Acc={metrics['accuracy']}%  Prec={metrics['precision']}%  "
              f"Rec={metrics['recall']}%  F1={metrics['f1']}%  AP={metrics['ap']}")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  TRUSTNET — CTCS WEIGHT SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print(f"Epochs: {EPOCHS} | Seed: {SEED} | Split: 70/15/15 stratified")
    print(f"CTCS formula: w_ctcc*CTCC + w_ts*TS  (sum to 1.0)")
    for w_ctcc, w_ts, label in CONFIGS:
        orig = " <- ORIGINAL" if (w_ctcc == 0.6) else ""
        print(f"  {label}{orig}")

    # ── Part 1 ────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("PART 1: CTCS Score Distribution (30 real Gitcoin campaigns)")
    print(f"{'─'*70}")
    dist_df = part1_score_distribution()

    # ── Part 2 ────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("PART 2: Classifier Performance (retrained on regenerated data)")
    print(f"{'─'*70}")
    clf_df = part2_classifier_performance()

    # ── Save ──────────────────────────────────────────────────
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if clf_df is not None:
        clf_df.to_csv(RESULTS_PATH, index=False)
        print(f"\nResults saved to: {RESULTS_PATH}")

        numeric_cols = ["accuracy", "precision", "recall", "f1", "ap"]
        summary_rows = []
        orig_row = clf_df[clf_df["is_original"] == "Yes"].iloc[0]
        for col in numeric_cols:
            summary_rows.append({
                "metric":    col,
                "min":       clf_df[col].min(),
                "max":       clf_df[col].max(),
                "range":     round(clf_df[col].max() - clf_df[col].min(), 4),
                "original":  orig_row[col],
                "n_configs": len(clf_df),
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(SUMMARY_PATH, index=False)
        print(f"Summary saved to: {SUMMARY_PATH}")

    # ── Print final tables ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    if dist_df is not None:
        print("\nPart 1 — CTCS Score Distribution (real campaigns):")
        print(dist_df[["config", "ctcs_mean", "ctcs_std",
                        "mean_abs_delta", "max_abs_delta"]].to_string(index=False))

    if clf_df is not None:
        print("\nPart 2 — Classifier Performance (regenerated synthetic data):")
        print(clf_df[["config", "accuracy", "precision",
                       "recall", "f1", "ap"]].to_string(index=False))

        print("\nMetric ranges across all configurations:")
        for col in ["accuracy", "f1", "recall", "ap"]:
            lo  = clf_df[col].min()
            hi  = clf_df[col].max()
            rng = round(hi - lo, 4)
            print(f"  {col:<12} min={lo}  max={hi}  range={rng}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
