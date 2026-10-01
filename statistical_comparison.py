"""
statistical_comparison.py
--------------------------
Repeated-run statistical analysis across all TrustNet baseline models.

Models compared:
  1. TrustNet Transformer
  2. MLP
  3. Logistic Regression
  4. Random Forest
  5. Gradient Boosting

Seeds:   42, 52, 62, 72, 82
Dataset: outputs/trustnet_train.csv

For EVERY seed, ALL five models use the EXACT SAME:
  - train/val/test split (generated with that seed)
  - normalization statistics (fit only on X_train for that seed)
  - test set indices (same 600 samples for every model at that seed)

This guarantees that metric differences between models at each seed
are attributable to the model only, not to data partitioning.

Statistical test:
  McNemar's test (scipy.stats, exact=False, continuity correction)
  Compares Transformer predictions vs each other model
  on the SAME test set (paired predictions).
  statsmodels NOT required — test is implemented via scipy.stats.

Outputs (created only when experiment is run):
  outputs/statistical_runs.csv      — per-model per-seed metrics
  outputs/statistical_summary.csv   — mean ± std across seeds
  outputs/mcnemar_results.csv        — McNemar test statistics

Usage:
    python statistical_comparison.py

DO NOT modify this file to change hyperparameters.
DO NOT use the test set for any form of tuning.
"""

import os
import sys
import random
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, confusion_matrix,
)
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")   # suppress convergence warnings in LR

# ──────────────────────────────────────────────────────────────
# CONFIG — identical to all existing experiments
# ──────────────────────────────────────────────────────────────
SEEDS     = [42, 52, 62, 72, 82]
THRESHOLD = 0.5
FEATURES  = ["sdi", "cti", "fvrs", "cfis", "ctcs"]
LABEL_COL = "is_fraud"

# Transformer / MLP training config
EPOCHS        = 50
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
D_MODEL       = 32
NUM_HEADS     = 4
NUM_LAYERS    = 2
DROPOUT       = 0.1

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(SCRIPT_DIR, "outputs", "trustnet_train.csv")
RUNS_PATH    = os.path.join(SCRIPT_DIR, "outputs", "statistical_runs.csv")
SUMMARY_PATH = os.path.join(SCRIPT_DIR, "outputs", "statistical_summary.csv")
MCNEMAR_PATH = os.path.join(SCRIPT_DIR, "outputs", "mcnemar_results.csv")


# ──────────────────────────────────────────────────────────────
# SEEDING HELPERS
# ──────────────────────────────────────────────────────────────
def set_all_seeds(seed: int):
    """
    Sets seeds for numpy, torch, and Python random.
    Enables torch determinism where available.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Use deterministic algorithms where possible
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass   # Not fatal — older PyTorch versions may not support this


# ──────────────────────────────────────────────────────────────
# DATA SPLIT & NORMALIZATION
# ──────────────────────────────────────────────────────────────
def make_split_and_normalize(X: np.ndarray, y: np.ndarray, seed: int):
    """
    For a given seed:
      1. Creates the IDENTICAL two-stage stratified split used by all
         existing experiments.
      2. Fits z-score normalization ONLY on X_train for this seed.
      3. Applies those statistics to val and test — no test leakage.

    All models at the same seed receive identical data partitions,
    ensuring metric differences are solely due to the model.
    """
    # Step 1: carve out 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.15,
        random_state=seed,   # seed-specific — produces different partition per seed
        stratify=y,
    )
    # Step 2: from 85%, take 17.647% as val → 15% overall
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.17647,
        random_state=seed,
        stratify=y_temp,
    )

    # Normalization: fit ONLY on training split for this seed
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_train_n = (X_train - mean) / std
    X_val_n   = (X_val   - mean) / std
    X_test_n  = (X_test  - mean) / std

    return X_train_n, X_val_n, X_test_n, y_train, y_val, y_test, mean, std


# ──────────────────────────────────────────────────────────────
# MODEL ARCHITECTURES (identical to existing implementations)
# ──────────────────────────────────────────────────────────────

class TrustNetTransformer(nn.Module):
    """
    Exact architecture from trustnet/genarate_pr_curve.py.
    d_model=32, 4 heads, 2 layers, classification head 32→16→1.
    """
    def __init__(self, num_features: int = 5):
        super().__init__()
        self.input_projection     = nn.Linear(1, D_MODEL)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.input_projection(x.unsqueeze(-1))
        tokens = tokens + self.positional_embedding
        return self.classifier(self.transformer_encoder(tokens).mean(dim=1))


class MLPBaseline(nn.Module):
    """
    Exact architecture from run_ablation.py.
    5→64→32→1 with ReLU and dropout.
    """
    def __init__(self, num_features: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(64, 32),           nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ──────────────────────────────────────────────────────────────
# TORCH MODEL TRAINING (shared for Transformer and MLP)
# ──────────────────────────────────────────────────────────────
def train_torch_model(
    model: nn.Module,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    pos_weight: float,
    device: torch.device,
) -> nn.Module:
    """
    Trains a PyTorch model (Transformer or MLP) with:
      - BCEWithLogitsLoss with positive class weighting
      - Adam optimizer, lr=0.001
      - 50 epochs
      - No early stopping, no scheduler
      - No test data seen during training
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader = DataLoader(
        TorchDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    for _ in range(EPOCHS):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            optimizer.step()

    return model


def predict_torch(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
        probs  = torch.sigmoid(logits).cpu().numpy().flatten()
    return probs


# ──────────────────────────────────────────────────────────────
# SKLEARN BASELINES
# ──────────────────────────────────────────────────────────────
def get_sklearn_baselines(pos_weight_ratio: float) -> dict:
    """
    Returns sklearn baseline instances with FIXED hyperparameters
    (identical to baseline_models.py — no test-set tuning).
    seed is passed separately per-run via random_state.
    Note: GradientBoosting receives sample_weight during fit().
    """
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced",
            solver="lbfgs"),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=None,
            class_weight="balanced", n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3),
    }


# ──────────────────────────────────────────────────────────────
# EVALUATE
# ──────────────────────────────────────────────────────────────
def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= THRESHOLD).astype(int)
    y_true = y_true.astype(int)
    return {
        "accuracy":          round(accuracy_score(y_true, y_pred) * 100, 4),
        "precision":         round(precision_score(y_true, y_pred, zero_division=0) * 100, 4),
        "recall":            round(recall_score(y_true, y_pred, zero_division=0) * 100, 4),
        "f1":                round(f1_score(y_true, y_pred, zero_division=0) * 100, 4),
        "average_precision": round(average_precision_score(y_true, y_prob), 6),
    }


# ──────────────────────────────────────────────────────────────
# McNEMAR TEST
# ──────────────────────────────────────────────────────────────
def mcnemar_test(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray) -> dict:
    """
    McNemar's test comparing two classifiers on the SAME test set.
    Uses the 2×2 contingency table of disagreements.

    Implementation uses scipy.stats.chi2_contingency on the
    discordant pairs table — equivalent to McNemar's test with
    continuity correction (Yates' correction for 2×2 tables).

    statsmodels is NOT required.

    Returns: {'statistic': float, 'p_value': float}
    """
    y_true  = y_true.astype(int)
    preds_a = preds_a.astype(int)
    preds_b = preds_b.astype(int)

    # Contingency table of disagreements
    # [A correct, B wrong] vs [A wrong, B correct]
    a_right_b_wrong = int(np.sum((preds_a == y_true) & (preds_b != y_true)))
    a_wrong_b_right = int(np.sum((preds_a != y_true) & (preds_b == y_true)))
    both_right      = int(np.sum((preds_a == y_true) & (preds_b == y_true)))
    both_wrong      = int(np.sum((preds_a != y_true) & (preds_b != y_true)))

    # Correct McNemar test with continuity correction.
    # McNemar uses ONLY the two discordant counts b and c.
    # b = A correct, B wrong (a_right_b_wrong)
    # c = A wrong,  B correct (a_wrong_b_right)
    b = a_right_b_wrong
    c = a_wrong_b_right

    if (b + c) == 0:
        # No disagreements — models are identical on this test set
        chi2_stat = 0.0
        p_value   = 1.0
    else:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        from scipy.stats import chi2 as chi2_dist
        p_value = 1.0 - chi2_dist.cdf(chi2_stat, df=1)

    return {
        "statistic": round(float(chi2_stat), 6),
        "p_value":   round(float(p_value), 8),
        "a_right_b_wrong": a_right_b_wrong,
        "a_wrong_b_right": a_wrong_b_right,
    }


# ──────────────────────────────────────────────────────────────
# MAIN EXPERIMENT LOOP
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("TRUSTNET STATISTICAL COMPARISON — 5 seeds × 5 models")
    print("=" * 70)

    # Verify dataset exists
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run fraud_data_simulator.py first.")

    # Load data
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype("float32")
    y = df[LABEL_COL].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype("float32")
    print(f"Loaded {len(y)} samples | fraud={int(y.sum())} | legit={int((y==0).sum())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(os.path.join(SCRIPT_DIR, "outputs"), exist_ok=True)

    all_runs    = []   # rows for statistical_runs.csv
    # Store per-seed predictions for McNemar
    # {seed: {model_name: (y_true, y_pred_binary, y_prob)}}
    all_preds   = {}

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"SEED = {seed}")
        print(f"{'='*70}")

        set_all_seeds(seed)

        # Split and normalize — identical for ALL models at this seed
        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         mean, std) = make_split_and_normalize(X, y, seed)

        print(f"  Split: train={len(X_train)} val={len(X_val)} test={len(X_test)}")
        print(f"  Test fraud: {int(y_test.sum())} / {len(y_test)}")

        # Positive class weight (same formula as Transformer)
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        pos_weight = neg / pos
        print(f"  pos_weight: {pos_weight:.4f}")

        all_preds[seed] = {}

        # ── 1. TrustNet Transformer ──────────────────────────
        print("\n  [1/5] TrustNet Transformer")
        set_all_seeds(seed)   # re-seed before each model init
        transformer = TrustNetTransformer(num_features=5)
        transformer = train_torch_model(
            transformer, X_train, y_train, X_val, y_val, pos_weight, device)
        probs_t = predict_torch(transformer, X_test, device)
        metrics_t = evaluate(y_test, probs_t)
        print(f"    F1={metrics_t['f1']}%  AP={metrics_t['average_precision']}")
        all_runs.append({"model": "Transformer", "seed": seed, **metrics_t})
        all_preds[seed]["Transformer"] = (
            y_test, (probs_t >= THRESHOLD).astype(int), probs_t)

        # ── 2. MLP ──────────────────────────────────────────
        print("  [2/5] MLP")
        set_all_seeds(seed)
        mlp = MLPBaseline(num_features=5)
        mlp = train_torch_model(
            mlp, X_train, y_train, X_val, y_val, pos_weight, device)
        probs_m = predict_torch(mlp, X_test, device)
        metrics_m = evaluate(y_test, probs_m)
        print(f"    F1={metrics_m['f1']}%  AP={metrics_m['average_precision']}")
        all_runs.append({"model": "MLP", "seed": seed, **metrics_m})
        all_preds[seed]["MLP"] = (
            y_test, (probs_m >= THRESHOLD).astype(int), probs_m)

        # ── 3. Logistic Regression ───────────────────────────
        print("  [3/5] Logistic Regression")
        set_all_seeds(seed)
        lr_model = LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=seed)
        lr_model.fit(X_train, y_train)
        probs_lr = lr_model.predict_proba(X_test)[:, 1]
        metrics_lr = evaluate(y_test, probs_lr)
        print(f"    F1={metrics_lr['f1']}%  AP={metrics_lr['average_precision']}")
        all_runs.append({"model": "Logistic Regression", "seed": seed, **metrics_lr})
        all_preds[seed]["Logistic Regression"] = (
            y_test, (probs_lr >= THRESHOLD).astype(int), probs_lr)

        # ── 4. Random Forest ────────────────────────────────
        print("  [4/5] Random Forest")
        set_all_seeds(seed)
        rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=None,
            class_weight="balanced", n_jobs=-1, random_state=seed)
        rf_model.fit(X_train, y_train)
        probs_rf = rf_model.predict_proba(X_test)[:, 1]
        metrics_rf = evaluate(y_test, probs_rf)
        print(f"    F1={metrics_rf['f1']}%  AP={metrics_rf['average_precision']}")
        all_runs.append({"model": "Random Forest", "seed": seed, **metrics_rf})
        all_preds[seed]["Random Forest"] = (
            y_test, (probs_rf >= THRESHOLD).astype(int), probs_rf)

        # ── 5. Gradient Boosting ─────────────────────────────
        print("  [5/5] Gradient Boosting")
        set_all_seeds(seed)
        gb_model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1,
            max_depth=3, random_state=seed)
        sample_weight = np.where(y_train == 1, pos_weight, 1.0)
        gb_model.fit(X_train, y_train, sample_weight=sample_weight)
        probs_gb = gb_model.predict_proba(X_test)[:, 1]
        metrics_gb = evaluate(y_test, probs_gb)
        print(f"    F1={metrics_gb['f1']}%  AP={metrics_gb['average_precision']}")
        all_runs.append({"model": "Gradient Boosting", "seed": seed, **metrics_gb})
        all_preds[seed]["Gradient Boosting"] = (
            y_test, (probs_gb >= THRESHOLD).astype(int), probs_gb)

    # ──────────────────────────────────────────────────────────
    # SAVE INDIVIDUAL RUNS
    # ──────────────────────────────────────────────────────────
    runs_df = pd.DataFrame(all_runs)
    runs_df.to_csv(RUNS_PATH, index=False)
    print(f"\nIndividual runs saved: {RUNS_PATH}")

    # ──────────────────────────────────────────────────────────
    # COMPUTE SUMMARY (mean ± std)
    # ──────────────────────────────────────────────────────────
    metric_cols = ["accuracy", "precision", "recall", "f1", "average_precision"]
    summary_rows = []
    for model_name in runs_df["model"].unique():
        sub = runs_df[runs_df["model"] == model_name]
        row = {"model": model_name}
        for col in metric_cols:
            row[f"mean_{col}"]   = round(sub[col].mean(), 4)
            row[f"std_{col}"]    = round(sub[col].std(ddof=1), 4)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    print(f"Summary saved: {SUMMARY_PATH}")

    # ──────────────────────────────────────────────────────────
    # McNEMAR TEST (Transformer vs each other model)
    # ──────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("McNEMAR TEST RESULTS (Transformer vs each model, per seed)")
    print("="*70)

    mcnemar_rows = []
    other_models = ["MLP", "Logistic Regression", "Random Forest", "Gradient Boosting"]

    for seed in SEEDS:
        y_true_t, y_pred_t, _ = all_preds[seed]["Transformer"]
        for other in other_models:
            y_true_o, y_pred_o, _ = all_preds[seed][other]
            result = mcnemar_test(y_true_t, y_pred_t, y_pred_o)
            mcnemar_rows.append({
                "seed":       seed,
                "model_a":    "Transformer",
                "model_b":    other,
                "statistic":  result["statistic"],
                "p_value":    result["p_value"],
                "a_right_b_wrong": result["a_right_b_wrong"],
                "a_wrong_b_right": result["a_wrong_b_right"],
            })
            sig = "**" if result["p_value"] < 0.05 else "  "
            print(f"  {sig} Transformer vs {other:<25} "
                  f"seed={seed}  chi2={result['statistic']:.4f}  "
                  f"p={result['p_value']:.4f}")

    mcnemar_df = pd.DataFrame(mcnemar_rows)
    mcnemar_df.to_csv(MCNEMAR_PATH, index=False)
    print(f"\nMcNemar results saved: {MCNEMAR_PATH}")

    # ──────────────────────────────────────────────────────────
    # PRINT SUMMARY TABLE
    # ──────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SUMMARY: mean (std) across 5 seeds")
    print("="*70)
    print(f"{'Model':<28} {'Accuracy':>14} {'F1':>14} {'AP':>14}")
    print("-"*70)
    for _, row in summary_df.iterrows():
        print(f"{row['model']:<28} "
              f"{row['mean_accuracy']:>6.2f}±{row['std_accuracy']:<6.2f}  "
              f"{row['mean_f1']:>6.2f}±{row['std_f1']:<6.2f}  "
              f"{row['mean_average_precision']:>6.4f}±{row['std_average_precision']:<6.4f}")

    print(f"\nTotal runs:  {len(all_runs)}  ({len(SEEDS)} seeds × 5 models)")
    print(f"Files created:")
    print(f"  {RUNS_PATH}")
    print(f"  {SUMMARY_PATH}")
    print(f"  {MCNEMAR_PATH}")
    print("\nNOTE: statsmodels not required. McNemar test implemented "
          "via scipy.stats.chi2_contingency with Yates continuity correction.")
    print("NOTE: XGBoost not included — package not installed.")


if __name__ == "__main__":
    main()
