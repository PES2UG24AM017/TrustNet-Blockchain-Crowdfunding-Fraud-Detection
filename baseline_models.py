"""
baseline_models.py
-------------------
Implements the reviewer-requested ML baselines for TrustNet.

Baselines implemented:
  1. Logistic Regression   (sklearn)
  2. Random Forest         (sklearn)
  3. Gradient Boosting     (sklearn)
  4. XGBoost               — NOT AVAILABLE (xgboost not installed)

CRITICAL: This script uses EXACTLY the same experimental setup as
genarate_pr_curve.py and run_ablation.py to ensure fair comparison:

  - Same dataset:     outputs/trustnet_train.csv
  - Same features:    sdi, cti, fvrs, cfis, ctcs
  - Same split:       70% train / 15% val / 15% test
                      train_test_split(test_size=0.15, random_state=42, stratify=y)
                      train_test_split(test_size=0.17647, random_state=42, stratify=y_temp)
  - Same normalization: z-score fit ONLY on X_train (never on val/test)
                        — prevents data leakage from test distribution
  - Same test set:    IDENTICAL 600 samples (same random_state=42 + stratify)
  - Same threshold:   0.5 for all binary predictions
  - No test-set tuning: all hyperparameters fixed before seeing test data

Why these choices matter for fair comparison:
  - Using the same split guarantees all models are evaluated on
    identical test samples, making metric differences attributable
    to the model, not the data partition.
  - Fitting normalization only on training data ensures no
    information from the test or validation set leaks into training.
  - Fixing hyperparameters (no grid search on test) ensures reported
    numbers are not optimistically biased.

Usage:
    python baseline_models.py

Outputs:
    outputs/baseline_results.csv       — metrics for all baselines
    outputs/baselines/logistic_regression.pkl
    outputs/baselines/random_forest.pkl
    outputs/baselines/gradient_boosting.pkl
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, confusion_matrix,
)

# ──────────────────────────────────────────────────────────────
# CONFIG — must match genarate_pr_curve.py and run_ablation.py
# ──────────────────────────────────────────────────────────────
SEED      = 42          # identical to all existing experiments
THRESHOLD = 0.5         # identical to all existing experiments
FEATURES  = ["sdi", "cti", "fvrs", "cfis", "ctcs"]
LABEL_COL = "is_fraud"

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(SCRIPT_DIR, "outputs", "trustnet_train.csv")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "outputs", "baseline_results.csv")
MODELS_DIR   = os.path.join(SCRIPT_DIR, "outputs", "baselines")

# ──────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ──────────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            f"Ensure outputs/trustnet_train.csv exists before running."
        )
    df = pd.read_csv(DATA_PATH)
    for col in FEATURES + [LABEL_COL]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from {DATA_PATH}")
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype("float32")
    y = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).values.astype("float32")
    print(f"Loaded {len(y)} samples | "
          f"fraud={int(y.sum())} ({y.mean()*100:.1f}%) | "
          f"legit={int((y==0).sum())}")
    return X, y


# ──────────────────────────────────────────────────────────────
# STEP 2 — SPLIT (IDENTICAL to genarate_pr_curve.py)
# ──────────────────────────────────────────────────────────────
def make_split(X, y):
    """
    Reproduces the EXACT split used by the Transformer experiment.

    Step 1: carve out 15% as test (stratified, seed=42)
    Step 2: from remaining 85%, take 17.647% as val  → 15% overall
    Result: 70% train / 15% val / 15% test
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.15,
        random_state=SEED,   # identical seed
        stratify=y,          # identical stratification
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.17647,   # 15/85 ≈ 17.647%
        random_state=SEED,
        stratify=y_temp,
    )
    print(f"Split → train:{len(X_train)} val:{len(X_val)} test:{len(X_test)}")
    print(f"  train fraud:{int(y_train.sum())}  val fraud:{int(y_val.sum())}  test fraud:{int(y_test.sum())}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ──────────────────────────────────────────────────────────────
# STEP 3 — NORMALIZATION (fit ONLY on X_train)
# ──────────────────────────────────────────────────────────────
def normalize(X_train, X_val, X_test):
    """
    Z-score normalization identical to the Transformer experiment.

    IMPORTANT: mean and std are computed ONLY from X_train.
    Applying the training statistics to val/test prevents any
    information from the test distribution leaking into training.
    """
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std[std == 0] = 1.0   # prevent divide-by-zero for constant features

    X_train_n = (X_train - mean) / std
    X_val_n   = (X_val   - mean) / std
    X_test_n  = (X_test  - mean) / std

    print(f"Normalization: mean={mean.round(4).tolist()} std={std.round(4).tolist()}")
    return X_train_n, X_val_n, X_test_n, mean, std


# ──────────────────────────────────────────────────────────────
# STEP 4 — EVALUATE HELPER
# ──────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_prob, threshold=THRESHOLD):
    """
    Computes all metrics on the SAME 600-sample test set.
    Threshold=0.5 matches the Transformer evaluation exactly.
    """
    y_pred = (y_prob >= threshold).astype(int)
    y_true = y_true.astype(int)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    ap   = average_precision_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    results = {
        "Model":     name,
        "Accuracy":  round(acc  * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall":    round(rec  * 100, 2),
        "F1":        round(f1   * 100, 2),
        "AP":        round(ap, 4),
        "TP": int(tp), "TN": int(tn),
        "FP": int(fp), "FN": int(fn),
        "Total_test": len(y_true),
    }

    print(f"\n  {name}")
    print(f"    Acc={results['Accuracy']}%  Prec={results['Precision']}%  "
          f"Rec={results['Recall']}%  F1={results['F1']}%  AP={results['AP']}")
    print(f"    CM: TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    return results


# ──────────────────────────────────────────────────────────────
# STEP 5 — BASELINE DEFINITIONS
# ──────────────────────────────────────────────────────────────
def get_baselines(pos_weight_ratio):
    """
    Returns a dict of {name: (model, save_filename)}.

    Class imbalance handling:
      - LogisticRegression: class_weight='balanced' — sklearn automatically
        weights each class inversely proportional to its frequency,
        equivalent to the pos_weight used by BCEWithLogitsLoss.
      - RandomForestClassifier: class_weight='balanced' — same principle,
        applied per tree bootstrap sample.
      - GradientBoostingClassifier: does NOT support class_weight directly.
        Instead we pass sample_weight during fit() — computed as
        pos_weight_ratio for fraud samples and 1.0 for legit samples,
        which is mathematically equivalent to BCEWithLogitsLoss pos_weight.
        pos_weight_ratio = legit_count / fraud_count = 2.7135 (same as Transformer).
    """
    baselines = {
        "Logistic Regression": (
            LogisticRegression(
                C=1.0,              # default regularization strength
                max_iter=1000,      # enough for convergence on normalized data
                random_state=SEED,
                class_weight="balanced",  # equivalent to pos_weight in Transformer
                solver="lbfgs",
            ),
            "logistic_regression.pkl",
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=100,   # standard default
                max_depth=None,     # allow full depth (no tuning)
                random_state=SEED,
                class_weight="balanced",
                n_jobs=-1,
            ),
            "random_forest.pkl",
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,  # standard default
                max_depth=3,        # standard default
                random_state=SEED,
                # class_weight not supported — use sample_weight in fit()
            ),
            "gradient_boosting.pkl",
        ),
    }
    return baselines


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("TRUSTNET — BASELINE MODELS")
    print("=" * 70)

    # Verify output directory
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Models will be saved to: {MODELS_DIR}")
    print(f"Results will be saved to: {RESULTS_PATH}")

    # Step 1: Load
    X, y = load_data()

    # Step 2: Split
    X_train, X_val, X_test, y_train, y_val, y_test = make_split(X, y)

    # Step 3: Normalize (fit on train only)
    X_train_n, X_val_n, X_test_n, mean, std = normalize(X_train, X_val, X_test)

    # Class weight ratio for GradientBoosting sample_weight
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    pos_weight_ratio = neg / pos
    print(f"\npos_weight ratio: {pos_weight_ratio:.4f} "
          f"(legit={neg} / fraud={pos}), same as Transformer")

    # Baselines
    baselines = get_baselines(pos_weight_ratio)

    print("\n" + "=" * 70)
    print("TRAINING AND EVALUATING BASELINES")
    print("=" * 70)

    all_results = []

    for name, (model, filename) in baselines.items():
        print(f"\n--- {name} ---")

        # GradientBoosting needs sample_weight instead of class_weight
        if isinstance(model, GradientBoostingClassifier):
            sample_weight = np.where(y_train == 1, pos_weight_ratio, 1.0)
            model.fit(X_train_n, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train_n, y_train)

        # Predict probabilities for the positive class (fraud = class 1)
        y_prob = model.predict_proba(X_test_n)[:, 1]

        # Evaluate on the SAME 600-sample test set
        result = evaluate(name, y_test, y_prob)
        all_results.append(result)

        # Save trained model
        model_path = os.path.join(MODELS_DIR, filename)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"    Saved: {model_path}")

    # Save results CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nAll results saved to: {RESULTS_PATH}")

    # Print summary table
    print("\n" + "=" * 70)
    print("BASELINE RESULTS SUMMARY")
    print(f"{'Model':<30} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AP':>7}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['Model']:<30} {r['Accuracy']:>6}% {r['Precision']:>6}% "
              f"{r['Recall']:>6}% {r['F1']:>6}% {r['AP']:>7}")

    print("\nFor comparison, TrustNet Transformer:")
    print(f"  Acc=96.17%  Prec=96.62%  Rec=88.82%  F1=92.56%  AP=0.9190")
    print(f"  CM: TN=434  FP=5  FN=18  TP=143")
    print("\nNote: XGBoost not available (module not installed).")
    print("=" * 70)


# ──────────────────────────────────────────────────────────────
# SYNTAX/IMPORT CHECK MODE
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
