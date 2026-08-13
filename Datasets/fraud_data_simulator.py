"""
Fraud Training Data Simulator
--------------------------------
Generates a labeled synthetic dataset for training TrustNet —
the PyTorch model that fuses SDI, CTI, FVRS, CFIS, and CTCS
into one final fraud probability.

Each row = one campaign with:
  sdi, cti, fvrs, cfis, ctcs  → the 5 trust metric scores (0 to 1)
  is_fraud                    → ground truth label (1 = fraud, 0 = legit)

Design principle:
  Fraudulent campaigns tend to have HIGH sdi/fvrs/cfis/ctcs and
  LOW cti — but with realistic noise and overlap, so the model
  actually has to learn patterns rather than a trivial threshold.

Usage:
    python fraud_data_simulator.py
"""

import os
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
NUM_CAMPAIGNS   = 5000
FRAUD_RATIO     = 0.25     # 25% of campaigns are fraudulent
NOISE_STD       = 0.12     # how much random noise/overlap to inject
SEED            = 42

OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../outputs")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "trustnet_training_data.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(SEED)


# ─────────────────────────────────────────────
# CLIP HELPER
# ─────────────────────────────────────────────
def clip01(x):
    return np.clip(x, 0.0, 1.0)


# ─────────────────────────────────────────────
# GENERATE LEGITIMATE CAMPAIGNS
# ─────────────────────────────────────────────
def generate_legit_campaigns(n):
    """
    Legit campaigns:
      sdi  → low  (original content)      mean ~0.20
      cti  → high (trustworthy signals)    mean ~0.70
      fvrs → low  (normal funding speed)   mean ~0.25
      cfis → low  (no collusion)           mean ~0.20
      ctcs → low  (no cross-campaign sync) mean ~0.20
    """
    sdi  = clip01(np.random.normal(0.20, NOISE_STD, n))
    cti  = clip01(np.random.normal(0.70, NOISE_STD, n))
    fvrs = clip01(np.random.normal(0.25, NOISE_STD, n))
    cfis = clip01(np.random.normal(0.20, NOISE_STD, n))
    ctcs = clip01(np.random.normal(0.20, NOISE_STD, n))

    return pd.DataFrame({
        'sdi': sdi, 'cti': cti, 'fvrs': fvrs,
        'cfis': cfis, 'ctcs': ctcs, 'is_fraud': 0
    })


# ─────────────────────────────────────────────
# GENERATE FRAUDULENT CAMPAIGNS
# ─────────────────────────────────────────────
def generate_fraud_campaigns(n):
    """
    Fraudulent campaigns are split into 3 sub-patterns so the
    model sees DIFFERENT kinds of fraud, not just one signature:

    Pattern A — Plagiarism-driven fraud (high SDI dominant)
      copied campaign text, otherwise looks fairly normal

    Pattern B — Velocity/collusion-driven fraud (high FVRS+CFIS)
      bot-funded, fast suspicious money movement, low-effort text

    Pattern C — Coordinated multi-campaign fraud (high CTCS)
      same actors running several synced campaigns

    Real fraud is rarely "high on everything" — mixing patterns
    makes the classification task realistic and non-trivial.
    """
    n_a = n // 3
    n_b = n // 3
    n_c = n - n_a - n_b

    # Pattern A: plagiarism-heavy
    a_sdi  = clip01(np.random.normal(0.80, NOISE_STD, n_a))
    a_cti  = clip01(np.random.normal(0.45, NOISE_STD, n_a))
    a_fvrs = clip01(np.random.normal(0.35, NOISE_STD, n_a))
    a_cfis = clip01(np.random.normal(0.30, NOISE_STD, n_a))
    a_ctcs = clip01(np.random.normal(0.30, NOISE_STD, n_a))

    # Pattern B: velocity/collusion-heavy
    b_sdi  = clip01(np.random.normal(0.35, NOISE_STD, n_b))
    b_cti  = clip01(np.random.normal(0.30, NOISE_STD, n_b))
    b_fvrs = clip01(np.random.normal(0.85, NOISE_STD, n_b))
    b_cfis = clip01(np.random.normal(0.80, NOISE_STD, n_b))
    b_ctcs = clip01(np.random.normal(0.40, NOISE_STD, n_b))

    # Pattern C: coordinated multi-campaign
    c_sdi  = clip01(np.random.normal(0.40, NOISE_STD, n_c))
    c_cti  = clip01(np.random.normal(0.35, NOISE_STD, n_c))
    c_fvrs = clip01(np.random.normal(0.50, NOISE_STD, n_c))
    c_cfis = clip01(np.random.normal(0.55, NOISE_STD, n_c))
    c_ctcs = clip01(np.random.normal(0.85, NOISE_STD, n_c))

    sdi  = np.concatenate([a_sdi,  b_sdi,  c_sdi])
    cti  = np.concatenate([a_cti,  b_cti,  c_cti])
    fvrs = np.concatenate([a_fvrs, b_fvrs, c_fvrs])
    cfis = np.concatenate([a_cfis, b_cfis, c_cfis])
    ctcs = np.concatenate([a_ctcs, b_ctcs, c_ctcs])

    return pd.DataFrame({
        'sdi': sdi, 'cti': cti, 'fvrs': fvrs,
        'cfis': cfis, 'ctcs': ctcs, 'is_fraud': 1
    })


# ─────────────────────────────────────────────
# ADD LABEL NOISE (realistic imperfect ground truth)
# ─────────────────────────────────────────────
def add_label_noise(df, flip_rate=0.03):
    """
    Real-world fraud labels are never 100% clean — some fraud
    goes undetected (false negatives) and some legit campaigns
    get incorrectly flagged historically (false positives).
    Flipping a small percentage of labels makes TrustNet more
    robust and avoids training on an unrealistically perfect
    separation between classes.
    """
    n_flip = int(len(df) * flip_rate)
    flip_idx = np.random.choice(df.index, size=n_flip, replace=False)
    df.loc[flip_idx, 'is_fraud'] = 1 - df.loc[flip_idx, 'is_fraud']
    print(f"  Flipped {n_flip} labels to simulate realistic "
          f"ground-truth noise ({flip_rate*100:.0f}%).", flush=True)
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60, flush=True)
    print("  FRAUD TRAINING DATA SIMULATOR", flush=True)
    print("  Generating labeled dataset for TrustNet", flush=True)
    print("=" * 60, flush=True)

    n_fraud = int(NUM_CAMPAIGNS * FRAUD_RATIO)
    n_legit = NUM_CAMPAIGNS - n_fraud

    print(f"\n  Total campaigns : {NUM_CAMPAIGNS}", flush=True)
    print(f"  Legitimate      : {n_legit} ({(1-FRAUD_RATIO)*100:.0f}%)", flush=True)
    print(f"  Fraudulent      : {n_fraud} ({FRAUD_RATIO*100:.0f}%)", flush=True)
    print(f"    → split across 3 fraud patterns "
          f"(plagiarism / velocity+collusion / coordinated)", flush=True)

    print("\n  Generating legitimate campaigns...", flush=True)
    legit_df = generate_legit_campaigns(n_legit)

    print("  Generating fraudulent campaigns...", flush=True)
    fraud_df = generate_fraud_campaigns(n_fraud)

    print("\n  Combining and shuffling...", flush=True)
    full_df = pd.concat([legit_df, fraud_df], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print("\n  Adding realistic label noise...", flush=True)
    full_df = add_label_noise(full_df)

    # Round scores for readability
    for col in ['sdi', 'cti', 'fvrs', 'cfis', 'ctcs']:
        full_df[col] = full_df[col].round(4)

    # ── Train/test split ──
    split_idx = int(len(full_df) * 0.8)
    train_df = full_df.iloc[:split_idx].reset_index(drop=True)
    test_df  = full_df.iloc[split_idx:].reset_index(drop=True)

    train_path = os.path.join(OUTPUT_DIR, "trustnet_train.csv")
    test_path  = os.path.join(OUTPUT_DIR, "trustnet_test.csv")

    full_df.to_csv(OUTPUT_PATH, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n" + "=" * 60, flush=True)
    print("  SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  Full dataset : {len(full_df)} rows → {OUTPUT_PATH}", flush=True)
    print(f"  Train split  : {len(train_df)} rows → {train_path}", flush=True)
    print(f"  Test split   : {len(test_df)} rows → {test_path}", flush=True)

    print(f"\n  Final label distribution:", flush=True)
    print(f"    Fraud (1) : {(full_df['is_fraud']==1).sum()} "
          f"({(full_df['is_fraud']==1).mean()*100:.1f}%)", flush=True)
    print(f"    Legit (0) : {(full_df['is_fraud']==0).sum()} "
          f"({(full_df['is_fraud']==0).mean()*100:.1f}%)", flush=True)

    print(f"\n  Mean scores by class:", flush=True)
    print(full_df.groupby('is_fraud')[
        ['sdi', 'cti', 'fvrs', 'cfis', 'ctcs']
    ].mean().round(3).to_string(), flush=True)

    print("\n  Sample rows:", flush=True)
    print(full_df.head(8).to_string(), flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
