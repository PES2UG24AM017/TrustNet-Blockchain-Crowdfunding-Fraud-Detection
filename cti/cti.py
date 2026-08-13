"""
CTI Batch Pipeline — Crowd Trust Index (All Campaigns At Once)
------------------------------------------------------------------
Computes CTI for every campaign in the dataset in one run,
no interactive prompts. Saves full results to CSV.

Reads from unified_dataset.csv (produced by dataset_simulator.py)
if it exists, otherwise falls back to gitcoin_grants.csv.

CTI = (A + T + M) / 3
  A = Approval Score
  T = Team Score        (normalized)
  M = Multi-chain Score

Usage:
    python cti_batch.py
"""

import os
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# CONFIG — dataset fallback logic
# ─────────────────────────────────────────────
UNIFIED_PATH = os.path.join(os.path.dirname(__file__), "../Datasets/unified_dataset.csv")
GITCOIN_PATH = os.path.join(os.path.dirname(__file__), "../Datasets/gitcoin_grants.csv")

USING_UNIFIED = os.path.exists(UNIFIED_PATH)
DATASET_PATH  = UNIFIED_PATH if USING_UNIFIED else GITCOIN_PATH

CTI_HIGH = 0.70
CTI_LOW  = 0.40

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "../outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUTS_DIR, "cti_features.csv")

# ─────────────────────────────────────────────
# COLUMN NAMES — differ depending on data source
# ─────────────────────────────────────────────
if USING_UNIFIED:
    COL_ID         = 'campaign_id'
    COL_TITLE      = 'title'
    COL_APPROVED   = 'approved'
    COL_TEAM_SIZE  = 'team_size'
    COL_MULTICHAIN = 'multichain'
else:
    COL_ID         = 'grant_id'
    COL_TITLE      = 'project_title'
    COL_APPROVED   = 'project_approved'
    COL_TEAM_SIZE  = 'project_team_size'
    COL_MULTICHAIN = 'live_on_other_chains'


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    print("=" * 60, flush=True)
    print("  CTI BATCH PIPELINE — All Campaigns", flush=True)
    print("=" * 60, flush=True)

    print(f"\n[1/3] Loading dataset...", flush=True)
    print(f"    Source: {DATASET_PATH}", flush=True)
    print(f"    Using {'unified/simulated' if USING_UNIFIED else 'Gitcoin raw'} "
          f"column schema.", flush=True)

    df = pd.read_csv(DATASET_PATH)

    # Unified dataset has one row per TRANSACTION —
    # CTI only needs one row per campaign
    if USING_UNIFIED:
        df = df.drop_duplicates(subset=COL_ID).reset_index(drop=True)

    print(f"    {len(df)} unique campaigns loaded.", flush=True)
    return df


# ─────────────────────────────────────────────
# COMPUTE CTI FOR ALL CAMPAIGNS
# ─────────────────────────────────────────────
def compute_all_cti(df):
    print("\n[2/3] Computing CTI scores for all campaigns...", flush=True)

    cti_df = df[[COL_ID, COL_TITLE, COL_APPROVED,
                 COL_TEAM_SIZE, COL_MULTICHAIN]].copy()
    cti_df.columns = ['campaign_id', 'title', 'approved',
                       'team_size', 'multichain']

    # A: Approval Score
    cti_df['A'] = cti_df['approved'].apply(
        lambda x: 1.0 if str(x).strip().lower() == 'true' else 0.0
    )

    # T: Team Size Score (normalized 0 to 1)
    team = pd.to_numeric(cti_df['team_size'], errors='coerce').fillna(1)
    team_range = team.max() - team.min()
    cti_df['T'] = ((team - team.min()) / (team_range + 1e-9)).round(4)

    # M: Multi-chain Score
    positive_keywords = ['yes', 'live', 'deployed', 'ethereum',
                         'polygon', 'bsc', 'avalanche', 'optimism',
                         'arbitrum', 'testnet', 'mainnet']
    def multichain_score(val):
        return 1.0 if any(
            kw in str(val).lower() for kw in positive_keywords
        ) else 0.0

    cti_df['M'] = cti_df['multichain'].apply(multichain_score)

    # Final CTI (raw)
    cti_df['cti_score'] = (
        (cti_df['A'] + cti_df['T'] + cti_df['M']) / 3
    ).round(4)

    # Normalize to [0, 1] across the whole dataset
    cti_min   = cti_df['cti_score'].min()
    cti_max   = cti_df['cti_score'].max()
    cti_range = cti_max - cti_min
    cti_df['cti_norm'] = (
        (cti_df['cti_score'] - cti_min) / (cti_range + 1e-9)
    ).round(4)

    # Trust label
    def label_trust(score):
        if score >= CTI_HIGH:
            return "HIGH TRUST"
        elif score >= CTI_LOW:
            return "MEDIUM TRUST"
        else:
            return "LOW TRUST"

    cti_df['trust_label'] = cti_df['cti_norm'].apply(label_trust)

    print(f"    CTI computed for {len(cti_df)} campaigns.", flush=True)
    return cti_df


# ─────────────────────────────────────────────
# SUMMARY STATS
# ─────────────────────────────────────────────
def print_summary(cti_df):
    print("\n[3/3] Summary", flush=True)
    print("-" * 60, flush=True)

    label_counts = cti_df['trust_label'].value_counts()
    print("  Trust distribution:", flush=True)
    for label in ["HIGH TRUST", "MEDIUM TRUST", "LOW TRUST"]:
        count = label_counts.get(label, 0)
        pct   = (count / len(cti_df) * 100) if len(cti_df) > 0 else 0
        print(f"    {label:<15} {count:>6}  ({pct:.1f}%)", flush=True)

    print(f"\n  CTI score stats:", flush=True)
    print(f"    Mean:   {cti_df['cti_norm'].mean():.4f}", flush=True)
    print(f"    Median: {cti_df['cti_norm'].median():.4f}", flush=True)
    print(f"    Min:    {cti_df['cti_norm'].min():.4f}", flush=True)
    print(f"    Max:    {cti_df['cti_norm'].max():.4f}", flush=True)

    print(f"\n  Top 5 most trusted campaigns:", flush=True)
    top5 = cti_df.nlargest(5, 'cti_norm')
    for _, r in top5.iterrows():
        print(f"    {r['cti_norm']:.4f}  {str(r['title'])[:50]}", flush=True)

    print(f"\n  Bottom 5 least trusted campaigns:", flush=True)
    bottom5 = cti_df.nsmallest(5, 'cti_norm')
    for _, r in bottom5.iterrows():
        print(f"    {r['cti_norm']:.4f}  {str(r['title'])[:50]}", flush=True)

    print("-" * 60, flush=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    df     = load_data()
    cti_df = compute_all_cti(df)
    print_summary(cti_df)

    # Save full results
    output_cols = ['campaign_id', 'title', 'A', 'T', 'M',
                   'cti_score', 'cti_norm', 'trust_label']
    cti_df[output_cols].to_csv(OUTPUT_PATH, index=False)

    print(f"\nAll {len(cti_df)} campaign CTI scores saved to:", flush=True)
    print(f"  {OUTPUT_PATH}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()