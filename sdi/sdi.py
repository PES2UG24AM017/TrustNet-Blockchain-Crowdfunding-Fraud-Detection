"""
SDI Batch Pipeline — Semantic Drift Index (All Campaigns At Once)
------------------------------------------------------------------
Computes SDI for every campaign in the dataset in one run,
no interactive prompts. Saves full results to CSV.

Reads from unified_dataset.csv (produced by dataset_simulator.py)
if it exists, otherwise falls back to gitcoin_grants.csv.

Uses all-MiniLM-L6-v2 Sentence Transformer + FAISS for
cosine similarity search across all campaign descriptions.

Usage:
    python sdi_batch.py
"""

import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# CONFIG — dataset fallback logic
# ─────────────────────────────────────────────
UNIFIED_PATH = os.path.join(os.path.dirname(__file__), "../Datasets/unified_dataset.csv")
GITCOIN_PATH = os.path.join(os.path.dirname(__file__), "../Datasets/gitcoin_grants.csv")

USING_UNIFIED = os.path.exists(UNIFIED_PATH)
DATASET_PATH  = UNIFIED_PATH if USING_UNIFIED else GITCOIN_PATH

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "../models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "../outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUTS_DIR, "sdi_features.csv")

SDI_HIGH   = 0.55   # tuned for small datasets
SDI_MEDIUM = 0.40
TOP_K      = 5

# ─────────────────────────────────────────────
# COLUMN NAMES — differ depending on data source
# ─────────────────────────────────────────────
if USING_UNIFIED:
    COL_ID          = 'campaign_id'
    COL_TITLE       = 'title'
    COL_DESCRIPTION = 'description'
    
else:
    COL_ID          = 'grant_id'
    COL_TITLE       = 'project_title'
    COL_DESCRIPTION = 'project_decription'
   


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    print("=" * 60, flush=True)
    print("  SDI BATCH PIPELINE — All Campaigns", flush=True)
    print("=" * 60, flush=True)

    print(f"\n[1/4] Loading dataset...", flush=True)
    print(f"    Source: {DATASET_PATH}", flush=True)
    print(f"    Using {'unified/simulated' if USING_UNIFIED else 'Gitcoin raw'} "
          f"column schema.", flush=True)

    df = pd.read_csv(DATASET_PATH)

    # Unified dataset has one row per TRANSACTION —
    # SDI only needs one row per campaign
    if USING_UNIFIED:
        df = df.drop_duplicates(subset=COL_ID).reset_index(drop=True)

    before = len(df)
    df = df.dropna(subset=[COL_DESCRIPTION]).copy()
    skipped = before - len(df)
    if skipped > 0:
        print(f"    Skipped {skipped} campaigns with no description "
              f"(SDI not computable for these).", flush=True)

    df[COL_DESCRIPTION] = df[COL_DESCRIPTION].astype(str).str.strip()
    df = df[df[COL_DESCRIPTION].str.len() > 20].reset_index(drop=True)

    print(f"    {len(df)} unique campaigns with valid descriptions loaded.",
          flush=True)
    return df


# ─────────────────────────────────────────────
# BUILD EMBEDDINGS + FAISS INDEX
# ─────────────────────────────────────────────
def build_index(df, model):
    print("\n[2/4] Building embeddings + FAISS index...", flush=True)

    embeddings = model.encode(
        df[COL_DESCRIPTION].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    print(f"    Index built with {index.ntotal} campaigns.", flush=True)
    return index, embeddings


# ─────────────────────────────────────────────
# COMPUTE SDI FOR ALL CAMPAIGNS
# ─────────────────────────────────────────────
def compute_all_sdi(df, index, embeddings):
    print("\n[3/4] Computing SDI scores for all campaigns...", flush=True)

    n = len(df)
    results = []

    # Search each campaign's embedding against the whole index at once
    scores_matrix, indices_matrix = index.search(embeddings, TOP_K + 1)

    for i in range(n):
        row = df.iloc[i]
        matches = []

        for score, idx in zip(scores_matrix[i], indices_matrix[i]):
            if idx == i:  # skip self-match
                continue
            matched = df.iloc[idx]
            matches.append({
                'title':      matched[COL_TITLE],
                'similarity': round(float(score), 4)
            })
            if len(matches) == TOP_K:
                break

        sdi_score = matches[0]['similarity'] if matches else 0.0

        if sdi_score >= SDI_HIGH:
            label = "HIGH PLAGIARISM RISK"
        elif sdi_score >= SDI_MEDIUM:
            label = "MODERATE RISK"
        else:
            label = "LIKELY ORIGINAL"

        results.append({
            'campaign_id':     row[COL_ID],
            'title':           row[COL_TITLE],
            'sdi_score':       sdi_score,
            'risk_label':      label,
            'top_match_1':     matches[0]['title'] if len(matches) > 0 else '',
            'top_match_1_sim': matches[0]['similarity'] if len(matches) > 0 else 0,
            'top_match_2':     matches[1]['title'] if len(matches) > 1 else '',
            'top_match_2_sim': matches[1]['similarity'] if len(matches) > 1 else 0,
        })

        if (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{n}...", flush=True)

    results_df = pd.DataFrame(results)
    print(f"    SDI computed for {len(results_df)} campaigns.", flush=True)
    return results_df


# ─────────────────────────────────────────────
# SUMMARY STATS
# ─────────────────────────────────────────────
def print_summary(results_df):
    print("\n[4/4] Summary", flush=True)
    print("-" * 60, flush=True)

    label_counts = results_df['risk_label'].value_counts()
    print("  Risk distribution:", flush=True)
    for label in ["HIGH PLAGIARISM RISK", "MODERATE RISK", "LIKELY ORIGINAL"]:
        count = label_counts.get(label, 0)
        pct   = (count / len(results_df) * 100) if len(results_df) > 0 else 0
        print(f"    {label:<22} {count:>6}  ({pct:.1f}%)", flush=True)

    print(f"\n  SDI score stats:", flush=True)
    print(f"    Mean:   {results_df['sdi_score'].mean():.4f}", flush=True)
    print(f"    Median: {results_df['sdi_score'].median():.4f}", flush=True)
    print(f"    Min:    {results_df['sdi_score'].min():.4f}", flush=True)
    print(f"    Max:    {results_df['sdi_score'].max():.4f}", flush=True)

    print(f"\n  Top 5 highest plagiarism risk:", flush=True)
    top5 = results_df.nlargest(5, 'sdi_score')
    for _, r in top5.iterrows():
        print(f"    {r['sdi_score']:.4f}  {str(r['title'])[:40]} "
              f"→ similar to {str(r['top_match_1'])[:35]}", flush=True)

    print(f"\n  Bottom 5 most original:", flush=True)
    bottom5 = results_df.nsmallest(5, 'sdi_score')
    for _, r in bottom5.iterrows():
        print(f"    {r['sdi_score']:.4f}  {str(r['title'])[:50]}", flush=True)

    print("-" * 60, flush=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    df = load_data()

    print("\n[0/4] Loading Sentence Transformer...", flush=True)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    try:
        dims = model.get_embedding_dimension()
    except AttributeError:
        dims = model.get_sentence_embedding_dimension()
    print(f"    Model ready ({dims} dims)", flush=True)

    index, embeddings = build_index(df, model)
    results_df = compute_all_sdi(df, index, embeddings)
    print_summary(results_df)

    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nAll {len(results_df)} campaign SDI scores saved to:", flush=True)
    print(f"  {OUTPUT_PATH}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()