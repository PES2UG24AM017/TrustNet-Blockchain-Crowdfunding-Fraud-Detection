"""
generate_cfis_graph.py
-----------------------
Runs the real CFIS pipeline on unified_dataset.csv and saves
an authentic wallet co-funding graph image for use in the paper.

Two images are saved to outputs/plots/:
  1. cfis_wallet_graph_full.png   - all campaigns, may be large
  2. cfis_wallet_graph_paper.png  - focused on campaigns with
                                    reused wallets (cleaner for paper)

Run from the project root:
    python generate_cfis_graph.py
"""

import os
import sys

# ── Path setup ──────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_DIR, "cfis"))

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from cfis_updated import (
    build_cofunding_graph,
    detect_communities,
    save_graph_visualization,
    graph_statistics,
)

# ── Config ──────────────────────────────────────────────────
DATASET_PATH  = os.path.join(PROJECT_DIR, "Datasets", "unified_dataset.csv")
PLOTS_DIR     = os.path.join(PROJECT_DIR, "outputs", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

OUTPUT_FULL   = os.path.join(PLOTS_DIR, "cfis_wallet_graph_full.png")
OUTPUT_PAPER  = os.path.join(PLOTS_DIR, "cfis_wallet_graph_paper.png")

# ── Load data ───────────────────────────────────────────────
print("Loading dataset:", DATASET_PATH)
df = pd.read_csv(DATASET_PATH)
print(f"  {len(df)} rows, {df['campaign_id'].nunique()} campaigns, "
      f"{df['wallet_id'].nunique()} unique wallets")

# ── Build campaign → wallet sets ────────────────────────────
campaign_wallets = (
    df.groupby("campaign_id")["wallet_id"]
      .apply(lambda w: set(w.str.lower()))
      .to_dict()
)

# ── Build the co-funding graph (min_shared_campaigns=2) ─────
print("\nBuilding wallet co-funding graph (min_shared_campaigns=2)...")
graph = build_cofunding_graph(campaign_wallets, min_shared_campaigns=2)
stats = graph_statistics(graph)
print(f"  Nodes (wallets): {stats['num_wallets']}")
print(f"  Edges (shared funding): {stats['num_edges']}")
print(f"  Isolated wallets: {stats['isolated_wallets']}")
print(f"  Largest connected component: {stats['largest_component_size']}")

communities = detect_communities(graph)
num_communities = len(set(communities.values()))
print(f"  Communities detected: {num_communities}")

# ── Save full graph using built-in function ─────────────────
print(f"\nSaving full graph to: {OUTPUT_FULL}")
save_graph_visualization(graph, path=OUTPUT_FULL, max_nodes=500)

# ── Build a paper-quality focused graph ─────────────────────
# Show only wallets that are actually connected (have edges)
# Color by community — this matches what the paper describes
print(f"\nBuilding paper-quality focused graph...")

connected_nodes = [n for n in graph.nodes() if graph.degree(n) > 0]
print(f"  Connected wallets (degree > 0): {len(connected_nodes)}")

if len(connected_nodes) == 0:
    print("  No connected wallets found — all wallets only appear in one campaign.")
    print("  Lowering threshold to min_shared_campaigns=1 for visualization...")
    graph = build_cofunding_graph(campaign_wallets, min_shared_campaigns=1)
    communities = detect_communities(graph)
    connected_nodes = [n for n in graph.nodes() if graph.degree(n) > 0]
    print(f"  Connected wallets (min=1): {len(connected_nodes)}")

# Take top campaigns with most wallet reuse for a clean diagram
subgraph = graph.subgraph(connected_nodes[:min(80, len(connected_nodes))])

fig, ax = plt.subplots(figsize=(10, 8))

# Layout
try:
    pos = nx.spring_layout(subgraph, seed=42, k=1.5)
except Exception:
    pos = nx.random_layout(subgraph, seed=42)

# Color nodes by community
community_ids   = [communities.get(n, 0) for n in subgraph.nodes()]
unique_comms    = list(set(community_ids))
cmap            = plt.cm.get_cmap("tab10", max(len(unique_comms), 2))
node_colors     = [cmap(unique_comms.index(c)) for c in community_ids]

# Edge widths by weight
weights = [subgraph[u][v].get("weight", 1) for u, v in subgraph.edges()]
max_w   = max(weights) if weights else 1
edge_widths = [1 + 3 * (w / max_w) for w in weights]

nx.draw_networkx_nodes(
    subgraph, pos, ax=ax,
    node_color=node_colors,
    node_size=120,
    alpha=0.85,
)
nx.draw_networkx_edges(
    subgraph, pos, ax=ax,
    width=edge_widths,
    edge_color="gray",
    alpha=0.5,
)

# Legend — one entry per community
patches = [
    mpatches.Patch(color=cmap(i), label=f"Community {unique_comms[i]}")
    for i in range(min(len(unique_comms), 8))
]
ax.legend(handles=patches, loc="upper right", fontsize=8,
          title="Wallet Communities", title_fontsize=9)

ax.set_title(
    "Wallet Co-funding Graph — TrustNet CFIS Module\n"
    f"({len(subgraph.nodes())} wallets · {len(subgraph.edges())} edges · "
    f"{len(unique_comms)} communities)",
    fontsize=12,
)
ax.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_PAPER, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Paper graph saved to: {OUTPUT_PAPER}")

# ── Print wallet reuse stats for paper ──────────────────────
print("\n── Wallet Reuse Statistics (for paper) ──")
wallet_campaign_counts = (
    df.groupby("wallet_id")["campaign_id"].nunique()
)
reused = wallet_campaign_counts[wallet_campaign_counts > 1]
print(f"  Total wallets: {len(wallet_campaign_counts)}")
print(f"  Wallets appearing in 2+ campaigns: {len(reused)}")
print(f"  Max campaigns per wallet: {wallet_campaign_counts.max()}")
if len(reused) > 0:
    print(f"  Top 5 most reused wallets:")
    for w, c in reused.nlargest(5).items():
        print(f"    {w[:20]}...  →  {c} campaigns")

print("\nDone. Images saved to outputs/plots/")
print("  Use cfis_wallet_graph_paper.png in the paper (Fig. 2)")
