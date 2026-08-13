import os
import ast
import random
import logging
import itertools
from typing import List, Dict, Any, Optional, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("cfis_module")

NULL_LIKE_WALLETS = {"", "none", "nan", "n/a", "na", "unknown", "null", "-"}


# ---------------------------------------------------------------------------
# Column auto-mapping helpers
# ---------------------------------------------------------------------------
def find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    """Finds the first matching column from a list of possible names."""
    columns = {c.lower(): c for c in df.columns}
    for name in possible_names:
        if name.lower() in columns:
            return columns[name.lower()]
    return None


def create_column_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Automatically map dataset columns to CFIS requirements."""
    return {
        "campaign_id": find_column(df, ["campaign_id", "campaign", "camp_id"]),
        "wallet_id": find_column(df, [
            "wallet_id", "wallet_address", "wallet", "user_id", "user_wallet"
        ]),
        "amount": find_column(df, ["amount", "amounts", "transaction_amount", "payments"]),
        "timestamp": find_column(df, [
            "timestamp", "timestamps", "transaction_time", "transaction_times",
            "time", "created_at", "funded_at"
        ]),
    }


def safe_str(value, default: str = "") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return str(value).strip()


def normalize_wallet_id(value: str) -> str:
    """
    Lowercases wallet identifiers so 0xABCDEF / 0xabcdef / 0xAbCdEf all
    collapse to the same graph node instead of being treated as three
    different wallets.
    """
    return value.strip().lower()


def is_null_like(value: str) -> bool:
    """Catches '', 'None', 'nan', 'N/A', 'unknown', etc. -- not just empty string."""
    return value.strip().lower() in NULL_LIKE_WALLETS


# ---------------------------------------------------------------------------
# Global wallet co-funding graph construction
# ---------------------------------------------------------------------------
def build_cofunding_graph(campaign_wallets: Dict[str, Set[str]],
                           min_shared_campaigns: int = 2,
                           campaign_wallet_amounts: Optional[Dict[str, Dict[str, float]]] = None,
                           weight_mode: str = "count") -> nx.Graph:
    """
    Builds the "Wallet Graph": an undirected, weighted graph where two
    wallets are connected ONLY if they've co-funded at least
    `min_shared_campaigns` campaigns together.

    Why not >=1: if the threshold were 1, every campaign's own wallets
    would trivially form a complete subgraph with each other just by
    funding THIS campaign together -- guaranteed by construction and
    carrying no fraud signal. >=2 means an edge only exists when two
    wallets have a genuine repeat relationship beyond the campaign
    currently being scored.

    weight_mode:
        "count"  (default) -> edge weight = number of shared campaigns.
        "amount" -> edge weight = sum, over every shared campaign c, of
                    (amount wallet-1 put into c + amount wallet-2 put
                    into c). Requires `campaign_wallet_amounts`, a dict
                    of {campaign_id: {wallet_id: total_amount}}.

    IMPORTANT: regardless of weight_mode, the min_shared_campaigns
    THRESHOLD is always evaluated on shared campaign COUNT, never on
    amount. Otherwise two wallets could bypass the "repeat relationship"
    requirement by making one single huge joint contribution together --
    which is a different signal (whale co-funding) from what this graph
    is meant to capture (recurring coordination).

    Node and edge insertion order is fully sorted so that community
    detection and PageRank produce the same result on every run,
    regardless of Python's set/dict iteration order (which is NOT
    guaranteed stable across runs for non-trivial cases).
    """
    if weight_mode not in ("count", "amount"):
        raise ValueError(f"weight_mode must be 'count' or 'amount', got '{weight_mode}'")
    if weight_mode == "amount" and campaign_wallet_amounts is None:
        raise ValueError("weight_mode='amount' requires campaign_wallet_amounts to be provided")

    pair_shared_campaigns: Dict[tuple, int] = {}
    pair_amount_weight: Dict[tuple, float] = {}
    all_wallets: Set[str] = set()

    for campaign_id, wallets in campaign_wallets.items():
        all_wallets.update(wallets)
        for w1, w2 in itertools.combinations(sorted(wallets), 2):
            key = (w1, w2)
            pair_shared_campaigns[key] = pair_shared_campaigns.get(key, 0) + 1
            if weight_mode == "amount":
                amounts_here = campaign_wallet_amounts.get(campaign_id, {})
                amt1 = float(amounts_here.get(w1, 0.0))
                amt2 = float(amounts_here.get(w2, 0.0))
                pair_amount_weight[key] = pair_amount_weight.get(key, 0.0) + amt1 + amt2

    graph = nx.Graph()
    graph.add_nodes_from(sorted(all_wallets))
    for (w1, w2) in sorted(pair_shared_campaigns.keys()):
        shared = pair_shared_campaigns[(w1, w2)]
        if shared < min_shared_campaigns:
            continue
        weight = pair_amount_weight[(w1, w2)] if weight_mode == "amount" else shared
        graph.add_edge(w1, w2, weight=weight, shared_campaigns=shared)

    return graph


def detect_communities(graph: nx.Graph) -> Dict[str, int]:
    """
    Runs greedy modularity community detection once on the whole graph.
    Deterministic given a deterministically-ordered graph (see
    build_cofunding_graph). Falls back to every wallet as its own
    community when the graph has no edges.
    """
    if graph.number_of_edges() == 0:
        return {node: i for i, node in enumerate(sorted(graph.nodes()))}

    communities = nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
    wallet_to_community = {}
    for idx, community in enumerate(communities):
        for wallet in sorted(community):
            wallet_to_community[wallet] = idx
    return wallet_to_community


def compute_pagerank(graph: nx.Graph) -> Dict[str, float]:
    """
    PageRank centrality on the wallet co-funding graph. Falls back to
    uniform scores if the graph has no edges.
    """
    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_edges() == 0:
        logger.warning(
            "Graph has no edges (no wallet pair shares >= min_shared_campaigns "
            "campaigns). WGD, CDS, and INF will carry limited information -- "
            "only COL/FTS/FAS will meaningfully vary across campaigns."
        )
        uniform = 1.0 / graph.number_of_nodes()
        return {node: uniform for node in graph.nodes()}
    return nx.pagerank(graph, weight="weight")


def graph_statistics(graph: nx.Graph) -> Dict[str, Any]:
    """
    Sanity-check statistics for debugging the wallet graph: size, edge
    count, average degree, density, isolated wallets, largest connected
    component, and average clustering coefficient.
    """
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    degrees = [d for _, d in graph.degree()]
    avg_degree = float(np.mean(degrees)) if degrees else 0.0
    isolated = sum(1 for d in degrees if d == 0)

    if n_nodes > 0:
        components = list(nx.connected_components(graph))
        largest_component = max((len(c) for c in components), default=0)
        num_components = len(components)
    else:
        largest_component = 0
        num_components = 0

    avg_clustering = float(nx.average_clustering(graph, weight="weight")) if n_edges > 0 else 0.0
    density = float(nx.density(graph)) if n_nodes > 0 else 0.0

    return {
        "num_wallets": n_nodes,
        "num_edges": n_edges,
        "avg_degree": round(avg_degree, 4),
        "density": round(density, 6),
        "isolated_wallets": isolated,
        "num_connected_components": num_components,
        "largest_component_size": largest_component,
        "avg_clustering": round(avg_clustering, 4),
    }


def validate_campaign_data(df: pd.DataFrame, campaign_wallets: Dict[str, Set[str]]) -> Dict[str, int]:
    """
    Explicit sanity checks on the campaign -> wallet structure, logged
    for visibility instead of being silently absorbed elsewhere:
      - duplicate wallet rows within the same campaign (already handled
        correctly by set() dedup, but counted here for transparency)
      - campaigns with exactly one wallet
      - campaigns with zero wallets (defensive check)
    """
    total_rows = len(df)
    total_unique_memberships = sum(len(w) for w in campaign_wallets.values())
    duplicate_rows = total_rows - total_unique_memberships

    zero_wallet_campaigns = [cid for cid, w in campaign_wallets.items() if len(w) == 0]
    single_wallet_campaigns = [cid for cid, w in campaign_wallets.items() if len(w) == 1]

    if zero_wallet_campaigns:
        logger.warning("%d campaign(s) have zero valid wallets: %s",
                        len(zero_wallet_campaigns), zero_wallet_campaigns[:10])

    if single_wallet_campaigns:
        shown = single_wallet_campaigns[:10]
        suffix = " ..." if len(single_wallet_campaigns) > 10 else ""
        logger.info("%d campaign(s) have exactly one wallet: %s%s",
                     len(single_wallet_campaigns), shown, suffix)

    return {
        "duplicate_wallet_rows": duplicate_rows,
        "campaigns_with_zero_wallets": len(zero_wallet_campaigns),
        "campaigns_with_one_wallet": len(single_wallet_campaigns),
    }


def save_graph_gexf(graph: nx.Graph, path: str = "wallet_graph.gexf") -> None:
    """
    One-call export of the wallet co-funding graph to GEXF for Gephi.
    """
    if graph.number_of_nodes() == 0:
        logger.warning("Graph is empty, nothing to export.")
        return
    nx.write_gexf(graph, path)
    logger.info("Graph exported to %s (%d nodes, %d edges). Open in Gephi to explore.",
                path, graph.number_of_nodes(), graph.number_of_edges())


def save_graph_visualization(graph: nx.Graph, path: str = "wallet_graph.png", max_nodes: int = 200) -> None:
    """
    Optional debugging utility -- draws the wallet co-funding graph and
    saves it to `path`. Not called automatically by the pipeline.
    """
    if graph.number_of_nodes() == 0:
        logger.warning("Graph is empty, nothing to visualize.")
        return
    if graph.number_of_nodes() > max_nodes:
        logger.warning(
            "Graph has %d nodes (> %d), skipping matplotlib render. "
            "Use save_graph_gexf(graph) and open the result in Gephi instead.",
            graph.number_of_nodes(), max_nodes
        )
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping graph visualization.")
        return

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(graph, seed=42)
    weights = [graph[u][v]["weight"] for u, v in graph.edges()]
    nx.draw(
        graph, pos, node_size=60, node_color="steelblue",
        edge_color="gray", width=[min(w, 5) for w in weights],
        with_labels=False
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Graph visualization saved to %s", path)


# ---------------------------------------------------------------------------
# Core CFIS calculations
# ---------------------------------------------------------------------------
class CFIS:
    def __init__(self, graph: nx.Graph, communities: Dict[str, int],
                 pagerank: Dict[str, float], wallet_reuse: Dict[str, int],
                 campaign_wallet_amounts: Optional[Dict[str, Dict[str, float]]] = None,
                 campaign_wallet_timestamps: Optional[Dict[str, Dict[str, pd.Timestamp]]] = None,
                 influence_top_k: int = 3, min_wallets_for_score: int = 3,
                 fts_time_window_seconds: float = 3600.0,
                 max_pairs_for_similarity: int = 2000):
        """
        wallet_reuse: {wallet_id: number_of_OTHER_campaigns_this_wallet_funded}.
        campaign_wallet_amounts: {campaign_id: {wallet_id: total_amount}},
            used by FAS. None if no amount column was found.
        campaign_wallet_timestamps: {campaign_id: {wallet_id: representative
            pd.Timestamp}}, used by FTS. None if no timestamp column was
            found.
        min_wallets_for_score: campaigns with fewer wallets than this have
            their WHOLE vector (wgd, cds, col, inf, fts, fas) zeroed --
            not just WGD -- since CDS/FAS/FTS are exactly as prone to a
            trivial "perfect" score at tiny n as WGD is.
        fts_time_window_seconds: business-defined "how close in time is
            suspicious" threshold used to normalize FTS. NOT derived
            from the campaign's own duration -- a fixed external
            parameter, matching how the handwritten spec treats
            "Max Allowed Time Window".
        max_pairs_for_similarity: FTS/FAS both average over all pairs of
            wallets in a campaign (O(n^2)). For campaigns with very many
            wallets this is capped by randomly sampling this many pairs
            instead of enumerating all of them, to keep runtime bounded.
        """
        self.graph = graph
        self.communities = communities
        self.pagerank = pagerank
        self.influence_top_k = influence_top_k
        self.min_wallets_for_score = min_wallets_for_score
        self.campaign_wallet_amounts = campaign_wallet_amounts or {}
        self.campaign_wallet_timestamps = campaign_wallet_timestamps or {}
        self.fts_time_window_seconds = fts_time_window_seconds
        self.max_pairs_for_similarity = max_pairs_for_similarity

        weights = [d["weight"] for _, _, d in graph.edges(data=True)]
        self._weight_cap = float(np.percentile(weights, 95)) if weights else 1.0
        self._weight_cap = max(self._weight_cap, 1e-9)
        self._max_weight = max(weights) if weights else 1  # diagnostics only

        pr_series = pd.Series(pagerank, dtype=float) if pagerank else pd.Series(dtype=float)
        # method="max": tied values all receive the percentile of the TOP
        # of their tie group. With the default "average" method, a
        # dataset where every wallet is equally (and maximally) reused
        # or equally central would land at ~0.5-0.6 percentile instead of
        # 1.0, since "average" splits the difference across the tied
        # group. That undercuts the whole point of using percentile rank
        # here -- these wallets genuinely ARE the most-connected/most-
        # reused in the dataset, and should be able to reach 1.0.
        self._pagerank_percentile = pr_series.rank(pct=True, method="max").to_dict() if not pr_series.empty else {}

        reuse_series = pd.Series(wallet_reuse, dtype=float) if wallet_reuse else pd.Series(dtype=float)
        self._collision_percentile = reuse_series.rank(pct=True, method="max").to_dict() if not reuse_series.empty else {}

    def calculate_wgd(self, wallets: Set[str]) -> float:
        """Weighted Wallet Graph Density, capped at the 95th percentile edge weight."""
        wallets = [w for w in wallets if w in self.graph]
        n = len(wallets)
        if n < 2:
            return 0.0

        subgraph = self.graph.subgraph(wallets)
        possible_pairs = n * (n - 1) / 2
        weighted_sum = sum(d["weight"] for _, _, d in subgraph.edges(data=True))
        if weighted_sum == 0:
            return 0.0

        score = weighted_sum / (possible_pairs * self._weight_cap)
        return float(np.clip(score, 0.0, 1.0))

    def calculate_cds(self, wallets: Set[str]) -> float:
        """Community Detection Score via normalized Shannon entropy."""
        wallets = [w for w in wallets if w in self.communities]
        n = len(wallets)
        if n == 0:
            return 0.0
        if n == 1:
            return 1.0

        community_ids = [self.communities[w] for w in wallets]
        counts = pd.Series(community_ids).value_counts()
        probs = counts / n

        entropy = -float(np.sum(probs * np.log(probs)))
        max_entropy = float(np.log(n))
        if max_entropy == 0:
            return 1.0

        normalized_entropy = entropy / max_entropy
        return float(np.clip(1.0 - normalized_entropy, 0.0, 1.0))

    def calculate_collision(self, wallets: Set[str]) -> float:
        """Frequency-weighted Collision Score via percentile rank of wallet reuse."""
        if not wallets:
            return 0.0
        scores = [self._collision_percentile.get(w, 0.0) for w in wallets]
        if not scores:
            return 0.0
        return float(np.clip(np.mean(scores), 0.0, 1.0))

    def calculate_influence(self, wallets: Set[str]) -> float:
        """Average of the top-k highest percentile-ranked PageRank scores."""
        scores = sorted(
            (self._pagerank_percentile.get(w, 0.0) for w in wallets),
            reverse=True
        )
        if not scores:
            return 0.0
        k = min(self.influence_top_k, len(scores))
        return float(np.mean(scores[:k]))

    def _sample_pairs(self, items: List) -> List[Tuple]:
        """
        Returns all pairwise combinations of `items`, unless that would
        exceed max_pairs_for_similarity, in which case a random sample
        of that many pairs is drawn instead (deterministic seed so
        results are reproducible).
        """
        n = len(items)
        total_pairs = n * (n - 1) // 2
        if total_pairs <= self.max_pairs_for_similarity:
            return list(itertools.combinations(items, 2))

        rng = random.Random(42)
        sampled = set()
        items_sorted = sorted(items) if all(isinstance(i, str) for i in items) else items
        while len(sampled) < self.max_pairs_for_similarity:
            i, j = rng.sample(range(n), 2)
            lo, hi = min(i, j), max(i, j)
            sampled.add((lo, hi))
        return [(items_sorted[i], items_sorted[j]) for (i, j) in sampled]

    def calculate_fts(self, campaign_id: str, wallets: Set[str]) -> float:
        """
        Funding Time Similarity (from handwritten spec, adapted to a
        whole-campaign average instead of a single wallet pair):

            FTS = 1 - (average pairwise time gap / fts_time_window_seconds)

        1.0 = every wallet in the campaign contributed within the same
        instant (synchronized, bot-like burst). 0.0 = contributions are
        spread out well beyond the configured time window (organic).
        Returns 0.0 if no timestamp data is available for this campaign.
        """
        campaign_times = self.campaign_wallet_timestamps.get(campaign_id, {})
        times = [campaign_times[w] for w in wallets if w in campaign_times and pd.notna(campaign_times[w])]
        if len(times) < 2:
            return 0.0

        pairs = self._sample_pairs(times)
        if not pairs:
            return 0.0

        gaps_seconds = [abs((t1 - t2).total_seconds()) for t1, t2 in pairs]
        avg_gap = float(np.mean(gaps_seconds))

        if self.fts_time_window_seconds <= 0:
            return 0.0

        fts = 1.0 - (avg_gap / self.fts_time_window_seconds)
        return float(np.clip(fts, 0.0, 1.0))

    def calculate_fas(self, campaign_id: str, wallets: Set[str]) -> float:
        """
        Funding Amount Similarity (from handwritten spec, adapted to a
        whole-campaign average instead of a single wallet pair):

            FAS = mean over all wallet pairs of
                  1 - |amount_1 - amount_2| / max(amount_1, amount_2)

        1.0 = every wallet contributed (near-)identical amounts
        (bot-like, e.g. everyone sends exactly $50). 0.0 = amounts vary
        naturally. Returns 0.0 if no amount data is available.
        """
        campaign_amounts = self.campaign_wallet_amounts.get(campaign_id, {})
        amounts = [campaign_amounts[w] for w in wallets if w in campaign_amounts and campaign_amounts[w] > 0]
        if len(amounts) < 2:
            return 0.0

        pairs = self._sample_pairs(amounts)
        if not pairs:
            return 0.0

        similarities = []
        for a1, a2 in pairs:
            max_amt = max(a1, a2)
            if max_amt <= 0:
                continue
            similarities.append(1.0 - abs(a1 - a2) / max_amt)

        if not similarities:
            return 0.0

        return float(np.clip(np.mean(similarities), 0.0, 1.0))

    def compute_vector(self, campaign_id: str, wallets: Set[str]) -> Dict[str, float]:
        # Below min_wallets_for_score, zero the whole vector -- CDS/FTS/FAS
        # are exactly as prone to a trivial "perfect" score at tiny n as
        # WGD is (e.g. a 2-wallet campaign with similar amounts is
        # definitionally FAS close to 1.0).
        if len(wallets) < self.min_wallets_for_score:
            return {"wgd": 0.0, "cds": 0.0, "col": 0.0, "inf": 0.0, "fts": 0.0, "fas": 0.0}

        return {
            "wgd": self.calculate_wgd(wallets),
            "cds": self.calculate_cds(wallets),
            "col": self.calculate_collision(wallets),
            "inf": self.calculate_influence(wallets),
            "fts": self.calculate_fts(campaign_id, wallets),
            "fas": self.calculate_fas(campaign_id, wallets),
        }


# ---------------------------------------------------------------------------
# Batch extraction from a full dataset
# ---------------------------------------------------------------------------
class CFISExtractor:
    def __init__(self, min_shared_campaigns: int = 2, use_cache: bool = True,
                 influence_top_k: int = 3, weight_mode: str = "count",
                 min_wallets_for_score: int = 3,
                 fts_time_window_seconds: float = 3600.0,
                 max_pairs_for_similarity: int = 2000):
        """
        min_shared_campaigns: threshold for a wallet-graph edge to exist.
        use_cache: reuse the previously built graph/communities/PageRank
            if the underlying campaign->wallet structure hasn't changed.
        influence_top_k: how many of a campaign's highest-ranked wallets
            to average for the Influence Score.
        weight_mode: "count" (default) or "amount" -- see
            build_cofunding_graph docstring.
        min_wallets_for_score: campaigns with fewer wallets than this
            have their entire CFIS vector zeroed instead of scored.
        fts_time_window_seconds: business-defined threshold (seconds)
            for "how close in time counts as suspicious" -- used to
            normalize Funding Time Similarity. Default 3600 (1 hour).
        max_pairs_for_similarity: caps the O(n^2) pairwise comparisons
            in FTS/FAS for very large campaigns via random sampling.
        """
        if weight_mode not in ("count", "amount"):
            raise ValueError(f"weight_mode must be 'count' or 'amount', got '{weight_mode}'")

        self.min_shared_campaigns = min_shared_campaigns
        self.use_cache = use_cache
        self.influence_top_k = influence_top_k
        self.weight_mode = weight_mode
        self.min_wallets_for_score = min_wallets_for_score
        self.fts_time_window_seconds = fts_time_window_seconds
        self.max_pairs_for_similarity = max_pairs_for_similarity

        self._cache_key = None
        self._graph: Optional[nx.Graph] = None
        self._communities: Optional[Dict[str, int]] = None
        self._pagerank: Optional[Dict[str, float]] = None

    @staticmethod
    def _cache_key_for(campaign_wallets: Dict[str, Set[str]],
                        campaign_wallet_amounts: Optional[Dict[str, Dict[str, float]]] = None) -> int:
        structure = hash(frozenset((cid, frozenset(wallets)) for cid, wallets in campaign_wallets.items()))
        if campaign_wallet_amounts is None:
            return structure
        amounts = hash(frozenset(
            (cid, frozenset(w_amounts.items())) for cid, w_amounts in campaign_wallet_amounts.items()
        ))
        return hash((structure, amounts))

    def _get_graph_bundle(self, campaign_wallets: Dict[str, Set[str]],
                           campaign_wallet_amounts_for_graph: Optional[Dict[str, Dict[str, float]]] = None):
        key = self._cache_key_for(campaign_wallets, campaign_wallet_amounts_for_graph)
        if self.use_cache and self._graph is not None and key == self._cache_key:
            logger.info("Reusing cached wallet graph / communities / PageRank.")
            return self._graph, self._communities, self._pagerank

        graph = build_cofunding_graph(
            campaign_wallets,
            min_shared_campaigns=self.min_shared_campaigns,
            campaign_wallet_amounts=campaign_wallet_amounts_for_graph,
            weight_mode=self.weight_mode,
        )
        communities = detect_communities(graph)
        pagerank = compute_pagerank(graph)

        if self.use_cache:
            self._graph, self._communities, self._pagerank, self._cache_key = graph, communities, pagerank, key

        return graph, communities, pagerank

    def clear_cache(self) -> None:
        self._cache_key = None
        self._graph = None
        self._communities = None
        self._pagerank = None

    def extract_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        mapping = create_column_mapping(df)
        for required in ["campaign_id", "wallet_id"]:
            if mapping[required] is None:
                raise ValueError(
                    f"Dataset does not contain a column for '{required}'. "
                    f"Please ensure your CSV has appropriate column names "
                    f"(e.g. campaign_id, wallet_id/wallet_address/user_id). "
                    f"Columns found in loaded file: {list(df.columns)}"
                )
        if self.weight_mode == "amount" and mapping["amount"] is None:
            raise ValueError(
                "weight_mode='amount' requires an amount column in the input data "
                "(e.g. amount, amounts, transaction_amount, payments)."
            )

        # Keep only the columns CFIS actually needs (campaign_id, wallet_id,
        # amount, timestamp). The unified dataset also carries unrelated
        # columns (title, description, team_size, current_funding, etc.)
        # that other pipelines use -- CFIS ignores them entirely.
        needed_cols = [c for c in mapping.values() if c is not None]
        df = df[needed_cols].copy()
        df["_campaign_id"] = df[mapping["campaign_id"]].apply(safe_str)
        df["_wallet_id"] = df[mapping["wallet_id"]].apply(safe_str).apply(normalize_wallet_id)

        before = len(df)
        df = df[~df["_campaign_id"].apply(is_null_like)]
        df = df[~df["_wallet_id"].apply(is_null_like)]
        removed = before - len(df)
        if removed:
            logger.info("Filtered out %d row(s) with missing/null-like campaign or wallet ids.", removed)

        if df.empty:
            logger.warning("No valid rows remain after filtering; returning empty feature set.")
            return pd.DataFrame(columns=["campaign_id", "n_wallets", "wgd", "cds", "col", "inf",
                                          "fts", "fas", "below_min_wallets"])

        campaign_wallets: Dict[str, Set[str]] = df.groupby("_campaign_id")["_wallet_id"].apply(set).to_dict()
        wallet_campaign_counts: Dict[str, int] = df.groupby("_wallet_id")["_campaign_id"].nunique().to_dict()
        wallet_reuse: Dict[str, int] = {w: max(c - 1, 0) for w, c in wallet_campaign_counts.items()}

        validate_campaign_data(df, campaign_wallets)

        # ---- amounts: needed for FAS always; also for the graph itself
        # when weight_mode == "amount" ----
        campaign_wallet_amounts: Optional[Dict[str, Dict[str, float]]] = None
        if mapping["amount"] is not None:
            df["_amount"] = pd.to_numeric(df[mapping["amount"]], errors="coerce").fillna(0.0)
            grouped_amounts = df.groupby(["_campaign_id", "_wallet_id"])["_amount"].sum()
            campaign_wallet_amounts = {}
            for (cid, wid), amt in grouped_amounts.items():
                campaign_wallet_amounts.setdefault(cid, {})[wid] = float(amt)
        else:
            logger.warning("No amount column found. FAS (Funding Amount Similarity) "
                            "will default to 0 for every campaign.")

        # ---- timestamps: needed for FTS ----
        campaign_wallet_timestamps: Optional[Dict[str, Dict[str, pd.Timestamp]]] = None
        if mapping["timestamp"] is not None:
            parsed = pd.to_datetime(df[mapping["timestamp"]], utc=True, errors="coerce")
            df["_timestamp"] = parsed.dt.tz_localize(None)
            grouped_times = df.groupby(["_campaign_id", "_wallet_id"])["_timestamp"].mean()
            campaign_wallet_timestamps = {}
            for (cid, wid), ts in grouped_times.items():
                campaign_wallet_timestamps.setdefault(cid, {})[wid] = ts
        else:
            logger.warning("No timestamp column found. FTS (Funding Time Similarity) "
                            "will default to 0 for every campaign.")

        campaign_wallet_amounts_for_graph = campaign_wallet_amounts if self.weight_mode == "amount" else None
        graph, communities, pagerank = self._get_graph_bundle(campaign_wallets, campaign_wallet_amounts_for_graph)

        stats = graph_statistics(graph)
        if stats["num_edges"] == 0:
            logger.warning("Graph has no edges. WGD/CDS/INF will have limited information.")
        elif stats["num_wallets"] > 0 and stats["isolated_wallets"] / stats["num_wallets"] > 0.8:
            logger.warning(
                "%.0f%% of wallets are isolated (no repeat co-funding relationships >= "
                "min_shared_campaigns=%d). WGD/CDS/INF will be near-zero for most campaigns.",
                100.0 * stats["isolated_wallets"] / stats["num_wallets"], self.min_shared_campaigns
            )

        cfis = CFIS(graph, communities, pagerank, wallet_reuse,
                    campaign_wallet_amounts=campaign_wallet_amounts,
                    campaign_wallet_timestamps=campaign_wallet_timestamps,
                    influence_top_k=self.influence_top_k,
                    min_wallets_for_score=self.min_wallets_for_score,
                    fts_time_window_seconds=self.fts_time_window_seconds,
                    max_pairs_for_similarity=self.max_pairs_for_similarity)

        records = []
        below_threshold_count = 0
        for campaign_id, wallets in campaign_wallets.items():
            vec = cfis.compute_vector(campaign_id, wallets)
            below_threshold = len(wallets) < self.min_wallets_for_score
            below_threshold_count += below_threshold
            records.append({
                "campaign_id": campaign_id,
                "n_wallets": len(wallets),
                "wgd": round(vec["wgd"], 4),
                "cds": round(vec["cds"], 4),
                "col": round(vec["col"], 4),
                "inf": round(vec["inf"], 4),
                "fts": round(vec["fts"], 4),
                "fas": round(vec["fas"], 4),
                "below_min_wallets": below_threshold,
            })

        if below_threshold_count:
            logger.info(
                "%d campaign(s) had fewer than min_wallets_for_score=%d wallets; "
                "their CFIS vector was zeroed rather than scored.",
                below_threshold_count, self.min_wallets_for_score
            )

        feature_df = pd.DataFrame(records)
        feature_df.fillna(0, inplace=True)

        self._validate_features(feature_df)

        return feature_df

    @staticmethod
    def _validate_features(feature_df: pd.DataFrame) -> None:
        """Confirms every core feature is within [0, 1]; logs a warning if not."""
        for col in ["wgd", "cds", "col", "inf", "fts", "fas"]:
            if col not in feature_df.columns:
                continue
            bad = feature_df[(feature_df[col] < 0) | (feature_df[col] > 1)]
            if len(bad) > 0:
                logger.warning(
                    "Feature '%s' has %d value(s) outside [0, 1]: campaigns %s",
                    col, len(bad), bad["campaign_id"].tolist()
                )

    def extract_features(self, input_csv: str, output_csv: str) -> pd.DataFrame:
        df = pd.read_csv(input_csv)
        feature_df = self.extract_dataframe(df)
        feature_df.to_csv(output_csv, index=False)

        feature_cols = ["wgd", "cds", "col", "inf", "fts", "fas"]

        print("\nCFIS Feature Statistics:")
        print(feature_df[feature_cols].describe())

        print("\nChecking for Infinite Values...")
        print(np.isfinite(feature_df[feature_cols]).all())

        return feature_df


def to_numpy(feature_df: pd.DataFrame) -> np.ndarray:
    """
    Converts the CFIS feature DataFrame into a clean (N, 6) float32
    array in [wgd, cds, col, inf, fts, fas] order (diagnostic columns
    like n_wallets/below_min_wallets are excluded). NaN/Inf are
    guaranteed to be replaced with 0.0.
    """
    array = feature_df[["wgd", "cds", "col", "inf", "fts", "fas"]].values.astype(np.float32)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Sample data generator (for local testing without a real dataset)
# ---------------------------------------------------------------------------
def generate_sample_csv(path: str, num_campaigns: int = 25, num_wallets: int = 150,
                         seed: Optional[int] = 42) -> None:
    """
    Generates a richer synthetic dataset covering several wallet-behavior
    archetypes, now including TIMESTAMPS so FTS/FAS have something
    meaningful to distinguish:

      - RING A / RING B: coordinated wallet clusters that repeatedly
        co-fund several campaigns together, contributing near-identical
        amounts within seconds of each other (high WGD/CDS/FTS/FAS).
      - BRIDGE wallet: links Ring A and Ring B.
      - HUB wallet: funds many otherwise-unrelated campaigns alongside
        different random wallets, at organic/spread times and varied
        amounts (high COL/INF, low WGD/CDS/FTS/FAS).
      - RANDOM wallets: organic funding patterns -- spread-out times,
        varied amounts.

    seed: fixes `random`/`numpy.random` state for reproducibility.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    rows = []
    all_wallets = [f"0xWallet{w:04d}" for w in range(num_wallets)]
    campaign_ids = iter(f"CAMP{i + 1:04d}" for i in range(num_campaigns))
    random_pool = all_wallets[13:]
    pool_cursor = itertools.count()
    base_time = pd.Timestamp("2026-01-01T00:00:00")

    def draw_random_wallets(n: int) -> List[str]:
        return [random_pool[next(pool_cursor) % len(random_pool)] for _ in range(n)]

    def organic_time() -> pd.Timestamp:
        return base_time + pd.Timedelta(minutes=int(np.random.randint(0, 60 * 24 * 10)))

    # RING A: near-simultaneous, near-identical-amount coordinated funding
    ring_a = all_wallets[0:6]
    ring_a_campaigns = [next(campaign_ids) for _ in range(3)]
    for cid in ring_a_campaigns:
        burst_time = organic_time()
        for w in ring_a:
            rows.append({
                "campaign_id": cid, "wallet_id": w,
                "amount": 1000 + random.randint(-5, 5),  # near-identical
                "timestamp": burst_time + pd.Timedelta(seconds=random.randint(0, 30)),  # near-simultaneous
            })

    ring_b = all_wallets[6:11]
    ring_b_campaigns = [next(campaign_ids) for _ in range(3)]
    for cid in ring_b_campaigns:
        burst_time = organic_time()
        for w in ring_b:
            rows.append({
                "campaign_id": cid, "wallet_id": w,
                "amount": 2500 + random.randint(-5, 5),
                "timestamp": burst_time + pd.Timedelta(seconds=random.randint(0, 30)),
            })

    bridge = all_wallets[11]
    rows.append({"campaign_id": ring_a_campaigns[0], "wallet_id": bridge,
                 "amount": random.randint(100, 5000), "timestamp": organic_time()})
    rows.append({"campaign_id": ring_b_campaigns[0], "wallet_id": bridge,
                 "amount": random.randint(100, 5000), "timestamp": organic_time()})

    hub = all_wallets[12]
    for _ in range(6):
        cid = next(campaign_ids)
        rows.append({"campaign_id": cid, "wallet_id": hub,
                     "amount": random.randint(100, 5000), "timestamp": organic_time()})
        for w in draw_random_wallets(random.randint(3, 6)):
            rows.append({"campaign_id": cid, "wallet_id": w,
                         "amount": random.randint(100, 5000), "timestamp": organic_time()})

    for cid in campaign_ids:
        n_contributors = random.randint(4, 12)
        contributors = random.sample(random_pool, min(n_contributors, len(random_pool)))
        for w in contributors:
            rows.append({"campaign_id": cid, "wallet_id": w,
                         "amount": random.randint(100, 5000), "timestamp": organic_time()})

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info("Sample dataset created: %s (%d rows, %d campaigns)",
                path, len(df), df["campaign_id"].nunique())


if __name__ == "__main__":
    # Resolve paths relative to THIS script's own folder, not the
    # process's current working directory. Code Runner / different IDEs
    # can launch a script with the working directory set to the project
    # root instead of the file's own folder -- if two files named
    # "unified_dataset.csv" exist, a plain relative path silently picks
    # up the wrong one. Anchoring to __file__'s directory makes this
    # deterministic.
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT  = os.path.join(SCRIPT_DIR, "..", "Datasets", "unified_dataset.csv")
    OUTPUT = os.path.join(SCRIPT_DIR, "..", "outputs", "cfis_features.csv")

    if not os.path.exists(INPUT):
        raise FileNotFoundError(
            f"Could not find unified_dataset.csv at: {INPUT}\n"
            f"Run dataset_simulator.py first to generate it."
        )

    # Ensure the outputs folder exists before writing to it
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    extractor = CFISExtractor(min_shared_campaigns=2, use_cache=True, influence_top_k=3)
    features = extractor.extract_features(INPUT, OUTPUT)

    print("\nCFIS Feature Extraction Completed")
    print("\nCFIS Features extracted:")
    print(features.head())
    print("\nOutput columns:", features.columns.tolist())
    print("\nClean CFIS feature CSV saved to:", OUTPUT)

    feature_matrix = to_numpy(features)
    print("\nFeature matrix shape for Transformer input:", feature_matrix.shape)