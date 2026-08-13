import os
import logging
import itertools
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ctcs_module")


# ---------------------------------------------------------------------------
# Column auto-mapping helpers
# ---------------------------------------------------------------------------
def find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    """
    Finds the first matching column from a list of possible names.

    Column names are matched after stripping whitespace and a leading
    UTF-8 BOM ("\ufeff") -- CSVs exported from Excel/Windows tools
    commonly attach a BOM to the FIRST header cell (e.g. "\ufeffamount")
    or leave stray leading/trailing spaces on header names, which
    silently breaks an exact-match lookup like `"amount" in columns`
    even though the column looks identical when printed.
    """
    def normalize(name: str) -> str:
        return name.strip().lstrip("\ufeff").lower()

    columns = {normalize(c): c for c in df.columns}
    for name in possible_names:
        key = normalize(name)
        if key in columns:
            return columns[key]
    return None


def create_column_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Automatically map dataset columns to CTCS requirements."""
    return {
        "campaign_id": find_column(df, ["campaign_id", "campaign", "camp_id"]),
        "amount": find_column(df, ["amount", "amounts", "transaction_amount", "payments"]),
        "timestamp": find_column(df, [
            "timestamp", "transaction_time", "funded_at", "created_at",
            "time", "datetime", "transaction_date"
        ]),
    }


def safe_str(value, default: str = "") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return str(value).strip()


def is_null_like(value: str) -> bool:
    return value.strip().lower() in {"", "none", "nan", "n/a", "na", "unknown", "null", "-"}


def parse_timestamp(value: Any) -> Optional[datetime]:
    """
    Parses a single timestamp value via pd.to_datetime, handling ISO
    strings, epoch seconds/ms/us, and mixed timezones. Mirrors
    fvrs_pipeline.parse_timestamp() / cfis_pipeline.parse_timestamp()
    so timestamp handling is consistent across all three pipelines.
    Returns a naive datetime normalized to UTC, or None if unparseable.
    """
    if isinstance(value, datetime):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value > 1e14:
                parsed = pd.to_datetime(value, unit="us", utc=True)
            elif value > 1e11:
                parsed = pd.to_datetime(value, unit="ms", utc=True)
            else:
                parsed = pd.to_datetime(value, unit="s", utc=True)
        else:
            parsed = pd.to_datetime(value, utc=True)
    except (ValueError, TypeError):
        return None

    if pd.isna(parsed):
        return None

    return parsed.tz_localize(None).to_pydatetime()


# ---------------------------------------------------------------------------
# Efficient synchronized-pair counting
# ---------------------------------------------------------------------------
def count_synced_pairs(sorted_ts_a: np.ndarray, sorted_ts_b: np.ndarray, omega: float) -> int:
    """
    Counts pairs (ta, tb), ta in sorted_ts_a, tb in sorted_ts_b, with
    |ta - tb| <= omega. Both inputs must already be sorted ascending
    (epoch-seconds float arrays). Uses binary search per element of
    sorted_ts_a against sorted_ts_b -- O(len_a * log(len_b)) -- instead
    of the O(len_a * len_b) brute-force nested loop the handwritten
    Sync() definition implies, which becomes impractical once campaigns
    have more than a few hundred timestamps each.
    """
    if len(sorted_ts_a) == 0 or len(sorted_ts_b) == 0:
        return 0

    lo_idx = np.searchsorted(sorted_ts_b, sorted_ts_a - omega, side="left")
    hi_idx = np.searchsorted(sorted_ts_b, sorted_ts_a + omega, side="right")
    return int(np.sum(hi_idx - lo_idx))


def build_time_bins(all_timestamps: np.ndarray, num_time_bins: int) -> np.ndarray:
    """
    Builds `num_time_bins` equal-width bin edges spanning the GLOBAL
    observation period (earliest to latest timestamp across the WHOLE
    dataset, not per campaign) -- this is what makes every campaign's
    FPV directly comparable. Returns num_time_bins + 1 edges. If every
    timestamp in the dataset is identical (zero-width period), widens
    it by 1 second so np.histogram doesn't choke on a degenerate range.
    """
    t_min, t_max = float(np.min(all_timestamps)), float(np.max(all_timestamps))
    if t_max <= t_min:
        logger.warning(
            "All timestamps in the dataset are identical (or the dataset has a single "
            "transaction). Widening the observation window by 1 second so time-bin "
            "construction doesn't degenerate; CTCC/TS will carry limited information."
        )
        t_max = t_min + 1.0
    return np.linspace(t_min, t_max, num_time_bins + 1)


def build_funding_pattern_vectors(campaign_bin_amounts: Dict[str, np.ndarray]) -> Tuple[List[str], np.ndarray]:
    """
    Stacks each campaign's FPV into a single (n_campaigns, num_time_bins)
    matrix, in a fixed campaign order, for vectorized correlation.
    Returns (ordered_campaign_ids, fpv_matrix).
    """
    campaign_ids = sorted(campaign_bin_amounts.keys())
    matrix = np.vstack([campaign_bin_amounts[cid] for cid in campaign_ids]) if campaign_ids else np.empty((0, 0))
    return campaign_ids, matrix


# ---------------------------------------------------------------------------
# Core CTCS calculations
# ---------------------------------------------------------------------------
class CTCS:
    def __init__(self, campaign_ids: List[str], fpv_matrix: np.ndarray,
                 campaign_timestamps: Dict[str, np.ndarray], sync_window_seconds: float = 3600.0,
                 min_timestamps_for_score: int = 2):
        """
        campaign_ids: campaign order matching fpv_matrix's rows.
        fpv_matrix: (n_campaigns, num_time_bins) funding pattern vectors.
        campaign_timestamps: {campaign_id: sorted np.ndarray of epoch-second
            floats} -- every parsed, valid timestamp for that campaign
            (NOT binned; TS needs the raw event times, CTCC needs the
            binned FPV).
        sync_window_seconds: TS's "omega" -- two timestamps from
            different campaigns within this many seconds of each other
            count as synchronized. Default 3600s (1 hour); tune to your
            domain the same way you would CFIS's max_time_window_seconds.
        min_timestamps_for_score: campaigns with fewer raw timestamps
            than this get TS (and, transitively, CTCS) zeroed rather
            than scored -- a campaign with 1 timestamp can't meaningfully
            participate in a "synchronization" measure at all.
        """
        self.campaign_ids = campaign_ids
        self.campaign_index = {cid: i for i, cid in enumerate(campaign_ids)}
        self.fpv_matrix = fpv_matrix
        self.campaign_timestamps = campaign_timestamps
        self.sync_window_seconds = max(float(sync_window_seconds), 1e-9)
        self.min_timestamps_for_score = min_timestamps_for_score
        self.n = len(campaign_ids)

        self._corr_matrix = self._compute_correlation_matrix()
        self._sync_count_matrix, self._total_pairs_matrix = self._compute_sync_matrices()

    def _compute_correlation_matrix(self) -> np.ndarray:
        """
        Vectorized Pearson correlation across every campaign pair via
        np.corrcoef -- mathematically identical to the handwritten
        Corr(F_i, F_j) formula, just computed for all pairs at once
        instead of a manual nested-loop accumulation. Zero-variance
        rows (a campaign funded in only one time bin) produce NaN from
        np.corrcoef; those are replaced with 0.0 here, with a one-time
        diagnostic log, since "undefined" isn't a value the Transformer
        (or the averaging in calculate_ctcc) can consume.
        """
        if self.n < 2:
            return np.zeros((self.n, self.n))

        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(self.fpv_matrix)

        nan_mask = np.isnan(corr)
        if nan_mask.any():
            zero_variance_campaigns = [
                self.campaign_ids[i] for i in range(self.n)
                if np.std(self.fpv_matrix[i]) == 0
            ]
            logger.info(
                "%d campaign(s) have zero-variance funding pattern vectors (funded in "
                "only one time bin, or not at all) -- their CTCC correlations are "
                "undefined and treated as 0.0: %s",
                len(zero_variance_campaigns), zero_variance_campaigns[:10]
            )
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        return corr

    def _compute_sync_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds the full (n, n) SyncCount and TotalPairs matrices once,
        using count_synced_pairs' binary-search approach for every
        campaign pair, so per-campaign TS lookups afterward are O(n)
        instead of recomputing pairwise sync counts on every call.
        """
        sync_count = np.zeros((self.n, self.n))
        total_pairs = np.zeros((self.n, self.n))

        ts_arrays = [self.campaign_timestamps.get(cid, np.array([])) for cid in self.campaign_ids]
        usable = [len(ts) >= self.min_timestamps_for_score for ts in ts_arrays]

        for i in range(self.n):
            if not usable[i]:
                continue
            for j in range(i + 1, self.n):
                if not usable[j]:
                    continue
                count = count_synced_pairs(ts_arrays[i], ts_arrays[j], self.sync_window_seconds)
                pairs = len(ts_arrays[i]) * len(ts_arrays[j])
                sync_count[i, j] = sync_count[j, i] = count
                total_pairs[i, j] = total_pairs[j, i] = pairs

        return sync_count, total_pairs

    def calculate_ctcc(self, campaign_id: str) -> float:
        """
        This campaign's average Pearson correlation with every OTHER
        campaign's funding pattern vector. See module docstring for why
        this is "average over j != i" rather than the global "average
        over all i<j pairs" (that global version is compute_global_scores()).
        """
        if campaign_id not in self.campaign_index or self.n < 2:
            return 0.0
        i = self.campaign_index[campaign_id]
        others = np.concatenate([self._corr_matrix[i, :i], self._corr_matrix[i, i + 1:]])
        if others.size == 0:
            return 0.0
        return float(np.clip(np.mean(others), -1.0, 1.0))

    def calculate_ts(self, campaign_id: str) -> float:
        """
        This campaign's fraction of cross-campaign timestamp pairs that
        fall within sync_window_seconds of each other, aggregated over
        every OTHER campaign (sum of SyncCount over j != i, divided by
        sum of TotalPairs over j != i).
        """
        if campaign_id not in self.campaign_index or self.n < 2:
            return 0.0
        i = self.campaign_index[campaign_id]
        total_pairs = np.sum(self._total_pairs_matrix[i, :])
        if total_pairs <= 0:
            return 0.0
        sync_count = np.sum(self._sync_count_matrix[i, :])
        return float(np.clip(sync_count / total_pairs, 0.0, 1.0))

    def calculate_ctcs(self, campaign_id: str) -> float:
        """Final score: CTCS_i = 0.6 * CTCC_i + 0.4 * TS_i, per the notes' weighting."""
        ctcc = self.calculate_ctcc(campaign_id)
        ts = self.calculate_ts(campaign_id)
        return float(0.6 * ctcc + 0.4 * ts)

    def compute_vector(self, campaign_id: str) -> Dict[str, float]:
        ctcc = self.calculate_ctcc(campaign_id)
        ts = self.calculate_ts(campaign_id)
        return {"ctcc": ctcc, "ts": ts, "ctcs": float(0.6 * ctcc + 0.4 * ts)}

    def compute_global_scores(self) -> Dict[str, float]:
        """
        The LITERAL handwritten formulas: one scalar for the entire
        dataset, averaged over every distinct campaign pair (i<j), not
        per-campaign. Useful as a single "how coordinated is this whole
        dataset" headline number alongside the per-campaign vector.
        """
        if self.n < 2:
            return {"global_ctcc": 0.0, "global_ts": 0.0, "global_ctcs": 0.0}

        iu = np.triu_indices(self.n, k=1)
        global_ctcc = float(np.mean(self._corr_matrix[iu])) if len(iu[0]) > 0 else 0.0

        total_sync = float(np.sum(self._sync_count_matrix[iu]))
        total_pairs = float(np.sum(self._total_pairs_matrix[iu]))
        global_ts = total_sync / total_pairs if total_pairs > 0 else 0.0

        global_ctcs = 0.6 * global_ctcc + 0.4 * global_ts
        return {"global_ctcc": global_ctcc, "global_ts": global_ts, "global_ctcs": global_ctcs}


# ---------------------------------------------------------------------------
# Batch extraction from a full dataset
# ---------------------------------------------------------------------------
class CTCSExtractor:
    def __init__(self, num_time_bins: int = 20, sync_window_seconds: float = 3600.0,
                 min_timestamps_for_score: int = 2):
        """
        num_time_bins: how many equal-width bins to divide the GLOBAL
            observation period into for the Funding Pattern Vector.
            More bins = finer-grained trend comparison but noisier/
            sparser vectors for short campaigns; fewer bins = smoother
            but coarser. 20 is a reasonable starting point; tune to your
            typical campaign duration (e.g. daily bins for month-long
            campaigns would be num_time_bins ~= observation_days).
        sync_window_seconds: TS's "omega" (see CTCS.__init__).
        min_timestamps_for_score: campaigns with fewer raw timestamps
            than this have TS (and therefore CTCS) zeroed -- see
            CTCS.__init__.
        """
        self.num_time_bins = num_time_bins
        self.sync_window_seconds = sync_window_seconds
        self.min_timestamps_for_score = min_timestamps_for_score
        self._last_global_scores: Optional[Dict[str, float]] = None

    @property
    def global_scores_(self) -> Optional[Dict[str, float]]:
        """The literal dataset-wide CTCC/TS/CTCS from the most recent extract_dataframe() call."""
        return self._last_global_scores

    def extract_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        mapping = create_column_mapping(df)
        for required in ["campaign_id", "amount", "timestamp"]:
            if mapping[required] is None:
                raise ValueError(
                    f"Dataset does not contain a column for '{required}'. CTCS requires "
                    f"campaign_id, amount, AND timestamp columns -- unlike CFIS's FTS/FAS, "
                    f"there's no meaningful degraded mode without both."
                )

        df = df.copy()
        df["_campaign_id"] = df[mapping["campaign_id"]].apply(safe_str)
        df["_amount"] = pd.to_numeric(df[mapping["amount"]], errors="coerce").fillna(0.0)
        df["_timestamp"] = df[mapping["timestamp"]].apply(parse_timestamp)
        df["_timestamp"] = pd.to_datetime(df["_timestamp"])

        before = len(df)
        df = df[~df["_campaign_id"].apply(is_null_like)]
        df = df.dropna(subset=["_timestamp"])
        removed = before - len(df)
        if removed:
            logger.info("Filtered out %d row(s) with missing campaign id or unparseable timestamp.", removed)

        if df.empty:
            logger.warning("No valid rows remain after filtering; returning empty feature set.")
            return pd.DataFrame(columns=["campaign_id", "n_transactions", "ctcc", "ts", "ctcs",
                                          "below_min_timestamps"])

        all_campaign_ids = sorted(df["_campaign_id"].unique())
        if len(all_campaign_ids) < 2:
            logger.warning(
                "Only %d distinct campaign(s) in the dataset; CTCS needs >= 2 campaigns "
                "to correlate/synchronize against each other. All scores will be 0.0.",
                len(all_campaign_ids)
            )

        # epoch-seconds float, needed for both binning and sync-window arithmetic
        # NOTE: do NOT use `.astype("int64") / 1e9` here. That assumes
        # datetime64[ns] resolution, but pandas (3.0+) can produce
        # datetime64[us] from a list of parsed Python datetimes -- which
        # silently makes every epoch value 1000x too small and collapses
        # sync_window_seconds comparisons (everything looks
        # "synchronized" once the whole timeline is squeezed by 1000x).
        # Timestamp.timestamp() is resolution-independent and always
        # correct regardless of the underlying dtype's unit.
        df["_epoch"] = df["_timestamp"].apply(lambda ts: ts.timestamp())

        bin_edges = build_time_bins(df["_epoch"].to_numpy(), self.num_time_bins)
        # np.digitize gives 1-indexed bins; clip into [0, num_time_bins-1] so the
        # single latest timestamp (which sits exactly on the last edge) still
        # lands in the final bin instead of spilling into a phantom extra bin.
        df["_bin"] = np.clip(np.digitize(df["_epoch"], bin_edges[1:-1], right=True), 0, self.num_time_bins - 1)

        campaign_bin_amounts: Dict[str, np.ndarray] = {
            cid: np.zeros(self.num_time_bins) for cid in all_campaign_ids
        }
        for (cid, b), amt in df.groupby(["_campaign_id", "_bin"])["_amount"].sum().items():
            campaign_bin_amounts[cid][int(b)] += float(amt)

        campaign_timestamps: Dict[str, np.ndarray] = {
            cid: np.sort(group["_epoch"].to_numpy())
            for cid, group in df.groupby("_campaign_id")
        }

        campaign_ids, fpv_matrix = build_funding_pattern_vectors(campaign_bin_amounts)

        ctcs = CTCS(campaign_ids, fpv_matrix, campaign_timestamps,
                    sync_window_seconds=self.sync_window_seconds,
                    min_timestamps_for_score=self.min_timestamps_for_score)

        self._last_global_scores = ctcs.compute_global_scores()
        logger.info("Global (dataset-wide) scores -- literal handwritten formulas: %s",
                    {k: round(v, 4) for k, v in self._last_global_scores.items()})

        n_transactions = df.groupby("_campaign_id").size().to_dict()

        records = []
        below_threshold_count = 0
        for cid in campaign_ids:
            below_threshold = len(campaign_timestamps.get(cid, [])) < self.min_timestamps_for_score
            below_threshold_count += below_threshold
            if below_threshold:
                vec = {"ctcc": 0.0, "ts": 0.0, "ctcs": 0.0}
            else:
                vec = ctcs.compute_vector(cid)
            records.append({
                "campaign_id": cid,
                "n_transactions": n_transactions.get(cid, 0),
                "ctcc": round(vec["ctcc"], 4),
                "ts": round(vec["ts"], 4),
                "ctcs": round(vec["ctcs"], 4),
                "below_min_timestamps": below_threshold,
            })

        if below_threshold_count:
            logger.info(
                "%d campaign(s) had fewer than min_timestamps_for_score=%d timestamps; "
                "their TS/CTCS were zeroed rather than scored.",
                below_threshold_count, self.min_timestamps_for_score
            )

        feature_df = pd.DataFrame(records)
        self._validate_features(feature_df)

        return feature_df

    @staticmethod
    def _validate_features(feature_df: pd.DataFrame) -> None:
        """ctcc in [-1,1], ts/ctcs in their documented ranges; logs a warning if violated."""
        bounds = {"ctcc": (-1.0, 1.0), "ts": (0.0, 1.0), "ctcs": (-0.6, 1.0)}
        for col, (lo, hi) in bounds.items():
            if col not in feature_df.columns or feature_df.empty:
                continue
            bad = feature_df[(feature_df[col] < lo - 1e-9) | (feature_df[col] > hi + 1e-9)]
            if len(bad) > 0:
                logger.warning(
                    "Feature '%s' has %d value(s) outside [%s, %s]: campaigns %s",
                    col, len(bad), lo, hi, bad["campaign_id"].tolist()
                )

    def extract_features(self, input_csv: str, output_csv: str) -> pd.DataFrame:
        df = pd.read_csv(input_csv)
        feature_df = self.extract_dataframe(df)
        feature_df.to_csv(output_csv, index=False)

        feature_cols = ["ctcc", "ts", "ctcs"]
        logger.info("CTCS Feature Statistics:\n%s", feature_df[feature_cols].describe())

        finite_mask = np.isfinite(feature_df[feature_cols])
        if finite_mask.all().all():
            logger.info("All CTCS feature values are finite.")
        else:
            logger.warning("Infinite or NaN values detected in CTCS features. "
                            "Fix these before feeding the Transformer.")

        return feature_df


def to_numpy(feature_df: pd.DataFrame) -> np.ndarray:
    """
    Converts the CTCS feature DataFrame into a clean (N, 3) float32
    array in [ctcc, ts, ctcs] order (diagnostic n_transactions and
    below_min_timestamps columns excluded). NaN/Inf replaced with 0.0.
    """
    array = feature_df[["ctcc", "ts", "ctcs"]].values.astype(np.float32)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Sample data generator (for local testing without a real dataset)
# ---------------------------------------------------------------------------
def generate_sample_csv(path: str, num_campaigns: int = 15, seed: Optional[int] = 42) -> None:
    """
    Generates a synthetic campaign-level transaction dataset covering
    two archetypes:

      - COORDINATED GROUP: several campaigns whose funding both trends
        together over time (same underlying wave shape, driving high
        CTCC) AND whose individual transactions repeatedly land within
        seconds of each other across campaigns (driving high TS).
      - ORGANIC campaigns: independent random funding trends and
        independently-timed transactions (should score low on both).

    seed: fixes `random`/`numpy.random` state for reproducibility.
    """
    import random as _random

    if seed is not None:
        _random.seed(seed)
        np.random.seed(seed)

    rows = []
    base_time = datetime(2026, 1, 1, 0, 0, 0)
    observation_days = 14

    # COORDINATED GROUP: 4 campaigns sharing one underlying funding wave
    # (a simple rising-then-falling shape), each with its own noise, plus
    # transactions clustered in short synchronized bursts across campaigns.
    coordinated_ids = [f"COORD{i}" for i in range(4)]
    num_bursts = 10
    for burst in range(num_bursts):
        # Bursts spread across the observation window, evenly spaced.
        burst_time = base_time + timedelta(
            seconds=int(burst / num_bursts * observation_days * 24 * 3600)
        )
        # Shared "wave" amount for this burst, common to all coordinated campaigns.
        wave_amount = 500 + 400 * np.sin(burst / num_bursts * np.pi)
        for cid in coordinated_ids:
            jitter_seconds = _random.randint(0, 30)  # near-identical, not exact
            amount = max(wave_amount + _random.uniform(-20, 20), 1)
            rows.append({
                "campaign_id": cid,
                "amount": round(amount, 2),
                "timestamp": (burst_time + timedelta(seconds=jitter_seconds)).isoformat(),
            })

    # ORGANIC campaigns: independent random trends, independently timed.
    for i in range(num_campaigns - len(coordinated_ids)):
        cid = f"ORGANIC{i}"
        n_tx = _random.randint(15, 40)
        # Each organic campaign gets its OWN random trend shape, uncorrelated
        # with the others and with the coordinated group's shared wave.
        trend_phase = _random.uniform(0, 2 * np.pi)
        for _ in range(n_tx):
            t_offset = _random.uniform(0, observation_days * 24 * 3600)
            trend_amount = 300 + 250 * np.sin(t_offset / (observation_days * 24 * 3600) * 2 * np.pi + trend_phase)
            amount = max(trend_amount + _random.uniform(-150, 150), 1)
            rows.append({
                "campaign_id": cid,
                "amount": round(amount, 2),
                "timestamp": (base_time + timedelta(seconds=t_offset)).isoformat(),
            })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle row order
    df.to_csv(path, index=False)
    logger.info("Sample dataset created: %s (%d rows, %d campaigns)",
                path, len(df), df["campaign_id"].nunique())


if __name__ == "__main__":
    # Resolve the input CSV relative to THIS script's own folder, not the
    # process's current working directory. Code Runner / different IDEs
    # can launch a script with the working directory set to the project
    # root instead of the file's own folder -- if two files named
    # "unified_dataset.csv" exist (e.g. one in the project root and one
    # in modules/), a plain relative path silently picks up the wrong
    # one. Anchoring to __file__'s directory makes this deterministic.
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT  = os.path.join(SCRIPT_DIR, "..", "Datasets", "unified_dataset.csv")
    OUTPUT = os.path.join(SCRIPT_DIR, "..", "outputs", "ctcs_features.csv")


    if not os.path.exists(INPUT):
        logger.info("Input CSV not found. Creating a sample dataset...")
        generate_sample_csv(INPUT)

    extractor = CTCSExtractor(num_time_bins=20, sync_window_seconds=3600.0)
    features = extractor.extract_features(INPUT, OUTPUT)

    logger.info("CTCS Feature Extraction Completed")
    logger.info("CTCS Features extracted:\n%s", features)
    logger.info("Output columns: %s", features.columns.tolist())
    logger.info("Clean CTCS feature CSV saved to: %s", OUTPUT)
    logger.info("Global (dataset-wide) scores: %s", extractor.global_scores_)

    feature_matrix = to_numpy(features)
    logger.info("Feature matrix shape for Transformer input: %s", feature_matrix.shape)