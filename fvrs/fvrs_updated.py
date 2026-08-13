"""
FVRS (Funding Velocity Risk Score) Feature Extraction Pipeline
==================================================================
Sub-features:  FGR, TBR, NUR, FTR, RAR  ->  feature vector fed into
the Dataset Builder alongside CTI, CFIS, CTCS, SDI, and ultimately
into the TrustNet Transformer Encoder as a (N, 5) float32 array.
"""

import os
import ast
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column auto-mapping helpers
# ---------------------------------------------------------------------------
def find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    """
    Finds the first matching column from a list of possible names.

    Example:
        current_funding may appear as:
        funding, current_amount, pledged, raised_amount
    """
    columns = {c.lower(): c for c in df.columns}

    for name in possible_names:
        if name.lower() in columns:
            return columns[name.lower()]

    return None


def create_column_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Automatically map dataset columns to FVRS requirements.
    """
    mapping = {
        "current_funding": find_column(df, [
            "current_funding", "funding", "current_amount",
            "pledged", "raised_amount"
        ]),
        "previous_funding": find_column(df, [
            "previous_funding", "initial_funding",
            "previous_amount", "start_amount"
        ]),
        "transactions_per_minute": find_column(df, [
            "transactions_per_minute", "tpm", "transaction_rate"
        ]),
        "total_transactions": find_column(df, [
            "total_transactions", "transaction_count", "num_transactions"
        ]),
        "new_users": find_column(df, [
            "new_users", "new_backers", "first_time_backers"
        ]),
        "total_users": find_column(df, [
            "total_users", "backers", "contributors"
        ]),
        "timestamps": find_column(df, [
            "timestamps", "transaction_times", "time_list"
        ]),
        "amounts": find_column(df, [
            "amounts", "transaction_amounts", "payments"
        ]),
        # Some datasets (like unified_dataset.csv) are transaction-level:
        # one row per transaction, with a single "timestamp" and "amount"
        # value per row instead of a pre-built list per campaign. These
        # get grouped into per-campaign lists in extract_dataframe().
        "timestamp": find_column(df, [
            "timestamp", "transaction_time", "tx_time", "date"
        ]),
        "amount": find_column(df, [
            "amount", "transaction_amount", "payment", "tx_amount"
        ])
    }

    return mapping


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def parse_list(value: Any) -> List:
    """
    Converts a CSV string representation of a list into an actual
    Python list. Only catches parsing-related exceptions (no bare except).
    """
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def parse_timestamp(value):
    """
    Parses a single timestamp value using pd.to_datetime, which -- unlike
    datetime.fromisoformat() -- handles ISO strings, UTC strings, epoch
    milliseconds/seconds, timezone-aware strings, and most other formats
    found in real-world public datasets, not just strict ISO 8601.
    """
    if isinstance(value, datetime):
        return value

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            # pd.to_datetime() treats a bare numeric value as nanoseconds
            # since epoch by default, which silently mis-parses ordinary
            # unix seconds/milliseconds timestamps (e.g. 1735689600 would
            # become a 1970 date instead of 2025). Infer the unit from
            # magnitude instead of trusting the default.
            if value > 1e14:        # microseconds since epoch
                parsed = pd.to_datetime(value, unit="us", utc=True)
            elif value > 1e11:      # milliseconds since epoch
                parsed = pd.to_datetime(value, unit="ms", utc=True)
            else:                   # seconds since epoch
                parsed = pd.to_datetime(value, unit="s", utc=True)
        else:
            # utc=True normalizes ANY input -- naive or tz-aware, and any
            # mix of offsets (+05:30, Z, -08:00, etc.) -- onto UTC. Without
            # this, a dataset mixing "2025-01-01T12:00:00" (naive) and
            # "2025-01-01T12:00:00Z" (aware) would raise
            # "can't compare offset-naive and offset-aware datetimes" the
            # moment two such timestamps are sorted together.
            parsed = pd.to_datetime(value, utc=True)
    except (ValueError, TypeError):
        return None

    if pd.isna(parsed):
        return None

    # Drop tzinfo AFTER normalizing to UTC, not before -- this keeps every
    # returned datetime naive (so it's a drop-in replacement everywhere
    # else in this module) while guaranteeing they're all on the same
    # absolute timeline, so sorting/subtracting them stays correct even
    # when the source data mixes timezones or naive/aware timestamps.
    return parsed.tz_localize(None).to_pydatetime()


def safe_float(value, default: float = 0.0) -> float:
    """Safely coerce any value to float, falling back to `default`."""
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def duration_and_tpm_from_timestamps(timestamps: List[datetime]):
    """
    Recomputes (duration_hours, transactions_per_minute) directly from
    a sorted timestamp list, so it never disagrees with a separately
    supplied CSV column. Returns (None, None) if there isn't enough
    timestamp data to compute this reliably (caller should fall back
    to CSV-provided values in that case).
    """
    ts = [t for t in timestamps if t is not None]
    if len(ts) < 2:
        return None, None

    ts = sorted(ts)
    span_seconds = (ts[-1] - ts[0]).total_seconds()
    if span_seconds <= 0:
        return None, None

    duration_hours = span_seconds / 3600.0
    tpm = len(ts) / (span_seconds / 60.0)
    return duration_hours, tpm


# ---------------------------------------------------------------------------
# Core FVRS calculations
# ---------------------------------------------------------------------------
class FVRS:
    def __init__(self):
        pass

    def calculate_fgr(self, current, previous):
        """
        Funding Growth Rate (FGR) -- raw value, NOT yet scaled.

        Measures growth in funding from previous to current campaign.

        - previous > 0  -> standard percentage growth, capped to
                            [-100, 1000] so outliers don't dominate.
        - previous <= 0 -> there is no valid baseline to compute a
                            percentage against. Returning a constant
                            (e.g. 100) for every such campaign destroys
                            information, since a campaign that raised
                            500 and one that raised 100,000 would look
                            identical. Instead we use log1p(current),
                            which preserves relative magnitude while
                            keeping the value on a compressed, bounded
                            scale that won't blow up the Transformer.

        NOTE: this method intentionally returns the raw/unscaled value.
        `compute_vector()` calls `_scale_fgr()` on this result before
        it goes into the feature vector, so that callers who want the
        raw percentage (e.g. for a dashboard or debugging) can still
        get it from this method directly.
        """
        current = safe_float(current, 0.0)
        previous = safe_float(previous, 0.0)

        if previous <= 0:
            if current <= 0:
                return 0.0
            return float(np.log1p(current))

        growth = ((current - previous) / previous) * 100
        return max(-100.0, min(1000.0, growth))

    def _scale_fgr(self, raw_fgr: float) -> float:
        """
        Rescales FGR onto roughly [0, 1] so it sits on the same scale as
        TBR, NUR, FTR, and RAR before hitting the Transformer. Without
        this, FGR (which can range from -100 to 1000, or 0-15ish for the
        log1p/no-baseline branch) numerically dominates a feature vector
        whose other four entries all live in [0, 1].

        Mapping: (raw_fgr + 100) / 1100, clipped to [0, 1].
        This means:
            raw_fgr = -100  (max drop)          -> 0.0
            raw_fgr =    0  (no change)          -> ~0.091
            raw_fgr = 1000  (capped max growth)  -> 1.0

        Caveat: the log1p(current) branch (previous <= 0) produces raw
        values roughly in [0, ~14] for realistic funding amounts, which
        this same mapping compresses into a narrow ~[0.091, 0.103] band.
        That's an intentional trade-off, not a bug -- campaigns with no
        valid previous-funding baseline are already a distinct, somewhat
        anomalous case, and relative ordering within that narrow band is
        still preserved. If your evaluation shows the Transformer can't
        distinguish within that band well enough, consider adding a
        separate binary "has_baseline" flag as its own feature instead
        of relying on FGR alone to carry that signal.
        """
        return float(np.clip((raw_fgr + 100.0) / 1100.0, 0.0, 1.0))

    def calculate_tbr(self, transactions_per_minute, total_transactions, duration_hours):
        """
        Transaction Burst Rate (TBR) -- approximation of burst rate.

        This is an APPROXIMATION, not a strict burst-detection statistic:
        it assumes transactions are uniformly distributed across the
        campaign duration to compute an "expected" rate, then compares
        the observed rate against it. Real campaigns may have uneven
        activity (e.g. launch-day spikes), so treat this as a coarse
        intensity signal rather than a precise burst detector.
        """
        transactions_per_minute = safe_float(transactions_per_minute, 0.0)
        total_transactions = safe_float(total_transactions, 0.0)
        duration_hours = safe_float(duration_hours, 0.0)

        if duration_hours <= 0 or transactions_per_minute < 0 or total_transactions < 0:
            return 0.0

        expected = total_transactions / (duration_hours * 60)
        expected = max(expected, 0.01)  # prevent division by zero

        burst_ratio = transactions_per_minute / expected
        return float(min(burst_ratio, 1.0))

    def calculate_nur(self, new_users, total_users):
        """
        New User Rate (NUR)

        Measures the proportion of new users relative to total users.
        """
        new_users = safe_float(new_users, 0.0)
        total_users = safe_float(total_users, 0.0)

        if total_users <= 0:
            return 0.0

        return float(min(new_users / total_users, 1.0))

    def calculate_ftr(self, timestamps):
        """
        Funding Timing Regularity (FTR)

        Measures how regular the transaction timing is.
        Higher values indicate more regular (bot-like) intervals.
        """
        timestamps = [parse_timestamp(x) for x in timestamps]
        timestamps = [x for x in timestamps if x is not None]

        if len(timestamps) < 3:
            return 0.0

        timestamps = sorted(timestamps)

        intervals = []
        for i in range(len(timestamps) - 1):
            delta = (timestamps[i + 1] - timestamps[i]).total_seconds()
            if delta > 0:  # ignore duplicate timestamps
                intervals.append(delta)

        if len(intervals) < 2:
            return 0.0

        mean = np.mean(intervals)
        if mean == 0:
            return 1.0

        std = np.std(intervals)
        cv = std / mean
        return float(1 / (1 + cv))

    def calculate_rar(self, amounts, threshold: int = 1000, tol: float = 1e-6):
        """
        Round Amount Ratio (RAR)

        Measures the proportion of transactions with round amounts.
        Only flags amounts divisible by `threshold` (default 1000),
        which naturally covers 1000 / 5000 / 10000-style round numbers.
        Small thresholds like 100 or 200 are deliberately excluded --
        they're common in legitimate donations and add noise rather
        than signal. Uses a floating-point tolerance instead of exact
        modulo equality to avoid float precision false negatives.
        """
        if len(amounts) == 0:
            return 0.0

        valid_amounts = []
        round_count = 0

        for amount in amounts:
            try:
                amount = float(amount)
            except (TypeError, ValueError):
                continue

            if amount <= 0:
                continue

            valid_amounts.append(amount)
            remainder = amount % threshold
            if remainder < tol or (threshold - remainder) < tol:
                round_count += 1

        if len(valid_amounts) == 0:
            return 0.0

        return round_count / len(valid_amounts)

    def compute_vector(self, campaign: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute FVRS feature vector for a single campaign.

        Returns a dict with: fgr, tbr, nur, ftr, rar -- all scaled to
        roughly [0, 1] so no single sub-feature numerically dominates
        the vector fed into the Transformer.
        """
        current = safe_float(campaign.get("current_funding", 0))
        previous = safe_float(campaign.get("previous_funding", 0))

        new_users = safe_float(campaign.get("new_users", 0))
        total_users = safe_float(campaign.get("total_users", 0))

        timestamps = campaign.get("timestamps", []) or []
        amounts = campaign.get("amounts", []) or []

        parsed_timestamps = [parse_timestamp(t) for t in timestamps]
        valid_timestamps = [t for t in parsed_timestamps if t is not None]

        # Recompute duration & TPM from the actual timestamps whenever
        # possible, so they can never disagree with a separately
        # supplied CSV column. Fall back to CSV-provided values only
        # when there isn't enough timestamp data to derive them.
        derived_duration, derived_tpm = duration_and_tpm_from_timestamps(parsed_timestamps)

        duration = derived_duration if derived_duration is not None \
            else safe_float(campaign.get("campaign_duration_hours", 24), 24)
        tpm = derived_tpm if derived_tpm is not None \
            else safe_float(campaign.get("transactions_per_minute", 0), 0)

        # total_transactions: prefer the count of actual parsed timestamps
        # over a separately supplied CSV column, since the two can
        # silently disagree (e.g. 40 timestamps logged but a CSV column
        # claiming 31 transactions). Only fall back to the CSV column /
        # len(amounts) when there are no usable timestamps to count.
        if valid_timestamps:
            total_tx = float(len(valid_timestamps))
        else:
            total_tx = safe_float(
                campaign.get("total_transactions", len(amounts)),
                len(amounts)
            )

        raw_fgr = self.calculate_fgr(current, previous)

        return {
            "fgr": self._scale_fgr(raw_fgr),
            "tbr": self.calculate_tbr(tpm, total_tx, duration),
            "nur": self.calculate_nur(new_users, total_users),
            "ftr": self.calculate_ftr(timestamps),
            "rar": self.calculate_rar(amounts)
        }


# ---------------------------------------------------------------------------
# Batch extraction from a full dataset
# ---------------------------------------------------------------------------
class FVRSExtractor:
    def __init__(self):
        self.fvrs = FVRS()

    def extract_dataframe(self, df):
        """
        Extract FVRS features from a pandas DataFrame.

        Supports two dataset shapes:
          1. Campaign-level: one row per campaign, with "timestamps"/
             "amounts" columns already holding a list per row (e.g.
             "[t1, t2, ...]" as a string).
          2. Transaction-level: one row per transaction (e.g.
             unified_dataset.csv), with a single "timestamp"/"amount"
             value per row and multiple rows sharing the same
             campaign_id. These rows are grouped by campaign_id and
             the timestamp/amount values collected into per-campaign
             lists before being handed to compute_vector().
        """
        mapping = create_column_mapping(df)

        has_list_columns = mapping["timestamps"] is not None and mapping["amounts"] is not None
        has_row_columns = mapping["timestamp"] is not None and mapping["amount"] is not None

        if not has_list_columns and not has_row_columns:
            raise ValueError(
                "Dataset does not contain a column for 'timestamps'/'amounts' "
                "(list-per-campaign) or 'timestamp'/'amount' (row-per-transaction). "
                "Please ensure your CSV has appropriate column names."
            )

        required = [
            "current_funding", "previous_funding", "transactions_per_minute",
            "total_transactions", "new_users", "total_users"
        ]

        for r in required:
            if mapping[r] is None:
                raise ValueError(
                    f"Dataset does not contain a column for '{r}'. "
                    f"Please ensure your CSV has appropriate column names."
                )

        if "campaign_id" not in df.columns:
            df["campaign_id"] = [f"CAMP{i:04d}" for i in range(len(df))]

        if "campaign_duration_hours" not in df.columns:
            df["campaign_duration_hours"] = 24  # default duration fallback only

        vectors = []

        if has_list_columns:
            # Campaign-level shape: one row per campaign already.
            df["timestamps"] = df[mapping["timestamps"]].apply(parse_list)
            df["amounts"] = df[mapping["amounts"]].apply(parse_list)

            for _, row in df.iterrows():
                campaign = {
                    "current_funding": row[mapping["current_funding"]],
                    "previous_funding": row[mapping["previous_funding"]],
                    "transactions_per_minute": row[mapping["transactions_per_minute"]],
                    "total_transactions": row[mapping["total_transactions"]],
                    "campaign_duration_hours": row["campaign_duration_hours"],
                    "new_users": row[mapping["new_users"]],
                    "total_users": row[mapping["total_users"]],
                    "timestamps": row["timestamps"],
                    "amounts": row["amounts"]
                }

                vec = self.fvrs.compute_vector(campaign)

                vectors.append({
                    "campaign_id": row["campaign_id"],
                    "fgr": vec["fgr"],
                    "tbr": vec["tbr"],
                    "nur": vec["nur"],
                    "ftr": vec["ftr"],
                    "rar": vec["rar"]
                })
        else:
            # Transaction-level shape: group rows sharing a campaign_id
            # into a single campaign record, collecting the per-row
            # timestamp/amount values into lists in their original
            # (row) order.
            for campaign_id, group in df.groupby("campaign_id", sort=False):
                campaign = {
                    # current_funding is cumulative -- the last row holds
                    # the final funding total for the campaign.
                    "current_funding": group[mapping["current_funding"]].iloc[-1],
                    # previous_funding is the baseline before the first
                    # transaction was recorded.
                    "previous_funding": group[mapping["previous_funding"]].iloc[0],
                    # transactions_per_minute, total_transactions, and
                    # campaign_duration_hours are constant per campaign
                    # in this dataset, so the first row's value applies.
                    "transactions_per_minute": group[mapping["transactions_per_minute"]].iloc[0],
                    "total_transactions": group[mapping["total_transactions"]].iloc[0],
                    "campaign_duration_hours": group["campaign_duration_hours"].iloc[0],
                    # new_users is a per-transaction flag (1 if that
                    # transaction was from a first-time backer) -- sum
                    # it to get the campaign's total new-user count.
                    "new_users": group[mapping["new_users"]].sum(),
                    # total_users is cumulative -- the last row holds the
                    # final distinct-user count for the campaign.
                    "total_users": group[mapping["total_users"]].iloc[-1],
                    "timestamps": group[mapping["timestamp"]].tolist(),
                    "amounts": group[mapping["amount"]].tolist()
                }

                vec = self.fvrs.compute_vector(campaign)

                vectors.append({
                    "campaign_id": campaign_id,
                    "fgr": vec["fgr"],
                    "tbr": vec["tbr"],
                    "nur": vec["nur"],
                    "ftr": vec["ftr"],
                    "rar": vec["rar"]
                })

        feature_df = pd.DataFrame(vectors)
        feature_df.fillna(0, inplace=True)

        return feature_df

    def extract_features(self, input_csv, output_csv):
        """
        Extract FVRS features from CSV and save to output.
        Returns a clean FVRS feature CSV with campaign_id + FVRS features.
        """
        df = pd.read_csv(input_csv)
        feature_df = self.extract_dataframe(df)

        feature_df.to_csv(output_csv, index=False)

        print("\nFVRS Feature Statistics:")
        feature_cols = ["fgr", "tbr", "nur", "ftr", "rar"]
        print(feature_df[feature_cols].describe())

        print("\nChecking for Infinite Values...")
        finite_mask = np.isfinite(feature_df[feature_cols])
        print(finite_mask.all())

        if not finite_mask.all().all():
            print("Warning: Infinite or NaN values detected in FVRS features. "
                  "Fix these before feeding the Transformer.")

        return feature_df


def to_numpy(feature_df: pd.DataFrame) -> np.ndarray:
    """
    Converts the FVRS feature DataFrame into a clean (N, 5) float32
    array in [fgr, tbr, nur, ftr, rar] order -- ready to hand straight
    to the TrustNet Transformer Encoder (or to be concatenated with
    CTI, CFIS, CTCS, SDI feature blocks first).

    Any NaN/+Inf/-Inf that slipped through upstream (e.g. from a
    malformed row) is replaced with 0.0 here, so the Transformer is
    guaranteed to never receive a non-finite value regardless of what
    the raw dataset contained.
    """
    array = feature_df[["fgr", "tbr", "nur", "ftr", "rar"]].values.astype(np.float32)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Sample data generator (for local testing without a real dataset)
# ---------------------------------------------------------------------------
def generate_sample_csv(path, num_campaigns=20):
    """
    Generate a sample CSV dataset with proper column names.
    """
    rows = []
    base_time = datetime.now()

    for i in range(num_campaigns):
        previous = random.choice([0, random.randint(1000, 50000)])
        current = previous + random.randint(500, 30000)
        total_tx = random.randint(10, 80)
        duration = random.randint(2, 72)
        tpm = total_tx / (duration * 60)
        total_users = random.randint(20, 100)
        new_users = random.randint(0, total_users)

        timestamps = []
        amounts = []
        current_time = base_time

        for j in range(total_tx):
            current_time = current_time + timedelta(seconds=random.randint(20, 300))
            timestamps.append(current_time.isoformat())
            amounts.append(random.choice([137, 250, 999, 1000, 5000, 10000]))

        rows.append({
            "campaign_id": f"CAMP{i + 1:04d}",
            "current_funding": current,
            "previous_funding": previous,
            "transactions_per_minute": tpm,
            "total_transactions": total_tx,
            "new_users": new_users,
            "total_users": total_users,
            "campaign_duration_hours": duration,
            "timestamps": str(timestamps),
            "amounts": str(amounts)
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print("Sample dataset created:", path)


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT  = os.path.join(SCRIPT_DIR, "..", "Datasets", "unified_dataset.csv")
    OUTPUT = os.path.join(SCRIPT_DIR, "..", "outputs", "fvrs_features.csv")

    if not os.path.exists(INPUT):
        print("Input CSV not found.")
        print("Creating sample dataset...")
        generate_sample_csv(INPUT)

    extractor = FVRSExtractor()
    features = extractor.extract_features(INPUT, OUTPUT)

    print("\nFVRS Feature Extraction Completed\n")
    print("FVRS Features extracted:")
    print(features.head())

    print("\nOutput columns:", features.columns.tolist())
    print("\nClean FVRS feature CSV saved to:", OUTPUT)

    # Ready for the TrustNet Transformer Encoder:
    feature_matrix = to_numpy(features)
    print("\nFeature matrix shape for Transformer input:", feature_matrix.shape)