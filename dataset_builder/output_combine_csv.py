"""
==============================================================================
TrustNet: Blockchain-Based Crowdfunding Fraud Detection Using Transformer
Networks

Module: Dataset Builder (Simplified)
==============================================================================
"""

from pathlib import Path
import os
import pandas as pd


# Shared primary key present in every TrustNet feature CSV file.
CAMPAIGN_ID_COLUMN = "campaign_id"

# Name of the optional fraud label column, which may appear in one or
# more of the five source files.
FRAUD_LABEL_COLUMN = "fraud_label"


# Same weights and threshold as TrustMetricsRegistry.sol's
# getCombinedRisk() function, so the label logic used to train
# TrustNet matches the on-chain risk formula exactly.
#   SDI  contributes 25%  (high SDI = high risk)
#   CTI  contributes 20%  (inverted — high CTI = LOW risk)
#   FVRS contributes 20%  (high FVRS = high risk)
#   CFIS contributes 20%  (high CFIS = high risk)
#   CTCS contributes 15%  (high CTCS = high risk)
METRIC_WEIGHTS = {
    'sdi':  0.25,
    'cti':  0.20,   # applied to (1 - cti), since high CTI = trust
    'fvrs': 0.20,
    'cfis': 0.20,
    'ctcs': 0.15,
}

# Instead of a FIXED risk cutoff (which only makes sense if your
# data's combined_risk values actually span the full 0-1 range),
# we label the riskiest TARGET_FRAUD_RATE% of campaigns in THIS
# dataset as fraud — whatever the actual numeric range of
# combined_risk happens to be. This threshold is recomputed fresh
# for every dataset you run this on, using that data's own
# distribution (via a percentile/quantile cutoff), so it never
# silently fails just because a dataset's risk scores cluster in
# a narrow band (e.g. 0.32–0.69) that never reaches a fixed 0.70.
#
# NOTE: this is separate from riskThreshold = 70 in
# TrustMetricsRegistry.sol. That fixed on-chain value is a policy
# decision for enforcement (freezing funds) set by platform
# admins. This percentile threshold is only used here, to
# bootstrap TRAINING labels when no real fraud_label exists yet.
TARGET_FRAUD_RATE = 0.225   # aims for ~20-25% of campaigns labeled fraud


# FVRS and CFIS pipelines output several sub-feature columns
# instead of one single combined score. When no single
# "fvrs"/"cfis"-named column exists, these known sub-columns are
# averaged into a composite score instead.
METRIC_SUBCOLUMNS = {
    'fvrs': ['fgr', 'tbr', 'nur', 'ftr', 'rar'],
    'cfis': ['wgd', 'cds', 'col', 'inf', 'fts', 'fas'],
}


def find_metric_column(columns: list, metric_keyword: str) -> str:
    """
    Finds the column that corresponds to a given metric, regardless
    of exact naming (sdi_score, sdi_norm, sdi, SDI_Score, etc.) — so
    this doesn't break if a teammate names their output column
    slightly differently.
    """
    matches = [
        c for c in columns
        if metric_keyword in c.lower() and c != FRAUD_LABEL_COLUMN
    ]
    if not matches:
        return None
    # Prefer a normalized column if multiple matches exist
    # (e.g. cti_score AND cti_norm both present — norm is preferred)
    normalized = [c for c in matches if 'norm' in c.lower()]
    return normalized[0] if normalized else matches[0]


def get_metric_series(dataframe: pd.DataFrame, metric: str) -> pd.Series:
    """
    Returns a single 0-1 series for the given metric, either from
    a directly-named column (e.g. 'sdi_score', 'ctcs'), or — for
    metrics like FVRS/CFIS whose pipelines output several
    sub-feature columns instead of one combined score — by
    averaging the known sub-columns into a composite score.
    """
    columns = dataframe.columns.tolist()

    direct_col = find_metric_column(columns, metric)
    if direct_col is not None:
        print(f"  {metric.upper():<5} → using column '{direct_col}'")
        return dataframe[direct_col].astype(float)

    sub_cols = METRIC_SUBCOLUMNS.get(metric, [])
    present_sub_cols = [c for c in sub_cols if c in columns]

    if present_sub_cols:
        print(f"  {metric.upper():<5} → no single column found, "
              f"averaging sub-features: {present_sub_cols}")
        return dataframe[present_sub_cols].astype(float).mean(axis=1)

    raise ValueError(
        f"Could not find a column (or known sub-columns) for metric "
        f"'{metric}' in the merged dataset. Available columns: {columns}"
    )


def generate_fraud_label_from_scores(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Derives a fraud_label using the same weighted combined-risk
    formula as TrustMetricsRegistry.sol's getCombinedRisk():

        combined_risk = 0.25*SDI + 0.20*(1-CTI) + 0.20*FVRS
                         + 0.20*CFIS + 0.15*CTCS

        fraud_label = 1 if combined_risk >= 0.70 else 0

    This is only used when no real fraud_label already exists in
    any of the 5 feature files. It's a bootstrapped/pseudo-label —
    TrustNet trained on this learns to refine and generalize this
    rule rather than discover ground-truth fraud independently.
    That distinction should be stated explicitly in the paper.
    """
    print("\nNo real fraud_label found in source files — deriving "
          "labels from the combined risk formula (same weights and "
          "threshold as the on-chain TrustMetricsRegistry contract).")

    columns = dataframe.columns.tolist()

    sdi  = get_metric_series(dataframe, 'sdi')
    cti  = get_metric_series(dataframe, 'cti')
    fvrs = get_metric_series(dataframe, 'fvrs')
    cfis = get_metric_series(dataframe, 'cfis')
    ctcs = get_metric_series(dataframe, 'ctcs')

    cti_risk = 1.0 - cti

    combined_risk = (
        METRIC_WEIGHTS['sdi']  * sdi +
        METRIC_WEIGHTS['cti']  * cti_risk +
        METRIC_WEIGHTS['fvrs'] * fvrs +
        METRIC_WEIGHTS['cfis'] * cfis +
        METRIC_WEIGHTS['ctcs'] * ctcs
    )

    dataframe['combined_risk_score'] = combined_risk.round(4)

    # Compute the cutoff FRESH from this dataset's own distribution
    # — e.g. if combined_risk only ranges 0.32 to 0.69 here, the
    # cutoff will land somewhere inside that range (not a useless
    # fixed 0.70 that no campaign could ever reach).
    risk_cutoff = combined_risk.quantile(1 - TARGET_FRAUD_RATE)

    dataframe[FRAUD_LABEL_COLUMN] = (
        combined_risk >= risk_cutoff
    ).astype(int)

    fraud_count = dataframe[FRAUD_LABEL_COLUMN].sum()
    total_count = len(dataframe)
    print(f"\n  combined_risk_score range in this dataset: "
          f"{combined_risk.min():.4f} to {combined_risk.max():.4f}")
    print(f"  Target fraud rate: {TARGET_FRAUD_RATE*100:.1f}% | "
          f"Computed cutoff: {risk_cutoff:.4f}")
    print(f"  Fraud labeled: {fraud_count}/{total_count} "
          f"({fraud_count/total_count*100:.1f}%)")

    return dataframe


def load_feature_file(file_path: Path) -> pd.DataFrame:
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {file_path}"
        )

    dataframe = pd.read_csv(file_path)

    if CAMPAIGN_ID_COLUMN not in dataframe.columns:
        raise ValueError(
            f"File '{file_path.name}' is missing the required "
            f"'{CAMPAIGN_ID_COLUMN}' column."
        )

    return dataframe


def keep_single_fraud_label_column(dataframes: list) -> list:
    
    fraud_label_already_kept = False

    for dataframe in dataframes:
        if FRAUD_LABEL_COLUMN in dataframe.columns:
            if fraud_label_already_kept:
                dataframe.drop(columns=[FRAUD_LABEL_COLUMN], inplace=True)
            else:
                fraud_label_already_kept = True

    return dataframes


def merge_feature_files(dataframes: list) -> pd.DataFrame:
    
    merged_dataframe = dataframes[0]

    for dataframe in dataframes[1:]:
        merged_dataframe = merged_dataframe.merge(
            dataframe, on=CAMPAIGN_ID_COLUMN, how="inner"
        )

    return merged_dataframe


def move_fraud_label_to_end(dataframe: pd.DataFrame) -> pd.DataFrame:
    
    if FRAUD_LABEL_COLUMN not in dataframe.columns:
        return dataframe

    other_columns = [
        column for column in dataframe.columns if column != FRAUD_LABEL_COLUMN
    ]
    return dataframe[other_columns + [FRAUD_LABEL_COLUMN]]


def build_trustnet_dataset(
    fvrs_path: Path,
    cti_path: Path,
    cfis_path: Path,
    ctcs_path: Path,
    sdi_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    
    fvrs = load_feature_file(fvrs_path)
    cti = load_feature_file(cti_path)
    cfis = load_feature_file(cfis_path)
    ctcs = load_feature_file(ctcs_path)
    sdi = load_feature_file(sdi_path)

    all_feature_dataframes = keep_single_fraud_label_column(
        [fvrs, cti, cfis, ctcs, sdi]
    )

    merged_dataset = merge_feature_files(all_feature_dataframes)

    # If none of the 5 feature files already provided a real
    # fraud_label, derive one from the combined risk formula
    # (same weights/threshold as the smart contract) so TrustNet
    # always has something to train on.
    if FRAUD_LABEL_COLUMN not in merged_dataset.columns:
        merged_dataset = generate_fraud_label_from_scores(merged_dataset)
    else:
        print(f"\nUsing real '{FRAUD_LABEL_COLUMN}' column found in "
              f"source files (not deriving from formula).")

    merged_dataset = move_fraud_label_to_end(merged_dataset)

    merged_dataset.to_csv(output_path, index=False)

    print("TrustNet dataset created successfully.")
    print(f"Output file: {output_path}")
    print(f"Total campaigns: {len(merged_dataset)}")
    print(f"Total columns: {len(merged_dataset.columns)}")

    return merged_dataset


if __name__ == "__main__":
    # Resolve paths relative to THIS script's own folder, not the
    # process's current working directory — same reasoning as the
    # SDI/CTI/FVRS/CFIS/CTCS scripts: anchoring to __file__'s
    # directory makes this deterministic regardless of where the
    # script is launched from.
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "..", "outputs")

    build_trustnet_dataset(
        fvrs_path=Path(os.path.join(OUTPUTS_DIR, "fvrs_features.csv")),
        cti_path=Path(os.path.join(OUTPUTS_DIR, "cti_features.csv")),
        cfis_path=Path(os.path.join(OUTPUTS_DIR, "cfis_features.csv")),
        ctcs_path=Path(os.path.join(OUTPUTS_DIR, "ctcs_features.csv")),
        sdi_path=Path(os.path.join(OUTPUTS_DIR, "sdi_features.csv")),
        output_path=Path(os.path.join(OUTPUTS_DIR, "trustnet_dataset.csv")),
    )