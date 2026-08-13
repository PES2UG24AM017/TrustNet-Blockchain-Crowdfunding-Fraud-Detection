"""
Dataset Validator + Simulator
--------------------------------
Prepares a dataset for the 5 trust metric pipelines
(SDI, CTI, FVRS, CFIS, CTCS).

Logic:
  1. If a CSV is uploaded → auto-detect which required
     columns exist using keyword matching.
  2. Missing columns are simulated realistically,
     EXCEPT 'description' which has special handling:
       - If dataset looks like Gitcoin (has grant_id /
         project_approved / project_decription-style
         columns) → fetch real description via Gitcoin API
       - Otherwise → description is set to None,
         SDI is skipped for those rows
  3. If NO CSV is uploaded at all → simulate everything,
     including description (demo/testing mode).
  4. Every simulated column prints a warning message.

Usage:
    python dataset_simulator.py --input path/to/file.csv
    python dataset_simulator.py                (no CSV → full simulation)
"""

import os
import sys
import argparse
import random
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
NUM_SIMULATED_CAMPAIGNS   = 30
NUM_SIMULATED_WALLETS     = 150
TRANSACTIONS_PER_CAMPAIGN = (10, 60)   # min, max range

GITCOIN_API_BASE = "https://gitcoin.co/grants/v1/api/grants"

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "unified_dataset.csv"
)

random.seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────
# COLUMN KEYWORD MAP (for auto-detection)
# ─────────────────────────────────────────────
COLUMN_KEYWORDS = {
    'campaign_id':    ['campaign_id', 'campaign', 'project_id', 'grant_id', 'id'],
    'title':          ['title', 'name', 'campaign_name', 'project_title'],
    'description':    ['description', 'desc', 'blurb', 'summary',
                        'details', 'project_decription', 'project_description'],
    'approved':       ['approved', 'verified', 'is_approved', 'status',
                        'project_approved'],
    'team_size':      ['team_size', 'team_count', 'members', 'team',
                        'project_team_size'],
    'multichain':     ['multichain', 'chains', 'live_on_other_chains',
                        'networks', 'cross_chain'],
    'wallet_id':      ['wallet_id', 'wallet', 'address', 'contributor',
                        'from_address', 'sender'],
    'amount':         ['amount', 'value', 'contribution', 'pledged'],
    'timestamp':      ['timestamp', 'time', 'date', 'created_at', 'tx_time'],
    'current_funding':    ['current_funding', 'funding', 'current_amount',
                            'pledged_total', 'raised_amount'],
    'previous_funding':   ['previous_funding', 'initial_funding',
                            'previous_amount', 'start_amount'],
    'transactions_per_minute': ['transactions_per_minute', 'tpm',
                                 'transaction_rate'],
    'total_transactions': ['total_transactions', 'transaction_count',
                            'num_transactions'],
    'new_users':      ['new_users', 'new_backers', 'first_time_backers'],
    'total_users':    ['total_users', 'backers', 'contributors'],
    'campaign_duration_hours': ['campaign_duration_hours', 'duration_hours',
                                  'duration'],
    'timestamps':     ['timestamps', 'transaction_times', 'time_list'],
    'amounts':        ['amounts', 'transaction_amounts', 'payments'],
}

# Gitcoin-specific signature columns (used to detect Gitcoin datasets)
GITCOIN_SIGNATURE_COLUMNS = [
    'grant_id', 'project_decription', 'project_approved',
    'project_team_size', 'live_on_other_chains'
]


# ─────────────────────────────────────────────
# COLUMN DETECTION
# ─────────────────────────────────────────────
def find_column(df, possible_names):
    """Case-insensitive column name matcher."""
    columns = {c.lower(): c for c in df.columns}
    for name in possible_names:
        if name.lower() in columns:
            return columns[name.lower()]
    return None


def detect_columns(df):
    """
    Maps every required field to whatever column exists in the
    uploaded dataframe (or None if missing).
    """
    mapping = {}
    for field, keywords in COLUMN_KEYWORDS.items():
        mapping[field] = find_column(df, keywords)
    return mapping


def is_gitcoin_dataset(df):
    """
    Detects if the uploaded CSV is Gitcoin-specific by checking
    for Gitcoin's signature column names.
    """
    existing_cols = [c.lower() for c in df.columns]
    matches = sum(
        1 for sig in GITCOIN_SIGNATURE_COLUMNS
        if sig.lower() in existing_cols
    )
    # If at least 2 signature columns are present, treat as Gitcoin data
    return matches >= 2


# ─────────────────────────────────────────────
# GITCOIN API LOOKUP
# ─────────────────────────────────────────────
def fetch_gitcoin_description(grant_id):
    """
    Attempts to fetch a real project description from Gitcoin's API
    using the grant_id. Returns None if lookup fails.
    """
    try:
        url = f"{GITCOIN_API_BASE}/{grant_id}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('description', None)
    except Exception:
        pass
    return None


def lookup_descriptions_from_gitcoin(df, campaign_id_col):
    """
    Fetches descriptions for every row using the Gitcoin API.
    Prints progress and falls back to None on failure.
    """
    print("  Attempting Gitcoin API lookup for missing descriptions...",
          flush=True)

    descriptions = []
    success_count = 0

    for i, campaign_id in enumerate(df[campaign_id_col]):
        desc = fetch_gitcoin_description(campaign_id)
        descriptions.append(desc)
        if desc is not None:
            success_count += 1

        if (i + 1) % 50 == 0:
            print(f"    Looked up {i+1}/{len(df)} campaigns...", flush=True)

    print(f"  Gitcoin API lookup complete: {success_count}/{len(df)} "
          f"descriptions found.", flush=True)
    return descriptions


# ─────────────────────────────────────────────
# REALISTIC SIMULATION HELPERS
# ─────────────────────────────────────────────

DESCRIPTION_TEMPLATES = [
    "{name} is a decentralized platform built on {chain} that enables "
    "users to {action} through {mechanism}. Our smart contracts have "
    "been designed to {benefit}.",

    "{name} aims to solve {problem} in the Web3 ecosystem by providing "
    "a {adjective} solution for {audience}. Built on {chain}, we focus "
    "on {benefit}.",

    "We are building {name}, a {adjective} protocol for {audience} "
    "that leverages {mechanism} to {action}. The project is live on "
    "{chain} and plans to expand to more networks.",

    "{name} is a community-driven initiative that helps {audience} "
    "{action} using blockchain technology. Our {mechanism} ensures "
    "{benefit} for all participants.",
]

WORD_BANK = {
    'chain':      ['Ethereum', 'Polygon', 'Arbitrum', 'Optimism',
                   'Avalanche', 'BNB Chain', 'Fantom'],
    'action':     ['manage digital assets', 'access decentralized finance',
                   'participate in governance', 'trade NFTs securely',
                   'earn yield on deposits', 'fund public goods'],
    'mechanism':  ['smart contract automation', 'a DAO governance model',
                   'liquidity pooling', 'on-chain reputation scoring',
                   'a staking mechanism'],
    'benefit':    ['maximum transparency', 'lower transaction costs',
                   'improved security', 'faster settlement times',
                   'greater financial inclusion'],
    'problem':    ['liquidity fragmentation', 'lack of transparency',
                   'high transaction fees', 'limited financial access',
                   'inefficient governance'],
    'adjective':  ['innovative', 'community-driven', 'secure',
                   'scalable', 'user-friendly'],
    'audience':   ['small businesses', 'independent developers',
                   'content creators', 'DeFi users', 'NFT collectors'],
}

NAME_PREFIXES  = ['Chain', 'Block', 'Yield', 'Vault', 'Nova', 'Meta',
                  'Zen', 'Flux', 'Aero', 'Proto']
NAME_SUFFIXES  = ['Protocol', 'Labs', 'DAO', 'Finance', 'Network',
                  'Swap', 'Chain', 'Hub']


def generate_realistic_name():
    return f"{random.choice(NAME_PREFIXES)}{random.choice(NAME_SUFFIXES)}"


def generate_realistic_description(name=None):
    """Generates a plausible Web3-style campaign description."""
    if name is None:
        name = generate_realistic_name()

    template = random.choice(DESCRIPTION_TEMPLATES)
    filled = template.format(
        name=name,
        chain=random.choice(WORD_BANK['chain']),
        action=random.choice(WORD_BANK['action']),
        mechanism=random.choice(WORD_BANK['mechanism']),
        benefit=random.choice(WORD_BANK['benefit']),
        problem=random.choice(WORD_BANK['problem']),
        adjective=random.choice(WORD_BANK['adjective']),
        audience=random.choice(WORD_BANK['audience']),
    )
    return filled


def generate_realistic_timestamps(n, start_date=None, duration_days=30):
    """
    Generates timestamps clustered realistically —
    more activity early and at the end (typical crowdfunding pattern),
    fewer in the middle.
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=duration_days)

    # U-shaped distribution: more activity at start and end
    weights = np.concatenate([
        np.linspace(3, 1, n // 3),
        np.linspace(1, 1, n - 2 * (n // 3)),
        np.linspace(1, 3, n // 3),
    ])
    weights = weights / weights.sum()

    offsets_days = np.sort(
        np.random.choice(
            np.linspace(0, duration_days, 500), size=n, p=None
        ) if False else
        np.random.choice(duration_days * 24, size=n, replace=True,
                          p=None)
    )

    timestamps = [
        start_date + timedelta(hours=int(h)) for h in sorted(offsets_days)
    ]
    return timestamps


def generate_realistic_amounts(n, fraud_pattern=False):
    """
    Generates realistic contribution amounts.
    Most contributions are small, few are large (log-normal).
    If fraud_pattern=True, injects suspiciously uniform amounts
    (simulating collusive/bot behavior).
    """
    if fraud_pattern and random.random() < 0.3:
        # Simulate collusion: many identical/similar amounts
        base = random.choice([10, 25, 50, 100])
        amounts = [base + random.uniform(-1, 1) for _ in range(n)]
    else:
        # Realistic log-normal distribution of contributions
        amounts = np.random.lognormal(mean=3.0, sigma=1.2, size=n)
        amounts = np.clip(amounts, 1, 5000)
    return [round(float(a), 2) for a in amounts]


def generate_wallet_pool(size=NUM_SIMULATED_WALLETS):
    """Generates a pool of realistic-looking Ethereum wallet addresses."""
    wallets = []
    for _ in range(size):
        addr = "0x" + "".join(
            random.choices(string.hexdigits.lower(), k=40)
        )
        wallets.append(addr)
    return wallets


# ─────────────────────────────────────────────
# FULL SIMULATION (no CSV uploaded at all)
# ─────────────────────────────────────────────
def simulate_full_dataset():
    """
    Builds a complete synthetic dataset with one row per
    transaction, covering all 5 metrics' required fields.
    """
    print("\n  No dataset provided — simulating full dataset from scratch.",
          flush=True)

    wallets = generate_wallet_pool()
    all_rows = []

    for cid in range(1, NUM_SIMULATED_CAMPAIGNS + 1):
        campaign_id = f"CAMP_{cid:04d}"
        name        = generate_realistic_name()
        description = generate_realistic_description(name)
        approved    = random.random() < 0.6          # 60% approved
        team_size   = max(1, int(np.random.exponential(scale=3)))
        multichain  = "Yes" if random.random() < 0.3 else "No"

        duration_days  = random.choice([14, 21, 30, 45, 60])
        n_transactions = random.randint(*TRANSACTIONS_PER_CAMPAIGN)

        # Inject fraud pattern into ~20% of campaigns
        is_fraud_campaign = random.random() < 0.2

        timestamps = generate_realistic_timestamps(
            n_transactions, duration_days=duration_days
        )
        amounts = generate_realistic_amounts(
            n_transactions, fraud_pattern=is_fraud_campaign
        )

        # For collusion simulation, reuse a small wallet subset
        if is_fraud_campaign and random.random() < 0.5:
            campaign_wallets = random.sample(wallets, k=min(5, len(wallets)))
            wallet_choices = random.choices(campaign_wallets, k=n_transactions)
        else:
            wallet_choices = random.choices(wallets, k=n_transactions)

        cumulative = 0
        unique_users_seen = set()

        for i in range(n_transactions):
            amount    = amounts[i]
            timestamp = timestamps[i]
            wallet    = wallet_choices[i]

            previous_funding = cumulative
            cumulative += amount
            current_funding  = cumulative

            unique_users_seen.add(wallet)
            new_user = wallet not in list(unique_users_seen)[:-1]

            row = {
                'campaign_id':       campaign_id,
                'title':             name,
                'description':       description,
                'approved':          approved,
                'team_size':         team_size,
                'multichain':        multichain,
                'wallet_id':         wallet,
                'amount':            amount,
                'timestamp':         timestamp,
                'current_funding':   round(current_funding, 2),
                'previous_funding':  round(previous_funding, 2),
                'transactions_per_minute': round(
                    n_transactions / (duration_days * 24 * 60), 4
                ),
                'total_transactions': n_transactions,
                'new_users':          1 if new_user else 0,
                'total_users':        len(unique_users_seen),
                'campaign_duration_hours': duration_days * 24,
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    print(f"  Simulated {NUM_SIMULATED_CAMPAIGNS} campaigns, "
          f"{len(df)} total transactions.", flush=True)
    return df


# ─────────────────────────────────────────────
# PARTIAL SIMULATION (CSV uploaded, some columns missing)
# ─────────────────────────────────────────────
def needs_transaction_expansion(df, mapping):
    """
    Detects the case where the uploaded dataset has ONE ROW PER
    CAMPAIGN (like raw Gitcoin data) rather than one row per
    transaction — i.e. wallet_id, amount, and timestamp are ALL
    missing, AND campaign_id is unique on every row.

    In this case, simulating "one transaction per row" would give
    every campaign exactly 1 fake transaction, which breaks
    FVRS/CFIS/CTCS (they need multiple transactions per campaign
    to compute velocity, collusion, and temporal patterns).
    """
    txn_cols_missing = (
        mapping.get('wallet_id') is None and
        mapping.get('amount') is None and
        mapping.get('timestamp') is None
    )

    campaign_id_col = mapping.get('campaign_id')
    if campaign_id_col is not None:
        unique_ratio = df[campaign_id_col].nunique() / max(len(df), 1)
        # Real-world "one row per campaign" data is rarely perfectly
        # unique (duplicate scrape entries, re-submissions, etc.) —
        # treat >=90% unique as "one row per campaign" rather than
        # requiring an exact 100% match.
        looks_like_one_row_per_campaign = (unique_ratio >= 0.90)
    else:
        # No campaign_id column at all — every row is its own
        # campaign by default, which has the same problem
        looks_like_one_row_per_campaign = True

    return txn_cols_missing and looks_like_one_row_per_campaign


def expand_campaigns_into_transactions(df, mapping, is_gitcoin):
    """
    Takes a dataframe with ONE ROW PER CAMPAIGN and expands it
    into MULTIPLE ROWS PER CAMPAIGN (one per simulated transaction),
    while preserving all real per-campaign columns (title,
    description, approved, team_size, multichain, etc.) by
    broadcasting them onto every transaction row.

    This fixes the "total_transactions=1 for every campaign" bug
    that happens when transaction-level data doesn't exist in the
    source dataset (e.g. raw Gitcoin campaign metadata).
    """
    print("\n  [EXPANSION] Detected one-row-per-campaign dataset with "
          "no transaction-level columns.", flush=True)
    print("  [EXPANSION] Expanding each campaign into multiple simulated "
          "transactions so FVRS/CFIS/CTCS have real transaction "
          "sequences to work with...", flush=True)

    campaign_id_col = mapping.get('campaign_id')
    wallets = generate_wallet_pool()

    # De-duplicate by campaign_id first — if the raw dataset has
    # accidental duplicate rows for the same campaign, we don't
    # want them treated as two separate campaigns sharing one ID
    if campaign_id_col is not None and df[campaign_id_col].duplicated().any():
        dup_count = df[campaign_id_col].duplicated().sum()
        print(f"  [EXPANSION] Found {dup_count} duplicate campaign_id "
              f"rows — keeping first occurrence of each.", flush=True)
        df = df.drop_duplicates(subset=campaign_id_col).reset_index(drop=True)

    expanded_rows = []

    for _, campaign_row in df.iterrows():
        campaign_id = (
            campaign_row[campaign_id_col]
            if campaign_id_col is not None
            else f"CAMP_{len(expanded_rows):04d}"
        )

        n_transactions = random.randint(*TRANSACTIONS_PER_CAMPAIGN)
        duration_days  = random.choice([14, 21, 30, 45, 60])
        is_fraud_campaign = random.random() < 0.2

        timestamps = generate_realistic_timestamps(
            n_transactions, duration_days=duration_days
        )
        amounts = generate_realistic_amounts(
            n_transactions, fraud_pattern=is_fraud_campaign
        )

        if is_fraud_campaign and random.random() < 0.5:
            campaign_wallets = random.sample(wallets, k=min(5, len(wallets)))
            wallet_choices = random.choices(campaign_wallets, k=n_transactions)
        else:
            wallet_choices = random.choices(wallets, k=n_transactions)

        cumulative = 0
        unique_users_seen = set()

        for i in range(n_transactions):
            amount    = amounts[i]
            timestamp = timestamps[i]
            wallet    = wallet_choices[i]

            previous_funding = cumulative
            cumulative += amount
            current_funding  = cumulative

            new_user = wallet not in unique_users_seen
            unique_users_seen.add(wallet)

            # Start with all REAL campaign-level fields, broadcast
            # onto this transaction row
            row = campaign_row.to_dict()

            # Overwrite/add transaction-level fields
            row['campaign_id']       = campaign_id
            row['wallet_id']         = wallet
            row['amount']            = amount
            row['timestamp']         = timestamp
            row['current_funding']   = round(current_funding, 2)
            row['previous_funding']  = round(previous_funding, 2)
            row['transactions_per_minute'] = round(
                n_transactions / (duration_days * 24 * 60), 4
            )
            row['total_transactions'] = n_transactions
            row['new_users']          = 1 if new_user else 0
            row['total_users']        = len(unique_users_seen)
            row['campaign_duration_hours'] = duration_days * 24

            expanded_rows.append(row)

    expanded_df = pd.DataFrame(expanded_rows)
    print(f"  [EXPANSION] Expanded {len(df)} campaigns into "
          f"{len(expanded_df)} transaction rows "
          f"(avg {len(expanded_df)/len(df):.1f} transactions/campaign).",
          flush=True)
    return expanded_df


def fill_missing_columns(df, mapping, is_gitcoin):
    """
    Fills in missing columns based on detected mapping.
    Prints a warning for every simulated column.
    Handles 'description' with special Gitcoin/None logic.
    """
    n = len(df)
    standardized = pd.DataFrame(index=df.index)

    # ── Direct mapping for columns that DO exist ──
    for field, actual_col in mapping.items():
        if actual_col is not None:
            standardized[field] = df[actual_col]

    # ── Fill PARTIAL gaps in columns that exist but have some
    #    missing cells (common in real-world data — e.g. a
    #    campaign that never filled in team_size when applying) ──
    def fill_partial_gaps(field, fill_fn, description_msg):
        if field in standardized.columns:
            missing_mask = standardized[field].isna() | (
                standardized[field].astype(str).str.strip() == ''
            )
            missing_count = missing_mask.sum()
            if missing_count > 0:
                print(f"  [SIMULATED] '{field}' column exists but has "
                      f"{missing_count} missing values — {description_msg}",
                      flush=True)
                standardized.loc[missing_mask, field] = fill_fn(missing_count)

    fill_partial_gaps(
        'approved',
        lambda n: np.random.choice([True, False], size=n, p=[0.6, 0.4]),
        "filling gaps with realistic 60/40 approval ratio."
    )
    fill_partial_gaps(
        'team_size',
        lambda n: np.clip(
            np.random.exponential(scale=3, size=n).astype(int) + 1, 1, 20
        ),
        "filling gaps with realistic exponential distribution."
    )
    fill_partial_gaps(
        'multichain',
        lambda n: np.random.choice(['Yes', 'No'], size=n, p=[0.3, 0.7]),
        "filling gaps with realistic 30/70 ratio."
    )

    # ── campaign_id must exist or be generated ──
    if 'campaign_id' not in standardized.columns:
        print("  [SIMULATED] 'campaign_id' column missing — "
              "generating sequential IDs.", flush=True)
        standardized['campaign_id'] = [f"CAMP_{i:04d}" for i in range(n)]

    # ── description: special handling ──
    if 'description' not in standardized.columns:
        if is_gitcoin:
            print("  Dataset detected as Gitcoin-specific. "
                  "Attempting API lookup for descriptions...", flush=True)
            descriptions = lookup_descriptions_from_gitcoin(
                standardized, 'campaign_id'
            )
            standardized['description'] = descriptions
        else:
            print("  [SKIPPED] 'description' column missing and dataset "
                  "is not Gitcoin-specific — SDI will be skipped for "
                  "these rows (description set to None).", flush=True)
            standardized['description'] = None

    # ── title ──
    if 'title' not in standardized.columns:
        print("  [SIMULATED] 'title' column missing — "
              "generating placeholder titles.", flush=True)
        standardized['title'] = [
            generate_realistic_name() for _ in range(n)
        ]

    # ── approved ──
    if 'approved' not in standardized.columns:
        print("  [SIMULATED] 'approved' column missing — "
              "simulating with realistic 60/40 approval ratio.", flush=True)
        standardized['approved'] = np.random.choice(
            [True, False], size=n, p=[0.6, 0.4]
        )

    # ── team_size ──
    if 'team_size' not in standardized.columns:
        print("  [SIMULATED] 'team_size' column missing — "
              "simulating with realistic exponential distribution.",
              flush=True)
        standardized['team_size'] = np.clip(
            np.random.exponential(scale=3, size=n).astype(int) + 1,
            1, 20
        )

    # ── multichain ──
    if 'multichain' not in standardized.columns:
        print("  [SIMULATED] 'multichain' column missing — "
              "simulating with realistic 30/70 ratio.", flush=True)
        standardized['multichain'] = np.random.choice(
            ['Yes', 'No'], size=n, p=[0.3, 0.7]
        )

    # ── wallet_id ──
    if 'wallet_id' not in standardized.columns:
        print("  [SIMULATED] 'wallet_id' column missing — "
              "generating realistic Ethereum-style addresses.", flush=True)
        wallets = generate_wallet_pool()
        standardized['wallet_id'] = np.random.choice(wallets, size=n)

    # ── amount ──
    if 'amount' not in standardized.columns:
        print("  [SIMULATED] 'amount' column missing — "
              "simulating with log-normal contribution distribution.",
              flush=True)
        standardized['amount'] = generate_realistic_amounts(n)

    # ── timestamp ──
    if 'timestamp' not in standardized.columns:
        print("  [SIMULATED] 'timestamp' column missing — "
              "simulating realistic timestamp distribution.", flush=True)
        standardized['timestamp'] = generate_realistic_timestamps(n)

    # ── current_funding / previous_funding ──
    if 'current_funding' not in standardized.columns:
        print("  [SIMULATED] 'current_funding' column missing — "
              "deriving from cumulative amount per campaign.", flush=True)
        standardized['current_funding'] = (
            standardized.groupby('campaign_id')['amount'].cumsum()
            if 'amount' in standardized.columns else 0
        )

    if 'previous_funding' not in standardized.columns:
        print("  [SIMULATED] 'previous_funding' column missing — "
              "deriving as current_funding shifted by one transaction.",
              flush=True)
        standardized['previous_funding'] = (
            standardized.groupby('campaign_id')['current_funding']
            .shift(1).fillna(0)
        )

    # ── transactions_per_minute ──
    if 'transactions_per_minute' not in standardized.columns:
        print("  [SIMULATED] 'transactions_per_minute' column missing — "
              "simulating realistic low-frequency rate.", flush=True)
        standardized['transactions_per_minute'] = np.random.uniform(
            0.001, 0.05, size=n
        )

    # ── total_transactions ──
    if 'total_transactions' not in standardized.columns:
        print("  [SIMULATED] 'total_transactions' column missing — "
              "deriving from per-campaign transaction counts.", flush=True)
        standardized['total_transactions'] = (
            standardized.groupby('campaign_id')['campaign_id']
            .transform('count')
        )

    # ── new_users / total_users ──
    if 'new_users' not in standardized.columns:
        print("  [SIMULATED] 'new_users' column missing — "
              "simulating with realistic binary flag.", flush=True)
        standardized['new_users'] = np.random.choice(
            [0, 1], size=n, p=[0.7, 0.3]
        )

    if 'total_users' not in standardized.columns:
        print("  [SIMULATED] 'total_users' column missing — "
              "deriving from cumulative new_users per campaign.", flush=True)
        standardized['total_users'] = (
            standardized.groupby('campaign_id')['new_users'].cumsum()
        )

    # ── campaign_duration_hours ──
    if 'campaign_duration_hours' not in standardized.columns:
        print("  [SIMULATED] 'campaign_duration_hours' column missing — "
              "simulating realistic campaign durations.", flush=True)
        unique_campaigns = standardized['campaign_id'].unique()
        duration_map = {
            cid: random.choice([14, 21, 30, 45, 60]) * 24
            for cid in unique_campaigns
        }
        standardized['campaign_duration_hours'] = (
            standardized['campaign_id'].map(duration_map)
        )

    return standardized


# ─────────────────────────────────────────────
# MAIN VALIDATION + SIMULATION ENTRY POINT
# ─────────────────────────────────────────────
def enforce_column_dtypes(df):
    """
    Forces every column to ONE consistent dtype before saving.
    Prevents the "mixed types" pandas warning downstream in
    FVRS/CFIS/CTCS, which happens when a column has a blend of
    int/float/str/NaN values (e.g. team_size = 5, 5.0, '5', NaN
    all mixed together after partial-gap filling).
    """
    print("\n  Normalizing column data types...", flush=True)

    INTEGER_COLUMNS = [
        'team_size', 'total_transactions', 'new_users', 'total_users',
        'campaign_duration_hours'
    ]
    FLOAT_COLUMNS = [
        'amount', 'current_funding', 'previous_funding',
        'transactions_per_minute'
    ]
    STRING_COLUMNS = [
        'campaign_id', 'title', 'description', 'multichain', 'wallet_id'
    ]
    BOOL_COLUMNS = ['approved']

    for col in INTEGER_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')

    for col in FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype('float64')

    for col in STRING_COLUMNS:
        if col in df.columns:
            # Description is allowed to stay NaN (SDI intentionally
            # skips rows with no description) — everything else
            # gets coerced to string.
            if col == 'description':
                df[col] = df[col].where(df[col].isna(), df[col].astype(str))
            else:
                df[col] = df[col].astype(str)

    for col in BOOL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: True if str(x).strip().lower() == 'true' else False
            )

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    print("    All columns normalized to consistent types.", flush=True)
    return df


def prepare_dataset(input_path=None):
    """
    Main entry point. Returns a fully standardized dataframe
    ready to be consumed by all 5 metric pipelines.
    """
    print("=" * 60, flush=True)
    print("  DATASET VALIDATOR + SIMULATOR", flush=True)
    print("=" * 60, flush=True)

    if input_path is None or not os.path.exists(input_path):
        print("\n  No valid CSV path provided.", flush=True)
        df = simulate_full_dataset()

    else:
        print(f"\n  Loading uploaded dataset: {input_path}", flush=True)
        raw_df = pd.read_csv(input_path)
        print(f"  Loaded {len(raw_df)} rows, {len(raw_df.columns)} columns.",
              flush=True)

        print("\n  Auto-detecting columns...", flush=True)
        mapping = detect_columns(raw_df)

        detected = {k: v for k, v in mapping.items() if v is not None}
        missing  = [k for k, v in mapping.items() if v is None]

        print(f"  Detected {len(detected)} columns: {list(detected.keys())}",
              flush=True)
        print(f"  Missing {len(missing)} columns: {missing}", flush=True)

        gitcoin_flag = is_gitcoin_dataset(raw_df)
        print(f"\n  Gitcoin-specific dataset: {gitcoin_flag}", flush=True)

        # Check if this is one-row-per-campaign data with no
        # transaction history at all (e.g. raw Gitcoin metadata) —
        # if so, expand into multiple simulated transactions per
        # campaign BEFORE filling other missing columns, otherwise
        # every campaign ends up with exactly 1 fake transaction.
        if needs_transaction_expansion(raw_df, mapping):
            raw_df = expand_campaigns_into_transactions(
                raw_df, mapping, gitcoin_flag
            )
            # Re-detect columns since the dataframe now has new/
            # overwritten transaction-level columns
            mapping = detect_columns(raw_df)

        print("\n  Filling missing columns...", flush=True)
        df = fill_missing_columns(raw_df, mapping, gitcoin_flag)

    print(f"\n  Final unified dataset: {len(df)} rows, "
          f"{len(df.columns)} columns.", flush=True)

    df = enforce_column_dtypes(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved to: {OUTPUT_PATH}", flush=True)

    return df


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate and simulate dataset for trust metrics."
    )
    parser.add_argument(
        '--input', type=str, default=None,
        help="Path to uploaded CSV file. Omit to fully simulate."
    )
    args = parser.parse_args()

    final_df = prepare_dataset(args.input)

    print("\n" + "=" * 60, flush=True)
    print("  SAMPLE OF FINAL DATASET", flush=True)
    print("=" * 60, flush=True)
    print(final_df.head(5).to_string(), flush=True)