"""
TrustNet Inference Pipeline
----------------------------
Recreates the EXACT preprocessing used during training (trustnet_transformer.py)
and exposes a clean predict() function for use by the Flask API.

Training preprocessing summary (from trustnet_transformer.py):
  - Load trustnet_dataset.csv after merging all 5 feature pipelines.
  - Feature columns = every column except campaign_id, fraud_label,
    combined_risk_score  (33 columns including noise columns like
    title_x, trust_label, top_match_1, etc.).
  - All feature columns are cast via pd.to_numeric(errors='coerce').fillna(0.0)
    which turns string / bool columns into floats:
      - String columns (title_x, title_y, trust_label, risk_label,
        top_match_1, top_match_2) → 0.0
      - Bool columns (below_min_wallets, below_min_timestamps) → 0.0 or 1.0
  - No StandardScaler, no LabelEncoder, no MinMaxScaler anywhere.
  - Converted to float32 tensor.
  - Model outputs raw logit; sigmoid applied in predict_proba().
  - Fraud threshold = 0.5 (DECISION_THRESHOLD from training config).

The inference pipeline must reproduce these exact steps:
  1. Build a dict of all 33 feature values (missing ones default to 0.0)
  2. Cast to float (strings → 0.0, bools → 0.0/1.0)
  3. Assemble into a float32 tensor in the exact feature_names order
  4. Run model.predict_proba(tensor) → fraud probability
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PATHS — resolve relative to this file's dir
# ─────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
MODEL_PATH  = os.path.join(PROJECT_DIR, "outputs", "trustnet_transformer.pt")

# ─────────────────────────────────────────────
# FRAUD DECISION THRESHOLD
# ─────────────────────────────────────────────
FRAUD_THRESHOLD = 0.5   # same as DECISION_THRESHOLD in trustnet_transformer.py


# ─────────────────────────────────────────────
# MODEL ARCHITECTURE (identical to training)
# ─────────────────────────────────────────────
class TrustNetTransformer(nn.Module):
    """
    Exact same architecture as trustnet/trustnet_transformer.py.
    Each scalar feature → Linear(1, d_model) token + positional embedding
    → TransformerEncoder → mean-pool → classifier head → raw logit.
    Sigmoid is applied at inference time via predict_proba().
    """
    def __init__(self, num_features: int, d_model: int = 32,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        self.input_projection = nn.Linear(1, d_model)
        self.positional_embedding = nn.Parameter(
            torch.randn(1, num_features, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        tokens = self.input_projection(x)
        tokens = tokens + self.positional_embedding
        attended = self.transformer_encoder(tokens)
        pooled = attended.mean(dim=1)
        return self.classifier(pooled)   # raw logit

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


# ─────────────────────────────────────────────
# MODEL LOADER (singleton)
# ─────────────────────────────────────────────
_model: Optional[TrustNetTransformer] = None
_feature_names: Optional[list] = None
_device: Optional[torch.device] = None


def load_model() -> tuple:
    """
    Loads and caches the TrustNet model + feature_names from checkpoint.
    Idempotent — safe to call multiple times (returns cached instance).
    """
    global _model, _feature_names, _device

    if _model is not None:
        return _model, _feature_names, _device

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at: {MODEL_PATH}\n"
            f"Run trustnet/trustnet_transformer.py first to train the model."
        )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=_device)

    _feature_names = checkpoint["feature_names"]

    _model = TrustNetTransformer(
        num_features=len(_feature_names),
        d_model=checkpoint["d_model"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
    ).to(_device)

    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()

    logger.info(
        "TrustNet model loaded: %d features, d_model=%d, device=%s",
        len(_feature_names), checkpoint["d_model"], _device
    )
    return _model, _feature_names, _device


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def _safe_float(value: Any) -> float:
    """
    Replicates pd.to_numeric(errors='coerce').fillna(0.0) on a single value.
    - Numeric types → float
    - Booleans  → 0.0 / 1.0
    - Strings   → 0.0  (non-numeric strings cannot be parsed → NaN → 0.0)
    - None/NaN  → 0.0
    """
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    try:
        v = float(value)
        return v if np.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def build_feature_vector(feature_dict: Dict[str, Any]) -> torch.Tensor:
    """
    Converts a dict of raw feature values into the float32 tensor
    expected by TrustNet.

    The dict keys must match the column names in trustnet_dataset.csv.
    Missing keys are silently defaulted to 0.0 (same as fillna(0.0)
    during training). Extra keys are ignored.

    Returns: float32 tensor of shape (1, num_features)
    """
    model, feature_names, device = load_model()

    row = [_safe_float(feature_dict.get(name, 0.0)) for name in feature_names]
    tensor = torch.tensor(row, dtype=torch.float32).unsqueeze(0).to(device)
    return tensor


# ─────────────────────────────────────────────
# TRUST SCORE CONVERSION
# ─────────────────────────────────────────────
def fraud_prob_to_trust_scores(
    fraud_prob: float,
    sdi: float = 0.0,
    cti: float = 0.5,
    fvrs: float = 0.0,
    cfis: float = 0.0,
    ctcs: float = 0.0,
) -> Dict[str, int]:
    """
    Converts [0,1] metric scores to on-chain integer scores (0-100)
    for TrustMetricsRegistry.updateAllScores().

    The Transformer's fraud_prob is a supplementary signal — the
    individual metric scores are what the contract stores and uses
    for getCombinedRisk(). Both are returned so the caller can decide
    what to push to the blockchain.

    Integer clipping ensures all values stay within [0, 100]
    as required by the contract's require() statements.
    """
    def to_int(x: float) -> int:
        return int(np.clip(round(x * 100), 0, 100))

    return {
        "sdi":  to_int(sdi),
        "cti":  to_int(cti),
        "fvrs": to_int(fvrs),
        "cfis": to_int(cfis),
        "ctcs": to_int(ctcs),
        "fraud_prob_score": to_int(fraud_prob),   # extra for dashboard
    }


# ─────────────────────────────────────────────
# MAIN INFERENCE ENTRY POINT
# ─────────────────────────────────────────────
def predict(feature_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full inference pipeline:
      feature_dict  (raw feature values, any keys present)
          ↓
      build_feature_vector  (33-element float32 tensor, missing=0.0)
          ↓
      TrustNetTransformer.predict_proba  (sigmoid of raw logit)
          ↓
      returns: {
          "fraud_probability": float in [0, 1],
          "is_fraud":          bool  (prob >= FRAUD_THRESHOLD),
          "fraud_threshold":   float,
          "trust_scores":      dict  (on-chain integer values 0-100),
          "feature_vector":    list  (for logging / debugging),
      }
    """
    model, feature_names, device = load_model()

    tensor = build_feature_vector(feature_dict)

    fraud_prob = float(model.predict_proba(tensor).item())
    is_fraud   = fraud_prob >= FRAUD_THRESHOLD

    # Extract metric scores for the blockchain call
    sdi  = _safe_float(feature_dict.get("sdi_score", feature_dict.get("sdi", 0.0)))
    cti  = _safe_float(feature_dict.get("cti_norm",  feature_dict.get("cti", 0.5)))
    fvrs = _safe_float(feature_dict.get("fvrs",      0.0))   # composite of sub-features
    cfis = _safe_float(feature_dict.get("cfis",      0.0))   # composite of sub-features
    ctcs = _safe_float(feature_dict.get("ctcs",      0.0))

    # If sub-features are present but no composite, compute averages
    if fvrs == 0.0:
        sub_fvrs = [feature_dict.get(k, 0.0) for k in ["fgr", "tbr", "nur", "ftr", "rar"]]
        if any(v != 0.0 for v in sub_fvrs):
            fvrs = float(np.mean([_safe_float(v) for v in sub_fvrs]))
    if cfis == 0.0:
        sub_cfis = [feature_dict.get(k, 0.0) for k in ["wgd", "cds", "col", "inf", "fts", "fas"]]
        if any(v != 0.0 for v in sub_cfis):
            cfis = float(np.mean([_safe_float(v) for v in sub_cfis]))

    trust_scores = fraud_prob_to_trust_scores(fraud_prob, sdi, cti, fvrs, cfis, ctcs)

    feature_vector = [_safe_float(feature_dict.get(name, 0.0)) for name in feature_names]

    return {
        "fraud_probability": round(fraud_prob, 6),
        "is_fraud":          bool(is_fraud),
        "fraud_threshold":   FRAUD_THRESHOLD,
        "trust_scores":      trust_scores,
        "feature_vector":    feature_vector,
        "feature_names":     feature_names,
    }


# ─────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Build a representative feature dict that mimics what the dataset builder
    # produces after all 5 feature pipelines have run on one campaign.
    sample_features = {
        # FVRS sub-features
        "fgr": 0.091,  "tbr": 0.15,  "nur": 0.08,  "ftr": 0.59,  "rar": 0.0,
        # CTI
        "title_x": "TestCampaign",  # string → becomes 0.0 in model
        "A": 0.0, "T": 0.0, "M": 0.0,
        "cti_score": 0.0, "cti_norm": 0.0,
        "trust_label": "LOW TRUST",   # string → 0.0
        # CFIS
        "n_wallets": 36, "wgd": 0.59, "cds": 0.80,
        "col": 0.70, "inf": 0.99, "fts": 0.0, "fas": 0.39,
        "below_min_wallets": False,   # bool → 0.0
        # CTCS
        "n_transactions": 44, "ctcc": 0.46, "ts": 0.003, "ctcs": 0.28,
        "below_min_timestamps": False,
        # SDI
        "title_y": "TestCampaign",    # string → 0.0
        "sdi_score": 0.71,
        "risk_label": "HIGH PLAGIARISM RISK",  # string → 0.0
        "top_match_1": "ZenLabs",     # string → 0.0
        "top_match_1_sim": 0.71,
        "top_match_2": "ZenHub",      # string → 0.0
        "top_match_2_sim": 0.65,
        # Combined risk (used by model despite the naming comment)
        "combined_risk_score": 0.57,
    }

    result = predict(sample_features)

    print("\n=== TrustNet Inference Result ===")
    print(f"  Fraud Probability  : {result['fraud_probability']:.4f}")
    print(f"  Is Fraud (>= {result['fraud_threshold']}) : {result['is_fraud']}")
    print(f"  On-chain scores    : {result['trust_scores']}")
    print(f"  Feature count      : {len(result['feature_names'])}")
