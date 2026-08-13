"""
TrustNet End-to-End Pipeline (Task 8)
--------------------------------------
Runs the complete workflow for a single campaign:

  Campaign data
      ↓
  Feature extraction (FVRS, CTI, CFIS, CTCS, SDI)
      ↓
  Preprocessing (identical to training)
      ↓
  TrustNet Transformer
      ↓
  Fraud probability
      ↓
  Blockchain update (updateAllScores)
      ↓
  Freeze if fraud_prob >= threshold
      ↓
  Return result

This script can be called:
  - Directly:   python pipeline.py
  - As a module: from pipeline import run_pipeline
  - By app.py:  imported and called per request

Usage:
    python pipeline.py --campaign_address 0xAbCd... [--dry_run]
"""

import os
import sys
import json
import logging
import argparse
from typing import Any, Dict, Optional

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, PROJECT_DIR)

from web3 import Web3
from config import (
    TRUST_METRICS_REGISTRY, SECURITY_MODULE, CROWDFUNDING_CORE,
    PRIVATE_KEY, WALLET_ADDRESS,
)
from blockchain import w3, registry, security
from predictor import predict, FRAUD_THRESHOLD

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(
    campaign_address: str,
    feature_dict: Dict[str, Any],
    campaign_id: str = "unknown",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Full end-to-end pipeline.

    Parameters
    ----------
    campaign_address : str
        On-chain address of the campaign creator.
    feature_dict : dict
        Pre-computed feature values. Any key from the 33-feature set is accepted;
        missing keys default to 0.0 (matching training fill logic).
    campaign_id : str
        Human-readable identifier for logging.
    dry_run : bool
        If True, runs inference but does NOT send any blockchain transactions.

    Returns
    -------
    dict with keys:
        campaign_id, campaign_address, fraud_probability, is_fraud,
        trust_scores, blockchain_update, freeze_decision,
        tx_update_hash, tx_freeze_hash
    """
    logger.info("=" * 60)
    logger.info("PIPELINE START | campaign=%s | address=%s", campaign_id, campaign_address)

    addr = Web3.to_checksum_address(campaign_address)

    # ── Step 1: Inference ──────────────────────────────────────────────
    logger.info("[1/4] Running TrustNet inference...")
    result     = predict(feature_dict)
    fraud_prob = result["fraud_probability"]
    is_fraud   = result["is_fraud"]
    ts         = result["trust_scores"]

    logger.info(
        "  → fraud_probability=%.4f | is_fraud=%s | threshold=%.2f",
        fraud_prob, is_fraud, FRAUD_THRESHOLD,
    )
    logger.info(
        "  → on-chain scores: SDI=%d CTI=%d FVRS=%d CFIS=%d CTCS=%d",
        ts["sdi"], ts["cti"], ts["fvrs"], ts["cfis"], ts["ctcs"],
    )

    tx_update_hash = None
    tx_freeze_hash = None
    update_receipt = None
    freeze_receipt = None

    # ── Step 2: Update blockchain scores ──────────────────────────────
    logger.info("[2/4] Updating TrustMetricsRegistry on Sepolia...")
    if dry_run:
        logger.info("  → DRY RUN — skipping blockchain transaction")
    else:
        try:
            update_receipt = _build_and_send(
                registry.functions.updateAllScores(
                    addr,
                    ts["sdi"], ts["cti"],
                    ts["fvrs"], ts["cfis"], ts["ctcs"],
                )
            )
            tx_update_hash = update_receipt["tx_hash"]
            logger.info(
                "  → updateAllScores tx: %s | status=%s | block=%s",
                tx_update_hash, update_receipt["status"], update_receipt["block"],
            )
            _verify_receipt(update_receipt, "updateAllScores")
        except Exception as e:
            logger.error("  ✗ updateAllScores failed: %s", e)
            update_receipt = {"error": str(e)}

    # ── Step 3: Freeze if fraud ────────────────────────────────────────
    logger.info("[3/4] Evaluating freeze decision...")
    freeze_decision = False

    if is_fraud:
        logger.info(
            "  → Fraud probability %.4f >= threshold %.2f — FREEZING campaign",
            fraud_prob, FRAUD_THRESHOLD,
        )
        if dry_run:
            logger.info("  → DRY RUN — skipping freeze transaction")
            freeze_decision = True
        else:
            try:
                # Check if already frozen to avoid redundant tx
                already_frozen = security.functions.isFrozen(addr).call()
                if already_frozen:
                    logger.info("  → Campaign already frozen, skipping freeze tx")
                    freeze_decision = True
                else:
                    freeze_receipt = _build_and_send(
                        security.functions.freezeCampaign(addr),
                        gas=200_000,
                    )
                    tx_freeze_hash  = freeze_receipt["tx_hash"]
                    freeze_decision = freeze_receipt["status"] == 1
                    logger.info(
                        "  → freezeCampaign tx: %s | status=%s | block=%s",
                        tx_freeze_hash, freeze_receipt["status"], freeze_receipt["block"],
                    )
                    _verify_receipt(freeze_receipt, "freezeCampaign")
            except Exception as e:
                logger.error("  ✗ freezeCampaign failed: %s", e)
                freeze_receipt  = {"error": str(e)}
    else:
        logger.info(
            "  → Fraud probability %.4f < threshold %.2f — campaign ALLOWED",
            fraud_prob, FRAUD_THRESHOLD,
        )

    # ── Step 4: Log complete decision ─────────────────────────────────
    logger.info("[4/4] Logging prediction decision...")
    logger.info(
        "PREDICTION_LOG | campaign=%s | fraud_prob=%.6f"
        " | tx_update=%s | frozen=%s | tx_freeze=%s",
        campaign_id, fraud_prob,
        tx_update_hash or "not_submitted",
        freeze_decision,
        tx_freeze_hash or "not_submitted",
    )

    logger.info("PIPELINE END | campaign=%s", campaign_id)
    logger.info("=" * 60)

    return {
        "campaign_id":       campaign_id,
        "campaign_address":  addr,
        "fraud_probability": fraud_prob,
        "is_fraud":          is_fraud,
        "fraud_threshold":   FRAUD_THRESHOLD,
        "trust_scores":      ts,
        "blockchain_update": update_receipt,
        "freeze_decision":   freeze_decision,
        "tx_update_hash":    tx_update_hash,
        "tx_freeze_hash":    tx_freeze_hash,
    }


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _build_and_send(fn, gas: int = 300_000) -> Dict[str, Any]:
    nonce    = w3.eth.get_transaction_count(WALLET_ADDRESS)
    tx       = fn.build_transaction({
        "from": WALLET_ADDRESS, "nonce": nonce,
        "gas": gas, "gasPrice": w3.eth.gas_price, "chainId": 11155111,
    })
    signed   = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash  = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt  = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    return {
        "tx_hash":  tx_hash.hex(),
        "status":   receipt.status,
        "block":    receipt.blockNumber,
        "gas_used": receipt.gasUsed,
    }


def _verify_receipt(receipt: Dict[str, Any], fn_name: str) -> None:
    if receipt.get("status") != 1:
        raise RuntimeError(
            f"{fn_name} transaction FAILED (status=0). "
            f"tx_hash={receipt.get('tx_hash')}"
        )
    logger.info("  ✓ %s confirmed on block %s", fn_name, receipt.get("block"))


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="TrustNet end-to-end pipeline for a single campaign"
    )
    parser.add_argument(
        "--campaign_address",
        default="0xDCdC29B6A4A477d3beAE4618F93535886137796D",
        help="On-chain campaign creator address",
    )
    parser.add_argument(
        "--campaign_id",
        default="CLI_TEST",
        help="Human-readable campaign ID for logging",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run inference but skip blockchain transactions",
    )
    args = parser.parse_args()

    # Example feature dict — replace with real feature extraction output
    sample_features = {
        "fgr": 0.097, "tbr": 0.156, "nur": 0.083, "ftr": 0.591, "rar": 0.0,
        "title_x": "TestCampaign",
        "A": 0.0, "T": 0.0, "M": 0.0, "cti_score": 0.0, "cti_norm": 0.0,
        "trust_label": "LOW TRUST",
        "n_wallets": 36, "wgd": 0.59, "cds": 0.80, "col": 0.70,
        "inf": 0.99, "fts": 0.0, "fas": 0.39, "below_min_wallets": False,
        "n_transactions": 44, "ctcc": 0.46, "ts": 0.003, "ctcs": 0.28,
        "below_min_timestamps": False,
        "title_y": "TestCampaign",
        "sdi_score": 0.71, "risk_label": "HIGH PLAGIARISM RISK",
        "top_match_1": "ZenLabs", "top_match_1_sim": 0.71,
        "top_match_2": "ZenHub",  "top_match_2_sim": 0.65,
        "combined_risk_score": 0.57,
    }

    result = run_pipeline(
        campaign_address = args.campaign_address,
        feature_dict     = sample_features,
        campaign_id      = args.campaign_id,
        dry_run          = args.dry_run,
    )

    print("\n=== Pipeline Result ===")
    print(json.dumps(result, indent=2, default=str))
