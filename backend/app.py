"""
TrustNet Flask API
-------------------
Endpoints:
  GET  /campaigns              List all on-chain campaigns
  GET  /campaign/<id>          Get single campaign details + trust scores
  POST /predict                Run TrustNet inference on feature input
  POST /update-blockchain      Push trust scores to TrustMetricsRegistry
  POST /freeze                 Freeze a campaign via SecurityModule
  POST /release                Attempt to release campaign funds
  GET  /metrics/<campaign>     Get on-chain trust scores for an address
  GET  /health                 Health check + model status
  POST /analyse-csv            Upload unified_dataset.csv â†’ risk for every campaign
  GET  /analyse-csv/<id>       Get full detail for one campaign from last CSV analysis
"""

import os
import sys
import io
import json
import logging
import traceback
import tempfile
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from flask import Flask, jsonify, request, Response
from flask_cors import CORS

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import (
    TRUST_METRICS_REGISTRY, SECURITY_MODULE, CROWDFUNDING_CORE,
    PRIVATE_KEY, WALLET_ADDRESS, RPC_URL,
)
from blockchain import w3, registry, security
from predictor import predict, load_model, FRAUD_THRESHOLD

# CSV analysis in-memory cache (populated when job completes)
_last_csv_results: Dict[str, Any] = {}

#feature extractor paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FVRS_DIR  = os.path.join(PROJECT_DIR, "fvrs")
CTI_DIR   = os.path.join(PROJECT_DIR, "cti")
CFIS_DIR  = os.path.join(PROJECT_DIR, "cfis")
CTCS_DIR  = os.path.join(PROJECT_DIR, "ctcs")
SDI_DIR   = os.path.join(PROJECT_DIR, "sdi")
for _d in [FVRS_DIR, CTI_DIR, CFIS_DIR, CTCS_DIR, SDI_DIR]:
    sys.path.insert(0, _d)

from fvrs_updated  import FVRSExtractor, FVRS
from cti           import compute_all_cti           # function, not class
from cfis_updated  import CFISExtractor, CFIS, build_cofunding_graph, detect_communities, compute_pagerank
from ctcs_updated  import CTCSExtractor

# In-memory store for the most-recent CSV analysis result
# { campaign_id: { all feature + prediction fields } }
_csv_analysis_cache: Dict[str, Any] = {}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# FLASK APP
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# LOGGING SETUP
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
LOGS_DIR = os.path.join(BACKEND_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

_file_handler = RotatingFileHandler(
    os.path.join(LOGS_DIR, "predictions.log"),
    maxBytes=5 * 1024 * 1024,   # 5 MB per file
    backupCount=5,
)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
))

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        _file_handler,
    ],
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# BLOCKCHAIN HELPERS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from web3 import Web3


def _build_and_send(fn, gas: int = 300_000) -> Dict[str, Any]:
    """
    Builds, signs and sends a transaction with retry logic for
    rate-limited RPC endpoints (Alchemy free tier).
    """
    import time

    # Use 1.5x current gas price to ensure the tx gets picked up
    base_gas_price = w3.eth.gas_price
    gas_price      = int(base_gas_price * 1.5)

    nonce = w3.eth.get_transaction_count(WALLET_ADDRESS, "pending")
    tx = fn.build_transaction({
        "from":     WALLET_ADDRESS,
        "nonce":    nonce,
        "gas":      gas,
        "gasPrice": gas_price,
        "chainId":  11155111,
    })
    signed  = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)

    logger.info("TX sent: %s | gas_price: %s gwei | nonce: %d",
                tx_hash.hex(), float(Web3.from_wei(gas_price, 'gwei')), nonce)

    # Wait up to 3 minutes, polling every 4 seconds
    for attempt in range(45):
        time.sleep(4)
        try:
            receipt = w3.eth.get_transaction_receipt(tx_hash)
            if receipt is not None:
                return {
                    "tx_hash":  tx_hash.hex(),
                    "status":   receipt.status,
                    "block":    receipt.blockNumber,
                    "gas_used": receipt.gasUsed,
                }
        except Exception:
            pass  # Receipt not available yet, keep polling

    raise TimeoutError(
        f"Transaction {tx_hash.hex()} not confirmed after 3 minutes. "
        f"Check https://sepolia.etherscan.io/tx/{tx_hash.hex()}"
    )


def _get_crowdfunding():
    """Lazy-load CrowdfundingCore contract (ABI must exist in backend/abi/)."""
    abi_path = os.path.join(BACKEND_DIR, "abi", "CrowdfundingCore.json")
    if not os.path.exists(abi_path):
        raise FileNotFoundError(f"CrowdfundingCore ABI not found: {abi_path}")
    with open(abi_path) as f:
        artifact = json.load(f)
    abi = artifact["abi"] if "abi" in artifact else artifact
    return w3.eth.contract(
        address=Web3.to_checksum_address(CROWDFUNDING_CORE),
        abi=abi,
    )


def _log_prediction(campaign_id: str, fraud_prob: float,
                    tx_hash: Optional[str], freeze_decision: bool) -> None:
    """
    Task 9: Every prediction is logged with campaign ID, fraud probability,
    blockchain transaction hash, and freeze decision.
    """
    logger.info(
        "PREDICTION | campaign=%s | fraud_prob=%.6f | tx_hash=%s | frozen=%s",
        campaign_id, fraud_prob,
        tx_hash if tx_hash else "not_submitted",
        freeze_decision,
    )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ROUTES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.route("/health", methods=["GET"])
def health():
    """Health check â€” confirms blockchain connection and model load status."""
    connected   = w3.is_connected()
    model_ok    = False
    feature_cnt = 0
    try:
        _, feature_names, _ = load_model()
        model_ok    = True
        feature_cnt = len(feature_names)
    except Exception as e:
        logger.warning("Model not loaded: %s", e)

    return jsonify({
        "status":        "ok" if (connected and model_ok) else "degraded",
        "blockchain":    "connected" if connected else "disconnected",
        "network":       "sepolia",
        "model":         "loaded" if model_ok else "error",
        "feature_count": feature_cnt,
        "fraud_threshold": FRAUD_THRESHOLD,
        "owner_address": WALLET_ADDRESS,
        "contracts": {
            "TrustMetricsRegistry": TRUST_METRICS_REGISTRY,
            "SecurityModule":       SECURITY_MODULE,
            "CrowdfundingCore":     CROWDFUNDING_CORE,
        },
    })


@app.route("/campaigns", methods=["GET"])
def get_campaigns():
    """
    Returns all campaigns from CrowdfundingCore, enriched with
    on-chain trust scores from TrustMetricsRegistry.
    """
    try:
        crowdfunding = _get_crowdfunding()
        count = crowdfunding.functions.campaignCount().call()

        campaigns = []
        for i in range(count):
            try:
                (creator, title, description, goal, deadline,
                 amount_raised, contributor_count,
                 goal_reached, funds_released) = crowdfunding.functions.getCampaign(i).call()

                # Fetch trust scores for this campaign's creator address
                scores = _fetch_trust_scores(creator)

                campaigns.append({
                    "id":               i,
                    "creator":          creator,
                    "title":            title,
                    "description":      description[:200] + ("..." if len(description) > 200 else ""),
                    "goal_wei":         goal,
                    "goal_eth":         float(Web3.from_wei(goal, "ether")),
                    "deadline":         deadline,
                    "amount_raised_wei": amount_raised,
                    "amount_raised_eth": float(Web3.from_wei(amount_raised, "ether")),
                    "contributor_count": contributor_count,
                    "goal_reached":     goal_reached,
                    "funds_released":   funds_released,
                    "is_frozen":        security.functions.isFrozen(creator).call(),
                    "is_flagged":       security.functions.isFlagged(creator).call(),
                    "trust_scores":     scores,
                })
            except Exception as e:
                logger.warning("Could not fetch campaign %d: %s", i, e)
                continue

        return jsonify({"campaigns": campaigns, "total": len(campaigns)})

    except Exception as e:
        logger.error("GET /campaigns error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/campaign/<int:campaign_id>", methods=["GET"])
def get_campaign(campaign_id: int):
    """Single campaign details with full trust metric breakdown."""
    try:
        crowdfunding = _get_crowdfunding()
        (creator, title, description, goal, deadline,
         amount_raised, contributor_count,
         goal_reached, funds_released) = crowdfunding.functions.getCampaign(campaign_id).call()

        scores   = _fetch_trust_scores(creator)
        combined = registry.functions.getCombinedRisk(creator).call()

        return jsonify({
            "id":               campaign_id,
            "creator":          creator,
            "title":            title,
            "description":      description,
            "goal_wei":         goal,
            "goal_eth":         float(Web3.from_wei(goal, "ether")),
            "deadline":         deadline,
            "amount_raised_wei": amount_raised,
            "amount_raised_eth": float(Web3.from_wei(amount_raised, "ether")),
            "contributor_count": contributor_count,
            "goal_reached":     goal_reached,
            "funds_released":   funds_released,
            "is_frozen":        security.functions.isFrozen(creator).call(),
            "is_flagged":       security.functions.isFlagged(creator).call(),
            "trust_scores":     scores,
            "combined_risk":    combined,
        })
    except Exception as e:
        logger.error("GET /campaign/%d error: %s", campaign_id, traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/metrics/<campaign_address>", methods=["GET"])
def get_metrics(campaign_address: str):
    """On-chain trust scores for a specific campaign (creator) address."""
    try:
        addr   = Web3.to_checksum_address(campaign_address)
        scores = _fetch_trust_scores(addr)
        return jsonify({
            "campaign": addr,
            "scores":   scores,
            "combined_risk": registry.functions.getCombinedRisk(addr).call(),
            "is_frozen":     security.functions.isFrozen(addr).call(),
            "is_flagged":    security.functions.isFlagged(addr).call(),
        })
    except Exception as e:
        logger.error("GET /metrics error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Run TrustNet inference.

    Request body (JSON):
    {
        "campaign_id": "CAMP_0001",           // for logging
        "features": {                         // any subset of the 33 feature columns
            "fgr": 0.097, "tbr": 0.156,       // FVRS sub-features
            "sdi_score": 0.71,                // SDI score
            "cti_norm": 0.06,                 // CTI normalized
            ...
        }
    }

    Alternatively, if the features are not pre-computed:
    {
        "campaign_id": "CAMP_0001",
        "title": "ZenSwap",
        "description": "...",
        "sdi_score": 0.71,
        ...
    }
    Any missing feature defaults to 0.0 (matching training fillna).
    """
    data        = request.get_json(force=True)
    campaign_id = data.get("campaign_id", "unknown")
    features    = data.get("features", data)   # allow flat dict too

    try:
        result = predict(features)

        _log_prediction(
            campaign_id      = campaign_id,
            fraud_prob       = result["fraud_probability"],
            tx_hash          = None,
            freeze_decision  = result["is_fraud"],
        )

        return jsonify({
            "campaign_id":     campaign_id,
            "fraud_probability": result["fraud_probability"],
            "is_fraud":        result["is_fraud"],
            "fraud_threshold": result["fraud_threshold"],
            "trust_scores":    result["trust_scores"],
        })

    except Exception as e:
        logger.error("POST /predict error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/update-blockchain", methods=["POST"])
def update_blockchain():
    """
    Push trust scores to TrustMetricsRegistry.updateAllScores().

    Request body (JSON):
    {
        "campaign_address": "0xAbCd...",   // on-chain campaign creator address
        "campaign_id": "CAMP_0001",        // for logging
        "features": { ... }                // same as /predict
    }

    Workflow:
      1. Run TrustNet inference
      2. Convert scores to on-chain integers (0-100)
      3. Call updateAllScores() on Sepolia
      4. Return tx_hash + fraud_probability
    """
    data             = request.get_json(force=True)
    campaign_address = data.get("campaign_address")
    campaign_id      = data.get("campaign_id", "unknown")
    features         = data.get("features", {})

    if not campaign_address:
        return jsonify({"error": "campaign_address is required"}), 400

    try:
        addr         = Web3.to_checksum_address(campaign_address)
        crowdfunding = _get_crowdfunding()

        # If campaign_id is a numeric on-chain ID, compute real features from chain.
        # This replaces whatever sparse form-input features the frontend sent.
        real_cid = None
        try:
            real_cid = int(campaign_id)
        except (TypeError, ValueError):
            pass

        if real_cid is not None:
            chain_features = _compute_features_from_chain(real_cid, crowdfunding)
            # Let any non-zero caller features override (manual form inputs)
            chain_features.update({k: v for k, v in features.items() if v != 0.0})
            features = chain_features

        result = predict(features)

        # Use composites from real extractor when available
        from predictor import fraud_prob_to_trust_scores as _fpts
        if "_sdi" in features:
            ts = _fpts(
                result["fraud_probability"],
                sdi  = features["_sdi"],
                cti  = features["_cti"],
                fvrs = features["_fvrs"],
                cfis = features["_cfis"],
                ctcs = features["_ctcs"],
            )
        else:
            ts = result["trust_scores"]

        logger.info(
            "UPDATE_BLOCKCHAIN | campaign=%s | "
            "SDI=%d CTI=%d FVRS=%d CFIS=%d CTCS=%d | fraud_prob=%.4f",
            addr,
            ts["sdi"], ts["cti"], ts["fvrs"], ts["cfis"], ts["ctcs"],
            result["fraud_probability"],
        )

        receipt_info = _build_and_send(
            registry.functions.updateAllScores(
                addr,
                ts["sdi"],
                ts["cti"],
                ts["fvrs"],
                ts["cfis"],
                ts["ctcs"],
            )
        )

        # Confirm on-chain state immediately after tx
        stored = _fetch_trust_scores(addr)
        logger.info(
            "UPDATE_BLOCKCHAIN | getScores AFTER tx | campaign=%s | "
            "SDI=%d CTI=%d FVRS=%d CFIS=%d CTCS=%d CombinedRisk=%d",
            addr,
            stored["sdi"], stored["cti"], stored["fvrs"],
            stored["cfis"], stored["ctcs"], stored["combined_risk"],
        )

        freeze_decision = result["is_fraud"] and receipt_info["status"] == 1

        _log_prediction(
            campaign_id     = campaign_id,
            fraud_prob      = result["fraud_probability"],
            tx_hash         = receipt_info["tx_hash"],
            freeze_decision = freeze_decision,
        )

        return jsonify({
            "campaign_id":       campaign_id,
            "campaign_address":  addr,
            "fraud_probability": result["fraud_probability"],
            "is_fraud":          result["is_fraud"],
            "trust_scores":      stored,   # confirmed on-chain values after tx
            "transaction":       receipt_info,
        })

    except Exception as e:
        logger.error("POST /update-blockchain error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/create-campaign", methods=["POST"])
def create_campaign():
    """
    Create a campaign on CrowdfundingCore via the oracle wallet.
    Request: { "title": "...", "description": "...", "goal_eth": 0.1, "duration_days": 30 }
    The oracle wallet (PRIVATE_KEY in .env) signs and pays gas.
    Returns: { "campaign_id": N, "transaction": {...} }
    """
    data         = request.get_json(force=True)
    title        = data.get("title", "").strip()
    description  = data.get("description", "").strip()
    goal_eth     = data.get("goal_eth", 0)
    duration_days= int(data.get("duration_days", 30))

    if not title:
        return jsonify({"error": "title is required"}), 400
    if not description:
        return jsonify({"error": "description is required"}), 400
    if not goal_eth or float(goal_eth) <= 0:
        return jsonify({"error": "goal_eth must be > 0"}), 400
    if duration_days < 1 or duration_days > 90:
        return jsonify({"error": "duration_days must be 1-90"}), 400

    try:
        goal_wei     = Web3.to_wei(float(goal_eth), "ether")
        crowdfunding = _get_crowdfunding()

        receipt_info = _build_and_send(
            crowdfunding.functions.createCampaign(
                title, description, goal_wei, duration_days
            ),
            gas=400_000,
        )

        # Read the new campaign count to get the ID
        campaign_id = crowdfunding.functions.campaignCount().call() - 1

        logger.info(
            "CREATE_CAMPAIGN | id=%d | title=%s | tx=%s",
            campaign_id, title, receipt_info["tx_hash"]
        )

        return jsonify({
            "campaign_id": campaign_id,
            "title":       title,
            "goal_eth":    float(goal_eth),
            "duration_days": duration_days,
            "transaction": receipt_info,
        })

    except Exception as e:
        logger.error("POST /create-campaign error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500



@app.route("/flag", methods=["POST"])
def flag_campaign():
    """
    Flag a campaign via SecurityModule.flagCampaign().
    Oracle or owner can call this.
    Request: { "campaign_address": "0x...", "reason": "..." }
    """
    data             = request.get_json(force=True)
    campaign_address = data.get("campaign_address")
    reason           = data.get("reason", "Flagged via dashboard")
    if not campaign_address:
        return jsonify({"error": "campaign_address is required"}), 400
    try:
        addr         = Web3.to_checksum_address(campaign_address)
        receipt_info = _build_and_send(
            security.functions.flagCampaign(addr, reason),
            gas=200_000,
        )
        logger.info("FLAG | campaign=%s | reason=%s | tx_hash=%s",
                    addr, reason, receipt_info["tx_hash"])
        return jsonify({
            "campaign_address": addr,
            "flagged":          True,
            "reason":           reason,
            "transaction":      receipt_info,
        })
    except Exception as e:
        logger.error("POST /flag error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/freeze", methods=["POST"])
def freeze_campaign():
    """
    Freeze a campaign via SecurityModule.freezeCampaign().
    Oracle or owner can call this.
    Request: { "campaign_address": "0x...", "reason": "..." }
    """
    data             = request.get_json(force=True)
    campaign_address = data.get("campaign_address")
    reason           = data.get("reason", "High fraud probability")
    if not campaign_address:
        return jsonify({"error": "campaign_address is required"}), 400
    try:
        addr         = Web3.to_checksum_address(campaign_address)
        receipt_info = _build_and_send(
            security.functions.freezeCampaign(addr),
            gas=200_000,
        )
        logger.info("FREEZE | campaign=%s | tx_hash=%s | reason=%s",
                    addr, receipt_info["tx_hash"], reason)
        return jsonify({
            "campaign_address": addr,
            "frozen":           True,
            "reason":           reason,
            "transaction":      receipt_info,
        })
    except Exception as e:
        logger.error("POST /freeze error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/unfreeze", methods=["POST"])
def unfreeze_campaign():
    """
    Unfreeze a campaign via SecurityModule.unfreezeCampaign().
    Only the contract owner (deployer wallet) can call this.
    Request: { "campaign_address": "0x...", "caller_address": "0x..." }
    The caller_address must match WALLET_ADDRESS (the deployer/owner).
    """
    data             = request.get_json(force=True)
    campaign_address = data.get("campaign_address")
    caller_address   = data.get("caller_address", "")

    if not campaign_address:
        return jsonify({"error": "campaign_address is required"}), 400

    # Enforce owner-only: caller must be the oracle/owner wallet.
    # Reject immediately if no caller address was provided.
    if not caller_address:
        logger.warning("UNFREEZE REJECTED | no caller_address supplied")
        return jsonify({
            "error": "Only the contract owner can unfreeze campaigns.",
        }), 403

    try:
        caller_norm = Web3.to_checksum_address(caller_address)
        owner_norm  = Web3.to_checksum_address(WALLET_ADDRESS)
    except Exception:
        logger.warning("UNFREEZE REJECTED | invalid address format | caller=%s", caller_address)
        return jsonify({"error": "Invalid address format"}), 400

    if caller_norm.lower() != owner_norm.lower():
        logger.warning("UNFREEZE REJECTED | caller=%s is not the owner=%s",
                       caller_address, WALLET_ADDRESS)
        return jsonify({
            "error": "Only the contract owner can unfreeze campaigns.",
        }), 403

    try:
        addr         = Web3.to_checksum_address(campaign_address)
        receipt_info = _build_and_send(
            security.functions.unfreezeCampaign(addr),
            gas=200_000,
        )
        logger.info("UNFREEZE | campaign=%s | tx_hash=%s",
                    addr, receipt_info["tx_hash"])
        return jsonify({
            "campaign_address": addr,
            "unfrozen":         True,
            "transaction":      receipt_info,
        })
    except Exception as e:
        logger.error("POST /unfreeze error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/unflag", methods=["POST"])
def unflag_campaign():
    """
    Unflag a campaign via SecurityModule.unflagCampaign().
    Only the contract owner can call this.
    Request: { "campaign_address": "0x..." }
    """
    data             = request.get_json(force=True)
    campaign_address = data.get("campaign_address")
    if not campaign_address:
        return jsonify({"error": "campaign_address is required"}), 400
    try:
        addr         = Web3.to_checksum_address(campaign_address)
        receipt_info = _build_and_send(
            security.functions.unflagCampaign(addr),
            gas=200_000,
        )
        logger.info("UNFLAG | campaign=%s | tx_hash=%s",
                    addr, receipt_info["tx_hash"])
        return jsonify({
            "campaign_address": addr,
            "unflagged":        True,
            "transaction":      receipt_info,
        })
    except Exception as e:
        logger.error("POST /unflag error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500



    """
    Freeze a campaign via SecurityModule.freezeCampaign().

    Request body (JSON):
    {
        "campaign_address": "0xAbCd...",
        "reason": "Fraud probability above threshold"  // optional
    }

    The oracle wallet (PRIVATE_KEY in .env) must be authorized
    as the oracle in SecurityModule.
    """
    data             = request.get_json(force=True)
    campaign_address = data.get("campaign_address")
    reason           = data.get("reason", "Fraud probability above threshold")

    if not campaign_address:
        return jsonify({"error": "campaign_address is required"}), 400

    try:
        addr         = Web3.to_checksum_address(campaign_address)
        receipt_info = _build_and_send(
            security.functions.freezeCampaign(addr),
            gas=200_000,
        )

        logger.info(
            "FREEZE | campaign=%s | tx_hash=%s | reason=%s",
            addr, receipt_info["tx_hash"], reason,
        )

        return jsonify({
            "campaign_address": addr,
            "frozen":           True,
            "reason":           reason,
            "transaction":      receipt_info,
        })

    except Exception as e:
        logger.error("POST /freeze error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/release", methods=["POST"])
def release_campaign():
    """
    Attempt to release campaign funds via CrowdfundingCore.releaseFunds().

    Request body (JSON):
    {
        "campaign_id": 0,          // on-chain integer campaign ID
        "caller_address": "0x..."  // must match WALLET_ADDRESS (owner/admin)
    }

    Note: The contract checks combined risk < riskThreshold before
    releasing -- if risk is too high it auto-freezes and reverts.
    """
    data           = request.get_json(force=True)
    campaign_id    = data.get("campaign_id")
    caller_address = data.get("caller_address", "")

    if campaign_id is None:
        return jsonify({"error": "campaign_id is required"}), 400

    # Enforce owner-only: caller must be the deployer/owner wallet
    if not caller_address:
        logger.warning("RELEASE REJECTED | no caller_address supplied")
        return jsonify({"error": "Only the contract owner can release campaign funds."}), 403

    try:
        caller_norm = Web3.to_checksum_address(caller_address)
        owner_norm  = Web3.to_checksum_address(WALLET_ADDRESS)
    except Exception:
        logger.warning("RELEASE REJECTED | invalid address format | caller=%s", caller_address)
        return jsonify({"error": "Invalid address format"}), 400

    if caller_norm.lower() != owner_norm.lower():
        logger.warning("RELEASE REJECTED | caller=%s is not the owner=%s",
                       caller_address, WALLET_ADDRESS)
        return jsonify({"error": "Only the contract owner can release campaign funds."}), 403

    try:
        crowdfunding = _get_crowdfunding()
        cid          = int(campaign_id)

        # Pre-check all contract conditions before sending tx
        c              = crowdfunding.functions.getCampaign(cid).call()
        creator        = c[0]
        goal_reached   = c[7]
        funds_released = c[8]
        frozen         = security.functions.isFrozen(creator).call()
        flagged        = security.functions.isFlagged(creator).call()
        paused         = security.functions.paused().call()
        risk           = registry.functions.getCombinedRisk(creator).call()
        threshold      = registry.functions.riskThreshold().call()

        if not goal_reached:
            return jsonify({"released": False, "error": "Funding goal not reached",
                            "campaign_id": cid, "blocked_by": "goal_not_reached"}), 200
        if funds_released:
            return jsonify({"released": False, "error": "Funds already released",
                            "campaign_id": cid, "blocked_by": "already_released"}), 200
        if frozen:
            return jsonify({"released": False, "error": "Campaign is frozen — unfreeze first",
                            "campaign_id": cid, "blocked_by": "frozen"}), 200
        if paused:
            return jsonify({"released": False, "error": "Platform is paused",
                            "campaign_id": cid, "blocked_by": "paused"}), 200
        if risk >= threshold:
            return jsonify({"released": False,
                            "error": f"Risk {risk} >= threshold {threshold} — will auto-freeze",
                            "campaign_id": cid, "blocked_by": "high_risk",
                            "risk": risk, "threshold": threshold}), 200

        receipt_info = _build_and_send(
            crowdfunding.functions.releaseFunds(cid),
            gas=400_000,
        )

        released = receipt_info["status"] == 1
        logger.info("RELEASE | campaign_id=%s | tx_hash=%s | released=%s",
                    cid, receipt_info["tx_hash"], released)

        return jsonify({
            "campaign_id": cid,
            "released":    released,
            "transaction": receipt_info,
        })

    except Exception as e:
        logger.error("POST /release error: %s", traceback.format_exc())
        return jsonify({"released": False, "error": str(e)}), 500


@app.route("/contribute", methods=["POST"])
def contribute():
    """
    Contribute ETH to a campaign via CrowdfundingCore.contribute(campaignId).
    The oracle wallet sends the contribution — for demo/testing purposes.
    Request: { "campaign_id": 0, "amount_eth": 0.01 }
    """
    data        = request.get_json(force=True)
    campaign_id = data.get("campaign_id")
    amount_eth  = data.get("amount_eth", 0)

    if campaign_id is None:
        return jsonify({"error": "campaign_id is required"}), 400
    if not amount_eth or float(amount_eth) <= 0:
        return jsonify({"error": "amount_eth must be > 0"}), 400

    try:
        crowdfunding = _get_crowdfunding()
        cid          = int(campaign_id)
        amount_wei   = Web3.to_wei(float(amount_eth), "ether")

        # Check campaign is active
        c = crowdfunding.functions.getCampaign(cid).call()
        if c[8]:  # funds_released
            return jsonify({"error": "Campaign funds already released"}), 400

        frozen  = security.functions.isFrozen(c[0]).call()
        flagged = security.functions.isFlagged(c[0]).call()
        if frozen:
            return jsonify({"error": "Campaign is frozen — contributions not accepted"}), 400
        if flagged:
            return jsonify({"error": "Campaign is flagged — contributions not accepted"}), 400

        # Build payable transaction
        import time
        nonce     = w3.eth.get_transaction_count(WALLET_ADDRESS, "pending")
        gas_price = int(w3.eth.gas_price * 1.5)
        tx = crowdfunding.functions.contribute(cid).build_transaction({
            "from":     WALLET_ADDRESS,
            "value":    amount_wei,
            "nonce":    nonce,
            "gas":      200_000,
            "gasPrice": gas_price,
            "chainId":  11155111,
        })
        signed  = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)

        logger.info("CONTRIBUTE | campaign=%d | amount=%.6f ETH | tx=%s",
                    cid, float(amount_eth), tx_hash.hex())

        # Poll for receipt
        for _ in range(45):
            time.sleep(4)
            try:
                receipt = w3.eth.get_transaction_receipt(tx_hash)
                if receipt:
                    return jsonify({
                        "campaign_id": cid,
                        "amount_eth":  float(amount_eth),
                        "success":     receipt.status == 1,
                        "transaction": {
                            "tx_hash":  tx_hash.hex(),
                            "status":   receipt.status,
                            "block":    receipt.blockNumber,
                            "gas_used": receipt.gasUsed,
                        }
                    })
            except Exception:
                pass

        raise TimeoutError(f"Contribution tx not confirmed: {tx_hash.hex()}")

    except Exception as e:
        logger.error("POST /contribute error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/refund", methods=["POST"])
def refund_campaign():
    """
    Prepare refund conditions (auto-flag if needed) and verify the contributor
    has a balance. The actual refund() call must be made by the contributor
    via MetaMask, because the contract uses msg.sender to identify who to refund.

    Request: { "campaign_id": 0, "contributor_address": "0x..." }
    Returns:
      - ready: true  → contributor has balance, conditions met, call refund() via MetaMask
      - ready: false → error message explaining why refund is not available
      - auto_flagged: true if the backend just flagged the campaign
    """
    data               = request.get_json(force=True)
    campaign_id        = data.get("campaign_id")
    contributor_address = data.get("contributor_address")

    if campaign_id is None:
        return jsonify({"error": "campaign_id is required"}), 400
    if not contributor_address:
        return jsonify({"error": "contributor_address is required — pass the MetaMask wallet address"}), 400

    try:
        crowdfunding = _get_crowdfunding()
        cid          = int(campaign_id)

        # Check current state
        c       = crowdfunding.functions.getCampaign(cid).call()
        creator = c[0]
        flagged = security.functions.isFlagged(creator).call()
        frozen  = security.functions.isFrozen(creator).call()
        deadline= c[4]
        goal_reached = c[7]
        funds_released = c[8]

        import time as _time
        goal_failed = _time.time() >= deadline and not goal_reached

        if funds_released:
            return jsonify({"ready": False, "error": "Funds already released — refund not possible"}), 200

        # If frozen but not flagged, auto-flag so the contract's refund() condition is met
        auto_flagged = False
        if frozen and not flagged:
            logger.info("Campaign %d is frozen but not flagged — auto-flagging for refund", cid)
            _build_and_send(
                security.functions.flagCampaign(
                    Web3.to_checksum_address(creator),
                    "Auto-flagged for refund after freeze"
                ),
                gas=200_000,
            )
            flagged = True
            auto_flagged = True

        if not flagged and not goal_failed:
            return jsonify({
                "ready": False,
                "error": "Refund not available: campaign must be flagged as fraudulent OR deadline passed with goal not reached",
                "is_flagged":  flagged,
                "is_frozen":   frozen,
                "goal_failed": goal_failed,
            }), 200

        # Verify the contributor actually has a balance on-chain
        contributor_addr = Web3.to_checksum_address(contributor_address)
        try:
            contrib = crowdfunding.functions.getContribution(cid, contributor_addr).call()
        except Exception:
            contrib = 0

        if contrib == 0:
            return jsonify({
                "ready": False,
                "error": f"Address {contributor_address[:10]}… has no contribution in campaign #{cid}.",
                "contributor": contributor_address,
                "campaign_id": cid,
            }), 200

        contrib_eth = float(Web3.from_wei(contrib, "ether"))
        logger.info(
            "REFUND_READY | campaign=%d | contributor=%s | amount=%.6f ETH | auto_flagged=%s",
            cid, contributor_address, contrib_eth, auto_flagged,
        )

        return jsonify({
            "ready":          True,
            "campaign_id":    cid,
            "contributor":    contributor_address,
            "amount_wei":     contrib,
            "amount_eth":     contrib_eth,
            "auto_flagged":   auto_flagged,
            "message":        "Conditions met. Sign the refund() transaction with MetaMask to claim your ETH.",
        })

    except Exception as e:
        logger.error("POST /refund error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/run-ai-detection", methods=["POST"])
def run_ai_detection():
    """
    Run TrustNet AI detection on a specific on-chain campaign.
    Fetches campaign data, runs inference, updates TrustMetricsRegistry.
    Request: { "campaign_id": 0 }
    Returns: fraud_probability, trust_scores, is_fraud, tx_hash
    """
    data        = request.get_json(force=True)
    campaign_id = data.get("campaign_id")
    if campaign_id is None:
        return jsonify({"error": "campaign_id is required"}), 400

    try:
        crowdfunding = _get_crowdfunding()
        cid          = int(campaign_id)

        c            = crowdfunding.functions.getCampaign(cid).call()
        creator      = c[0]
        title        = c[1]
        description  = c[2]
        goal         = c[3]
        raised       = c[5]

        # Compute all five feature scores from actual on-chain data
        features = _compute_features_from_chain(cid, crowdfunding)

        # Allow caller to override specific features (e.g. from the detection form)
        caller_features = data.get("features", {})
        features.update({k: v for k, v in caller_features.items() if v != 0.0})

        result = predict(features)
        ts     = result["trust_scores"]

        # Override the individual metric scores with the real computed values
        # (predictor.py extracts sdi/cti/fvrs/cfis/ctcs from the feature dict;
        # the _fvrs/_cti/_cfis/_ctcs/_sdi keys carry the composites we computed)
        from predictor import fraud_prob_to_trust_scores as _fpts
        ts = _fpts(
            result["fraud_probability"],
            sdi  = features["_sdi"],
            cti  = features["_cti"],
            fvrs = features["_fvrs"],
            cfis = features["_cfis"],
            ctcs = features["_ctcs"],
        )

        logger.info(
            "AI_DETECTION | campaign=%d | creator=%s | "
            "SDI=%d CTI=%d FVRS=%d CFIS=%d CTCS=%d | fraud_prob=%.4f | "
            "raw: sdi=%.3f cti=%.3f fvrs=%.3f cfis=%.3f ctcs=%.3f",
            cid, creator,
            ts["sdi"], ts["cti"], ts["fvrs"], ts["cfis"], ts["ctcs"],
            result["fraud_probability"],
            features["_sdi"], features["_cti"], features["_fvrs"],
            features["_cfis"], features["_ctcs"],
        )

        # Push scores on-chain
        receipt_info = _build_and_send(
            registry.functions.updateAllScores(
                Web3.to_checksum_address(creator),
                ts["sdi"], ts["cti"], ts["fvrs"], ts["cfis"], ts["ctcs"],
            )
        )

        # Immediately read back from chain to confirm what was stored
        stored = _fetch_trust_scores(Web3.to_checksum_address(creator))
        logger.info(
            "AI_DETECTION | getScores AFTER tx | campaign=%d | "
            "SDI=%d CTI=%d FVRS=%d CFIS=%d CTCS=%d CombinedRisk=%d LastUpdated=%s",
            cid,
            stored["sdi"], stored["cti"], stored["fvrs"],
            stored["cfis"], stored["ctcs"],
            stored["combined_risk"], stored.get("last_updated", "N/A"),
        )

        logger.info("AI_DETECTION | campaign=%d | fraud_prob=%.4f | tx=%s",
                    cid, result["fraud_probability"], receipt_info["tx_hash"])

        return jsonify({
            "campaign_id":       cid,
            "campaign_address":  creator,
            "title":             title,
            "fraud_probability": result["fraud_probability"],
            "is_fraud":          result["is_fraud"],
            "fraud_threshold":   result["fraud_threshold"],
            "trust_scores":      stored,   # confirmed on-chain values, not pre-tx estimates
            "transaction":       receipt_info,
            "recommendation":    "FREEZE" if result["is_fraud"] else "APPROVE",
        })

    except Exception as e:
        logger.error("POST /run-ai-detection error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# INTERNAL HELPERS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _fetch_trust_scores(campaign_address: str) -> Dict[str, Any]:
    """Fetch and format all trust scores from TrustMetricsRegistry."""
    try:
        (sdi, cti, fvrs, cfis, ctcs,
         combined_risk, last_updated) = registry.functions.getScores(campaign_address).call()
        return {
            "sdi":          sdi,
            "cti":          cti,
            "fvrs":         fvrs,
            "cfis":         cfis,
            "ctcs":         ctcs,
            "combined_risk": combined_risk,
            "last_updated": last_updated,
        }
    except Exception:
        return {
            "sdi": 0, "cti": 0, "fvrs": 0, "cfis": 0, "ctcs": 0,
            "combined_risk": 0, "last_updated": 0,
        }


def _compute_features_from_chain(cid: int, crowdfunding) -> Dict[str, Any]:
    """
    Fetch on-chain data for a single campaign and compute all five
    TrustNet feature scores (FVRS, CTI, CFIS, CTCS, SDI) using the
    actual extractor classes — no hardcoded defaults.
    """
    import time as _time
    import pandas as pd_inner

    # 1. Campaign base data
    c              = crowdfunding.functions.getCampaign(cid).call()
    title          = c[1]
    description    = c[2]
    goal_wei       = c[3]
    deadline_ts    = c[4]
    raised_wei     = c[5]
    n_contributors = c[6]

    goal_eth   = float(Web3.from_wei(goal_wei,   "ether"))
    raised_eth = float(Web3.from_wei(raised_wei, "ether"))

    # 2. ContributionReceived events → amounts + timestamps
    events = []
    try:
        events = crowdfunding.events.ContributionReceived.get_logs(
            from_block=0, to_block="latest",
            argument_filters={"campaignId": cid},
        )
    except Exception:
        try:
            events = crowdfunding.events.ContributionReceived.create_filter(
                from_block=0, to_block="latest",
                argument_filters={"campaignId": cid},
            ).get_all_entries()
        except Exception:
            pass

    amounts_eth = [float(Web3.from_wei(e["args"]["amount"], "ether")) for e in events]
    timestamps  = [int(e["args"]["timestamp"]) for e in events]

    # 3. FVRS
    from fvrs_updated import FVRS as _FVRS
    fvrs_calc = _FVRS()
    if timestamps:
        span_secs      = max(timestamps) - min(timestamps)
        duration_hours = max(span_secs / 3600.0, 1.0)
    else:
        duration_hours = max((deadline_ts - _time.time()) / 3600.0, 1.0)
    tpm    = len(timestamps) / (duration_hours * 60.0) if duration_hours > 0 else 0.0
    ts_iso = [datetime.fromtimestamp(t, tz=timezone.utc).isoformat() for t in timestamps]
    fvrs_vec = fvrs_calc.compute_vector({
        "current_funding":         raised_eth,
        "previous_funding":        0.0,
        "transactions_per_minute": tpm,
        "total_transactions":      float(len(timestamps)),
        "campaign_duration_hours": duration_hours,
        "new_users":               float(n_contributors),
        "total_users":             float(max(n_contributors, 1)),
        "timestamps":              ts_iso,
        "amounts":                 amounts_eth,
    })
    fvrs_score = float(np.mean([fvrs_vec["fgr"], fvrs_vec["tbr"],
                                 fvrs_vec["nur"], fvrs_vec["ftr"], fvrs_vec["rar"]]))

    # 4. CTI — boost if creator wallet has prior frozen/flagged campaigns
    approved  = raised_eth >= goal_eth * 0.5
    team_size = max(1, n_contributors // 3)
    A         = 1.0 if approved else 0.0
    T         = min(float(team_size) / 10.0, 1.0)
    M         = 0.0

    # Check if this creator wallet has other campaigns that are frozen or flagged.
    # If so, raise CTI to signal elevated threat from a repeat-offender wallet.
    creator_addr = c[0]
    wallet_threat_boost = 0.0
    try:
        total_campaigns = crowdfunding.functions.campaignCount().call()
        for other_id in range(total_campaigns):
            if other_id == cid:
                continue
            try:
                oc = crowdfunding.functions.getCampaign(other_id).call()
                if oc[0].lower() == creator_addr.lower():
                    other_frozen  = security.functions.isFrozen(oc[0]).call()
                    other_flagged = security.functions.isFlagged(oc[0]).call()
                    if other_frozen or other_flagged:
                        wallet_threat_boost = 0.4   # lift CTI by 40 points (out of 1.0)
                        break
            except Exception:
                continue
    except Exception:
        pass

    cti_norm = min((A + T + M) / 3.0 + wallet_threat_boost, 1.0)

    # 5. CFIS
    wallets    = set()
    cfis_vec   = {"wgd": 0.0, "cds": 0.0, "col": 0.0, "inf": 0.0, "fts": 0.0, "fas": 0.0}
    cfis_score = 0.0
    try:
        contributor_addrs = crowdfunding.functions.getContributors(cid).call()
        wallets = {w.lower() for w in contributor_addrs if w and w != "0x" + "0"*40}
    except Exception:
        pass

    if len(wallets) >= 2:
        from cfis_updated import (
            CFIS as _CFIS,
            build_cofunding_graph as _bcg,
            detect_communities as _dc,
            compute_pagerank as _cp,
        )
        cw  = {str(cid): wallets}
        wr  = {w: 0 for w in wallets}
        cwa: Dict[str, Dict[str, float]] = {}
        cwt: Dict[str, Any] = {}
        for e in events:
            key = e["args"]["contributor"].lower()
            amt = float(Web3.from_wei(e["args"]["amount"], "ether"))
            ts  = int(e["args"]["timestamp"])
            cwa.setdefault(str(cid), {})
            cwa[str(cid)][key] = cwa[str(cid)].get(key, 0.0) + amt
            cwt.setdefault(str(cid), {})
            cwt[str(cid)][key] = pd_inner.Timestamp(ts, unit="s")
        g        = _bcg(cw, min_shared_campaigns=1)
        cm       = _dc(g)
        pr       = _cp(g)
        cfis_obj = _CFIS(
            graph=g, communities=cm, pagerank=pr, wallet_reuse=wr,
            campaign_wallet_amounts=cwa, campaign_wallet_timestamps=cwt,
            min_wallets_for_score=2,
        )
        cfis_vec   = cfis_obj.compute_vector(str(cid), wallets)
        cfis_score = float(np.mean([cfis_vec["wgd"], cfis_vec["cds"],
                                     cfis_vec["col"], cfis_vec["fts"], cfis_vec["fas"]]))

    # 6. CTCS — cross-campaign temporal correlation using CTCSExtractor
    # Use all campaigns from this creator wallet as the comparison set.
    # Falls back to single-campaign timing regularity if only one campaign exists.
    ctcc_score = 0.0
    ts_score   = 0.0
    ctcs_score = 0.0
    try:
        total_campaigns = crowdfunding.functions.campaignCount().call()
        # Collect transaction rows for all campaigns from this creator
        ctcs_rows = []
        for oid in range(total_campaigns):
            try:
                oc = crowdfunding.functions.getCampaign(oid).call()
                if oc[0].lower() != creator_addr.lower():
                    continue
                # Get events for this campaign
                try:
                    oc_events = crowdfunding.events.ContributionReceived.get_logs(
                        from_block=0, to_block="latest",
                        argument_filters={"campaignId": oid},
                    )
                except Exception:
                    oc_events = []
                for e in oc_events:
                    ctcs_rows.append({
                        "campaign_id": str(oid),
                        "amount":      float(Web3.from_wei(e["args"]["amount"], "ether")),
                        "timestamp":   int(e["args"]["timestamp"]),
                    })
            except Exception:
                continue

        if ctcs_rows:
            import pandas as pd_ctcs
            ctcs_df = pd_ctcs.DataFrame(ctcs_rows)
            # Need at least 1 row with a valid timestamp
            if len(ctcs_df) >= 1:
                extractor = CTCSExtractor(num_time_bins=10, sync_window_seconds=3600.0,
                                          min_timestamps_for_score=1)
                try:
                    ctcs_feat = extractor.extract_dataframe(ctcs_df)
                    row = ctcs_feat[ctcs_feat["campaign_id"] == str(cid)]
                    if not row.empty:
                        ctcc_score = float(row["ctcc"].iloc[0])
                        ts_score   = float(row["ts"].iloc[0])
                        ctcs_score = float(row["ctcs"].iloc[0])
                except Exception:
                    pass

        # Fallback: single-campaign timing regularity when no extractor result
        if ctcs_score == 0.0 and len(timestamps) >= 2:
            ts_arr     = np.array(sorted(timestamps), dtype=float)
            gaps       = np.diff(ts_arr)
            mean_gap   = float(np.mean(gaps)) if len(gaps) > 0 else 1.0
            std_gap    = float(np.std(gaps))  if len(gaps) > 0 else 0.0
            cv         = std_gap / (mean_gap + 1e-9)
            ts_score   = float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))
            ctcs_score = ts_score * 0.4

    except Exception as ctcs_err:
        logger.warning("CTCS computation failed: %s", ctcs_err)

    # 7. SDI — semantic similarity vs other campaigns
    sdi_score = 0.0
    try:
        from csv_analyzer import _get_sdi_model
        import faiss
        count       = crowdfunding.functions.campaignCount().call()
        other_descs = []
        for oid in range(count):
            if oid == cid:
                continue
            try:
                oc = crowdfunding.functions.getCampaign(oid).call()
                if oc[2] and len(str(oc[2]).strip()) > 10:
                    other_descs.append(str(oc[2]))
            except Exception:
                continue
        if other_descs and description and len(description.strip()) > 10:
            sdi_model = _get_sdi_model()   # reuse cached model — no reload
            all_texts = [description] + other_descs
            embs      = sdi_model.encode(
                all_texts, batch_size=32,
                convert_to_numpy=True, normalize_embeddings=True,
            ).astype(np.float32)
            idx = faiss.IndexFlatIP(embs.shape[1])
            idx.add(embs[1:])
            scores, _ = idx.search(embs[:1], min(5, len(other_descs)))
            sdi_score = float(np.clip(scores[0][0], 0.0, 1.0)) if len(scores[0]) > 0 else 0.0
    except Exception as sdi_err:
        logger.warning("SDI computation skipped: %s", sdi_err)

    logger.info(
        "CHAIN_FEATURES | campaign=%d | SDI=%.3f CTI=%.3f FVRS=%.3f CFIS=%.3f CTCS=%.3f | contributors=%d txs=%d",
        cid, sdi_score, cti_norm, fvrs_score, cfis_score, ctcs_score,
        n_contributors, len(timestamps),
    )

    return {
        # FVRS sub-features (used by model tensor) + composite key predictor.py reads
        "fgr": fvrs_vec["fgr"], "tbr": fvrs_vec["tbr"],
        "nur": fvrs_vec["nur"], "ftr": fvrs_vec["ftr"], "rar": fvrs_vec["rar"],
        "fvrs": fvrs_score,          # predictor.py get("fvrs", 0.0)
        # CTI
        "A": A, "T": T, "M": M,
        "cti_score": cti_norm, "cti_norm": cti_norm,  # predictor.py get("cti_norm", ...)
        # CFIS sub-features + composite key predictor.py reads
        "wgd": cfis_vec["wgd"], "cds": cfis_vec["cds"],
        "col": cfis_vec["col"], "inf": cfis_vec["inf"],
        "fts": cfis_vec["fts"], "fas": cfis_vec["fas"],
        "cfis": cfis_score,          # predictor.py get("cfis", 0.0)
        # CTCS
        "ctcc": ctcc_score, "ts": ts_score,
        "ctcs": ctcs_score,          # predictor.py get("ctcs", 0.0)
        # SDI
        "sdi_score": sdi_score,      # predictor.py get("sdi_score", ...)
        "title_x": title, "title_y": title,
        "combined_risk_score": float(np.mean([fvrs_score, cti_norm,
                                               cfis_score, ctcs_score, sdi_score])),
        # Private composites used by run_ai_detection to call fraud_prob_to_trust_scores
        "_fvrs": fvrs_score,
        "_cti":  cti_norm,
        "_cfis": cfis_score,
        "_ctcs": ctcs_score,
        "_sdi":  sdi_score,
    }



# ─────────────────────────────────────────────
# SIMULATOR HELPER
# ─────────────────────────────────────────────
def _run_simulator_and_load() -> pd.DataFrame:
    """
    Runs Datasets/dataset_simulator.py which regenerates
    unified_dataset.csv, then loads and returns it as a DataFrame.
    Called automatically when the uploaded CSV is missing required columns.
    """
    import subprocess
    sim_script  = os.path.join(PROJECT_DIR, "Datasets", "dataset_simulator.py")
    output_csv  = os.path.join(PROJECT_DIR, "Datasets", "unified_dataset.csv")

    if not os.path.exists(sim_script):
        raise FileNotFoundError(f"dataset_simulator.py not found at {sim_script}")

    logger.info("Running dataset_simulator.py ...")
    result = subprocess.run(
        [sys.executable, sim_script],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"dataset_simulator.py failed (exit {result.returncode}):\n{result.stderr[-500:]}"
        )
    logger.info("Simulator finished. stdout: %s", result.stdout[-300:])

    if not os.path.exists(output_csv):
        raise FileNotFoundError(f"Simulator ran but {output_csv} was not created.")

    df = pd.read_csv(output_csv)
    logger.info("Loaded simulated dataset: %d rows, %d campaigns",
                len(df), df["campaign_id"].nunique())
    return df

# CSV ANALYSIS ROUTES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.route("/analyse-csv", methods=["POST"])
def analyse_csv_route():
    """
    Accepts CSV upload two ways:
      1. multipart/form-data  field 'file'  (curl / non-browser)
      2. application/json     {"filename": "x.csv", "data": "<base64>"}
         (browser fetch — avoids CORS preflight on multipart)
    """
    try:
        # ── Decode CSV from request ────────────────────────────────────
        ct = request.content_type or ""

        if "multipart" in ct:
            if "file" not in request.files:
                return jsonify({"error": "No file field in multipart upload."}), 400
            f = request.files["file"]
            df = pd.read_csv(f)

        elif "json" in ct or "application/json" in ct:
            body = request.get_json(force=True, silent=True) or {}
            import base64, io
            raw = body.get("data", "")
            if not raw:
                return jsonify({"error": "JSON body missing 'data' field (base64 CSV)."}), 400
            csv_bytes = base64.b64decode(raw)
            df = pd.read_csv(io.BytesIO(csv_bytes))

        else:
            # Try reading raw body as CSV
            try:
                df = pd.read_csv(io.BytesIO(request.data))
            except Exception:
                return jsonify({"error": f"Unsupported Content-Type: {ct}"}), 415

        # ── Check / fix columns ────────────────────────────────────────
        required = {"campaign_id", "title", "description",
                    "wallet_id", "amount", "timestamp"}
        missing  = required - set(df.columns)

        if missing:
            logger.warning(
                "CSV missing %s — running dataset_simulator.py to regenerate.",
                sorted(missing)
            )
            df = _run_simulator_and_load()
            still_missing = required - set(df.columns)
            if still_missing:
                return jsonify({
                    "error": f"Simulator ran but still missing: {sorted(still_missing)}"
                }), 500

        # ── Start async job ────────────────────────────────────────────
        from csv_analyzer import analyse_csv_async
        job_id = analyse_csv_async(df)
        return jsonify({
            "job_id":    job_id,
            "status":    "running",
            "message":   "Analysis started",
            "simulated": bool(missing),
        })

    except Exception as e:
        logger.error("POST /analyse-csv error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/analyse-csv/status/<job_id>", methods=["GET"])
def analyse_csv_status(job_id: str):
    """Poll analysis job status. Returns progress or final results."""
    from csv_analyzer import get_job, STEPS
    job = get_job(job_id)
    if not job:
        return jsonify({"error": f"Job '{job_id}' not found."}), 404

    if job["status"] == "running":
        return jsonify({
            "job_id":      job_id,
            "status":      "running",
            "step":        job["step"],
            "step_name":   job["step_name"],
            "total_steps": job["total_steps"],
            "steps":       STEPS,
        })
    elif job["status"] == "error":
        return jsonify({"job_id": job_id, "status": "error", "error": job["error"]}), 500
    else:  # done
        results = job["results"] or []
        global _last_csv_results
        _last_csv_results = {r["campaign_id"]: r for r in results}
        return jsonify({
            "job_id":      job_id,
            "status":      "done",
            "campaigns":   results,
            "total":       job["total"],
            "fraud_count": job["fraud_count"],
            "summary":     job["summary"],
        })


@app.route("/analyse-csv/<campaign_id>", methods=["GET"])
def get_csv_campaign(campaign_id: str):
    """Return full detail for one campaign from the last CSV analysis."""
    if not _last_csv_results:
        return jsonify({"error": "No analysis in memory. Upload a CSV first."}), 404
    result = _last_csv_results.get(campaign_id)
    if not result:
        return jsonify({"error": f"Campaign '{campaign_id}' not found."}), 404
    return jsonify(result)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ENTRY POINT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    logging.info("Starting TrustNet API with Waitress (multi-threaded WSGI)...")
    logging.info("Blockchain connected: %s", w3.is_connected())

    # Pre-load TrustNet model
    try:
        _, feature_names, _ = load_model()
        logging.info("TrustNet model pre-loaded: %d features", len(feature_names))
    except Exception as e:
        logging.warning("Model pre-load failed: %s", e)

    # Pre-load SDI sentence-transformer in main thread BEFORE starting server
    # This avoids the WinError 10038 socket issue in worker threads on Windows
    try:
        from csv_analyzer import _get_sdi_model
        _get_sdi_model()
        logging.info("SDI sentence-transformer pre-loaded.")
    except Exception as e:
        logging.warning("SDI pre-load failed (will load on first analysis): %s", e)

    from waitress import serve
    logging.info("Serving on http://0.0.0.0:5000 with 8 threads")
    serve(app, host="0.0.0.0", port=5000, threads=8)
