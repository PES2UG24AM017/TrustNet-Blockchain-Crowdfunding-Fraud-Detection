"""
csv_analyzer.py  —  TrustNet full-pipeline CSV analysis
---------------------------------------------------------
Runs FVRS → CTI → CFIS → CTCS → SDI → TrustNet on every
campaign in a unified_dataset.csv DataFrame.

Key design decisions:
  • SDI sentence-transformer is cached after first load
    so repeated calls don't re-download/re-init the model.
  • analyse_csv_async() runs the pipeline in a background
    thread and writes progress to a shared status dict so
    the Flask endpoint can return immediately and the
    frontend polls /analyse-csv/status.
"""

import os
import sys
import uuid
import logging
import threading
import multiprocessing
import traceback
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Path setup ──────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ["fvrs", "cti", "cfis", "ctcs", "sdi"]:
    _p = os.path.join(PROJECT_DIR, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fvrs_updated import FVRSExtractor
from cfis_updated import CFISExtractor
from ctcs_updated import CTCSExtractor

import importlib.util as _ilu

def _load_module(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_cti_mod = _load_module("cti_module",  os.path.join(PROJECT_DIR, "cti", "cti.py"))
_sdi_mod = _load_module("sdi_module",  os.path.join(PROJECT_DIR, "sdi", "sdi.py"))
compute_all_cti_fn = _cti_mod.compute_all_cti
compute_all_sdi_fn = _sdi_mod.compute_all_sdi
build_index_fn     = _sdi_mod.build_index

from predictor import predict, FRAUD_THRESHOLD

# ── SDI model singleton — loaded once, reused across all jobs ───────────
_sdi_model = None
_sdi_model_lock = threading.Lock()

def _get_sdi_model():
    global _sdi_model
    if _sdi_model is not None:
        return _sdi_model
    with _sdi_model_lock:
        if _sdi_model is None:
            logger.info("Loading SDI sentence-transformer (first time only)…")
            from sentence_transformers import SentenceTransformer
            _sdi_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("SDI model loaded and cached.")
    return _sdi_model

# ── Job store — use a regular dict + threading.Lock (jobs run in threads,
# but waitress gives each request its own thread so polling works fine)
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()

STEPS = ["FVRS", "CTI", "CFIS", "CTCS", "SDI", "TrustNet"]

def _new_job() -> str:
    job_id = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[job_id] = {
            "status":      "running",
            "step":        0,
            "step_name":   STEPS[0],
            "total_steps": len(STEPS),
            "results":     None,
            "summary":     None,
            "error":       None,
        }
    return job_id

def get_job(job_id: str) -> Optional[Dict]:
    return _jobs.get(job_id)

def _set_step(job_id: str, idx: int):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["step"]      = idx
            _jobs[job_id]["step_name"] = STEPS[idx] if idx < len(STEPS) else "Done"

# ── Combined risk (mirrors smart contract formula) ──────────────────────
def _combined_risk(sdi, cti, fvrs, cfis, ctcs):
    return (0.25 * sdi + 0.20 * (1.0 - cti) + 0.20 * fvrs +
            0.20 * cfis + 0.15 * ctcs)

# ── Core pipeline ───────────────────────────────────────────────────────
def _run_pipeline(df: pd.DataFrame, job_id: str) -> list:
    try:
        # 1. FVRS
        _set_step(job_id, 0)
        logger.info("[%s] Running FVRS…", job_id)
        fvrs_df = FVRSExtractor().extract_dataframe(df.copy())
        fvrs_df["fvrs_score"] = fvrs_df[["fgr","tbr","nur","ftr","rar"]].mean(axis=1)

        # 2. CTI
        _set_step(job_id, 1)
        logger.info("[%s] Running CTI…", job_id)
        cti_input = df.drop_duplicates("campaign_id").reset_index(drop=True)
        _cti_mod.COL_ID         = "campaign_id"
        _cti_mod.COL_TITLE      = "title"
        _cti_mod.COL_APPROVED   = "approved"
        _cti_mod.COL_TEAM_SIZE  = "team_size"
        _cti_mod.COL_MULTICHAIN = "multichain"
        cti_df = compute_all_cti_fn(cti_input)[
            ["campaign_id","cti_score","cti_norm","trust_label","A","T","M"]
        ].copy()

        # 3. CFIS
        _set_step(job_id, 2)
        logger.info("[%s] Running CFIS…", job_id)
        cfis_df = CFISExtractor(min_shared_campaigns=2).extract_dataframe(df.copy())
        cfis_df["cfis_score"] = cfis_df[["wgd","cds","col","inf","fts","fas"]].mean(axis=1)

        # 4. CTCS
        _set_step(job_id, 3)
        logger.info("[%s] Running CTCS…", job_id)
        ctcs_df = CTCSExtractor(num_time_bins=20).extract_dataframe(df.copy())

        # 5. SDI
        _set_step(job_id, 4)
        logger.info("[%s] Running SDI…", job_id)
        sdi_input = (df.drop_duplicates("campaign_id")
                       .dropna(subset=["description"])
                       .reset_index(drop=True))
        sdi_input = sdi_input[sdi_input["description"].astype(str).str.len() > 20]

        _sdi_mod.COL_ID          = "campaign_id"
        _sdi_mod.COL_TITLE       = "title"
        _sdi_mod.COL_DESCRIPTION = "description"

        try:
            sdi_model  = _get_sdi_model()
            sdi_index, sdi_embeddings = build_index_fn(sdi_input, sdi_model)
            sdi_df = compute_all_sdi_fn(sdi_input, sdi_index, sdi_embeddings)[
                ["campaign_id","sdi_score","risk_label",
                 "top_match_1","top_match_1_sim","top_match_2","top_match_2_sim"]
            ].copy()
        except Exception as e:
            logger.warning("[%s] SDI failed (%s) — defaulting to 0", job_id, e)
            campaigns = df["campaign_id"].unique()
            sdi_df = pd.DataFrame({
                "campaign_id": campaigns, "sdi_score": 0.0,
                "risk_label": "UNKNOWN", "top_match_1": "",
                "top_match_1_sim": 0.0, "top_match_2": "", "top_match_2_sim": 0.0,
            })

        # 6. Merge
        merged = (fvrs_df
                  .merge(cti_df,  on="campaign_id", how="left")
                  .merge(cfis_df, on="campaign_id", how="left")
                  .merge(ctcs_df, on="campaign_id", how="left")
                  .merge(sdi_df,  on="campaign_id", how="left"))
        merged.fillna(0, inplace=True)

        titles = df.drop_duplicates("campaign_id")[["campaign_id","title"]].set_index("campaign_id")
        merged["title"] = merged["campaign_id"].map(titles["title"]).fillna("Unknown")
        merged["combined_risk_score"] = merged.apply(
            lambda r: _combined_risk(
                float(r.get("sdi_score",0)), float(r.get("cti_norm",0.5)),
                float(r.get("fvrs_score",0)), float(r.get("cfis_score",0)),
                float(r.get("ctcs",0))
            ), axis=1)

        # 7. TrustNet inference
        _set_step(job_id, 5)
        logger.info("[%s] Running TrustNet inference on %d campaigns…", job_id, len(merged))
        results = []
        for _, row in merged.iterrows():
            feat = {
                "fgr": float(row.get("fgr",0)), "tbr": float(row.get("tbr",0)),
                "nur": float(row.get("nur",0)), "ftr": float(row.get("ftr",0)),
                "rar": float(row.get("rar",0)),
                "title_x":     str(row.get("title","")),
                "A":           float(row.get("A",0)), "T": float(row.get("T",0)),
                "M":           float(row.get("M",0)),
                "cti_score":   float(row.get("cti_score",0)),
                "cti_norm":    float(row.get("cti_norm",0)),
                "trust_label": str(row.get("trust_label","")),
                "n_wallets":   float(row.get("n_wallets",0)),
                "wgd":  float(row.get("wgd",0)),  "cds": float(row.get("cds",0)),
                "col":  float(row.get("col",0)),  "inf": float(row.get("inf",0)),
                "fts":  float(row.get("fts",0)),  "fas": float(row.get("fas",0)),
                "below_min_wallets": bool(row.get("below_min_wallets",False)),
                "n_transactions":       float(row.get("n_transactions",0)),
                "ctcc": float(row.get("ctcc",0)), "ts": float(row.get("ts",0)),
                "ctcs": float(row.get("ctcs",0)),
                "below_min_timestamps": bool(row.get("below_min_timestamps",False)),
                "title_y":        str(row.get("title","")),
                "sdi_score":      float(row.get("sdi_score",0)),
                "risk_label":     str(row.get("risk_label","")),
                "top_match_1":    str(row.get("top_match_1","")),
                "top_match_1_sim": float(row.get("top_match_1_sim",0)),
                "top_match_2":    str(row.get("top_match_2","")),
                "top_match_2_sim": float(row.get("top_match_2_sim",0)),
                "combined_risk_score": float(row.get("combined_risk_score",0)),
            }
            inf   = predict(feat)
            prob  = inf["fraud_probability"]
            ts_sc = inf["trust_scores"]
            cr100 = int(round(row["combined_risk_score"] * 100))
            results.append({
                "campaign_id":       str(row["campaign_id"]),
                "title":             str(row.get("title","Unknown")),
                "fraud_probability": round(prob, 4),
                "is_fraud":          bool(inf["is_fraud"]),
                "fraud_threshold":   FRAUD_THRESHOLD,
                "trust_scores": {
                    "sdi":  ts_sc["sdi"], "cti":  ts_sc["cti"],
                    "fvrs": ts_sc["fvrs"],"cfis": ts_sc["cfis"],
                    "ctcs": ts_sc["ctcs"],"combined_risk": cr100,
                },
                "features": {
                    k: round(float(row.get(k,0)),4)
                    for k in ["fgr","tbr","nur","ftr","rar",
                               "wgd","cds","col","inf","fts","fas",
                               "ctcc","ts","ctcs","cti_score","cti_norm",
                               "sdi_score","combined_risk_score"]
                },
                "sdi_details": {
                    "risk_label":    str(row.get("risk_label","UNKNOWN")),
                    "top_match_1":   str(row.get("top_match_1","")),
                    "top_match_1_sim": round(float(row.get("top_match_1_sim",0)),4),
                    "top_match_2":   str(row.get("top_match_2","")),
                    "top_match_2_sim": round(float(row.get("top_match_2_sim",0)),4),
                },
            })

        results.sort(key=lambda x: x["fraud_probability"], reverse=True)
        logger.info("[%s] Pipeline complete: %d campaigns", job_id, len(results))
        return results

    except Exception:
        raise


def _worker(job_id: str, df: pd.DataFrame):
    """Background thread entry point."""
    try:
        results = _run_pipeline(df, job_id)
        probs   = [r["fraud_probability"] for r in results]
        summary = {
            "mean_fraud_probability": round(float(np.mean(probs)), 4) if probs else 0,
            "max_fraud_probability":  round(float(np.max(probs)),  4) if probs else 0,
            "high_risk_count": sum(
                1 for r in results if r["trust_scores"]["combined_risk"] >= 70
            ),
        }
        with _jobs_lock:
            _jobs[job_id]["status"]  = "done"
            _jobs[job_id]["results"] = results
            _jobs[job_id]["summary"] = summary
            _jobs[job_id]["fraud_count"] = sum(1 for r in results if r["is_fraud"])
            _jobs[job_id]["total"]   = len(results)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("[%s] Pipeline failed:\n%s", job_id, tb)
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"]  = str(e)


def analyse_csv_async(df: pd.DataFrame) -> str:
    """
    Start the pipeline in a background thread.
    Returns job_id immediately — poll get_job(job_id) for status.
    """
    job_id = _new_job()
    t = threading.Thread(target=_worker, args=(job_id, df), daemon=True)
    t.start()
    logger.info("Analysis job %s started in background thread.", job_id)
    return job_id
