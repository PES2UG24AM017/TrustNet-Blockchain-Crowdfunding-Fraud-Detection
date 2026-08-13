# TrustNet: Blockchain Crowdfunding Fraud Detection

A Transformer-based framework for detecting fraud in blockchain crowdfunding campaigns using five novel trust metrics (SDI, CTI, FVRS, CFIS, CTCS), deployed on Ethereum Sepolia testnet.

**Paper:** TrustNet: A Transformer-Based Framework for Blockchain Crowdfunding Fraud Detection Using Multi-Dimensional Trust Features — PES University, Bengaluru.

---

## Folder Structure

```
Crowdfunding-Fraud-Detection/
├── contracts/                  Solidity smart contracts
│   ├── CrowdfundingCore.sol    Campaign creation, contributions, fund release/refund
│   ├── SecurityModule.sol      Freeze/flag campaigns, oracle auth, platform pause
│   └── TrustMetricsRegistry.sol  Stores 5 trust metric scores per campaign (0-100)
│
├── backend/                    Python backend (Flask API + inference + blockchain)
│   ├── app.py                  Flask REST API (all endpoints)
│   ├── predictor.py            TrustNet inference pipeline (preprocessing + model)
│   ├── pipeline.py             End-to-end: inference → blockchain → freeze
│   ├── blockchain.py           Web3.py connection to Sepolia
│   ├── config.py               Loads .env (RPC URL, keys, contract addresses)
│   ├── update_scores.py        Standalone script to call updateAllScores()
│   ├── model_loader.py         Standalone model loader / sanity check
│   ├── logs/predictions.log    All prediction logs (auto-created on first run)
│   ├── abi/                    Contract ABIs (CrowdfundingCore, SecurityModule, TrustMetricsRegistry)
│   └── requirements.txt        Python dependencies
│
├── trustnet/
│   └── trustnet_transformer.py  Training script for the TrustNet model
│
├── fvrs/fvrs_updated.py        FVRS feature extractor (FGR, TBR, NUR, FTR, RAR)
├── cti/cti.py                  CTI feature extractor (A, T, M → cti_score, cti_norm)
├── cfis/cfis_updated.py        CFIS feature extractor (WGD, CDS, COL, INF, FTS, FAS)
├── ctcs/ctcs_updated.py        CTCS feature extractor (CTCC, TS, CTCS)
├── sdi/sdi.py                  SDI feature extractor (FAISS + sentence-transformers)
│
├── dataset_builder/
│   └── output_combine_csv.py   Merges all 5 feature CSVs → trustnet_dataset.csv
│
├── Datasets/
│   ├── dataset_simulator.py    Generates synthetic unified_dataset.csv
│   └── fraud_data_simulator.py  Generates labeled training data (sdi,cti,fvrs,cfis,ctcs)
│
├── outputs/                    All generated CSVs + trained model
│   ├── fvrs_features.csv
│   ├── cti_features.csv
│   ├── cfis_features.csv
│   ├── ctcs_features.csv
│   ├── sdi_features.csv
│   ├── trustnet_dataset.csv    Final merged training dataset (30 campaigns, 35 cols)
│   └── trustnet_transformer.pt  Trained model checkpoint (33 features, d_model=32)
│
├── frontend/                   Dashboard (vanilla HTML/CSS/JS)
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── scripts/                    Hardhat deploy scripts (TypeScript)
├── ignition/                   Hardhat Ignition modules
├── test/                       Hardhat + Solidity unit tests
├── hardhat.config.ts
├── package.json
└── .env                        Root env (Hardhat: SEPOLIA_RPC_URL, SEPOLIA_PRIVATE_KEY)
```

---

## Reproducibility

### Reproduce model results (Fig 5, 6, 7 and Table III in paper)
```bash
python trustnet/genarate_pr_curve.py
# Saves plots to outputs/plots/
# Prints: Accuracy, Precision, Recall, F1, AP
```

### Reproduce ablation study (Table V in paper)
```bash
python run_ablation.py
# Trains 8 configurations and prints the complete ablation table
```

---

The trained model is saved at `outputs/trustnet_transformer_5features.pt` (5-feature model — used for all paper results) and `outputs/trustnet_transformer.pt` (32-feature production model used by the backend). Both are PyTorch checkpoints. Paper results (96.17% accuracy) come from the 5-feature model trained via `genarate_pr_curve.py`.

```python
{
    "model_state_dict":   <weights>,
    "feature_names":      <list of 33 column names>,
    "d_model":            32,
    "num_heads":          4,
    "num_layers":         2,
    "decision_threshold": 0.5,
}
```

Load it:

```python
from backend.predictor import load_model
model, feature_names, device = load_model()
```

---

## Preprocessing / Inference

The preprocessing during training was minimal — no sklearn scalers, no label encoding:

1. Load `outputs/trustnet_dataset.csv`
2. Drop `campaign_id`, `fraud_label`, `combined_risk_score`
3. Cast ALL remaining columns via `pd.to_numeric(errors='coerce').fillna(0.0)`
   - This turns string columns (`title_x`, `title_y`, `trust_label`, `risk_label`, `top_match_1`, `top_match_2`) into `0.0`
   - Bool columns (`below_min_wallets`, `below_min_timestamps`) become `0.0` or `1.0`
4. Convert to `float32` tensor
5. Feed to model; apply `torch.sigmoid()` to get fraud probability

The inference pipeline in `backend/predictor.py` reproduces these exact steps:

```python
from backend.predictor import predict

result = predict({
    "fgr": 0.097, "tbr": 0.15, "sdi_score": 0.71,
    "cti_norm": 0.06, "wgd": 0.59, "ctcs": 0.28,
    "combined_risk_score": 0.57,
    # ... any subset of the 33 features; missing keys default to 0.0
})

print(result["fraud_probability"])  # e.g. 0.2545
print(result["is_fraud"])           # True / False
print(result["trust_scores"])       # {"sdi": 71, "cti": 6, ...} (0-100 ints for blockchain)
```

### The 32 Feature Columns — Production Model (in exact order)

```
FVRS:  fgr, tbr, nur, ftr, rar
CTI:   title_x (→0.0), A, T, M, cti_score, cti_norm, trust_label (→0.0)
CFIS:  n_wallets, wgd, cds, col, inf, fts, fas, below_min_wallets
CTCS:  n_transactions, ctcc, ts, ctcs, below_min_timestamps
SDI:   title_y (→0.0), sdi_score, risk_label (→0.0),
       top_match_1 (→0.0), top_match_1_sim, top_match_2 (→0.0), top_match_2_sim
```

Note: `combined_risk_score` (the 33rd column in the dataset) is deliberately excluded
from training to prevent data leakage — it is derived from the other 5 metrics.

---

## Blockchain Integration

### Contracts (Sepolia)

| Contract              | Address                                      |
|-----------------------|----------------------------------------------|
| SecurityModule        | `0xf7559D4875be123119C0FFAb4ca690D23747d32B` |
| TrustMetricsRegistry  | `0x2181c36c169061c5b0465166732cAc3C0A9e9085` |
| CrowdfundingCore      | `0x582c7F0721A0A5b572233F2e1583412b946fF43A` |

### Score Formula (on-chain)

```
combined_risk = (SDI*25 + (100-CTI)*20 + FVRS*20 + CFIS*20 + CTCS*15) / 100
```

Campaigns with `combined_risk >= 70` are frozen automatically by `CrowdfundingCore.releaseFunds()`.

---

## API Endpoints

| Method | Endpoint                   | Description                                       |
|--------|----------------------------|---------------------------------------------------|
| GET    | `/health`                  | Check blockchain + model status                   |
| GET    | `/campaigns`               | List all on-chain campaigns with trust scores     |
| GET    | `/campaign/<id>`           | Single campaign detail + full trust breakdown     |
| GET    | `/metrics/<address>`       | On-chain trust scores for a campaign address      |
| POST   | `/predict`                 | Run TrustNet inference on feature input           |
| POST   | `/update-blockchain`       | Inference + push scores to TrustMetricsRegistry   |
| POST   | `/freeze`                  | Freeze campaign via SecurityModule                |
| POST   | `/release`                 | Attempt to release funds via CrowdfundingCore     |

---

## How to Run

### Prerequisites

- Python 3.10+
- Node.js 18+ (for Hardhat / frontend dev server)
- MetaMask browser extension
- A Sepolia RPC endpoint (Alchemy / Infura)

### 1. Configure environment

Create `backend/.env`:

```env
SEPOLIA_RPC_URL=https://eth-sepolia.g.alchemy.com/v2/YOUR_KEY
PRIVATE_KEY=0xYOUR_ORACLE_PRIVATE_KEY
WALLET_ADDRESS=0xYOUR_ORACLE_WALLET_ADDRESS
TRUST_METRICS_REGISTRY=0x2181c36c169061c5b0465166732cAc3C0A9e9085
SECURITY_MODULE=0xf7559D4875be123119C0FFAb4ca690D23747d32B
CROWDFUNDING_CORE=0x582c7F0721A0A5b572233F2e1583412b946fF43A
```

### 2. Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Verify model is present

```bash
python model_loader.py
# Should print: d_model: 32, Number of features: 33
```

### 4. Start the Flask API

```bash
cd backend
python app.py
# → Running on http://0.0.0.0:5000
```

### 5. Open the dashboard

Open `frontend/index.html` in a browser (or serve it with any static server):

```bash
# Simple one-liner (Python built-in):
cd frontend
python -m http.server 8080
# → Open http://localhost:8080
```

### 6. Run the full pipeline (CLI)

```bash
cd backend
python pipeline.py --campaign_address 0xDCdC29B6A4A477d3beAE4618F93535886137796D --dry_run
# Remove --dry_run to send real transactions
```

---

## Complete Workflow (Task 8)

```
Campaign data (on-chain or input form)
        ↓
Feature extraction (FVRS, CTI, CFIS, CTCS, SDI pipelines)
        ↓
predictor.py: build_feature_vector()
  → 33-element float32 tensor (strings/bools → 0.0, no scaling)
        ↓
TrustNetTransformer.predict_proba()
  → fraud probability ∈ [0, 1]
        ↓
/update-blockchain: push SDI/CTI/FVRS/CFIS/CTCS (0-100) to TrustMetricsRegistry
  → updateAllScores() tx on Sepolia
        ↓
fraud_prob >= 0.5 → /freeze: call SecurityModule.freezeCampaign()
fraud_prob <  0.5 → campaign allowed, releaseFunds() passes trust check
        ↓
Dashboard updates with risk meter, metric boxes, freeze status
        ↓
logs/predictions.log: campaign_id | fraud_prob | tx_hash | frozen
```

---

## Logging

Every prediction is logged to `backend/logs/predictions.log`:

```
2026-07-13 11:30:00 [INFO] PREDICTION | campaign=CAMP_0001 | fraud_prob=0.714200 | tx_hash=0xabc...def | frozen=True
```

---

## Integration Summary

**Files added:**
- `backend/predictor.py` — inference pipeline (preprocessing + model)
- `backend/pipeline.py` — end-to-end workflow (inference → blockchain → freeze)
- `backend/app.py` — Flask REST API (was empty)
- `frontend/index.html` — dashboard (MetaMask login, browse campaigns, predict, pipeline)
- `frontend/style.css` — dark theme UI
- `frontend/app.js` — frontend logic
- `README.md` — this file

**Files NOT modified:**
- All Solidity contracts
- Hardhat project
- Dataset builder
- Feature extraction modules (FVRS, CTI, CFIS, CTCS, SDI)
- Transformer architecture
- Trained model (`outputs/trustnet_transformer.pt`)
- Deployment addresses
- `blockchain.py`, `config.py`, `update_scores.py`, `model_loader.py`
