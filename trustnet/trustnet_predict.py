"""
TrustNet Predictor — Dashboard Integration Module
-----------------------------------------------------
A clean, ready-to-call interface for the dashboard team to get
fraud predictions from the trained TrustNet Transformer, without
needing to understand PyTorch internals.

Requires:
    outputs/trustnet_transformer.pt   (trained model, from teammate)
    trustnet_transformer.py           (model architecture, same folder)

Usage (as a library):
    from trustnet_predict import TrustNetPredictor

    predictor = TrustNetPredictor()
    result = predictor.predict({
        'sdi_score': 0.82,
        'cti_norm': 0.31,
        'fgr': 0.6, 'tbr': 0.7, 'nur': 0.5, 'ftr': 0.4, 'rar': 0.6,
        'wgd': 0.5, 'cds': 0.6, 'col': 0.4, 'inf': 0.5, 'fts': 0.6, 'fas': 0.5,
        'ctcs': 0.7,
    })
    print(result)
    # {'fraud_probability': 0.87, 'is_fraud': True, 
    #  'risk_level': 'HIGH', 'confidence': 0.87}

Usage (from command line, for a quick manual test):
    python trustnet_predict.py
"""

import os
import torch
from trustnet_transformer import TrustNetTransformer

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "..", "outputs", "trustnet_transformer.pt")


class TrustNetPredictor:
    """
    Loads the trained TrustNet model once, then exposes a simple
    .predict(scores_dict) method for the dashboard to call on
    every campaign it needs a fraud score for.

    Loading the model is somewhat slow — this class is designed
    to be instantiated ONCE when the dashboard/backend starts up,
    not once per request.
    """

    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at: {model_path}\n"
                f"Make sure trustnet_transformer.pt has been placed "
                f"in the outputs/ folder."
            )

        checkpoint = torch.load(model_path, map_location='cpu')

        self.feature_names = checkpoint['feature_names']
        self.decision_threshold = checkpoint.get('decision_threshold', 0.5)

        self.model = TrustNetTransformer(
            num_features=len(self.feature_names),
            d_model=checkpoint['d_model'],
            num_heads=checkpoint['num_heads'],
            num_layers=checkpoint['num_layers'],
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()   # inference mode — disables dropout etc.

        print(f"TrustNet model loaded. Expecting {len(self.feature_names)} "
              f"features: {self.feature_names}")

    def predict(self, scores: dict) -> dict:
        """
        Takes a dictionary of trust metric scores for ONE campaign
        and returns a fraud prediction.

        Parameters
        ----------
        scores : dict
            Must contain a key for every name in self.feature_names,
            each mapped to a float between 0 and 1. Extra keys are
            ignored. Missing keys raise a clear error rather than
            silently defaulting to 0 (a silent default could produce
            a misleading prediction).

        Returns
        -------
        dict with keys:
            fraud_probability : float, 0 to 1
            is_fraud           : bool
            risk_level         : str, 'LOW' / 'MEDIUM' / 'HIGH'
            confidence         : float, how far the probability is
                                  from the decision boundary (0.5),
                                  rescaled to 0-1 — higher means the
                                  model is more certain either way
        """
        missing = [f for f in self.feature_names if f not in scores]
        if missing:
            raise ValueError(
                f"Missing required score(s): {missing}. "
                f"predict() needs all of: {self.feature_names}"
            )

        values = [float(scores[f]) for f in self.feature_names]

        x = torch.tensor([values], dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(x)
            probability = torch.sigmoid(logits).item()

        is_fraud = probability >= self.decision_threshold

        if probability >= 0.75:
            risk_level = "HIGH"
        elif probability >= 0.45:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        confidence = abs(probability - 0.5) * 2   # 0 (uncertain) to 1 (certain)

        return {
            'fraud_probability': round(probability, 4),
            'is_fraud':          bool(is_fraud),
            'risk_level':        risk_level,
            'confidence':        round(confidence, 4),
        }

    def predict_batch(self, campaigns: list) -> list:
        """
        Convenience method for scoring MULTIPLE campaigns at once
        — e.g. the dashboard's "all campaigns" view.

        Parameters
        ----------
        campaigns : list of dicts, each shaped like the `scores`
                    argument to .predict(). Should also include a
                    'campaign_id' key if you want it echoed back
                    in the results for easy matching.

        Returns
        -------
        list of dicts — same shape as .predict()'s return value,
        with 'campaign_id' included if it was present in the input.
        """
        results = []
        for campaign in campaigns:
            campaign_id = campaign.get('campaign_id', None)
            scores = {k: v for k, v in campaign.items() if k != 'campaign_id'}

            result = self.predict(scores)
            if campaign_id is not None:
                result = {'campaign_id': campaign_id, **result}

            results.append(result)
        return results


# ─────────────────────────────────────────────
# QUICK MANUAL TEST (run this file directly)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    predictor = TrustNetPredictor()

    print("\nExpected feature order:", predictor.feature_names)
    print("Decision threshold:", predictor.decision_threshold)

    # Build a dummy example matching whatever features the model
    # actually expects (fill each with 0.5 as a placeholder)
    example_scores = {name: 0.5 for name in predictor.feature_names}

    print("\nRunning a test prediction with placeholder scores (all 0.5)...")
    result = predictor.predict(example_scores)
    print("Result:", result)

    print("\n" + "-" * 60)
    print("To use this in the dashboard backend:")
    print("-" * 60)
    print("""
from trustnet_predict import TrustNetPredictor

predictor = TrustNetPredictor()   # load once at startup

# ... later, for each campaign needing a score:
result = predictor.predict({
    'sdi_score': 0.82, 'cti_norm': 0.31, ...
})
print(result['fraud_probability'], result['risk_level'])
""")
