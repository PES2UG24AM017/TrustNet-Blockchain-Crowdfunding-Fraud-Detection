"""
xai_integrated_gradients.py
-----------------------------
TrustNet — Explainability Analysis via Integrated Gradients

REVIEWER REQUEST:
  "Attention/XAI explanation missing; include attention visualizations
   or established XAI methods showing how trust signals influence
   fraud decisions."

METHOD SELECTED: Integrated Gradients (Sundararajan, Taly & Yan, 2017)
----------------------------------------------------------------------
Integrated Gradients (IG) is a well-established, gradient-based
feature attribution method that satisfies two important axioms:

  Sensitivity: if changing a single input changes the output, it
    receives a non-zero attribution.
  Implementation Invariance: attributions are the same for
    functionally equivalent networks.
  Completeness: attributions sum to the difference between the
    model output at the input and at a chosen baseline.

IG is appropriate here because:
  - The model takes exactly 5 scalar inputs (SDI, CTI, FVRS, CFIS, CTCS)
  - It is implementable directly in PyTorch without extra dependencies
  - It avoids the known limitation of raw Transformer attention
    weights, which are not directly equivalent to feature importance
    (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019)
  - captum/SHAP are unavailable in this environment

WHAT INTEGRATED GRADIENTS MEASURES:
  For a given test sample x and baseline x' (here: all-zeros in
  normalised space, representing a neutral/average-risk profile):

    IG_i(x) = (x_i - x'_i) * integral[dF/dx_i along path from x' to x]

  A positive IG attribution means the feature pushed the model's
  fraud probability UP relative to the baseline.
  A negative IG attribution means it pushed it DOWN.
  This is a model-contribution measure, NOT a causal claim.

IMPORTANT LIMITATIONS:
  - Attributions are relative to the chosen baseline (zero vector
    in normalised space, corresponding to the training-set mean in
    original space), not absolute causal importance.
  - The model was trained on synthetic data; attributions reflect
    learned patterns in that data.
  - Do NOT interpret "SDI has the largest attribution" as
    "SDI causes fraud". It means SDI had the largest average
    contribution to the model's fraud score on this test set.

COMPLETENESS CHECK:
  For each sample, sum(IG attributions) ≈ F(x) - F(x_baseline).
  Max absolute completeness error is reported as a quality metric.

DATA USED:
  - Checkpoint:  outputs/trustnet_transformer_5features.pt
  - Dataset:     outputs/trustnet_train.csv
  - Split:       Same 70/15/15 stratified split (seed=42) as primary
  - Norm stats:  Loaded from checkpoint (feature_mean, feature_std)
  - Test set:    Same 600 samples as primary evaluation

OUTPUTS:
  outputs/xai/feature_attributions.csv
  outputs/xai/feature_importance_summary.csv
  outputs/xai/mean_attribution_barplot.png  (300 dpi)

Usage:
    python xai_integrated_gradients.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

# ─────────────────────────────────────────────────────────────
# PATHS & CONFIG
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR  = os.path.join(SCRIPT_DIR, 'outputs')
CKPT_PATH    = os.path.join(OUTPUTS_DIR, 'trustnet_transformer_5features.pt')
DATA_PATH    = os.path.join(OUTPUTS_DIR, 'trustnet_train.csv')
XAI_DIR      = os.path.join(OUTPUTS_DIR, 'xai')

SEED       = 42
FEATURES   = ['sdi', 'cti', 'fvrs', 'cfis', 'ctcs']
LABEL_COL  = 'is_fraud'
THRESHOLD  = 0.5
IG_STEPS   = 300   # number of Riemann approximation steps (>= 50 is standard)

os.makedirs(XAI_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# MODEL — exact architecture matching the saved checkpoint
# ─────────────────────────────────────────────────────────────
class TrustNetTransformer(nn.Module):
    """Exact architecture from trustnet_transformer.py."""
    def __init__(self, num_features=5, d_model=32,
                 num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_projection   = nn.Linear(1, d_model)
        self.positional_embedding = nn.Parameter(
            torch.randn(1, num_features, d_model) * 0.02
        )
        el = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(el, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        x      = x.unsqueeze(-1)                    # (B, 5, 1)
        tokens = self.input_projection(x)            # (B, 5, 32)
        tokens = tokens + self.positional_embedding  # (B, 5, 32)
        H      = self.transformer_encoder(tokens)    # (B, 5, 32)
        z      = H.mean(dim=1)                       # (B, 32)
        return  self.classifier(z)                   # (B, 1)  raw logit

    def fraud_probability(self, x):
        """Return sigmoid-activated fraud probability."""
        return torch.sigmoid(self.forward(x))


# ─────────────────────────────────────────────────────────────
# INTEGRATED GRADIENTS
# ─────────────────────────────────────────────────────────────
def integrated_gradients(model, x, baseline, steps=IG_STEPS):
    """
    Compute Integrated Gradients for a batch of inputs.

    Parameters
    ----------
    model    : nn.Module  (must output raw logit or probability; we use sigmoid)
    x        : torch.Tensor  shape (B, F)  — normalised test inputs
    baseline : torch.Tensor  shape (1, F)  — reference input (zero vector)
    steps    : int  — Riemann approximation steps

    Returns
    -------
    attributions : np.ndarray  shape (B, F)  — signed IG attributions
    delta_pred   : np.ndarray  shape (B,)    — F(x) - F(baseline) for completeness
    """
    model.eval()
    x        = x.float()
    baseline = baseline.float()

    # Interpolated inputs: shape (steps, B, F)
    alphas = torch.linspace(0, 1, steps + 1).view(-1, 1, 1)  # (S+1,1,1)
    interpolated = baseline + alphas * (x.unsqueeze(0) - baseline)  # (S+1, B, F)

    # Flatten batch and steps for a single forward pass
    S1, B, F = interpolated.shape
    inp_flat = interpolated.view(S1 * B, F)
    inp_flat.requires_grad_(True)

    # Forward pass — use fraud_probability (sigmoid output)
    out_flat = model.fraud_probability(inp_flat)  # (S1*B, 1)
    out_flat = out_flat.squeeze(-1)               # (S1*B,)

    # Gradient of output w.r.t. each input dimension
    grads_flat = torch.autograd.grad(
        outputs=out_flat.sum(),
        inputs=inp_flat,
        create_graph=False,
    )[0]  # (S1*B, F)

    grads = grads_flat.view(S1, B, F)  # (S+1, B, F)

    # Trapezoidal rule: average over steps axis
    avg_grads = (grads[:-1] + grads[1:]).mean(dim=0) / 2  # (B, F)

    # IG = (x - baseline) * avg_gradients
    diff        = (x - baseline).detach()
    attributions = (avg_grads * diff).detach().numpy()  # (B, F)

    # Completeness check: sum(IG_i) ≈ F(x) - F(x_baseline)
    with torch.no_grad():
        f_x    = model.fraud_probability(x).squeeze().numpy()
        f_base = model.fraud_probability(
            baseline.expand(x.shape[0], -1)
        ).squeeze().numpy()
    delta_pred = f_x - f_base  # (B,)

    return attributions, delta_pred, f_x


# ─────────────────────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────────────────────
def load_data_and_model():
    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURES].values.astype('float32')
    y  = df[LABEL_COL].values.astype('float32')

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    X_train, _, _, _ = train_test_split(
        X_temp, y_temp, test_size=0.17647, random_state=SEED, stratify=y_temp
    )

    # Load normalisation stats from checkpoint (match primary experiment exactly)
    ckpt  = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    mean  = np.array(ckpt['feature_mean'])
    std   = np.array(ckpt['feature_std'])

    X_test_n = (X_test - mean) / std

    model = TrustNetTransformer()
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    return model, X_test_n, X_test, y_test, mean, std


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  TRUSTNET — INTEGRATED GRADIENTS XAI ANALYSIS")
    print("=" * 70)

    # ── Load ──────────────────────────────────────────────────
    model, X_test_n, X_test_raw, y_test, mean, std = load_data_and_model()

    Xt = torch.tensor(X_test_n, dtype=torch.float32)

    # ── Verify checkpoint reproduces primary results ──────────
    with torch.no_grad():
        probs_check = model.fraud_probability(Xt).squeeze().numpy()
    preds = (probs_check >= THRESHOLD).astype(float)
    tp = int(((preds==1)&(y_test==1)).sum()); tn = int(((preds==0)&(y_test==0)).sum())
    fp = int(((preds==1)&(y_test==0)).sum()); fn = int(((preds==0)&(y_test==1)).sum())
    acc  = (tp+tn)/len(y_test)
    prec = tp/(tp+fp) if tp+fp > 0 else 0
    rec  = tp/(tp+fn) if tp+fn > 0 else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0
    ap   = average_precision_score(y_test, probs_check)
    print(f"\n[1/5] Checkpoint verification:")
    print(f"  TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"  Acc={acc*100:.2f}% Prec={prec*100:.2f}% Rec={rec*100:.2f}% "
          f"F1={f1*100:.2f}% AP={ap:.4f}")
    assert tp==143 and tn==434 and fp==5 and fn==18, "Checkpoint mismatch — aborting"
    print("  [OK] Primary results confirmed.")

    # ── Baseline: zero vector in normalised space ─────────────
    # Corresponds to (mean - mean)/std = 0 for each feature,
    # i.e. the training-set average of each feature.
    baseline = torch.zeros(1, len(FEATURES))

    print(f"\n[2/5] Computing Integrated Gradients ({IG_STEPS} steps)...")
    print(f"  Baseline: zero vector in normalised space")
    print(f"  (= training-set mean in original scale: {mean.round(4)})")
    print(f"  Test samples: {len(Xt)}")

    attributions, delta_pred, f_x = integrated_gradients(
        model, Xt, baseline, steps=IG_STEPS
    )
    # attributions: (600, 5)  — signed IG per sample per feature

    # ── Completeness check ────────────────────────────────────
    attr_sum   = attributions.sum(axis=1)
    completeness_err = np.abs(attr_sum - delta_pred)
    print(f"\n[3/5] Completeness check:")
    print(f"  Max  |sum(IG) - (F(x)-F(baseline))| = {completeness_err.max():.6f}")
    print(f"  Mean |sum(IG) - (F(x)-F(baseline))| = {completeness_err.mean():.6f}")
    print(f"  (Values < 0.001 are acceptable for {IG_STEPS} integration steps)")

    # ── Feature importance summary ────────────────────────────
    print(f"\n[4/5] Feature importance summary:")
    mean_abs_attr = np.abs(attributions).mean(axis=0)
    mean_sig_attr = attributions.mean(axis=0)         # signed mean
    total         = mean_abs_attr.sum()
    rel_pct       = (mean_abs_attr / total * 100)

    print(f"  {'Feature':<8} {'MeanAbsAttr':>12} {'RelContrib%':>12} {'MeanSignedAttr':>15}")
    print(f"  {'-'*50}")
    order = np.argsort(mean_abs_attr)[::-1]
    for i in order:
        print(f"  {FEATURES[i]:<8} {mean_abs_attr[i]:>12.6f} {rel_pct[i]:>11.2f}% {mean_sig_attr[i]:>15.6f}")

    # ── Build per-sample attribution dataframe ────────────────
    rows = []
    for idx in range(len(Xt)):
        row = {'sample_index': idx}
        for fi, feat in enumerate(FEATURES):
            row[feat + '_attribution'] = round(float(attributions[idx, fi]), 6)
        row['prediction_probability'] = round(float(f_x[idx]), 6)
        row['predicted_label']        = int(preds[idx])
        row['actual_label']           = int(y_test[idx])
        # Raw (un-normalised) feature values for interpretability
        for fi, feat in enumerate(FEATURES):
            row[feat + '_raw'] = round(float(X_test_raw[idx, fi]), 4)
        rows.append(row)
    attr_df = pd.DataFrame(rows)

    # ── Representative examples ───────────────────────────────
    print(f"\n[5/5] Representative examples (one each of TP, TN, FP, FN):")

    def find_sample(pred_val, actual_val, label):
        mask = (attr_df['predicted_label'] == pred_val) & \
               (attr_df['actual_label']    == actual_val)
        if mask.sum() == 0:
            print(f"  {label}: none found")
            return None
        # Pick the sample closest to median fraud probability
        sub = attr_df[mask]
        median_p = sub['prediction_probability'].median()
        idx = (sub['prediction_probability'] - median_p).abs().idxmin()
        return attr_df.loc[idx]

    rep_samples = {}
    for label, pv, av in [('TP',1,1),('TN',0,0),('FP',1,0),('FN',0,1)]:
        s = find_sample(pv, av, label)
        if s is not None:
            rep_samples[label] = s
            print(f"\n  {label} (sample {int(s['sample_index'])}, "
                  f"p={s['prediction_probability']:.4f}):")
            print(f"    Raw features: " +
                  ", ".join(f"{f}={s[f+'_raw']:.3f}" for f in FEATURES))
            print(f"    Attributions: " +
                  ", ".join(f"{f}={s[f+'_attribution']:.4f}" for f in FEATURES))

    # ── Save outputs ──────────────────────────────────────────
    # 1. Per-sample attributions
    attr_path = os.path.join(XAI_DIR, 'feature_attributions.csv')
    attr_df.to_csv(attr_path, index=False)
    print(f"\n  Saved: {attr_path}")

    # 2. Summary
    summary_rows = []
    for i in range(len(FEATURES)):
        summary_rows.append({
            'feature':                     FEATURES[i],
            'mean_absolute_attribution':   round(float(mean_abs_attr[i]), 6),
            'mean_signed_attribution':     round(float(mean_sig_attr[i]), 6),
            'relative_contribution_percent': round(float(rel_pct[i]), 2),
            'rank':                        int(np.argsort(mean_abs_attr)[::-1].tolist().index(i) + 1),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values('rank').reset_index(drop=True)
    summary_path = os.path.join(XAI_DIR, 'feature_importance_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    # 3. Bar plot — mean absolute attribution
    feat_labels = ['SDI', 'CTI', 'FVRS', 'CFIS', 'CTCS']
    feat_map    = dict(zip(FEATURES, feat_labels))
    ordered_features = [FEATURES[i] for i in order]
    ordered_labels   = [feat_map[f] for f in ordered_features]
    ordered_vals     = [mean_abs_attr[i] for i in order]
    ordered_pct      = [rel_pct[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bar_colors = ['#2c7bb6' if feat_map[f] != 'CTI' else '#d7191c'
                  for f in ordered_features]
    bars = ax.barh(ordered_labels[::-1], ordered_vals[::-1],
                   color=bar_colors[::-1], edgecolor='black', linewidth=0.6)

    # Add percentage labels on bars
    for bar, pct in zip(bars, ordered_pct[::-1]):
        ax.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', ha='left', fontsize=9)

    ax.set_xlabel('Mean Absolute Integrated Gradient Attribution', fontsize=10)
    ax.set_title('Mean Absolute Feature Attributions — TrustNet\n'
                 '(Integrated Gradients, 600-sample held-out test set)', fontsize=10)
    ax.set_xlim(0, max(ordered_vals) * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend note
    ax.text(0.98, 0.02,
            'Note: blue bars = features positively correlated with\n'
            'fraud score on average; red = CTI (higher CTI → lower\n'
            'fraud risk by design). Attribution = model contribution,\n'
            'not causal influence.',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7, style='italic', color='#555555',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.tight_layout()
    plot_path = os.path.join(XAI_DIR, 'mean_attribution_barplot.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {plot_path} (300 dpi)")

    # ── Final summary ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nMethod: Integrated Gradients ({IG_STEPS} Riemann steps)")
    print(f"Baseline: zero vector in normalised space")
    print(f"Test samples: {len(Xt)} (same 600-sample split as primary evaluation)")
    print(f"Completeness max error: {completeness_err.max():.6f}")
    print(f"\nFeature Importance (ranked by mean absolute attribution):")
    print(f"  {'Rank':<5} {'Feature':<8} {'MeanAbsAttr':>12} {'RelContrib%':>12} {'MeanSignedAttr':>15}")
    print(f"  {'-'*55}")
    for row in summary_rows:
        print(f"  {row['rank']:<5} {row['feature']:<8} "
              f"{row['mean_absolute_attribution']:>12.6f} "
              f"{row['relative_contribution_percent']:>11.2f}% "
              f"{row['mean_signed_attribution']:>15.6f}")

    print(f"\nInterpretation:")
    top = summary_df.iloc[0]
    print(f"  {top['feature'].upper()} exhibited the largest mean absolute "
          f"attribution ({top['mean_absolute_attribution']:.6f}, "
          f"{top['relative_contribution_percent']:.1f}% of total), indicating "
          f"it had the largest average contribution to the model output on "
          f"the test set under this attribution method.")
    print(f"\n  A positive mean signed attribution indicates the feature")
    print(f"  pushed predictions toward fraud on average; a negative value")
    print(f"  indicates it pushed toward legitimate on average.")
    print(f"  These attributions reflect learned model behaviour, not")
    print(f"  causal relationships between features and fraud.")

    print("\n" + "=" * 70)
    return summary_df, attr_df


if __name__ == '__main__':
    main()
