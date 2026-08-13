import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

D_MODEL = 32
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1

THRESHOLD = 0.5

FEATURES = [
    "sdi",
    "cti",
    "fvrs",
    "cfis",
    "ctcs"
]

LABEL_COL = "is_fraud"


# ============================================================
# RANDOM SEED
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUTS_DIR = os.path.join(
    SCRIPT_DIR,
    "..",
    "outputs"
)

DATA_PATH = os.path.join(
    OUTPUTS_DIR,
    "trustnet_train.csv"
)

MODEL_PATH = os.path.join(
    OUTPUTS_DIR,
    "trustnet_transformer_5features.pt"
)

PLOTS_DIR = os.path.join(
    OUTPUTS_DIR,
    "plots"
)

os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)

PR_CURVE_PATH = os.path.join(
    PLOTS_DIR,
    "trustnet_precision_recall_curve.png"
)

CONFUSION_MATRIX_PATH = os.path.join(
    PLOTS_DIR,
    "trustnet_confusion_matrix.png"
)

TRAINING_CURVE_PATH = os.path.join(
    PLOTS_DIR,
    "trustnet_training_curve.png"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("TRUSTNET TRANSFORMER")
print("TRAINING + PR CURVE + CONFUSION MATRIX")
print("=" * 70)

print("\nDevice:", DEVICE)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset:")
print(DATA_PATH)

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}"
    )


df = pd.read_csv(DATA_PATH)


print("\nDataset shape:")
print(df.shape)


print("\nDataset columns:")

for col in df.columns:
    print("-", col)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = FEATURES + [LABEL_COL]

for col in required_columns:

    if col not in df.columns:

        raise ValueError(
            f"\nRequired column '{col}' "
            f"not found in dataset."
        )


# ============================================================
# FEATURES AND LABEL
# ============================================================

X = df[FEATURES].copy()

y = df[LABEL_COL].copy()


X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

y = pd.to_numeric(
    y,
    errors="coerce"
)


X = X.fillna(0)

y = y.fillna(0)


X = X.values.astype(
    np.float32
)

y = y.values.astype(
    np.float32
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\nFeatures used:")

for feature in FEATURES:
    print("-", feature)


total_samples = len(y)

fraud_samples = int(
    np.sum(y == 1)
)

legitimate_samples = int(
    np.sum(y == 0)
)


print(
    "\nTotal samples:",
    total_samples
)

print(
    "Fraud samples:",
    fraud_samples
)

print(
    "Legitimate samples:",
    legitimate_samples
)

print(
    "Fraud percentage:",
    f"{fraud_samples / total_samples * 100:.2f}%"
)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First:
# 85% temporary
# 15% test

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=SEED,
    stratify=y
)


# From the remaining 85%:
# 70/85 = approximately 82.35% train
# 15/85 = approximately 17.65% validation
#
# Overall:
# Train      = 70%
# Validation = 15%
# Test       = 15%

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.17647,
    random_state=SEED,
    stratify=y_temp
)


print("\n")
print("=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Validation samples:",
    len(X_val)
)

print(
    "Test samples:",
    len(X_test)
)


print(
    "\nTraining fraud:",
    int(np.sum(y_train == 1))
)

print(
    "Validation fraud:",
    int(np.sum(y_val == 1))
)

print(
    "Test fraud:",
    int(np.sum(y_test == 1))
)


# ============================================================
# NORMALIZATION
# ============================================================

# IMPORTANT:
# Calculate normalization only from training data.

feature_mean = X_train.mean(
    axis=0
)

feature_std = X_train.std(
    axis=0
)

feature_std[
    feature_std == 0
] = 1.0


X_train = (
    X_train - feature_mean
) / feature_std


X_val = (
    X_val - feature_mean
) / feature_std


X_test = (
    X_test - feature_mean
) / feature_std


# ============================================================
# PYTORCH DATASET
# ============================================================

class TrustNetDataset(Dataset):

    def __init__(
        self,
        X,
        y
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        ).unsqueeze(1)


    def __len__(self):

        return len(self.X)


    def __getitem__(
        self,
        index
    ):

        return (
            self.X[index],
            self.y[index]
        )


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = TrustNetDataset(
    X_train,
    y_train
)

val_dataset = TrustNetDataset(
    X_val,
    y_val
)

test_dataset = TrustNetDataset(
    X_test,
    y_test
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# TRUSTNET TRANSFORMER
# ============================================================

class TrustNetTransformer(
    nn.Module
):

    def __init__(
        self,
        num_features=5,
        d_model=32,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    ):

        super(
            TrustNetTransformer,
            self
        ).__init__()


        # ----------------------------------------------------
        # Linear Projection
        # ----------------------------------------------------

        self.input_projection = nn.Linear(
            1,
            d_model
        )


        # ----------------------------------------------------
        # Positional Encoding
        # ----------------------------------------------------

        self.positional_embedding = nn.Parameter(
            torch.randn(
                1,
                num_features,
                d_model
            ) * 0.02
        )


        # ----------------------------------------------------
        # Transformer Encoder
        # ----------------------------------------------------

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="relu"
            )
        )


        self.transformer_encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )
        )


        # ----------------------------------------------------
        # Classification Head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                d_model,
                d_model // 2
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                d_model // 2,
                1
            )
        )


    def forward(
        self,
        x
    ):

        # [batch, 5]
        #
        # ->
        #
        # [batch, 5, 1]

        x = x.unsqueeze(-1)


        # Linear projection

        tokens = self.input_projection(
            x
        )


        # Positional encoding

        tokens = (
            tokens +
            self.positional_embedding
        )


        # Transformer

        H = self.transformer_encoder(
            tokens
        )


        # Mean pooling

        z = H.mean(
            dim=1
        )


        # Classification head

        logit = self.classifier(
            z
        )


        return logit


# ============================================================
# CREATE MODEL
# ============================================================

model = TrustNetTransformer(
    num_features=5,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)


model = model.to(
    DEVICE
)


print("\nModel:")
print(model)


# ============================================================
# CLASS WEIGHT
# ============================================================

positive_count = np.sum(
    y_train == 1
)

negative_count = np.sum(
    y_train == 0
)


if positive_count > 0:

    pos_weight = (
        negative_count /
        positive_count
    )

else:

    pos_weight = 1.0


print(
    "\nPositive class weight:",
    pos_weight
)


criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [pos_weight],
        dtype=torch.float32,
        device=DEVICE
    )
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# STORE TRAINING HISTORY
# ============================================================

train_losses = []

val_losses = []


# ============================================================
# TRAINING
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING")
print("=" * 70)


for epoch in range(
    EPOCHS
):

    # ========================================================
    # TRAINING
    # ========================================================

    model.train()

    train_loss = 0.0


    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(
            DEVICE
        )

        y_batch = y_batch.to(
            DEVICE
        )


        # Forward pass

        logits = model(
            X_batch
        )


        # Calculate loss

        loss = criterion(
            logits,
            y_batch
        )


        # Clear old gradients

        optimizer.zero_grad()


        # Backpropagation

        loss.backward()


        # Update weights

        optimizer.step()


        train_loss += (
            loss.item()
            *
            len(X_batch)
        )


    train_loss = (
        train_loss /
        len(train_dataset)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    validation_loss = 0.0


    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )


            logits = model(
                X_batch
            )


            loss = criterion(
                logits,
                y_batch
            )


            validation_loss += (
                loss.item()
                *
                len(X_batch)
            )


    validation_loss = (
        validation_loss /
        len(val_dataset)
    )


    # Store losses

    train_losses.append(
        train_loss
    )

    val_losses.append(
        validation_loss
    )


    # Print progress

    if (
        epoch == 0
        or
        (epoch + 1) % 5 == 0
    ):

        print(
            f"Epoch "
            f"{epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: "
            f"{train_loss:.6f} | "
            f"Validation Loss: "
            f"{validation_loss:.6f}"
        )


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "feature_names":
            FEATURES,

        "d_model":
            D_MODEL,

        "num_heads":
            NUM_HEADS,

        "num_layers":
            NUM_LAYERS,

        "feature_mean":
            feature_mean,

        "feature_std":
            feature_std
    },
    MODEL_PATH
)


print("\nModel saved:")
print(MODEL_PATH)


# ============================================================
# TRAINING CURVE
# ============================================================

plt.figure(
    figsize=(7, 5)
)


epochs = range(
    1,
    EPOCHS + 1
)


plt.plot(
    epochs,
    train_losses,
    linewidth=2,
    label="Training Loss"
)


plt.plot(
    epochs,
    val_losses,
    linewidth=2,
    label="Validation Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Binary Cross-Entropy Loss"
)


plt.title(
    "TrustNet Transformer Training Curve"
)


plt.legend()


plt.grid(
    True,
    alpha=0.25
)


plt.tight_layout()


plt.savefig(
    TRAINING_CURVE_PATH,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nTraining curve saved:"
)

print(
    TRAINING_CURVE_PATH
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("TEST EVALUATION")
print("=" * 70)


model.eval()


all_labels = []

all_probabilities = []


with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(
            DEVICE
        )


        logits = model(
            X_batch
        )


        probabilities = torch.sigmoid(
            logits
        )


        all_probabilities.extend(
            probabilities
            .cpu()
            .numpy()
            .flatten()
        )


        all_labels.extend(
            y_batch
            .numpy()
            .flatten()
        )


all_labels = np.array(
    all_labels
)

all_probabilities = np.array(
    all_probabilities
)


# ============================================================
# CLASS PREDICTIONS
# ============================================================

all_predictions = (
    all_probabilities >= THRESHOLD
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


precision_value = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)


recall_value = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)


f1_value = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)


roc_auc = roc_auc_score(
    all_labels,
    all_probabilities
)


average_precision = (
    average_precision_score(
        all_labels,
        all_probabilities
    )
)


print(
    "\nAccuracy:",
    f"{accuracy:.4f}"
)

print(
    "Precision:",
    f"{precision_value:.4f}"
)

print(
    "Recall:",
    f"{recall_value:.4f}"
)

print(
    "F1-score:",
    f"{f1_value:.4f}"
)

print(
    "ROC-AUC:",
    f"{roc_auc:.4f}"
)

print(
    "Average Precision:",
    f"{average_precision:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]


print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)


print("\n")
print(cm)


print("\nTrue Negatives :", TN)
print("False Positives:", FP)
print("False Negatives:", FN)
print("True Positives :", TP)


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(6, 5)
)


plt.imshow(
    cm,
    interpolation="nearest"
)


plt.title(
    "Confusion Matrix - TrustNet Transformer"
)


plt.colorbar()


tick_marks = np.arange(2)


plt.xticks(
    tick_marks,
    ["Legitimate", "Fraud"]
)


plt.yticks(
    tick_marks,
    ["Legitimate", "Fraud"]
)


plt.xlabel(
    "Predicted Label"
)


plt.ylabel(
    "True Label"
)


# Put numbers inside cells

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=14
        )


plt.tight_layout()


plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nConfusion matrix saved:"
)

print(
    CONFUSION_MATRIX_PATH
)


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

precision, recall, thresholds = (
    precision_recall_curve(
        all_labels,
        all_probabilities
    )
)


fraud_prevalence = (
    all_labels.mean()
)


plt.figure(
    figsize=(7, 5.5)
)


plt.plot(
    recall,
    precision,
    linewidth=2,
    label=(
        f"TrustNet Transformer "
        f"(AP = {average_precision:.3f})"
    )
)


plt.axhline(
    fraud_prevalence,
    linestyle="--",
    linewidth=1.5,
    label=(
        f"Fraud prevalence = "
        f"{fraud_prevalence:.3f}"
    )
)


plt.xlabel(
    "Recall"
)


plt.ylabel(
    "Precision"
)


plt.title(
    "Precision-Recall Curve "
    "of TrustNet Transformer"
)


plt.xlim(
    0,
    1
)


plt.ylim(
    0,
    1.05
)


plt.grid(
    True,
    alpha=0.25
)


plt.legend(
    loc="lower left"
)


plt.tight_layout()


plt.savefig(
    PR_CURVE_PATH,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nPR curve saved:"
)

print(
    PR_CURVE_PATH
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)


print(
    "\nAverage Precision:",
    f"{average_precision:.4f}"
)

print(
    "ROC-AUC:",
    f"{roc_auc:.4f}"
)

print(
    "Accuracy:",
    f"{accuracy:.4f}"
)

print(
    "Precision:",
    f"{precision_value:.4f}"
)

print(
    "Recall:",
    f"{recall_value:.4f}"
)

print(
    "F1-score:",
    f"{f1_value:.4f}"
)


print("\nFiles generated:")

print(
    "1.",
    TRAINING_CURVE_PATH
)

print(
    "2.",
    CONFUSION_MATRIX_PATH
)

print(
    "3.",
    PR_CURVE_PATH
)

print("\nDone.")