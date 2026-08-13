import torch
import pandas as pd
import torch.nn as nn

# ----------------------------
# Model Definition
# ----------------------------

D_MODEL = 32
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1


class TrustNetTransformer(nn.Module):
    def __init__(self, num_features,
                 d_model=D_MODEL,
                 num_heads=NUM_HEADS,
                 num_layers=NUM_LAYERS,
                 dropout=DROPOUT):
        super().__init__()

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
            activation="relu"
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_projection(x)
        x = x + self.positional_embedding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

    def predict_proba(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


# ----------------------------
# Load model
# ----------------------------

checkpoint = torch.load(
    "../outputs/trustnet_transformer.pt",
    map_location="cpu"
)

model = TrustNetTransformer(
    num_features=len(checkpoint["feature_names"]),
    d_model=checkpoint["d_model"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ----------------------------
# Load dataset
# ----------------------------

df = pd.read_csv("../outputs/trustnet_dataset.csv")

print("Dataset shape:", df.shape)

feature_names = checkpoint["feature_names"]

X = df[feature_names]

sample = torch.tensor(
    X.iloc[0].values,
    dtype=torch.float32
).unsqueeze(0)

prob = model.predict_proba(sample)

print("Fraud Probability:", prob.item())