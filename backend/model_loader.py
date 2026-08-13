import torch
import torch.nn as nn

# Model parameters (same values used during training)
D_MODEL = 32
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1   # If your training file uses a different value, replace this.

class TrustNetTransformer(nn.Module):
    def __init__(
        self,
        num_features,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ):
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
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        tokens = self.input_projection(x)
        tokens = tokens + self.positional_embedding
        attended = self.transformer_encoder(tokens)
        pooled = attended.mean(dim=1)
        return self.classifier(pooled)

    def predict_proba(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


# ------------------------
# Load checkpoint
# ------------------------

checkpoint = torch.load(
    "../outputs/trustnet_transformer.pt",
    map_location="cpu",
)

model = TrustNetTransformer(
    num_features=len(checkpoint["feature_names"]),
    d_model=checkpoint["d_model"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"],
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded successfully")
print("Features:", len(checkpoint["feature_names"]))

import torch

# Create a dummy feature vector with 33 features
sample = torch.rand(1, len(checkpoint["feature_names"]))

# Get fraud probability
probability = model.predict_proba(sample)

print("\nFraud Probability:")
print(probability.item())