import torch

checkpoint = torch.load(
    "../outputs/trustnet_transformer.pt",
    map_location="cpu"
)

print("d_model:", checkpoint["d_model"])
print("num_heads:", checkpoint["num_heads"])
print("num_layers:", checkpoint["num_layers"])
print("Number of features:", len(checkpoint["feature_names"]))