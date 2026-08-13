from huggingface_hub import hf_hub_download
import pandas as pd
import shutil
import os

print("Downloading Gitcoin Grants dataset...")

file_path = hf_hub_download(
    repo_id="Poupou/Gitcoin-Grant-DataBuilder",
    filename="df_application_normalized.csv",
    repo_type="dataset"
)

# Copy to your Datasets folder
dest = "gitcoin_grants.csv"
shutil.copy(file_path, dest)

# Quick check
df = pd.read_csv(dest)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nSample description:")
print(df['project_decription'].dropna().iloc[0][:200])