import json
from web3 import Web3
from config import *

# Connect to Sepolia
w3 = Web3(Web3.HTTPProvider(RPC_URL))

if w3.is_connected():
    print("Connected to Sepolia")
else:
    print("Connection Failed")
    exit()

# Load ABI files
with open("abi/TrustMetricsRegistry.json", "r") as f:
    registry_json = json.load(f)

with open("abi/SecurityModule.json", "r") as f:
    security_json = json.load(f)

# Hardhat artifact format: ABI is inside the "abi" field
registry_abi = registry_json["abi"]
security_abi = security_json["abi"]

# Create contract objects
registry = w3.eth.contract(
    address=Web3.to_checksum_address(TRUST_METRICS_REGISTRY),
    abi=registry_abi
)

security = w3.eth.contract(
    address=Web3.to_checksum_address(SECURITY_MODULE),
    abi=security_abi
)

print("Contracts Loaded Successfully")