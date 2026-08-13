import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Blockchain Configuration
RPC_URL = os.getenv("SEPOLIA_RPC_URL")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS")

# Contract Addresses
TRUST_METRICS_REGISTRY = os.getenv("TRUST_METRICS_REGISTRY")
SECURITY_MODULE = os.getenv("SECURITY_MODULE")
CROWDFUNDING_CORE = os.getenv("CROWDFUNDING_CORE")