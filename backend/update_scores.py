from web3 import Web3
from blockchain import w3, registry
from config import PRIVATE_KEY, WALLET_ADDRESS

campaign = Web3.to_checksum_address(
    "0xDCdC29B6A4A477d3beAE4618F93535886137796D"
)

# Example AI scores (0-100)
sdi = 90
cti = 85
fvrs = 15
cfis = 10
ctcs = 95

nonce = w3.eth.get_transaction_count(WALLET_ADDRESS)

tx = registry.functions.updateAllScores(
    campaign,
    sdi,
    cti,
    fvrs,
    cfis,
    ctcs
).build_transaction({
    "from": WALLET_ADDRESS,
    "nonce": nonce,
    "gas": 300000,
    "gasPrice": w3.eth.gas_price,
    "chainId": 11155111   # Sepolia
})

signed_tx = w3.eth.account.sign_transaction(
    tx,
    PRIVATE_KEY
)

tx_hash = w3.eth.send_raw_transaction(
    signed_tx.raw_transaction
)

print("Transaction Hash:", tx_hash.hex())

receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

print("Status:", receipt.status)
print("Block:", receipt.blockNumber)