"""
Unfreeze and unflag all campaigns on CrowdfundingCore.
Run once: python unfreeze_all.py
"""
import json, time
from blockchain import w3, security
from config import WALLET_ADDRESS, PRIVATE_KEY, CROWDFUNDING_CORE
from web3 import Web3

with open('abi/CrowdfundingCore.json') as f:
    abi = json.load(f)['abi']
core = w3.eth.contract(address=Web3.to_checksum_address(CROWDFUNDING_CORE), abi=abi)

count = core.functions.campaignCount().call()
print("Total campaigns:", count)

def send_and_wait(fn):
    nonce = w3.eth.get_transaction_count(WALLET_ADDRESS, 'pending')
    tx = fn.build_transaction({
        'from': WALLET_ADDRESS, 'nonce': nonce,
        'gas': 150000,
        'gasPrice': int(w3.eth.gas_price * 1.5),
        'chainId': 11155111
    })
    signed  = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print("  TX:", tx_hash.hex()[:20] + "...")
    for _ in range(30):
        time.sleep(4)
        try:
            r = w3.eth.get_transaction_receipt(tx_hash)
            if r:
                print("  Confirmed block:", r.blockNumber, "status:", r.status)
                return r
        except Exception:
            pass
    print("  Timed out waiting for receipt")
    return None

seen_creators = set()
for i in range(count):
    c       = core.functions.getCampaign(i).call()
    creator = Web3.to_checksum_address(c[0])
    title   = c[1]

    frozen  = security.functions.isFrozen(creator).call()
    flagged = security.functions.isFlagged(creator).call()
    print("Campaign", i, title, "| frozen:", frozen, "| flagged:", flagged)

    if frozen and creator not in seen_creators:
        print("  Unfreezing", creator)
        send_and_wait(security.functions.unfreezeCampaign(creator))
        seen_creators.add(creator)

    if flagged:
        print("  Unflagging", creator)
        send_and_wait(security.functions.unflagCampaign(creator))

print("\nFinal state check:")
for i in range(count):
    c       = core.functions.getCampaign(i).call()
    creator = Web3.to_checksum_address(c[0])
    print("Campaign", i, c[1], "| frozen:", security.functions.isFrozen(creator).call(),
          "| flagged:", security.functions.isFlagged(creator).call())

print("Done.")
