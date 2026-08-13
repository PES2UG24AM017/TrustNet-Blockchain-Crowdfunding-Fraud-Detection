"""
Diagnose why releaseFunds() is blocked for each campaign.
Run: python diagnose_release.py
"""
import json, time
from web3 import Web3
from blockchain import w3, security, registry
from config import CROWDFUNDING_CORE, WALLET_ADDRESS

with open('abi/CrowdfundingCore.json') as f:
    abi = json.load(f)['abi']
core = w3.eth.contract(address=Web3.to_checksum_address(CROWDFUNDING_CORE), abi=abi)

count   = core.functions.campaignCount().call()
thresh  = registry.functions.riskThreshold().call()
paused  = security.functions.paused().call()
now     = int(time.time())

print("=== Campaign Release Diagnostics ===")
print("Platform paused:", paused)
print("Risk threshold :", thresh)
print("Total campaigns:", count)
print()

for i in range(count):
    c = core.functions.getCampaign(i).call()
    creator        = c[0]
    title          = c[1]
    goal           = c[3]
    deadline       = c[4]
    raised         = c[5]
    goal_reached   = c[7]
    funds_released = c[8]

    frozen  = security.functions.isFrozen(creator).call()
    flagged = security.functions.isFlagged(creator).call()
    risk    = registry.functions.getCombinedRisk(creator).call()

    goal_eth   = float(Web3.from_wei(goal,   'ether'))
    raised_eth = float(Web3.from_wei(raised, 'ether'))

    print(f"Campaign #{i}: {title}")
    print(f"  Creator       : {creator}")
    print(f"  Raised / Goal : {raised_eth:.4f} / {goal_eth:.4f} ETH")
    print(f"  Goal reached  : {goal_reached}")
    print(f"  Funds released: {funds_released}")
    print(f"  Frozen        : {frozen}")
    print(f"  Flagged       : {flagged}")
    print(f"  Risk / Thresh : {risk} / {thresh}")
    print(f"  Deadline past : {now >= deadline}")

    reasons = []
    if not goal_reached:    reasons.append("Goal NOT reached")
    if funds_released:      reasons.append("Already released")
    if frozen:              reasons.append("Campaign is FROZEN")
    if flagged:             reasons.append("Campaign is FLAGGED")
    if paused:              reasons.append("Platform is PAUSED")
    if risk >= thresh:      reasons.append(f"Risk {risk} >= threshold {thresh}")

    if reasons:
        print(f"  STATUS: BLOCKED — {', '.join(reasons)}")
    elif creator.lower() != WALLET_ADDRESS.lower():
        print(f"  STATUS: OK — but must be called by creator ({creator[:10]}...), not oracle")
    else:
        print(f"  STATUS: READY TO RELEASE")
    print()
