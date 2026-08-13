import json
from web3 import Web3
from blockchain import w3
from config import CROWDFUNDING_CORE

with open("abi/CrowdfundingCore.json", "r") as f:
    core_json = json.load(f)

core = w3.eth.contract(
    address=Web3.to_checksum_address(CROWDFUNDING_CORE),
    abi=core_json["abi"]
)

count = core.functions.campaignCount().call()
print("Campaign Count:", count)