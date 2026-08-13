import json

with open("abi/TrustMetricsRegistry.json", "r") as f:
    abi = json.load(f)["abi"]

for item in abi:
    if item["type"] == "function" and item["name"] == "updateAllScores":
        print(item)