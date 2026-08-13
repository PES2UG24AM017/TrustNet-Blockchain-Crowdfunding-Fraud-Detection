import json

with open("abi/CrowdfundingCore.json", "r") as f:
    data = json.load(f)

print("Available functions:\n")

for item in data["abi"]:
    if item["type"] == "function":
        print(item["name"])