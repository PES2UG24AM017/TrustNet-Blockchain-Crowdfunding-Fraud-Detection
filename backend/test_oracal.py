from blockchain import registry, security

print("Registry Oracle :", registry.functions.oracle().call())
print("Security Oracle :", security.functions.oracle().call())