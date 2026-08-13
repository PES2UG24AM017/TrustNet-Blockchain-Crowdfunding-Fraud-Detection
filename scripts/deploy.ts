import { network } from "hardhat";

async function main() {
  const { viem } = await network.connect();

  const publicClient = await viem.getPublicClient();
  const [walletClient] = await viem.getWalletClients();

  console.log("Deploying SecurityModule...");

  const securityModule = await viem.deployContract("SecurityModule", [
    "0xDCdC29B6A4A477d3beAE4618F93535886137796D",
  ]);

  await publicClient.waitForTransactionReceipt({
    hash: securityModule.deploymentTransaction.hash,
  });

  console.log("SecurityModule:", securityModule.address);

  console.log("Deploying TrustMetricsRegistry...");

  const trustRegistry = await viem.deployContract("TrustMetricsRegistry", [
    "0xDCdC29B6A4A477d3beAE4618F93535886137796D",
  ]);

  await publicClient.waitForTransactionReceipt({
    hash: trustRegistry.deploymentTransaction.hash,
  });

  console.log("TrustMetricsRegistry:", trustRegistry.address);

  console.log("Deploying CrowdfundingCore...");

  const crowdfundingCore = await viem.deployContract("CrowdfundingCore", [
    securityModule.address,
    trustRegistry.address,
  ]);

  await publicClient.waitForTransactionReceipt({
    hash: crowdfundingCore.deploymentTransaction.hash,
  });

  console.log("CrowdfundingCore:", crowdfundingCore.address);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});