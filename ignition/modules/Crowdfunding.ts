import { buildModule } from "@nomicfoundation/hardhat-ignition/modules";

const CrowdfundingModule = buildModule("CrowdfundingModule", (m) => {
  // Your wallet address (oracle)
  const oracle = "0xDCdC29B6A4A477d3beAE4618F93535886137796D";

  const securityModule = m.contract("SecurityModule", [oracle]);

  const trustRegistry = m.contract("TrustMetricsRegistry", [oracle]);

  const crowdfundingCore = m.contract("CrowdfundingCore", [
    securityModule,
    trustRegistry,
  ]);

  return {
    securityModule,
    trustRegistry,
    crowdfundingCore,
  };
});

export default CrowdfundingModule;