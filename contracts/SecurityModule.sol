// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";

/**
 * @title SecurityModule
 * @dev Handles access control, pausing, and fraud flagging.
 *
 * Roles:
 *   owner  → deploys contracts, can pause/unpause, appoint oracle
 *   oracle → trusted backend wallet that pushes trust metric scores
 *
 * ─────────────────────────────────────────────────────────────────
 * CENTRALISATION RISKS & FUTURE MITIGATIONS (referenced in paper)
 * ─────────────────────────────────────────────────────────────────
 * Current design (research prototype on Sepolia testnet):
 *   - A single oracle address (the Python Flask backend wallet)
 *     has exclusive write access to trust scores and freeze/flag
 *     actions. If this server is compromised or unavailable, fraud
 *     enforcement halts — a single point of failure.
 *   - A single owner address can pause the entire platform and
 *     approve unfreezes without requiring consensus from any other
 *     party — a governance centralisation risk.
 *
 * Recommended mitigations for production deployment:
 *   1. Decentralised Oracle (Chainlink Functions):
 *      Replace the Flask oracle wallet with a Chainlink Functions
 *      subscription that executes TrustNet inference inside a
 *      decentralised oracle network (DON). The DON aggregates
 *      results from multiple independent nodes before writing
 *      on-chain, eliminating the single-server dependency and
 *      providing cryptographic proof of computation integrity.
 *      See: https://docs.chain.link/chainlink-functions
 *
 *   2. Multi-Signature Governance (Gnosis Safe / OpenZeppelin Governor):
 *      Replace the single-owner pause/unfreeze permissions with a
 *      Multi-Sig scheme requiring k-of-n keyholders (e.g. 2-of-3)
 *      to approve any enforcement action. This prevents a single
 *      compromised key from freezing or unfreezing all campaigns.
 *      See: https://docs.safe.global/
 *
 * These extensions are left as future work. The current prototype
 * validates the core fraud detection pipeline and enforcement logic.
 * ─────────────────────────────────────────────────────────────────
 */
contract SecurityModule is Ownable, Pausable {

    // ─────────────────────────────────────────────
    // STATE
    // ─────────────────────────────────────────────

    // Oracle address — your Python backend wallet
    address public oracle;

    // Campaigns flagged as fraudulent
    mapping(address => bool) public flaggedCampaigns;

    // Campaigns frozen (funds locked pending review)
    mapping(address => bool) public frozenCampaigns;

    // ─────────────────────────────────────────────
    // EVENTS
    // ─────────────────────────────────────────────

    event OracleUpdated(address indexed oldOracle, address indexed newOracle);
    event CampaignFlagged(address indexed campaign, string reason);
    event CampaignUnflagged(address indexed campaign);
    event CampaignFrozen(address indexed campaign);
    event CampaignUnfrozen(address indexed campaign);

    // ─────────────────────────────────────────────
    // MODIFIERS
    // ─────────────────────────────────────────────

    modifier onlyOracle() {
        require(msg.sender == oracle, "SecurityModule: caller is not the oracle");
        _;
    }

    modifier notFlagged(address campaign) {
        require(!flaggedCampaigns[campaign], "SecurityModule: campaign is flagged");
        _;
    }

    modifier notFrozen(address campaign) {
        require(!frozenCampaigns[campaign], "SecurityModule: campaign is frozen");
        _;
    }

    // ─────────────────────────────────────────────
    // CONSTRUCTOR
    // ─────────────────────────────────────────────

    constructor(address _oracle) Ownable(msg.sender) {
        require(_oracle != address(0), "SecurityModule: invalid oracle address");
        oracle = _oracle;
        emit OracleUpdated(address(0), _oracle);
    }

    // ─────────────────────────────────────────────
    // ORACLE MANAGEMENT
    // ─────────────────────────────────────────────

    function setOracle(address _newOracle) external onlyOwner {
        require(_newOracle != address(0), "SecurityModule: invalid oracle address");
        emit OracleUpdated(oracle, _newOracle);
        oracle = _newOracle;
    }

    // ─────────────────────────────────────────────
    // PAUSE / UNPAUSE
    // ─────────────────────────────────────────────

    function pause() external onlyOwner {
        _pause();
    }

    function unpause() external onlyOwner {
        _unpause();
    }

    // ─────────────────────────────────────────────
    // FRAUD FLAGGING
    // ─────────────────────────────────────────────

    /**
     * @dev Flag a campaign as fraudulent.
     * Called by backend when TrustNet score exceeds threshold.
     */
    function flagCampaign(address campaign, string calldata reason) external {
        require(
            msg.sender == oracle || msg.sender == owner(),
            "SecurityModule: not authorized"
        );
        flaggedCampaigns[campaign] = true;
        emit CampaignFlagged(campaign, reason);
    }

    function unflagCampaign(address campaign) external onlyOwner {
        flaggedCampaigns[campaign] = false;
        emit CampaignUnflagged(campaign);
    }

    // ─────────────────────────────────────────────
    // FUND FREEZING
    // ─────────────────────────────────────────────

    /**
     * @dev Freeze campaign funds pending review.
     * Called automatically when combined risk score is too high.
     */
    function freezeCampaign(address campaign) external {
        require(
            msg.sender == oracle || msg.sender == owner(),
            "SecurityModule: not authorized"
        );
        frozenCampaigns[campaign] = true;
        emit CampaignFrozen(campaign);
    }

    function unfreezeCampaign(address campaign) external onlyOwner {
        frozenCampaigns[campaign] = false;
        emit CampaignUnfrozen(campaign);
    }

    // ─────────────────────────────────────────────
    // VIEW FUNCTIONS
    // ─────────────────────────────────────────────

    function isFlagged(address campaign) external view returns (bool) {
        return flaggedCampaigns[campaign];
    }

    function isFrozen(address campaign) external view returns (bool) {
        return frozenCampaigns[campaign];
    }
}
