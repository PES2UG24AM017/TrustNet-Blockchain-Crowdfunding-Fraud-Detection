// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "./SecurityModule.sol";
import "./TrustMetricsRegistry.sol";

/**
 * @title CrowdfundingCore
 * @dev Main crowdfunding contract. Handles campaign creation,
 * contributions, fund release, and refunds.
 *
 * Integrates with:
 *   SecurityModule       → checks if campaign is flagged/frozen
 *   TrustMetricsRegistry → checks combined risk score before
 *                          releasing funds
 */
contract CrowdfundingCore is ReentrancyGuard {

    // ─────────────────────────────────────────────
    // STRUCTS
    // ─────────────────────────────────────────────

    struct Campaign {
        address payable creator;    // campaign creator wallet
        string  title;              // campaign title
        string  description;        // campaign description (for SDI)
        uint256 goal;               // funding goal in wei
        uint256 deadline;           // unix timestamp
        uint256 amountRaised;       // total ETH raised
        uint256 contributorCount;   // number of unique contributors
        bool    goalReached;        // whether goal was met
        bool    fundsReleased;      // whether creator was paid
        bool    exists;             // whether campaign exists
    }

    // ─────────────────────────────────────────────
    // STATE
    // ─────────────────────────────────────────────

    // Contract references
    SecurityModule       public securityModule;
    TrustMetricsRegistry public trustRegistry;

    // Campaign ID counter
    uint256 public campaignCount;

    // Campaign ID → Campaign
    mapping(uint256 => Campaign) public campaigns;

    // Campaign ID → contributor address → amount contributed
    mapping(uint256 => mapping(address => uint256)) public contributions;

    // Campaign ID → list of contributor addresses (for CFIS)
    mapping(uint256 => address[]) public contributorList;

    // ─────────────────────────────────────────────
    // EVENTS
    // ─────────────────────────────────────────────

    event CampaignCreated(
        uint256 indexed campaignId,
        address indexed creator,
        string  title,
        uint256 goal,
        uint256 deadline
    );
    event ContributionReceived(
        uint256 indexed campaignId,
        address indexed contributor,
        uint256 amount,
        uint256 timestamp
    );
    event FundsReleased(
        uint256 indexed campaignId,
        address indexed creator,
        uint256 amount
    );
    event RefundIssued(
        uint256 indexed campaignId,
        address indexed contributor,
        uint256 amount
    );
    event FundsFrozen(
        uint256 indexed campaignId,
        uint256 combinedRisk
    );

    // ─────────────────────────────────────────────
    // CONSTRUCTOR
    // ─────────────────────────────────────────────

    constructor(
        address _securityModule,
        address _trustRegistry
    ) {
        require(_securityModule != address(0), "Invalid security module");
        require(_trustRegistry  != address(0), "Invalid trust registry");
        securityModule = SecurityModule(_securityModule);
        trustRegistry  = TrustMetricsRegistry(_trustRegistry);
    }

    // ─────────────────────────────────────────────
    // CAMPAIGN CREATION
    // ─────────────────────────────────────────────

    /**
     * @dev Create a new crowdfunding campaign.
     * Description is stored on-chain so SDI can reference it.
     */
    function createCampaign(
        string calldata _title,
        string calldata _description,
        uint256 _goal,
        uint256 _durationDays
    ) external returns (uint256) {
        require(bytes(_title).length       > 0,  "Title required");
        require(bytes(_description).length > 0,  "Description required");
        require(_goal                      > 0,  "Goal must be > 0");
        require(_durationDays              > 0,  "Duration must be > 0");
        require(_durationDays              <= 90, "Max duration is 90 days");

        // Check platform is not paused
        require(
            !securityModule.paused(),
            "CrowdfundingCore: platform is paused"
        );

        uint256 campaignId = campaignCount++;
        uint256 deadline   = block.timestamp + (_durationDays * 1 days);

        campaigns[campaignId] = Campaign({
            creator:          payable(msg.sender),
            title:            _title,
            description:      _description,
            goal:             _goal,
            deadline:         deadline,
            amountRaised:     0,
            contributorCount: 0,
            goalReached:      false,
            fundsReleased:    false,
            exists:           true
        });

        emit CampaignCreated(campaignId, msg.sender, _title, _goal, deadline);
        return campaignId;
    }

    // ─────────────────────────────────────────────
    // CONTRIBUTIONS
    // ─────────────────────────────────────────────

    /**
     * @dev Contribute ETH to a campaign.
     * Logs timestamp and contributor address for
     * FVRS and CFIS metric computation.
     */
    function contribute(uint256 _campaignId)
        external
        payable
        nonReentrant
    {
        require(msg.value > 0, "Contribution must be > 0");

        Campaign storage campaign = campaigns[_campaignId];
        require(campaign.exists,                           "Campaign does not exist");
        require(block.timestamp < campaign.deadline,       "Campaign has ended");
        require(!campaign.fundsReleased,                   "Funds already released");

        // Check platform not paused
        require(
            !securityModule.paused(),
            "CrowdfundingCore: platform is paused"
        );

        // Check campaign not flagged
        require(
            !securityModule.isFlagged(campaign.creator),
            "CrowdfundingCore: campaign is flagged as fraudulent"
        );

        // Check campaign not frozen
        require(
            !securityModule.isFrozen(campaign.creator),
            "CrowdfundingCore: campaign is frozen"
        );

        // Track new contributors
        if (contributions[_campaignId][msg.sender] == 0) {
            campaign.contributorCount++;
            contributorList[_campaignId].push(msg.sender);
        }

        contributions[_campaignId][msg.sender] += msg.value;
        campaign.amountRaised                  += msg.value;

        if (campaign.amountRaised >= campaign.goal) {
            campaign.goalReached = true;
        }

        // Emit event with timestamp for FVRS computation
        emit ContributionReceived(
            _campaignId,
            msg.sender,
            msg.value,
            block.timestamp
        );
    }

    // ─────────────────────────────────────────────
    // FUND RELEASE
    // ─────────────────────────────────────────────

    /**
     * @dev Release funds to campaign creator.
     * Checks TrustMetricsRegistry before releasing.
     * If risk score too high → freeze instead of release.
     */
    function releaseFunds(uint256 _campaignId)
        external
        nonReentrant
    {
        Campaign storage campaign = campaigns[_campaignId];
        require(campaign.exists,                     "Campaign does not exist");
        require(msg.sender == campaign.creator,      "Only creator can release funds");
        require(campaign.goalReached,                "Funding goal not reached");
        require(!campaign.fundsReleased,             "Funds already released");

        // Check if frozen
        require(
            !securityModule.isFrozen(campaign.creator),
            "CrowdfundingCore: campaign funds are frozen"
        );

        // Check combined risk score from TrustMetricsRegistry
        uint256 riskScore = trustRegistry.getCombinedRisk(campaign.creator);

        if (riskScore >= trustRegistry.riskThreshold()) {
            // Auto-freeze if risk too high
            securityModule.freezeCampaign(campaign.creator);
            emit FundsFrozen(_campaignId, riskScore);
            revert("CrowdfundingCore: high fraud risk, funds frozen for review");
        }

        // Safe to release
        campaign.fundsReleased = true;
        uint256 amount         = campaign.amountRaised;

        (bool success, ) = payable(campaign.creator).call{value: amount}("");
        require(success, "Transfer failed");
        emit FundsReleased(_campaignId, campaign.creator, amount);
    }

    // ─────────────────────────────────────────────
    // REFUNDS
    // ─────────────────────────────────────────────

    /**
     * @dev Refund contributor if goal not met after deadline,
     * or if campaign was confirmed fraudulent by admin.
     */
    function refund(uint256 _campaignId) external nonReentrant {
        Campaign storage campaign = campaigns[_campaignId];
        require(campaign.exists, "Campaign does not exist");

        bool    isFraud       = securityModule.isFlagged(campaign.creator);
        bool    goalFailed    = block.timestamp >= campaign.deadline
                                && !campaign.goalReached;

        require(
            isFraud || goalFailed,
            "Refund not available: campaign active or goal reached"
        );
        require(!campaign.fundsReleased, "Funds already released");

        uint256 amount = contributions[_campaignId][msg.sender];
        require(amount > 0, "No contribution to refund");

        contributions[_campaignId][msg.sender] = 0;
        campaign.amountRaised -= amount;
        (bool success, ) = payable(msg.sender).call{value: amount}("");
        require(success, "Refund failed");

        emit RefundIssued(_campaignId, msg.sender, amount);
    }

    // ─────────────────────────────────────────────
    // VIEW FUNCTIONS
    // ─────────────────────────────────────────────

    function getCampaign(uint256 _campaignId)
        external
        view
        returns (
            address creator,
            string  memory title,
            string  memory description,
            uint256 goal,
            uint256 deadline,
            uint256 amountRaised,
            uint256 contributorCount,
            bool    goalReached,
            bool    fundsReleased
        )
    {
        Campaign memory c = campaigns[_campaignId];
        require(c.exists, "Campaign does not exist");
        return (
            c.creator,
            c.title,
            c.description,
            c.goal,
            c.deadline,
            c.amountRaised,
            c.contributorCount,
            c.goalReached,
            c.fundsReleased
        );
    }

    function getContributors(uint256 _campaignId)
        external
        view
        returns (address[] memory)
    {
        return contributorList[_campaignId];
    }

    function getContribution(uint256 _campaignId, address _contributor)
        external
        view
        returns (uint256)
    {
        return contributions[_campaignId][_contributor];
    }
}
