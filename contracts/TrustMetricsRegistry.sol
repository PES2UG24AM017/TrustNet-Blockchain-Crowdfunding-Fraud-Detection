// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title TrustMetricsRegistry
 * @dev Stores all 5 trust metric scores per campaign.
 * Scores are pushed by the trusted oracle (Python backend).
 *
 * All scores stored as integers scaled by 100
 * e.g. 0.73 → 73, 1.00 → 100
 *
 * Metrics:
 *   SDI  → Semantic Drift Index      (plagiarism detection)
 *   CTI  → Crowd Trust Index         (campaign credibility)
 *   FVRS → Funding Velocity Risk     (abnormal funding speed)
 *   CFIS → Collusive Funding Index   (coordinated fraud)
 *   CTCS → Cross Campaign Temporal   (pattern correlation)
 */
contract TrustMetricsRegistry is Ownable {

    // ─────────────────────────────────────────────
    // STRUCTS
    // ─────────────────────────────────────────────

    struct TrustScores {
        uint256 sdi;    // 0-100 (high = plagiarism risk)
        uint256 cti;    // 0-100 (high = trusted campaign)
        uint256 fvrs;   // 0-100 (high = abnormal velocity)
        uint256 cfis;   // 0-100 (high = collusion risk)
        uint256 ctcs;   // 0-100 (high = temporal anomaly)
        uint256 lastUpdated;  // block timestamp of last update
        bool    exists;       // whether scores have been set
    }

    // ─────────────────────────────────────────────
    // STATE
    // ─────────────────────────────────────────────

    // Oracle address — Python backend wallet
    address public oracle;

    // Campaign address → trust scores
    mapping(address => TrustScores) public scores;

    // Combined risk threshold (0-100)
    // Above this → campaign gets frozen
    uint256 public riskThreshold = 70;

    // ─────────────────────────────────────────────
    // EVENTS
    // ─────────────────────────────────────────────

    event ScoresUpdated(
        address indexed campaign,
        uint256 sdi,
        uint256 cti,
        uint256 fvrs,
        uint256 cfis,
        uint256 ctcs,
        uint256 combinedRisk
    );
    event OracleUpdated(address indexed oldOracle, address indexed newOracle);
    event ThresholdUpdated(uint256 oldThreshold, uint256 newThreshold);

    // ─────────────────────────────────────────────
    // MODIFIERS
    // ─────────────────────────────────────────────

    modifier onlyOracle() {
        require(
            msg.sender == oracle,
            "TrustMetricsRegistry: caller is not the oracle"
        );
        _;
    }

    // ─────────────────────────────────────────────
    // CONSTRUCTOR
    // ─────────────────────────────────────────────

    constructor(address _oracle) Ownable(msg.sender) {
        require(_oracle != address(0), "TrustMetricsRegistry: invalid oracle");
        oracle = _oracle;
        emit OracleUpdated(address(0), _oracle);
    }

    // ─────────────────────────────────────────────
    // ORACLE MANAGEMENT
    // ─────────────────────────────────────────────

    function setOracle(address _newOracle) external onlyOwner {
        require(_newOracle != address(0), "TrustMetricsRegistry: invalid oracle");
        emit OracleUpdated(oracle, _newOracle);
        oracle = _newOracle;
    }

    function setRiskThreshold(uint256 _threshold) external onlyOwner {
        require(_threshold <= 100, "TrustMetricsRegistry: threshold must be <= 100");
        emit ThresholdUpdated(riskThreshold, _threshold);
        riskThreshold = _threshold;
    }

    // ─────────────────────────────────────────────
    // SCORE UPDATES (called by oracle/backend)
    // ─────────────────────────────────────────────

    /**
     * @dev Push all 5 scores for a campaign at once.
     * Called by Python backend after computing all metrics.
     */
    function updateAllScores(
        address campaign,
        uint256 _sdi,
        uint256 _cti,
        uint256 _fvrs,
        uint256 _cfis,
        uint256 _ctcs
    ) external onlyOracle {
        require(campaign != address(0), "TrustMetricsRegistry: invalid campaign");
        require(_sdi  <= 100, "SDI out of range");
        require(_cti  <= 100, "CTI out of range");
        require(_fvrs <= 100, "FVRS out of range");
        require(_cfis <= 100, "CFIS out of range");
        require(_ctcs <= 100, "CTCS out of range");

        scores[campaign] = TrustScores({
            sdi:         _sdi,
            cti:         _cti,
            fvrs:        _fvrs,
            cfis:        _cfis,
            ctcs:        _ctcs,
            lastUpdated: block.timestamp,
            exists:      true
        });

        uint256 risk = getCombinedRisk(campaign);
        emit ScoresUpdated(campaign, _sdi, _cti, _fvrs, _cfis, _ctcs, risk);
    }

    /**
     * @dev Update individual scores (for partial updates)
     */
    function updateSDI(address campaign, uint256 _sdi) external onlyOracle {
        require(_sdi <= 100, "SDI out of range");
        scores[campaign].sdi         = _sdi;
        scores[campaign].lastUpdated = block.timestamp;
        scores[campaign].exists      = true;
    }

    function updateCTI(address campaign, uint256 _cti) external onlyOracle {
        require(_cti <= 100, "CTI out of range");
        scores[campaign].cti         = _cti;
        scores[campaign].lastUpdated = block.timestamp;
        scores[campaign].exists      = true;
    }

    function updateFVRS(address campaign, uint256 _fvrs) external onlyOracle {
        require(_fvrs <= 100, "FVRS out of range");
        scores[campaign].fvrs        = _fvrs;
        scores[campaign].lastUpdated = block.timestamp;
        scores[campaign].exists      = true;
    }

    function updateCFIS(address campaign, uint256 _cfis) external onlyOracle {
        require(_cfis <= 100, "CFIS out of range");
        scores[campaign].cfis        = _cfis;
        scores[campaign].lastUpdated = block.timestamp;
        scores[campaign].exists      = true;
    }

    function updateCTCS(address campaign, uint256 _ctcs) external onlyOracle {
        require(_ctcs <= 100, "CTCS out of range");
        scores[campaign].ctcs        = _ctcs;
        scores[campaign].lastUpdated = block.timestamp;
        scores[campaign].exists      = true;
    }

    // ─────────────────────────────────────────────
    // RISK CALCULATION
    // ─────────────────────────────────────────────

    /**
     * @dev Compute combined risk score (0-100).
     *
     * SDI  → high score = HIGH risk  (weight: 25%)
     * CTI  → high score = LOW risk   (inverted, weight: 20%)
     * FVRS → high score = HIGH risk  (weight: 20%)
     * CFIS → high score = HIGH risk  (weight: 20%)
     * CTCS → high score = HIGH risk  (weight: 15%)
     *
     * Combined = (SDI*25 + (100-CTI)*20 + FVRS*20 + CFIS*20 + CTCS*15) / 100
     */
    function getCombinedRisk(address campaign)
        public
        view
        returns (uint256)
    {
        TrustScores memory s = scores[campaign];
        if (!s.exists) return 0;

        uint256 ctiRisk = 100 - s.cti; // invert CTI (high CTI = low risk)

        uint256 combined = (
            s.sdi    * 25 +
            ctiRisk  * 20 +
            s.fvrs   * 20 +
            s.cfis   * 20 +
            s.ctcs   * 15
        ) / 100;

        return combined;
    }

    /**
     * @dev Check if a campaign exceeds the risk threshold
     */
    function isHighRisk(address campaign) external view returns (bool) {
        return getCombinedRisk(campaign) >= riskThreshold;
    }

    // ─────────────────────────────────────────────
    // VIEW FUNCTIONS
    // ─────────────────────────────────────────────

    function getScores(address campaign)
        external
        view
        returns (
            uint256 sdi,
            uint256 cti,
            uint256 fvrs,
            uint256 cfis,
            uint256 ctcs,
            uint256 combinedRisk,
            uint256 lastUpdated
        )
    {
        TrustScores memory s = scores[campaign];
        return (
            s.sdi,
            s.cti,
            s.fvrs,
            s.cfis,
            s.ctcs,
            getCombinedRisk(campaign),
            s.lastUpdated
        );
    }
}
