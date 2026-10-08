/* TrustNet-5 Dashboard — app.js */
'use strict';

const API = 'http://localhost:5000';
const SEPOLIA_ID = '0xaa36a7';

let walletAddress = null;
let ownerAddress  = null;   // set from /health — only this wallet can unfreeze/release
let lastDetectionResult = null;
let allCampaigns = [];

// Returns true only when the connected MetaMask account is the contract owner
function isOwner() {
  return ownerAddress && walletAddress &&
    walletAddress.toLowerCase() === ownerAddress.toLowerCase();
}

// CSV analysis state — declared here so showPage() can reference them safely
let _csvFile     = null;
let _csvResults  = [];
let _csvFiltered = [];

// ═══════════════════════════════════════
// CLOCK
// ═══════════════════════════════════════
function tickClock() {
  const el = document.getElementById('topbar-time');
  if (el) el.textContent = new Date().toLocaleTimeString('en-US', { hour12: false });
}
setInterval(tickClock, 1000);
tickClock();

// ═══════════════════════════════════════
// METAMASK — only login method
// ═══════════════════════════════════════
async function connectWallet() {
  const btn      = document.getElementById('mm-btn');
  const statusEl = document.getElementById('auth-status');

  function setStatus(msg, type) {
    statusEl.textContent = msg;
    statusEl.className   = 'auth-status-box ' + type;
    statusEl.classList.remove('hidden');
  }

  if (typeof window.ethereum === 'undefined') {
    setStatus('⚠ MetaMask not found. Please install the extension from metamask.io', 'error');
    return;
  }

  if (btn) { btn.disabled = true; btn.querySelector('.mm-btn-text').textContent = 'Connecting…'; }
  setStatus('⏳ Requesting account access…', 'connecting');

  try {
    // wallet_requestPermissions forces MetaMask to show the account picker
    // even if the site was previously approved — user must actively select/confirm
    await window.ethereum.request({
      method: 'wallet_requestPermissions',
      params: [{ eth_accounts: {} }],
    });
    const accounts = await window.ethereum.request({ method: 'eth_accounts' });
    if (!accounts.length) throw new Error('No account selected.');
    walletAddress = accounts[0];

    setStatus('⏳ Switching to Sepolia…', 'connecting');
    try {
      await window.ethereum.request({
        method: 'wallet_switchEthereumChain',
        params: [{ chainId: SEPOLIA_ID }],
      });
    } catch (_) {
      setStatus('⚠ Please switch MetaMask to Sepolia testnet manually.', 'error');
    }

    setStatus('✓ Connected — loading dashboard…', 'success');
    setTimeout(() => onWalletConnected(walletAddress), 600);

  } catch (err) {
    setStatus('✗ ' + (err.message || 'Connection rejected.'), 'error');
    if (btn) { btn.disabled = false; btn.querySelector('.mm-btn-text').textContent = 'Connect with MetaMask'; }
  }
}

function logout() {
  walletAddress = null;

  // Mark as logged out — blocks auto-reconnect until user explicitly clicks Connect
  sessionStorage.setItem('trustnet_logged_out', '1');

  // Hide app, show login screen
  document.getElementById('app').classList.add('hidden');
  document.getElementById('auth-screen').classList.remove('hidden');

  // Scroll to top so the auth screen is fully visible
  window.scrollTo({ top: 0, behavior: 'instant' });
  document.body.style.overflow = '';

  // Reset sidebar
  const dot = document.getElementById('sb-dot');
  if (dot) dot.classList.remove('connected');
  const sbAddr = document.getElementById('sb-addr');
  if (sbAddr) sbAddr.textContent = 'Not connected';
  const sbNet = document.getElementById('sb-net');
  if (sbNet) sbNet.textContent = '--';

  // Reset auth status
  const statusEl = document.getElementById('auth-status');
  if (statusEl) { statusEl.textContent = ''; statusEl.classList.add('hidden'); }
  const btn = document.getElementById('mm-btn');
  if (btn) { btn.disabled = false; btn.querySelector('.mm-btn-text').textContent = 'Connect with MetaMask'; }

  // Reset CSV results
  _csvResults  = [];
  _csvFiltered = [];

  showToast('Disconnected. Click Connect to log in again.');
}

function onWalletConnected(addr) {  walletAddress = addr;
  const short = addr.slice(0, 6) + '…' + addr.slice(-4);

  // Hide login, show app
  document.getElementById('auth-screen').classList.add('hidden');
  document.getElementById('app').classList.remove('hidden');

  // Update sidebar
  const dot = document.getElementById('sb-dot');
  if (dot) dot.classList.add('connected');
  const sbAddr = document.getElementById('sb-addr');
  if (sbAddr) sbAddr.textContent = short;
  const sbNet = document.getElementById('sb-net');
  if (sbNet) sbNet.textContent = 'Sepolia';
  const avatar = document.getElementById('topbar-avatar');
  if (avatar) avatar.textContent = addr.slice(2, 3).toUpperCase();

  showPage('home');
  checkApiHealth();

  window.ethereum.on('accountsChanged', accs => {
    if (accs.length > 0) onWalletConnected(accs[0]);
  });
}

// ═══════════════════════════════════════
// PAGE ROUTING
// ═══════════════════════════════════════
const PAGE_TITLES = {
  home: 'Start Detection', campaigns: 'Browse Campaigns',
  'campaign-detail': 'Campaign Details', features: 'Feature Analysis',
  detection: 'AI Detection Results', contract: 'Smart Contract Decision',
  blockchain: 'Blockchain Status', admin: 'Admin Dashboard',
  create: 'Create Campaign',
};

function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  const pg = document.getElementById('page-' + name);
  if (pg) pg.classList.add('active');

  document.querySelectorAll('.nav-item').forEach(n => {
    n.classList.toggle('active', n.dataset.page === name);
  });
  const tt = document.getElementById('topbar-title');
  if (tt) tt.textContent = PAGE_TITLES[name] || 'TrustNet-5';

  if (name === 'campaigns')  loadCampaigns();
  if (name === 'blockchain') loadBlockchainStatus();
  if (name === 'admin')      loadAdmin();
  if (name === 'contract')   loadContractPage();
  // Features page: only reset if no results are loaded yet
  if (name === 'features' && _csvResults.length === 0) resetCsvAnalysis();

  // Create Campaign page — init form listeners + wallet badge
  if (name === 'create') {
    setTimeout(initCreateFormListeners, 50);
    const wb = document.getElementById('create-wallet-badge');
    if (wb && walletAddress) {
      wb.textContent = walletAddress.slice(0, 6) + '…' + walletAddress.slice(-4);
      wb.className = 'badge-mini green';
    } else if (wb) {
      wb.textContent = 'Wallet not connected';
      wb.className = 'badge-mini';
    }
  }
}

// ═══════════════════════════════════════
// API HELPERS
// ═══════════════════════════════════════
async function apiFetch(path, opts = {}, timeoutMs = 300000) {
  const controller = new AbortController();
  const tid = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(API + path, {
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      ...opts,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'HTTP ' + res.status);
    return data;
  } catch (e) {
    if (e.name === 'AbortError') throw new Error('Request timed out — the operation is taking too long. Please try again.');
    throw e;
  } finally {
    clearTimeout(tid);
  }
}

async function checkApiHealth() {
  try {
    const h = await apiFetch('/health');
    const pill = document.getElementById('api-pill');
    if (pill) {
      const dot = pill.querySelector('.status-dot');
      if (h.status === 'ok') {
        if (dot) dot.classList.add('green');
        pill.innerHTML = '<span class="status-dot green"></span> API Online · ' + h.feature_count + ' features';
      }
    }
    // store owner address so we can gate admin actions
    if (h.owner_address) ownerAddress = h.owner_address.toLowerCase();
    // populate contract addresses on blockchain page
    setContractAddresses(h.contracts);
  } catch (_) {
    const pill = document.getElementById('api-pill');
    if (pill) pill.innerHTML = '<span class="status-dot"></span> API Offline';
  }
}

function setContractAddresses(contracts) {
  if (!contracts) return;
  const set = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };
  set('cpa-registry', contracts.TrustMetricsRegistry || '--');
  set('cpa-security',  contracts.SecurityModule || '--');
  set('cpa-core',      contracts.CrowdfundingCore || '--');
  set('sc-contract-addr', contracts.SecurityModule || '--');
}

// ═══════════════════════════════════════
// CAMPAIGNS
// ═══════════════════════════════════════
async function loadCampaigns() {
  const grid = document.getElementById('campaigns-grid');
  grid.innerHTML = '<div class="spinner-wrap"><div class="spinner"></div></div>';
  try {
    const data = await apiFetch('/campaigns');
    allCampaigns = data.campaigns || [];
    if (!allCampaigns.length) {
      grid.innerHTML = '<p style="color:var(--muted);padding:48px;text-align:center;grid-column:1/-1">No campaigns found on-chain yet.</p>';
      return;
    }
    grid.innerHTML = allCampaigns.map(renderCampaignCard).join('');
  } catch (e) {
    grid.innerHTML = `<p style="color:var(--danger);padding:32px;text-align:center;grid-column:1/-1">Error: ${e.message}</p>`;
  }
}

function riskClass(r) { return r >= 70 ? 'risk-high' : r >= 40 ? 'risk-med' : 'risk-low'; }
function riskChip(r)  { return r >= 70 ? 'chip-red' : r >= 40 ? 'chip-yellow' : 'chip-green'; }
function statusChip(c) {
  if (c.is_frozen)  return '<span class="chip chip-red">Frozen</span>';
  if (c.is_flagged) return '<span class="chip chip-yellow">Flagged</span>';
  if (c.goal_reached) return '<span class="chip chip-green">Goal Met</span>';
  return '<span class="chip chip-blue">Active</span>';
}

function renderCampaignCard(c) {
  const risk = c.trust_scores?.combined_risk ?? 0;
  const pct  = c.goal_eth > 0 ? Math.min(c.amount_raised_eth / c.goal_eth * 100, 100) : 0;
  return `
  <div class="campaign-card ${c.is_frozen ? 'frozen' : c.is_flagged ? 'flagged' : ''}"
       onclick="showCampaignDetail(${c.id})">
    <div class="cc-header">
      <div class="cc-title">${esc(c.title)}</div>
      ${statusChip(c)}
    </div>
    <div class="cc-meta">ID #${c.id} · ${c.contributor_count} contributors · ${new Date(c.deadline * 1000).toLocaleDateString()}</div>
    <div class="cc-progress-wrap"><div class="cc-progress" style="width:${pct.toFixed(1)}%"></div></div>
    <div class="cc-funding">
      <span>${c.amount_raised_eth.toFixed(4)} ETH raised</span>
      <span>${pct.toFixed(0)}% of ${c.goal_eth.toFixed(4)} ETH</span>
    </div>
    <div class="cc-risk-row">
      <span style="font-size:11px;color:var(--muted);width:68px;flex-shrink:0">Risk ${risk}/100</span>
      <div class="cc-risk-bar-wrap"><div class="cc-risk-bar ${riskClass(risk)}" style="width:${risk}%"></div></div>
    </div>
    <div class="cc-footer">
      <span class="chip ${riskChip(risk)}">${risk >= 70 ? 'High Risk' : risk >= 40 ? 'Medium Risk' : 'Low Risk'}</span>
      ${!c.funds_released && !c.is_frozen
        ? `<button class="btn-donate-sm" data-cid="${c.id}" data-ctitle="${esc(c.title).replace(/"/g,'&quot;')}" onclick="event.stopPropagation();openDonateModal(this.dataset.cid, this.dataset.ctitle)">💜 Donate</button>`
        : `<span style="font-size:11px;color:var(--muted)">SDI: ${c.trust_scores?.sdi ?? '--'} · CTI: ${c.trust_scores?.cti ?? '--'}</span>`}
    </div>
  </div>`;
}

// ═══════════════════════════════════════
// CAMPAIGN DETAIL
// ═══════════════════════════════════════
async function showCampaignDetail(id) {
  showPage('campaign-detail');
  const wrap = document.getElementById('campaign-detail-content');
  wrap.innerHTML = '<div class="spinner-wrap"><div class="spinner"></div></div>';
  try {
    const c = await apiFetch('/campaign/' + id);
    const risk = c.combined_risk ?? 0;
    const pct  = c.goal_eth > 0 ? Math.min(c.amount_raised_eth / c.goal_eth * 100, 100) : 0;
    const ts   = c.trust_scores;
    const scores = [
      { name: 'SDI', val: ts.sdi, sub: 'Semantic Drift', color: '#a78bfa' },
      { name: 'CTI', val: ts.cti, sub: 'Crowd Trust',   color: 'var(--success)' },
      { name: 'FVRS',val: ts.fvrs,sub: 'Velocity Risk', color: 'var(--accent2)' },
      { name: 'CFIS',val: ts.cfis,sub: 'Collusion',     color: 'var(--warning)' },
      { name: 'CTCS',val: ts.ctcs,sub: 'Temporal',      color: 'var(--danger)' },
    ];
    wrap.innerHTML = `
    <div class="cd-two-col">
      <div>
        <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:16px">
          <div>
            <h2 style="font-size:24px;font-weight:700;margin-bottom:6px">${esc(c.title)}</h2>
            <div style="display:flex;gap:6px;flex-wrap:wrap">${statusChip(c)}${c.is_frozen ? '<span class="chip chip-red">Frozen</span>' : ''}${c.is_flagged ? '<span class="chip chip-yellow">Flagged</span>' : ''}${c.goal_reached ? '<span class="chip chip-green">Goal Reached</span>' : ''}</div>
          </div>
        </div>
        <div class="card"><div class="card-title" style="margin-bottom:8px">Description</div><p style="font-size:13px;color:var(--text2);line-height:1.7">${esc(c.description)}</p><p style="margin-top:10px;font-size:11px;color:var(--muted);font-family:var(--mono)">${c.creator}</p></div>
        <div class="card">
          <div class="card-title" style="margin-bottom:10px">Funding Progress</div>
          <div class="cc-progress-wrap" style="height:8px;border-radius:4px"><div class="cc-progress" style="width:${pct.toFixed(1)}%;height:8px;border-radius:4px"></div></div>
          <div style="display:flex;justify-content:space-between;margin-top:8px;font-size:13px">
            <span>${c.amount_raised_eth.toFixed(6)} ETH raised</span><span>${pct.toFixed(1)}% funded</span>
          </div>
          <div style="font-size:12px;color:var(--muted);margin-top:4px">${c.contributor_count} contributors · deadline ${new Date(c.deadline * 1000).toLocaleString()}</div>
        </div>
        <div class="card">
          <div class="card-title" style="margin-bottom:10px">Trust Metrics</div>
          <div class="score-hexes">
            ${scores.map(s => `<div class="score-hex"><div class="sh-name">${s.name}</div><div class="sh-val" style="color:${s.color}">${s.val}</div><div class="sh-sub">${s.sub}</div></div>`).join('')}
          </div>
        </div>
      </div>
      <div>
        <div class="card">
          <div class="card-title" style="margin-bottom:12px">Combined Risk Score</div>
          <div style="text-align:center;padding:16px 0">
            <svg viewBox="0 0 200 120" width="200" height="120" style="display:block;margin:0 auto">
              <path d="M20,110 A90,90 0 0,1 180,110" fill="none" stroke="var(--border)" stroke-width="16" stroke-linecap="round"/>
              <path d="M20,110 A90,90 0 0,1 180,110" fill="none" stroke="${risk >= 70 ? 'var(--danger)' : risk >= 40 ? 'var(--warning)' : 'var(--success)'}" stroke-width="16" stroke-linecap="round" stroke-dasharray="283" stroke-dashoffset="${283 - (risk / 100) * 283}"/>
            </svg>
            <div style="font-size:40px;font-weight:800;margin-top:-20px;color:${risk >= 70 ? 'var(--danger)' : risk >= 40 ? 'var(--warning)' : 'var(--success)'}">${risk}</div>
            <div style="font-size:13px;color:var(--muted)">/ 100 · Threshold: 70</div>
          </div>
        </div>
        <div class="card">
          <div class="card-title" style="margin-bottom:12px">Actions</div>
          <div style="display:flex;flex-direction:column;gap:8px" id="campaign-actions-panel">
          </div>
          <div id="det-action-result" style="margin-top:10px"></div>
        </div>
        <div class="card"><div class="card-title" style="margin-bottom:8px">Last Updated</div><div style="font-size:12px;color:var(--muted)">Block ${ts.last_updated || 'N/A'}</div></div>
      </div>
    </div>`;

    // Build action buttons via DOM to avoid nested template literal issues
    _buildCampaignActions(c);

  } catch (e) {
    wrap.innerHTML = `<p style="color:var(--danger);padding:32px">Error: ${e.message}</p>`;
  }
}

// ═══════════════════════════════════════
// AI DETECTION
// ═══════════════════════════════════════
async function runDetection() {
  showToast('Running TrustNet inference…');
  const features = {
    sdi_score:           parseFloatOrZero('det-sdi'),
    cti_norm:            parseFloatOrZero('det-cti'),
    fgr:                 parseFloatOrZero('det-fgr'),
    wgd:                 parseFloatOrZero('det-wgd'),
    ctcs:                parseFloatOrZero('det-ctcs'),
    combined_risk_score: parseFloatOrZero('det-combined'),
  };
  const campaignId = document.getElementById('det-campaign-id')?.value || 'DETECT_TEST';

  try {
    const data = await apiFetch('/predict', {
      method: 'POST',
      body: JSON.stringify({ campaign_id: campaignId, features }),
    });
    lastDetectionResult = { ...data, campaign_id: campaignId, features };
    renderDetectionResult(data, features);
    showToast(data.is_fraud ? '⚠ Fraud detected!' : '✓ Detection complete', data.is_fraud ? 'error' : 'success');
  } catch (e) {
    showToast('Detection failed: ' + e.message, 'error');
  }
}

function renderDetectionResult(data, features) {
  const prob = data.fraud_probability;
  const pct  = (prob * 100).toFixed(1);
  const isF  = data.is_fraud;
  const ts   = data.trust_scores;

  // Verdict banner
  const banner = document.getElementById('verdict-banner');
  banner.classList.remove('hidden');
  banner.className = 'verdict-banner ' + (isF ? 'fraud' : 'legit');
  document.getElementById('vb-title').textContent = isF ? 'FRAUD DETECTED' : 'CAMPAIGN APPEARS LEGITIMATE';
  document.getElementById('vb-sub').textContent   = isF
    ? `High fraud probability ${pct}% — campaign flagged for review.`
    : `Low fraud probability ${pct}% — campaign within acceptable thresholds.`;
  document.getElementById('vb-prob').textContent  = pct + '%';

  // Gauge
  const offset = 283 - (prob * 283);
  const arc = document.getElementById('gauge-arc');
  if (arc) {
    arc.style.stroke = isF ? 'var(--danger)' : prob >= 0.4 ? 'var(--warning)' : 'var(--success)';
    arc.style.strokeDashoffset = offset;
  }
  const gv = document.getElementById('gauge-value');
  if (gv) gv.textContent = pct + '%';
  const chip = document.getElementById('verdict-label');
  if (chip) { chip.textContent = isF ? '⚠ FRAUD' : '✓ LEGITIMATE'; chip.className = 'verdict-chip ' + (isF ? 'fraud' : 'legit'); }

  // Metric bars
  const metrics = { sdi: ts.sdi, cti: ts.cti, fvrs: ts.fvrs, cfis: ts.cfis, ctcs: ts.ctcs };
  for (const [k, v] of Object.entries(metrics)) {
    const bar = document.getElementById('md-' + k);
    const val = document.getElementById('mdv-' + k);
    if (bar) bar.style.width = v + '%';
    if (val) val.textContent = v;
  }

  // Feature score decomp
  const fsd = {
    fgr: features.fgr, tbr: features.tbr, nur: features.nur,
    ftr: features.ftr, rar: features.rar, wgd: features.wgd,
    cds: features.cds, col: features.col, fts: features.fts, fas: features.fas,
  };
  for (const [k, v] of Object.entries(fsd)) {
    const el = document.getElementById('fsd-' + k);
    if (el) el.textContent = v !== undefined && v !== null ? parseFloat(v).toFixed(2) : '--';
  }

  // Intelligence
  const set = (id, val, color) => {
    const e = document.getElementById(id);
    if (e) { e.textContent = val; if (color) e.style.color = color; }
  };
  set('intel-risk',     ts.combined_risk + '/100', isF ? 'var(--danger)' : 'var(--success)');
  set('intel-decision', isF ? 'FREEZE' : 'ALLOW',  isF ? 'var(--danger)' : 'var(--success)');
  set('intel-sdi',      ts.sdi + '/100', ts.sdi >= 70 ? 'var(--danger)' : 'var(--success)');
  set('intel-cfis',     ts.cfis + '/100', ts.cfis >= 70 ? 'var(--danger)' : 'var(--success)');

  const freezeBtn = document.getElementById('btn-freeze-detect');
  if (freezeBtn) freezeBtn.disabled = !isF;
}

function clearDetection() {
  ['det-sdi','det-cti','det-fgr','det-wgd','det-ctcs','det-combined'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = '';
  });
  document.getElementById('verdict-banner')?.classList.add('hidden');
  const chip = document.getElementById('verdict-label');
  if (chip) { chip.textContent = 'Run Detection'; chip.className = 'verdict-chip pending'; }
  const gv = document.getElementById('gauge-value');
  if (gv) gv.textContent = '--';
  lastDetectionResult = null;
}

async function freezeFromDetection() {
  if (!lastDetectionResult?.campaign_id) { showToast('No campaign to freeze.', 'error'); return; }
  const addr = prompt('Enter campaign creator address to freeze:');
  if (!addr) return;
  await doFreeze(addr);
}

// ═══════════════════════════════════════
// SMART CONTRACT DECISION
// ═══════════════════════════════════════
function loadContractPage() {
  const next = document.getElementById('eg-next');
  if (next) next.textContent = new Date(Date.now() + 120000).toLocaleTimeString();
  const block = document.getElementById('eg-block');
  if (block) block.textContent = 'Current + 2';

  // Pre-fill from last detection
  if (lastDetectionResult) {
    const prob = lastDetectionResult.fraud_probability;
    const el = document.getElementById('sc-fraud-prob');
    if (el) el.value = prob.toFixed(4);
    const badge = document.getElementById('sc-campaign-badge');
    if (badge) badge.textContent = 'Campaign: ' + (lastDetectionResult.campaign_id || '—');
    const rbadge = document.getElementById('sc-risk-badge');
    if (rbadge) {
      rbadge.textContent = 'Risk: ' + (lastDetectionResult.trust_scores?.combined_risk ?? '—') + '/100';
      rbadge.className = 'badge-mini ' + (prob >= 0.5 ? 'red' : 'green');
    }
    const banner = document.getElementById('sc-banner');
    if (banner) {
      banner.classList.remove('hidden');
      document.getElementById('sc-banner-title').innerHTML =
        `Fraud Probability ${(prob * 100).toFixed(0)}% <span class="badge-red-sm">${prob >= 0.5 ? 'HIGH RISK' : 'LOW RISK'}</span>`;
      const sysStatus = document.getElementById('sc-system-status');
      if (sysStatus) sysStatus.textContent = prob >= 0.5 ? 'SYSTEM LOCKED 🔒' : 'SYSTEM CLEAR ✓';
    }
  }

  // Update timeline dots based on real on-chain state across all campaigns
  _refreshTimelineDots();
}

async function _refreshTimelineDots() {
  try {
    const data = await apiFetch('/campaigns');
    const campaigns = data.campaigns || [];

    const anyFrozen  = campaigns.some(c => c.is_frozen);
    const anyFlagged = campaigns.some(c => c.is_flagged);

    const setDot = (id, active, subId, subText) => {
      const el = document.getElementById(id);
      if (!el) return;
      if (active) {
        el.classList.add('pending');
        el.querySelector('.at-dot').style.background = 'var(--accent)';
      } else {
        el.classList.remove('pending');
        el.querySelector('.at-dot').style.background = '';
      }
      const sub = document.getElementById(subId);
      if (sub && subText) sub.textContent = subText;
    };

    setDot('at-freeze', anyFrozen,
      'at-freeze-sub',
      anyFrozen ? 'freezeCampaign() — executed ✓' : 'freezeCampaign() — pending risk confirmation');

    setDot('at-flag', anyFlagged,
      'at-flag-sub',
      anyFlagged ? 'flagCampaign() — executed ✓' : 'flagCampaign() — conditional on admin approval');

    // Refund dot: active if any campaign is both frozen and flagged
    const anyRefundable = campaigns.some(c => c.is_frozen && c.is_flagged);
    const refundEl = document.getElementById('at-refund');
    if (refundEl) {
      if (anyRefundable) {
        refundEl.classList.add('pending');
        refundEl.querySelector('.at-dot').style.background = 'var(--accent)';
      } else {
        refundEl.classList.remove('pending');
        refundEl.querySelector('.at-dot').style.background = '';
      }
    }
  } catch (_) {}
}

async function executeSmartContract() {
  const addr = document.getElementById('sc-campaign-input')?.value?.trim();
  const prob = parseFloat(document.getElementById('sc-fraud-prob')?.value || '0');
  if (!addr) { showToast('Enter campaign address.', 'error'); return; }

  const result = document.getElementById('sc-tx-result');
  const content = document.getElementById('sc-tx-content');

  try {
    showToast('Executing smart contract…');
    const btn = document.getElementById('sc-confirm-btn');
    if (btn) btn.disabled = true;

    // Step 1: update scores
    const features = lastDetectionResult?.features || {};
    const updateData = await apiFetch('/update-blockchain', {
      method: 'POST',
      body: JSON.stringify({ campaign_address: addr, campaign_id: addr.slice(0, 8), features }),
    });

    let freezeData = null;
    if (prob >= 0.5) {
      freezeData = await apiFetch('/freeze', {
        method: 'POST',
        body: JSON.stringify({ campaign_address: addr, reason: `Fraud probability ${(prob * 100).toFixed(1)}%` }),
      });
    }

    if (content) content.innerHTML = `
      <div style="margin-bottom:10px">
        <div class="card-title" style="margin-bottom:6px">Scores Updated ✓</div>
        <div style="font-size:12px;font-family:var(--mono);color:var(--muted)">${updateData.transaction?.tx_hash || '--'}</div>
        <div style="font-size:12px;color:var(--text2);margin-top:4px">Block: ${updateData.transaction?.block || '--'}</div>
      </div>
      ${freezeData ? `<div><div class="card-title" style="color:var(--danger);margin-bottom:6px">Campaign Frozen ✓</div><div style="font-size:12px;font-family:var(--mono);color:var(--muted)">${freezeData.transaction?.tx_hash || '--'}</div></div>` : '<div style="color:var(--success);font-size:13px">✓ Campaign allowed — risk below threshold</div>'}`;
    result?.classList.remove('hidden');
    showToast(freezeData ? 'Campaign frozen on-chain.' : 'Scores updated.', freezeData ? 'error' : 'success');
    if (btn) btn.disabled = false;
  } catch (e) {
    if (content) content.innerHTML = `<p style="color:var(--danger)">${e.message}</p>`;
    result?.classList.remove('hidden');
    showToast('Execution failed: ' + e.message, 'error');
    const btn = document.getElementById('sc-confirm-btn');
    if (btn) btn.disabled = false;
  }
}

// ═══════════════════════════════════════
// BLOCKCHAIN STATUS
// ═══════════════════════════════════════
async function loadBlockchainStatus() {
  try {
    const h = await apiFetch('/health');
    if (h.owner_address) ownerAddress = h.owner_address.toLowerCase();
    setContractAddresses(h.contracts);

    const set = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };
    set('bcd-chain', 'Chain ID: 11155111');
    set('bcd-oracle', walletAddress ? walletAddress.slice(0, 10) + '…' + walletAddress.slice(-6) : '--');
  } catch (_) {}

  try {
    const data = await apiFetch('/campaigns');
    const camps = data.campaigns || [];
    const frozen = camps.filter(c => c.is_frozen).length;
    const set = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };
    set('bcs-campaigns', camps.length);
    set('bcs-frozen', frozen);

    // Build recent transactions table from campaign data
    const rows = document.getElementById('tx-rows');
    if (rows) {
      if (!camps.length) {
        rows.innerHTML = '<div class="tx-row" style="grid-column:1/-1;color:var(--muted);text-align:center">No transactions yet.</div>';
      } else {
        rows.innerHTML = camps.slice(0, 8).map(c => `
          <div class="tx-row">
            <span class="tx-fn">${c.is_frozen ? 'freezeCampaign' : 'updateAllScores'}</span>
            <span class="tx-addr">${c.creator.slice(0, 10)}…${c.creator.slice(-6)}</span>
            <span>${c.trust_scores?.last_updated || '--'}</span>
            <span style="color:var(--muted)">Recent</span>
            <span class="${c.is_frozen ? 'chip chip-red' : 'chip chip-green'}">${c.is_frozen ? 'Frozen' : 'Active'}</span>
          </div>`).join('');
      }
    }
    set('last-sync', new Date().toLocaleTimeString());
  } catch (_) {
    const rows = document.getElementById('tx-rows');
    if (rows) rows.innerHTML = '<div class="tx-row" style="grid-column:1/-1;color:var(--muted);text-align:center">Connect API to view transactions.</div>';
  }

  // Try to get block number from window.ethereum
  if (window.ethereum) {
    try {
      const hex = await window.ethereum.request({ method: 'eth_blockNumber' });
      const blockNum = parseInt(hex, 16);
      const set = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };
      set('bcd-block', blockNum.toLocaleString());
    } catch (_) {}
  }
}

// ═══════════════════════════════════════
// ADMIN DASHBOARD — full workflow
// ═══════════════════════════════════════

// Per-campaign AI detection cache: { campaignId: { fraud_probability, is_fraud, trust_scores, recommendation } }
const _aiResults = {};

async function loadAdmin() {
  try {
    const data = await apiFetch('/campaigns');
    const camps = data.campaigns || [];
    allCampaigns = camps;

    const set = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };
    set('kpi-total', camps.length);
    set('kpi-total-sub', camps.length + ' on Sepolia');
    const frozen   = camps.filter(c => c.is_frozen).length;
    const released = camps.filter(c => c.funds_released).length;
    const highRisk = camps.filter(c => (c.trust_scores?.combined_risk ?? 0) >= 70).length;
    set('kpi-fraud',    highRisk);
    set('kpi-fraud-sub', highRisk + ' high-risk detected');
    set('kpi-released', released);
    set('kpi-frozen',   frozen);
    set('last-sync',    new Date().toLocaleTimeString());

    renderAdminTable(camps);
  } catch (e) {
    const tbody = document.getElementById('admin-tbody');
    if (tbody) tbody.innerHTML = `<tr><td colspan="6" class="tbl-loading" style="color:var(--danger)">${e.message}</td></tr>`;
  }
}

function _campaignWorkflowStatus(c) {
  if (c.funds_released) return { label: 'Released',       cls: 'chip-green' };
  if (c.is_frozen)      return { label: 'Frozen',         cls: 'chip-red' };
  if (c.is_flagged)     return { label: 'Flagged',        cls: 'chip-yellow' };
  if (c.goal_reached)   return { label: 'Goal Reached',   cls: 'chip-blue' };
  return                       { label: 'Funding Active', cls: 'chip-gray' };
}

function renderAdminTable(camps) {
  const tbody = document.getElementById('admin-tbody');
  if (!tbody) return;
  if (!camps.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="tbl-loading">No campaigns found.</td></tr>';
    return;
  }
  tbody.innerHTML = camps.map(c => {
    const ts       = c.trust_scores;
    const risk     = ts?.combined_risk ?? 0;
    const ai       = _aiResults[c.id];
    const wf       = _campaignWorkflowStatus(c);
    const riskColor= risk >= 70 ? 'var(--danger)' : risk >= 40 ? 'var(--warning)' : 'var(--success)';

    // AI verdict display
    const aiDisplay = ai
      ? `<span class="chip ${ai.is_fraud ? 'chip-red' : 'chip-green'}">${(ai.fraud_probability * 100).toFixed(1)}% ${ai.is_fraud ? 'FRAUD' : 'LEGIT'}</span>`
      : `<span style="font-size:11px;color:var(--muted)">Not run</span>`;

    // Button logic — enable/disable based on workflow state
    const canRelease = ai && !ai.is_fraud && c.goal_reached && !c.is_frozen && !c.funds_released;
    const canFreeze  = ai && !c.is_frozen && !c.funds_released;
    const canRefund  = c.is_frozen && !c.funds_released;

    return `
    <tr>
      <td>
        <div style="font-weight:600;max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(c.title)}</div>
        <div style="font-size:11px;font-family:var(--mono);color:var(--muted)">#${c.id} · ${c.creator.slice(0,8)}…</div>
      </td>
      <td>
        <div style="font-size:13px">${c.amount_raised_eth.toFixed(4)} / ${c.goal_eth.toFixed(4)} ETH</div>
        <div style="height:4px;background:var(--border);border-radius:2px;margin-top:4px;max-width:80px;overflow:hidden">
          <div style="height:4px;background:var(--accent);border-radius:2px;width:${Math.min(c.goal_eth > 0 ? c.amount_raised_eth / c.goal_eth * 100 : 0, 100)}%"></div>
        </div>
      </td>
      <td><span class="chip ${wf.cls}">${wf.label}</span></td>
      <td>
        <div style="display:flex;align-items:center;gap:6px">
          <div style="width:40px;height:4px;background:var(--border);border-radius:2px;overflow:hidden">
            <div style="height:4px;width:${risk}%;background:${riskColor};border-radius:2px"></div>
          </div>
          <span style="font-size:12px;font-weight:700;color:${riskColor}">${risk}</span>
        </div>
      </td>
      <td>
        <div class="admin-actions" style="flex-wrap:wrap;gap:4px">
          <button class="btn-xs" onclick="adminRunAI(${c.id})" title="Run AI Detection">🔍 AI</button>
          <button class="btn-xs" onclick="showCampaignDetail(${c.id})" title="View Details">View</button>
          ${canRelease
            ? `<button class="btn-xs" style="background:var(--success-bg);color:var(--success)" onclick="adminRelease(${c.id})" title="Approve & Release Funds">✓ Release</button>`
            : `<button class="btn-xs" style="opacity:.4;cursor:not-allowed" disabled title="${!c.goal_reached ? 'Goal not reached' : ai && ai.is_fraud ? 'AI flagged as fraud' : c.is_frozen ? 'Frozen' : c.funds_released ? 'Already released' : 'Run AI first'}">✓ Release</button>`}
          ${canFreeze
            ? `<button class="btn-xs" style="background:var(--danger-bg);color:var(--danger)" onclick="adminFreeze('${c.creator}',${c.id})" title="Freeze Campaign">🔒 Freeze</button>`
            : c.is_frozen
              ? `<button class="btn-xs" style="background:var(--success-bg);color:var(--success)" onclick="doUnfreeze('${c.creator}')" title="Unfreeze Campaign">🔓 Unfreeze</button>`
              : ''}
          ${canRefund
            ? `<button class="btn-xs" style="background:var(--warning-bg);color:var(--warning)" onclick="adminRefund(${c.id})" title="Refund Contributors">↩ Refund</button>`
            : ''}
        </div>
      </td>
    </tr>`;
  }).join('');
}

// ── Run AI detection on a single campaign ───────────────────────────
async function adminRunAI(campaignId) {
  showToast('Running AI detection on campaign #' + campaignId + '…');
  try {
    const data = await apiFetch('/run-ai-detection', {
      method: 'POST',
      body: JSON.stringify({ campaign_id: campaignId }),
    });

    // Cache result
    _aiResults[campaignId] = {
      fraud_probability: data.fraud_probability,
      is_fraud:          data.is_fraud,
      trust_scores:      data.trust_scores,
      recommendation:    data.recommendation,
    };

    // Show inline panel
    const panel = document.getElementById('admin-ai-panel');
    const title = document.getElementById('aip-title');
    const content = document.getElementById('aip-content');
    if (panel && content) {
      const prob   = (data.fraud_probability * 100).toFixed(1);
      const isF    = data.is_fraud;
      const ts     = data.trust_scores;
      if (title) title.textContent = `AI Result — Campaign #${campaignId}: ${data.title || ''}`;
      content.innerHTML = `
        <div style="display:flex;align-items:center;gap:20px;padding:12px 0;border-bottom:1px solid var(--border);margin-bottom:12px">
          <div style="text-align:center">
            <div style="font-size:32px;font-weight:800;color:${isF ? 'var(--danger)' : 'var(--success)'}">${prob}%</div>
            <div style="font-size:12px;color:var(--muted)">Fraud Probability</div>
          </div>
          <div>
            <div class="chip ${isF ? 'chip-red' : 'chip-green'}" style="font-size:14px;padding:6px 14px;margin-bottom:8px">
              ${isF ? '⚠ FRAUD DETECTED — Recommend FREEZE' : '✓ LEGITIMATE — Recommend APPROVE'}
            </div>
            <div style="font-size:12px;color:var(--muted)">
              TX: <code>${data.transaction?.tx_hash?.slice(0, 20) ?? 'N/A'}…</code> · Block: ${data.transaction?.block ?? '?'}
            </div>
          </div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:12px">
          ${[['SDI',ts.sdi,'Semantic Drift'],['CTI',ts.cti,'Crowd Trust'],['FVRS',ts.fvrs,'Velocity'],['CFIS',ts.cfis,'Collusion'],['CTCS',ts.ctcs,'Temporal']].map(([n,v,s]) =>
            `<div class="mfg-item"><div class="mfg-lbl">${n}</div><div class="mfg-val">${v}</div><div class="mfg-sub">${s}</div></div>`
          ).join('')}
        </div>
        <div style="display:flex;gap:8px">
          ${isF
            ? `<button class="btn-danger-lg" style="flex:1;margin-top:0" onclick="adminFreeze('${data.campaign_address}',${campaignId})">🔒 Freeze Campaign</button>`
            : `<button class="btn-success" style="flex:1;padding:10px;border:none;border-radius:6px;font-weight:700;cursor:pointer" onclick="adminRelease(${campaignId})">✓ Approve & Release Funds</button>`}
          <button class="btn-outline-lg" style="flex:1" onclick="document.getElementById('admin-ai-panel').classList.add('hidden')">Dismiss</button>
        </div>`;
      panel.classList.remove('hidden');
      panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Re-render table to update button states — reload from API to get fresh on-chain scores
    await loadAdmin();

    // If campaign detail is open for this campaign, refresh it with updated scores
    if (document.getElementById('page-campaign-detail')?.classList.contains('active')) {
      // Update Trust Metrics immediately from AI result (confirmed on-chain values)
      _updateDetailTrustMetrics(data.trust_scores);
    }

    showToast('AI detection complete for campaign #' + campaignId, data.is_fraud ? 'error' : 'success');
  } catch (e) {
    showToast('AI detection failed: ' + e.message, 'error');
  }
}

// ── Admin: Approve & Release ─────────────────────────────────────────
async function adminRelease(campaignId) {
  if (!walletAddress) { showToast('Connect MetaMask first.', 'error'); return; }
  if (!isOwner()) {
    showToast('Only the contract owner can release campaign funds.', 'error');
    return;
  }
  if (!confirm('Approve and release funds for campaign #' + campaignId + '?\n\nThis action cannot be undone.')) return;
  try {
    showToast('Sending releaseFunds transaction…');
    const data = await apiFetch('/release', {
      method: 'POST',
      body: JSON.stringify({ campaign_id: campaignId, caller_address: walletAddress }),
    });
    if (data.released) {
      showToast('Funds released ✓  TX: ' + data.transaction.tx_hash.slice(0, 16) + '…', 'success');
      await loadAdmin();
    } else {
      showToast('Release blocked: ' + _releaseBlockReason(data.error || ''), 'error');
    }
  } catch (e) {
    showToast('Release failed: ' + _releaseBlockReason(e.message), 'error');
  }
}

// ── Admin: Freeze ────────────────────────────────────────────────────
async function adminFreeze(address, campaignId) {
  if (!confirm('Freeze campaign #' + campaignId + '?\nContributors will be eligible for refunds.')) return;
  try {
    showToast('Sending freeze transaction…');
    const data = await apiFetch('/freeze', {
      method: 'POST',
      body: JSON.stringify({ campaign_address: address, reason: 'AI fraud detection — manual freeze by admin' }),
    });
    showToast('Campaign frozen ✓  TX: ' + data.transaction.tx_hash.slice(0, 16) + '…', 'error');
    document.getElementById('admin-ai-panel')?.classList.add('hidden');
    await loadAdmin();
  } catch (e) { showToast('Freeze failed: ' + e.message, 'error'); }
}

// ── Admin: Refund ────────────────────────────────────────────────────
async function adminRefund(campaignId) {
  // Always resolve the current wallet — walletAddress may be stale if MetaMask
  // account changed after page load without triggering our accountsChanged handler
  let contributor = walletAddress;
  if (!contributor && window.ethereum) {
    try {
      const accounts = await window.ethereum.request({ method: 'eth_accounts' });
      contributor = accounts?.[0] || null;
      if (contributor) walletAddress = contributor; // keep in sync
    } catch (_) {}
  }

  console.log('[adminRefund] campaignId:', campaignId, '| contributor:', contributor, '| walletAddress:', walletAddress);

  if (!contributor) {
    showToast('Connect MetaMask first — refunds must be signed by the wallet that donated.', 'error');
    return;
  }

  if (!confirm(
    'Claim your refund for campaign #' + campaignId + '?\n\n' +
    'MetaMask will open. Sign the transaction with the wallet you donated from.\n' +
    'If the campaign is only frozen (not flagged), the backend will auto-flag it first.'
  )) return;

  const ar = document.getElementById('det-action-result');

  try {
    showToast('Checking refund conditions…');

    // Step 1 — backend pre-flight: auto-flag if needed & verify contribution balance
    const check = await apiFetch('/refund', {
      method: 'POST',
      body: JSON.stringify({ campaign_id: campaignId, contributor_address: contributor }),
    });

    if (!check.ready) {
      const msg = check.error || 'Refund not available';
      showToast(msg, 'error');
      if (ar) ar.innerHTML = '<div class="chip chip-red" style="margin-top:8px">' + msg + '</div>';
      return;
    }

    if (check.auto_flagged) {
      showToast('Campaign auto-flagged ✓ — now signing refund…', 'success');
    }

    // Step 2 — user signs refund() with MetaMask (msg.sender = their wallet)
    showToast('Opening MetaMask — sign to claim ' + check.amount_eth.toFixed(6) + ' ETH…');

    const provider = new ethers.providers.Web3Provider(window.ethereum);
    const signer   = provider.getSigner();
    const abi      = ['function refund(uint256 _campaignId) external'];
    const contract = new ethers.Contract(CROWDFUNDING_CORE_ADDR, abi, signer);

    const tx = await contract.refund(campaignId);
    if (ar) ar.innerHTML = '<div class="chip chip-yellow" style="margin-top:8px">⏳ Waiting for confirmation…</div>';
    showToast('Transaction sent — waiting for confirmation…');

    const receipt = await tx.wait();

    if (receipt.status === 1) {
      showToast('Refund claimed ✓  TX: ' + receipt.transactionHash.slice(0, 16) + '…', 'success');
      if (ar) ar.innerHTML = `<div class="chip chip-green" style="margin-top:8px">Refund Claimed ✓ · ${check.amount_eth.toFixed(6)} ETH · ${receipt.transactionHash.slice(0,16)}…</div>`;
      await loadAdmin();
      if (document.getElementById('page-campaign-detail')?.classList.contains('active')) {
        setTimeout(() => showCampaignDetail(campaignId), 1000);
      }
    } else {
      showToast('Transaction reverted — check MetaMask for details.', 'error');
      if (ar) ar.innerHTML = '<div class="chip chip-red" style="margin-top:8px">Transaction reverted</div>';
    }
  } catch (e) {
    const msg = e?.reason || e?.data?.message || e?.message || 'Refund failed';
    showToast(msg, 'error');
    if (ar) ar.innerHTML = '<div class="chip chip-red" style="margin-top:8px">' + msg.slice(0, 80) + '</div>';
  }
}

function filterAdminTable() {
  const search  = document.getElementById('admin-search')?.value?.toLowerCase()  || '';
  const statusF = document.getElementById('admin-filter-status')?.value || '';
  const riskF   = document.getElementById('admin-filter-risk')?.value   || '';

  const filtered = allCampaigns.filter(c => {
    const risk = c.trust_scores?.combined_risk ?? 0;
    const matchSearch = !search || c.title.toLowerCase().includes(search) || c.creator.toLowerCase().includes(search);
    const matchStatus = !statusF ||
      (statusF === 'frozen'   && c.is_frozen) ||
      (statusF === 'flagged'  && c.is_flagged) ||
      (statusF === 'released' && c.funds_released) ||
      (statusF === 'active'   && !c.is_frozen && !c.is_flagged && !c.funds_released);
    const matchRisk = !riskF ||
      (riskF === 'high'   && risk >= 70) ||
      (riskF === 'medium' && risk >= 40 && risk < 70) ||
      (riskF === 'low'    && risk < 40);
    return matchSearch && matchStatus && matchRisk;
  });
  renderAdminTable(filtered);
}

// ═══════════════════════════════════════
// BLOCKCHAIN ACTIONS
// ═══════════════════════════════════════
async function doFreeze(address) {
  if (!address) { showToast('No address provided.', 'error'); return; }
  try {
    showToast('Sending freeze transaction…');
    const data = await apiFetch('/freeze', {
      method: 'POST',
      body: JSON.stringify({ campaign_address: address, reason: 'Manual freeze via dashboard' }),
    });
    showToast('Frozen ✓  TX: ' + data.transaction.tx_hash.slice(0, 14) + '…', 'error');
    const ar = document.getElementById('det-action-result');
    if (ar) ar.innerHTML = `<div class="chip chip-red" style="margin-top:8px">Frozen · ${data.transaction.tx_hash.slice(0, 16)}…</div>`;
  } catch (e) { showToast('Freeze failed: ' + e.message, 'error'); }
}

async function doUnfreeze(address) {
  if (!address) { showToast('No address provided.', 'error'); return; }
  if (!walletAddress) { showToast('Connect MetaMask first.', 'error'); return; }
  if (!isOwner()) {
    showToast('Only the contract owner can unfreeze campaigns.', 'error');
    return;
  }
  try {
    showToast('Sending unfreeze transaction…');
    const data = await apiFetch('/unfreeze', {
      method: 'POST',
      body: JSON.stringify({ campaign_address: address, caller_address: walletAddress }),
    });
    showToast('Unfrozen ✓  TX: ' + data.transaction.tx_hash.slice(0, 14) + '…', 'success');
    const ar = document.getElementById('det-action-result');
    if (ar) ar.innerHTML = `<div class="chip chip-green" style="margin-top:8px">Unfrozen · ${data.transaction.tx_hash.slice(0, 16)}…</div>`;
    if (document.getElementById('page-campaigns')?.classList.contains('active')) loadCampaigns();
    if (document.getElementById('page-admin')?.classList.contains('active')) loadAdmin();
  } catch (e) { showToast('Unfreeze failed: ' + e.message, 'error'); }
}

async function doFlag(address) {
  if (!address) { showToast('No address provided.', 'error'); return; }
  const reason = prompt('Reason for flagging (optional):') || 'Flagged via dashboard';
  if (reason === null) return; // cancelled
  try {
    showToast('Sending flag transaction…');
    const data = await apiFetch('/flag', {
      method: 'POST',
      body: JSON.stringify({ campaign_address: address, reason }),
    });
    showToast('Flagged ✓  TX: ' + data.transaction.tx_hash.slice(0, 14) + '…', 'error');
    const ar = document.getElementById('det-action-result');
    if (ar) ar.innerHTML = `<div class="chip chip-yellow" style="margin-top:8px">Flagged · ${data.transaction.tx_hash.slice(0, 16)}…</div>`;
    if (document.getElementById('page-campaigns')?.classList.contains('active')) loadCampaigns();
    if (document.getElementById('page-admin')?.classList.contains('active')) loadAdmin();
  } catch (e) { showToast('Flag failed: ' + e.message, 'error'); }
}

async function doUnflag(address) {
  if (!address) { showToast('No address provided.', 'error'); return; }
  try {
    showToast('Sending unflag transaction…');
    const data = await apiFetch('/unflag', {
      method: 'POST',
      body: JSON.stringify({ campaign_address: address }),
    });
    showToast('Unflagged ✓  TX: ' + data.transaction.tx_hash.slice(0, 14) + '…', 'success');
    const ar = document.getElementById('det-action-result');
    if (ar) ar.innerHTML = `<div class="chip chip-green" style="margin-top:8px">Unflagged · ${data.transaction.tx_hash.slice(0, 16)}…</div>`;
    if (document.getElementById('page-campaigns')?.classList.contains('active')) loadCampaigns();
    if (document.getElementById('page-admin')?.classList.contains('active')) loadAdmin();
  } catch (e) { showToast('Unflag failed: ' + e.message, 'error'); }
}

async function doRelease(campaignId) {
  try {
    showToast('Sending release transaction…');
    const data = await apiFetch('/release', {
      method: 'POST',
      body: JSON.stringify({ campaign_id: campaignId }),
    });
    const ar = document.getElementById('det-action-result');
    if (data.released) {
      showToast('Funds released!', 'success');
      if (ar) ar.innerHTML = '<div class="chip chip-green" style="margin-top:8px">Released ✓</div>';
      // Reload detail and admin if open
      if (document.getElementById('page-campaign-detail')?.classList.contains('active')) {
        setTimeout(() => showCampaignDetail(campaignId), 1000);
      }
      if (document.getElementById('page-admin')?.classList.contains('active')) loadAdmin();
    } else {
      const hint = _releaseBlockReason(data.error || '');
      showToast('Blocked: ' + hint, 'error');
      if (ar) ar.innerHTML = '<div class="chip chip-red" style="margin-top:8px">Blocked — ' + hint + '</div>';
    }
  } catch (e) {
    const hint = _releaseBlockReason(e.message || '');
    showToast('Release failed: ' + hint, 'error');
    const ar = document.getElementById('det-action-result');
    if (ar) ar.innerHTML = '<div class="chip chip-red" style="margin-top:8px">' + hint + '</div>';
  }
}

function _releaseBlockReason(msg) {
  msg = (msg || '').toLowerCase();
  if (msg.includes('goal not reached') || msg.includes('goal'))   return 'Funding goal not yet reached';
  if (msg.includes('already released'))                           return 'Funds already released';
  if (msg.includes('frozen'))                                     return 'Campaign is frozen — unfreeze first';
  if (msg.includes('creator'))                                    return 'Only the campaign creator can release funds';
  if (msg.includes('risk') || msg.includes('fraud'))              return 'Risk score too high — campaign auto-frozen';
  if (msg.includes('paused'))                                     return 'Platform is paused';
  return msg.slice(0, 80) || 'Release blocked by contract';
}

async function runPipelineForCampaign(creatorAddr, campaignId) {
  const features = lastDetectionResult?.features || {};
  showToast('Running full pipeline for campaign #' + campaignId + '…');
  try {
    const data = await apiFetch('/update-blockchain', {
      method: 'POST',
      body: JSON.stringify({ campaign_address: creatorAddr, campaign_id: String(campaignId), features }),
    });

    // Only freeze when the on-chain Combined Risk Score exceeds the contract threshold (70).
    // data.is_fraud is the raw ML probability flag — it does NOT replace the on-chain
    // risk threshold. The contract itself auto-freezes on releaseFunds() if risk >= 70.
    const combinedRisk = data.trust_scores?.combined_risk ?? 0;
    const RISK_THRESHOLD = 70;
    if (combinedRisk >= RISK_THRESHOLD) {
      await doFreeze(creatorAddr);
    }

    // Refresh campaign detail with confirmed on-chain scores from the response
    if (data.trust_scores) {
      _updateDetailTrustMetrics(data.trust_scores);
    }
    // Also do a full reload to sync all UI state (frozen status, etc.)
    setTimeout(() => showCampaignDetail(campaignId), 500);

    showToast('Pipeline complete.', 'success');
  } catch (e) { showToast('Pipeline error: ' + e.message, 'error'); }
}

// ═══════════════════════════════════════
// FILE UPLOAD
// ═══════════════════════════════════════
function handleDrop(ev) {
  ev.preventDefault();
  const file = ev.dataTransfer?.files?.[0];
  if (file) processUploadedFile(file);
}
function handleFileSelect(ev) {
  const file = ev.target?.files?.[0];
  if (file) processUploadedFile(file);
}
function processUploadedFile(file) {
  const info = document.getElementById('upload-file-info');
  if (info) info.textContent = file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
  showToast('File loaded: ' + file.name, 'success');
  const zone = document.getElementById('upload-zone');
  if (zone) zone.style.borderColor = 'var(--success)';
}
function loadUploadedCampaign() { showToast('Using uploaded file for analysis.'); }

// ═══════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════
function parseFloatOrZero(id) {
  const v = parseFloat(document.getElementById(id)?.value);
  return isNaN(v) ? 0 : v;
}
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

let _toastTimer;
function showToast(msg, type = '') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast ' + type;
  t.classList.remove('hidden');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => t.classList.add('hidden'), 4500);
}

// ── Update Trust Metrics in campaign detail without full reload ──────
function _updateDetailTrustMetrics(trustScores) {
  if (!trustScores) return;

  // The score-hexes are rendered inside showCampaignDetail as .sh-val elements
  // Find them by their parent .score-hex and update by order: SDI, CTI, FVRS, CFIS, CTCS
  const hexes = document.querySelectorAll('#campaign-detail-content .score-hex .sh-val');
  const order = ['sdi', 'cti', 'fvrs', 'cfis', 'ctcs'];
  const colors = {
    sdi:  '#a78bfa',
    cti:  'var(--success)',
    fvrs: 'var(--accent2)',
    cfis: 'var(--warning)',
    ctcs: 'var(--danger)',
  };

  hexes.forEach((el, i) => {
    const key = order[i];
    if (key && trustScores[key] !== undefined) {
      el.textContent = trustScores[key];
      el.style.color = colors[key];
    }
  });

  // Also update the combined risk gauge and number
  const combined = trustScores.combined_risk ?? trustScores.fraud_prob_score ?? 0;
  const gaugeNum = document.querySelector('#campaign-detail-content [style*="font-size:40px"]');
  if (gaugeNum) {
    const color = combined >= 70 ? 'var(--danger)' : combined >= 40 ? 'var(--warning)' : 'var(--success)';
    gaugeNum.textContent = combined;
    gaugeNum.style.color = color;

    // Update SVG arc
    const arc = document.querySelector('#campaign-detail-content path[stroke-dasharray]');
    if (arc) {
      arc.style.stroke = color;
      arc.setAttribute('stroke-dashoffset', 283 - (combined / 100) * 283);
    }
  }

  // Add a "Updated by AI" badge
  const lastUpdated = document.querySelector('#campaign-detail-content .card:last-child .card-title + div');
  if (lastUpdated) {
    lastUpdated.innerHTML = '<span style="color:var(--success);font-size:12px">✓ Updated by AI detection · just now</span>';
  }
}
function _buildCampaignActions(c) {
  const container = document.getElementById('campaign-actions-panel');
  if (!container) { console.warn('[actions] container not found'); return; }
  if (!container) return;
  container.innerHTML = ''; // clear

  const mk = (text, cls, style, onClick, disabled) => {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.className   = cls;
    btn.style.cssText = 'width:100%;' + (style || '');
    if (disabled) { btn.disabled = true; btn.style.opacity = '0.5'; btn.style.cursor = 'not-allowed'; }
    else if (onClick) btn.addEventListener('click', onClick);
    return btn;
  };

  // 1. Run AI Detection — always
  container.appendChild(mk('🔍 Run AI Detection', 'btn-primary-lg', '', () => adminRunAI(c.id)));

  // 2. Donate — show for all non-released campaigns (let backend validate)
  const donateBtn = document.createElement('button');
  donateBtn.textContent = '💜 Donate to Campaign';
  donateBtn.className   = 'btn-donate';
  donateBtn.style.cssText = 'width:100%';
  donateBtn.addEventListener('click', () => openDonateModal(c.id, c.title));
  container.appendChild(donateBtn);

  console.log('[donate check] funds_released:', c.funds_released, '| is_frozen:', c.is_frozen);

  // 3. Run Full Pipeline
  container.appendChild(mk('⚡ Run Full Pipeline', 'btn-outline-lg', '', () => runPipelineForCampaign(c.creator, c.id)));

  // 4. Release Funds
  if (c.funds_released) {
    container.appendChild(mk('✓ Funds Already Released', 'btn-outline-lg', '', null, true));
  } else if (c.goal_reached && !c.is_frozen) {
    const rel = mk('✓ Approve & Release Funds', 'btn-success', 'padding:10px;border:none;border-radius:6px;font-weight:600;', () => doRelease(c.id));
    container.appendChild(rel);
  } else {
    const msg = !c.goal_reached ? '⏳ Goal Not Reached Yet' : '🔒 Unfreeze Before Releasing';
    container.appendChild(mk(msg, 'btn-outline-lg', '', null, true));
  }

  // 5. Freeze / Unfreeze
  if (c.is_frozen) {
    container.appendChild(mk('🔓 Unfreeze Campaign', 'btn-success', 'padding:10px;border:none;border-radius:6px;font-weight:600;', () => doUnfreeze(c.creator)));
  } else {
    container.appendChild(mk('🔒 Freeze Campaign', 'btn-danger-lg', '', () => doFreeze(c.creator)));
  }

  // 6. Refund — only when frozen and not released
  if (c.is_frozen && !c.funds_released) {
    container.appendChild(mk('↩ Refund Contributors', 'btn-outline-lg', 'border-color:var(--warning);color:var(--warning);', () => adminRefund(c.id)));
  }

  // 7. Flag / Unflag
  if (c.is_flagged) {
    container.appendChild(mk('✓ Unflag Campaign', 'btn-outline-lg', '', () => doUnflag(c.creator)));
  } else {
    container.appendChild(mk('⚑ Flag as Fraudulent', 'btn-outline-lg', 'border-color:var(--warning);color:var(--warning);', () => doFlag(c.creator)));
  }
}
// ═══════════════════════════════════════

let _donateCampaignId = null;

function openDonateModal(campaignId, title) {
  _donateCampaignId = parseInt(campaignId, 10);
  const sub = document.getElementById('donate-modal-sub');
  if (sub) sub.textContent = 'Campaign #' + campaignId + ' — ' + (title || '');
  const amt = document.getElementById('donate-amount');
  if (amt) { amt.value = '10000000000000000'; updateDonatePreview(); }
  const res = document.getElementById('donate-result');
  if (res) { res.classList.add('hidden'); res.textContent = ''; }
  const btn = document.getElementById('donate-submit-btn');
  const btnText = document.getElementById('donate-btn-text');
  if (btn) btn.disabled = false;
  if (btnText) btnText.textContent = '💜 Contribute ETH';
  document.getElementById('donate-backdrop')?.classList.remove('hidden');
  document.getElementById('donate-modal')?.classList.remove('hidden');
  document.body.style.overflow = 'hidden';
  setTimeout(() => document.getElementById('donate-amount')?.focus(), 100);
}

function closeDonateModal() {
  document.getElementById('donate-backdrop')?.classList.add('hidden');
  document.getElementById('donate-modal')?.classList.add('hidden');
  document.body.style.overflow = '';
  _donateCampaignId = null;
}

function setDonateAmount(wei) {
  const el = document.getElementById('donate-amount');
  if (el) { el.value = wei; updateDonatePreview(); }
}

function updateDonatePreview() {
  const val = parseFloat(document.getElementById('donate-amount')?.value || '0');
  const el  = document.getElementById('donate-wei-preview');
  if (el) {
    if (!isNaN(val) && val > 0) {
      const ethVal = (val / 1e18).toFixed(6);
      el.textContent = `≈ ${ethVal} ETH`;
    } else {
      el.textContent = '≈ 0 ETH';
    }
  }
}

// Wire up live preview
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('donate-amount')?.addEventListener('input', updateDonatePreview);
});

async function submitDonation() {
  if (!_donateCampaignId && _donateCampaignId !== 0) {
    showToast('No campaign selected.', 'error'); return;
  }
  // Input is now in Wei — read as integer directly
  const amountWei = parseInt(document.getElementById('donate-amount')?.value || '0');
  if (!amountWei || amountWei <= 0) {
    showToast('Enter a valid amount in Wei.', 'error'); return;
  }
  if (!walletAddress) {
    showToast('Connect MetaMask first.', 'error'); return;
  }

  const btn     = document.getElementById('donate-submit-btn');
  const btnText = document.getElementById('donate-btn-text');
  const res     = document.getElementById('donate-result');
  if (btn) btn.disabled = true;
  if (btnText) btnText.innerHTML = '<span class="btn-spinner"></span> Waiting for MetaMask…';
  if (res) res.classList.add('hidden');

  try {
    const provider = new ethers.providers.Web3Provider(window.ethereum);
    const signer   = provider.getSigner();

    const abi = ['function contribute(uint256 _campaignId) payable'];
    const contract = new ethers.Contract(CROWDFUNDING_CORE_ADDR, abi, signer);

    const valueWei = ethers.BigNumber.from(amountWei.toString());

    // Dry-run first to surface the exact revert reason before prompting MetaMask
    try {
      await contract.callStatic.contribute(_donateCampaignId, { value: valueWei });
    } catch (staticErr) {
      // Decode the revert reason (ethers v5 puts it in .reason or nested .error)
      const reason =
        staticErr.reason ||
        staticErr.error?.message ||
        staticErr.data?.message ||
        staticErr.message ||
        'Transaction would revert on-chain';
      throw new Error(reason);
    }

    showToast('Confirm donation in MetaMask…');

    // MetaMask popup appears here — user sees amount + gas
    const tx = await contract.contribute(_donateCampaignId, { value: valueWei });

    if (btnText) btnText.innerHTML = '<span class="btn-spinner"></span> Waiting for confirmation…';
    showToast('Transaction sent — confirming…');

    const receipt = await tx.wait();

    if (receipt.status === 1) {
      if (res) {
        res.style.background = 'var(--success-bg)';
        res.style.border     = '1px solid var(--success)';
        res.style.color      = 'var(--success)';
        res.innerHTML = `✓ Donated ${amountWei} Wei! &nbsp;<span style="font-size:11px;font-family:var(--mono)">${receipt.transactionHash.slice(0,18)}…</span> · Block ${receipt.blockNumber}`;
        res.classList.remove('hidden');
      }
      showToast('Donation confirmed! ' + amountWei + ' Wei sent.', 'success');

      // Refresh campaign data
      setTimeout(() => {
        if (document.getElementById('page-campaigns')?.classList.contains('active')) loadCampaigns();
        if (document.getElementById('page-campaign-detail')?.classList.contains('active')) showCampaignDetail(_donateCampaignId);
      }, 1000);
    } else {
      throw new Error('Transaction reverted');
    }

  } catch (err) {
    const msg = err.message || String(err);
    const isRejected = msg.includes('user rejected') || msg.includes('User denied') || msg.includes('ACTION_REJECTED');
    const display = isRejected ? 'Transaction cancelled in MetaMask.' : msg.slice(0, 100);

    if (res) {
      res.style.background = 'var(--danger-bg)';
      res.style.border     = '1px solid var(--danger)';
      res.style.color      = 'var(--danger)';
      res.textContent      = '✗ ' + display;
      res.classList.remove('hidden');
    }
    showToast(isRejected ? 'Cancelled.' : 'Donation failed: ' + display, 'error');
  } finally {
    if (btn) btn.disabled = false;
    if (btnText) btnText.textContent = '💜 Contribute ETH';
  }
}

// ═══════════════════════════════════════
// INIT
// ═══════════════════════════════════════
(async () => {
  // Never auto-connect — always require explicit button click.
  // MetaMask's eth_accounts returns approved accounts silently,
  // which bypasses the login screen entirely. We block this deliberately.
  // Users must click "Connect with MetaMask" every time.
})();

// ═══════════════════════════════════════
// CREATE CAMPAIGN — calls CrowdfundingCore
// directly via window.ethereum (no backend)
// ═══════════════════════════════════════

const CROWDFUNDING_CORE_ADDR = '0x07eC712c3974Fb7d756B7A939010157392eeb4d2';

// Minimal ABI — only what we need in the browser
const CROWDFUNDING_ABI = [
  {
    name: 'createCampaign',
    type: 'function',
    stateMutability: 'nonpayable',
    inputs: [
      { name: '_title',        type: 'string'  },
      { name: '_description',  type: 'string'  },
      { name: '_goal',         type: 'uint256' },
      { name: '_durationDays', type: 'uint256' },
    ],
    outputs: [{ name: '', type: 'uint256' }],
  },
  {
    name: 'campaignCount',
    type: 'function',
    stateMutability: 'view',
    inputs: [],
    outputs: [{ name: '', type: 'uint256' }],
  },
];

// ── ABI encoder helpers (no ethers.js needed) ──────────────────────────
function toHex32(n) {
  // pad a BigInt or number to 32-byte hex
  return BigInt(n).toString(16).padStart(64, '0');
}

function encodeString(str) {
  const bytes = new TextEncoder().encode(str);
  const len   = toHex32(bytes.length);
  // pad to 32-byte boundary
  const padded = Math.ceil(bytes.length / 32) * 32;
  const hex = Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join('');
  return len + hex.padEnd(padded * 2, '0');
}

function encodeCreateCampaign(title, description, goalWei, durationDays) {
  // function selector: keccak256("createCampaign(string,string,uint256,uint256)")[0:4]
  const selector = '380ef3fe';

  // Dynamic types: offset pointers (4 params, 2 dynamic strings at end)
  // param 0 (title)       → dynamic, offset = 4*32 = 0x80
  // param 1 (description) → dynamic, offset = 0x80 + len(encoded title)
  // param 2 (goal)        → static uint256
  // param 3 (duration)    → static uint256

  const titleEnc = encodeString(title);
  const descEnc  = encodeString(description);

  const offset0 = 4 * 32;                    // 128 = 0x80
  const offset1 = offset0 + titleEnc.length / 2;

  return '0x' + selector +
    toHex32(offset0) +      // offset to title
    toHex32(offset1) +      // offset to description
    toHex32(goalWei) +      // goal (wei)
    toHex32(durationDays) + // duration days
    titleEnc +
    descEnc;
}

// ── ETH → Wei conversion ──────────────────────────────────────────────
function ethToWei(eth) {
  // avoid float precision: split on decimal point
  const [whole, frac = ''] = String(eth).split('.');
  const fracPadded = (frac + '0'.repeat(18)).slice(0, 18);
  return BigInt(whole) * BigInt('1000000000000000000') + BigInt(fracPadded);
}

// ── Live char counter + wei preview ──────────────────────────────────
function initCreateFormListeners() {
  const desc = document.getElementById('cf-desc');
  if (desc) {
    desc.addEventListener('input', () => {
      const c = document.getElementById('cf-desc-count');
      if (c) c.textContent = desc.value.length;
    });
  }
  const goal = document.getElementById('cf-goal');
  if (goal) {
    goal.addEventListener('input', () => {
      const el = document.getElementById('cf-goal-wei');
      if (!el) return;
      try {
        const wei = ethToWei(goal.value || '0');
        el.textContent = '≈ ' + wei.toLocaleString() + ' wei';
      } catch (_) {
        el.textContent = 'Invalid amount';
      }
    });
  }
  // update wallet badge
  const wb = document.getElementById('create-wallet-badge');
  if (wb && walletAddress) {
    wb.textContent = walletAddress.slice(0, 6) + '…' + walletAddress.slice(-4);
    wb.className = 'badge-mini green';
  }
}

// ── Main create function ──────────────────────────────────────────────
async function createCampaign() {
  document.getElementById('create-result-card')?.classList.add('hidden');
  document.getElementById('create-error-card')?.classList.add('hidden');

  const title   = document.getElementById('cf-title')?.value?.trim()  || '';
  const desc    = document.getElementById('cf-desc')?.value?.trim()   || '';
  const goalEth = document.getElementById('cf-goal')?.value           || '';
  const days    = parseInt(document.getElementById('cf-days')?.value  || '0');

  if (!title)        { showFieldError('cf-title', 'Title is required');        return; }
  if (!desc)         { showFieldError('cf-desc',  'Description is required');  return; }
  if (!goalEth || parseFloat(goalEth) <= 0) { showFieldError('cf-goal', 'Enter a valid goal'); return; }
  if (!days || days < 1 || days > 90)       { showFieldError('cf-days', '1–90 days');          return; }
  if (!walletAddress) { showToast('Connect MetaMask first.', 'error'); return; }

  const btn     = document.getElementById('cf-submit-btn');
  const btnText = document.getElementById('cf-btn-text');
  btn.disabled  = true;
  btnText.innerHTML = '<span class="btn-spinner"></span> Waiting for MetaMask…';

  try {
    // Use ethers.js + MetaMask provider to call createCampaign
    const provider = new ethers.providers.Web3Provider(window.ethereum);
    const signer   = provider.getSigner();

    const abi = [
      'function createCampaign(string _title, string _description, uint256 _goal, uint256 _durationDays) returns (uint256)'
    ];
    const contract = new ethers.Contract(CROWDFUNDING_CORE_ADDR, abi, signer);

    const goalWei = ethers.utils.parseEther(String(goalEth));

    showToast('Confirm the transaction in MetaMask…');

    // This triggers MetaMask popup
    const tx = await contract.createCampaign(title, desc, goalWei, days);
    btnText.innerHTML = '<span class="btn-spinner"></span> Waiting for confirmation…';
    showToast('Transaction sent — waiting for confirmation…');

    const receipt = await tx.wait();   // waits for 1 confirmation

    if (receipt.status !== 1) {
      throw new Error('Transaction reverted (status 0). Check MetaMask for details.');
    }

    // Get new campaign ID from event log or campaignCount
    let campaignId = '?';
    try {
      // CampaignCreated event: topic[0] = event sig, topic[1] = campaignId (indexed)
      const created = receipt.events?.find(e => e.event === 'CampaignCreated');
      if (created) {
        campaignId = created.args[0].toNumber();
      } else {
        // Fallback: read campaignCount - 1
        const countAbi = ['function campaignCount() view returns (uint256)'];
        const coreView = new ethers.Contract(CROWDFUNDING_CORE_ADDR, countAbi, provider);
        const count    = await coreView.campaignCount();
        campaignId     = count.toNumber() - 1;
      }
    } catch (_) {}

    showCreateSuccess(
      receipt.transactionHash,
      { blockNumber: receipt.blockNumber, gasUsed: receipt.gasUsed?.toNumber?.() ?? 0, status: '0x1' },
      campaignId, title, goalEth, days
    );
    showToast('Campaign #' + campaignId + ' created on Sepolia!', 'success');

    // Auto-run fraud detection
    setTimeout(() => autoRunDetectionOnNewCampaign(campaignId, title, desc), 1500);

    // Refresh campaign list
    setTimeout(loadCampaigns, 2000);

  } catch (err) {
    const msg = err.message || String(err);
    // MetaMask rejection is not an error to display loudly
    if (msg.includes('user rejected') || msg.includes('User denied')) {
      showToast('Transaction cancelled.', 'error');
    } else {
      document.getElementById('create-error-content').textContent = msg;
      document.getElementById('create-error-card')?.classList.remove('hidden');
      showToast('Failed: ' + msg.slice(0, 80), 'error');
    }
  } finally {
    btn.disabled = false;
    btnText.innerHTML = '🚀 Deploy Campaign on Sepolia';
  }
}

// ── Poll for receipt every 3s (max 2 min) ────────────────────────────
async function waitForReceipt(txHash, maxAttempts = 40) {
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise(r => setTimeout(r, 3000));
    const receipt = await window.ethereum.request({
      method: 'eth_getTransactionReceipt',
      params: [txHash],
    });
    if (receipt) return receipt;
  }
  throw new Error('Transaction not confirmed within 2 minutes. Check Etherscan: https://sepolia.etherscan.io/tx/' + txHash);
}

// ── Show success card ────────────────────────────────────────────────
function showCreateSuccess(txHash, receipt, campaignId, title, goalEth, days) {
  const card    = document.getElementById('create-result-card');
  const content = document.getElementById('create-result-content');
  card.classList.remove('hidden');

  const etherscanUrl = 'https://sepolia.etherscan.io/tx/' + txHash;
  const deadline = new Date(Date.now() + days * 86400000).toLocaleDateString();

  content.innerHTML = `
    <div class="create-result-row"><span class="crr-lbl">Campaign ID</span><span class="crr-val">#${campaignId}</span></div>
    <div class="create-result-row"><span class="crr-lbl">Title</span><span class="crr-val">${esc(title)}</span></div>
    <div class="create-result-row"><span class="crr-lbl">Goal</span><span class="crr-val">${goalEth} ETH</span></div>
    <div class="create-result-row"><span class="crr-lbl">Duration</span><span class="crr-val">${days} days · ends ${deadline}</span></div>
    <div class="create-result-row"><span class="crr-lbl">Block</span><span class="crr-val">${typeof receipt.blockNumber === 'string' ? parseInt(receipt.blockNumber, 16) : receipt.blockNumber}</span></div>
    <div class="create-result-row"><span class="crr-lbl">Gas Used</span><span class="crr-val">${(typeof receipt.gasUsed === 'string' ? parseInt(receipt.gasUsed, 16) : receipt.gasUsed).toLocaleString()}</span></div>
    <div class="create-result-row">
      <span class="crr-lbl">TX Hash</span>
      <a href="${etherscanUrl}" target="_blank" style="font-size:11px;font-family:var(--mono);color:var(--accent2)">
        ${txHash.slice(0, 18)}… ↗
      </a>
    </div>
    <div style="margin-top:12px;padding:8px 12px;background:rgba(79,110,247,.1);border:1px solid rgba(79,110,247,.2);border-radius:6px;font-size:12px;color:var(--text2)">
      ⏳ TrustNet is running fraud detection on campaign #${campaignId}…
    </div>
    <div style="display:flex;gap:8px;margin-top:12px">
      <button class="btn-sm-primary" onclick="showPage('campaigns')">View All Campaigns</button>
      <button class="btn-sm-outline" onclick="showCampaignDetail(${campaignId})">View This Campaign</button>
    </div>`;

  // scroll to result
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Auto-run TrustNet detection on the new campaign ───────────────────
async function autoRunDetectionOnNewCampaign(campaignId, title, desc) {
  try {
    // fetch the created campaign's data from backend to get creator address
    const data = await apiFetch('/campaign/' + campaignId);
    // run detection with description as SDI signal (sdi_score defaults to 0 until SDI pipeline runs)
    const features = { combined_risk_score: 0 };
    const result   = await apiFetch('/predict', {
      method: 'POST',
      body: JSON.stringify({ campaign_id: 'CAMP_' + campaignId, features }),
    });

    const resultEl = document.querySelector('#create-result-content div:last-of-type div:last-child');
    const infoBox  = document.querySelector('#create-result-content div[style*="rgba(79,110,247"]');
    if (infoBox) {
      const isF = result.is_fraud;
      infoBox.style.background = isF ? 'var(--danger-bg)' : 'var(--success-bg)';
      infoBox.style.borderColor = isF ? 'var(--danger)' : 'var(--success)';
      infoBox.style.color = isF ? 'var(--danger)' : 'var(--success)';
      infoBox.textContent = isF
        ? `⚠ TrustNet: Fraud probability ${(result.fraud_probability * 100).toFixed(1)}% — campaign flagged for review`
        : `✓ TrustNet: Fraud probability ${(result.fraud_probability * 100).toFixed(1)}% — campaign appears legitimate`;
    }
  } catch (_) {
    // silent — detection is a bonus, not critical to campaign creation
  }
}

// ── Helpers ────────────────────────────────────────────────────────────
function showFieldError(id, msg) {
  const el = document.getElementById(id);
  if (el) el.classList.add('error');
  showToast(msg, 'error');
  el?.focus();
  setTimeout(() => el?.classList.remove('error'), 3000);
}

function clearCreateForm() {
  ['cf-title','cf-goal','cf-days'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = '';
  });
  const desc = document.getElementById('cf-desc');
  if (desc) desc.value = '';
  const cnt = document.getElementById('cf-desc-count');
  if (cnt) cnt.textContent = '0';
  const wei = document.getElementById('cf-goal-wei');
  if (wei) wei.textContent = '≈ 0 wei';
  document.getElementById('create-result-card')?.classList.add('hidden');
  document.getElementById('create-error-card')?.classList.add('hidden');
}

// initialise listeners when page shown — merged into showPage above

// ═══════════════════════════════════════
// CSV ANALYSIS — Feature Analysis page
// ═══════════════════════════════════════

// ── File selection ────────────────────────────────────────────────────
function handleCsvDrop(ev) {
  ev.preventDefault();
  const file = ev.dataTransfer?.files?.[0];
  if (file) setCsvFile(file);
}
function handleCsvSelect(ev) {
  const file = ev.target?.files?.[0];
  if (file) setCsvFile(file);
}
function setCsvFile(file) {
  if (!file.name.endsWith('.csv')) {
    showToast('Only CSV files are accepted.', 'error'); return;
  }
  _csvFile = file;

  // Update upload zone visuals
  const icon  = document.getElementById('fa-upload-icon');
  const title = document.getElementById('fa-upload-title');
  const sub   = document.getElementById('fa-upload-sub');
  const zone  = document.getElementById('fa-upload-zone');
  if (icon)  icon.textContent  = '✓';
  if (title) title.textContent = file.name;
  if (sub)   sub.textContent   = (file.size / 1024).toFixed(1) + ' KB · ready to analyse';
  if (zone)  zone.style.borderColor = 'var(--success)';

  const info = document.getElementById('fa-file-info');
  if (info) {
    info.textContent = '✓ ' + file.name + ' · ' + (file.size / 1024).toFixed(1) + ' KB';
    info.classList.remove('hidden');
  }
  const btn = document.getElementById('fa-analyse-btn');
  if (btn) btn.style.display = 'block';
}

// ── Animate pipeline steps during upload ─────────────────────────────
const STEPS = ['fvrs','cti','cfis','ctcs','sdi','trustnet'];
let _stepTimer = null; // kept for safety — animation is server-driven

function startPipelineAnimation() {
  STEPS.forEach(s => setStep(s, '', 'Waiting'));
  document.getElementById('fa-progress-wrap')?.classList.remove('hidden');
  document.getElementById('fa-progress-msg').textContent = 'Uploading CSV to server…';
  // Mark first step active immediately as visual feedback
  setStep(STEPS[0], 'active', 'Running…');
}

function stopPipelineAnimation(success) {
  STEPS.forEach(s => setStep(s, success ? 'done' : 'error', success ? 'Done \u2713' : 'Error'));
  const msg = document.getElementById('fa-progress-msg');
  if (msg) msg.textContent = success ? 'Analysis complete!' : 'Analysis failed.';
}

function setStep(id, cls, label) {
  const el = document.getElementById('fp-' + id);
  if (!el) return;
  el.className = 'fp-step ' + cls;
  el.querySelector('.fp-status').textContent = label;
}

// ── Run analysis — async with polling ────────────────────────────────
async function runCsvAnalysis() {
  if (!_csvFile) { showToast('Select a CSV file first.', 'error'); return; }

  const btn     = document.getElementById('fa-analyse-btn');
  const btnText = document.getElementById('fa-analyse-text');
  btn.disabled  = true;
  btnText.innerHTML = '<span class="btn-spinner"></span> Uploading…';
  document.getElementById('fa-error-box')?.classList.add('hidden');

  startPipelineAnimation();
  showToast('Reading CSV and starting analysis…');

  try {
    // Step 1 — Read file as base64, send as JSON (avoids CORS preflight on multipart)
    const base64data = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload  = () => resolve(reader.result.split(',')[1]); // strip data URL prefix
      reader.onerror = reject;
      reader.readAsDataURL(_csvFile);
    });

    let startRes, startData;
    try {
      startRes  = await fetch(API + '/analyse-csv', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ filename: _csvFile.name, data: base64data }),
      });
      startData = await startRes.json();
    } catch (netErr) {
      throw new Error(
        'Network error contacting ' + API + '/analyse-csv — is the Flask server running? (' + netErr.message + ')'
      );
    }
    if (!startRes.ok) throw new Error(startData.error || 'Upload failed: HTTP ' + startRes.status);

    const jobId = startData.job_id;
    if (!jobId) throw new Error('Server did not return a job_id. Response: ' + JSON.stringify(startData));

    // Show notice if simulator was used
    if (startData.simulated) {
      const info = document.getElementById('fa-file-info');
      if (info) {
        info.textContent = '⚠ Some columns were missing — dataset_simulator.py was run automatically. Analysing simulated data.';
        info.style.background = 'rgba(245,166,35,.15)';
        info.style.borderColor = 'var(--warning)';
        info.style.color = 'var(--warning)';
      }
    }
    showToast('Analysis running… polling for results.');
    btnText.innerHTML = '<span class="btn-spinner"></span> Analysing…';

    // Step 2 — Poll status every 3 seconds until done or error
    const result = await pollJobStatus(jobId);

    stopPipelineAnimation(true);
    _csvResults  = result.campaigns || [];
    _csvFiltered = [..._csvResults];

    showCsvResults(result);
    showToast('✓ ' + _csvResults.length + ' campaigns analysed!', 'success');

  } catch (e) {
    stopPipelineAnimation(false);
    const errBox = document.getElementById('fa-error-box');
    if (errBox) { errBox.textContent = '✗ ' + e.message; errBox.classList.remove('hidden'); }
    showToast('Analysis failed: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btnText.innerHTML = '⚡ Run Full TrustNet Analysis';
  }
}

// ── Poll /analyse-csv/status/<job_id> until done ──────────────────────
async function pollJobStatus(jobId, intervalMs = 3000, maxWaitMs = 300000) {
  const STEP_NAMES = ['FVRS', 'CTI', 'CFIS', 'CTCS', 'SDI', 'TrustNet'];
  const started = Date.now();

  while (Date.now() - started < maxWaitMs) {
    await new Promise(r => setTimeout(r, intervalMs));

    let res, data;
    try {
      res  = await fetch(API + '/analyse-csv/status/' + jobId);
      data = await res.json();
    } catch (fetchErr) {
      // Network hiccup — just retry next interval
      const msg = document.getElementById('fa-progress-msg');
      if (msg) msg.textContent = 'Waiting for server... retrying';
      continue;
    }

    if (data.status === 'error') {
      throw new Error(data.error || 'Server-side pipeline error');
    }

    if (data.status === 'running') {
      // Update pipeline step animation to match server progress
      const step = data.step ?? 0;
      STEP_NAMES.forEach((name, i) => {
        if (i < step)        setStep(name.toLowerCase(), 'done',   'Done');
        else if (i === step) setStep(name.toLowerCase(), 'active', 'Running...');
        else                 setStep(name.toLowerCase(), '',       'Waiting');
      });
      const msg = document.getElementById('fa-progress-msg');
      if (msg) msg.textContent = 'Running ' + (data.step_name || '') + ' pipeline…';
      continue;
    }

    if (data.status === 'done') {
      if (!data.campaigns || !data.summary) {
        throw new Error(data.error || 'Unexpected response from server');
      }
      return data;
    }
  }
  throw new Error('Analysis timed out after 5 minutes. Check Flask terminal for errors.');
}

// ── Render results ────────────────────────────────────────────────────
function showCsvResults(data) {
  // switch panels
  document.getElementById('fa-upload-state').classList.add('hidden');
  document.getElementById('fa-results-state').classList.remove('hidden');
  document.getElementById('fa-reset-btn').style.display = 'block';

  // KPIs — guard against missing summary
  const summary = data.summary || {};
  const kpiEl = document.getElementById('fa-kpis');
  const avg   = ((summary.mean_fraud_probability || 0) * 100).toFixed(1);
  const max   = ((summary.max_fraud_probability  || 0) * 100).toFixed(1);
  kpiEl.innerHTML = `
    <div class="kpi-card"><div class="kpi-val">${data.total || 0}</div><div class="kpi-lbl">TOTAL CAMPAIGNS</div></div>
    <div class="kpi-card fraud-kpi"><div class="kpi-val">${data.fraud_count || 0}</div><div class="kpi-lbl">FRAUD DETECTED</div><div class="kpi-sub">${summary.high_risk_count || 0} high-risk (≥70)</div></div>
    <div class="kpi-card"><div class="kpi-val">${avg}%</div><div class="kpi-lbl">AVG FRAUD PROBABILITY</div></div>
    <div class="kpi-card"><div class="kpi-val">${max}%</div><div class="kpi-lbl">HIGHEST FRAUD PROB</div></div>`;

  // Badge
  const badge = document.getElementById('fa-summary-badge');
  if (badge) {
    badge.textContent = data.fraud_count + ' fraud / ' + data.total + ' campaigns';
    badge.className   = 'badge-mini ' + (data.fraud_count > 0 ? 'red' : 'green');
    badge.classList.remove('hidden');
  }

  renderCsvTable(_csvResults);
}

function renderCsvTable(rows) {
  const tbody = document.getElementById('fa-tbody');
  if (!tbody) return;
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="10" class="tbl-loading">No campaigns match the filter.</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map(r => {
    const prob   = (r.fraud_probability * 100).toFixed(1);
    const ts     = r.trust_scores;
    const cr     = ts.combined_risk;
    const rcls   = cr >= 70 ? 'risk-high' : cr >= 40 ? 'risk-med' : 'risk-low';
    // Flag as fraud if ML model says so OR if combined risk >= threshold (70)
    const isF    = r.is_fraud || cr >= 70;
    return `
    <tr style="cursor:pointer" onclick="openCsvModal('${r.campaign_id}')">
      <td><code style="font-size:12px;color:var(--accent2)">${esc(r.campaign_id)}</code></td>
      <td style="font-weight:600;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(r.title)}</td>
      <td>
        <div style="display:flex;align-items:center;gap:6px">
          <div style="width:50px;height:5px;background:var(--border);border-radius:3px;overflow:hidden">
            <div style="width:${prob}%;height:5px;background:${isF ? 'var(--danger)' : 'var(--success)'};border-radius:3px"></div>
          </div>
          <span style="font-weight:700;color:${isF ? 'var(--danger)' : 'var(--success)'}">${prob}%</span>
        </div>
      </td>
      <td>${ts.sdi}</td>
      <td>${ts.cti}</td>
      <td>${ts.fvrs}</td>
      <td>${ts.cfis}</td>
      <td>${ts.ctcs}</td>
      <td>
        <div style="display:flex;align-items:center;gap:6px">
          <div style="width:40px;height:5px;background:var(--border);border-radius:3px;overflow:hidden">
            <div class="${rcls}" style="width:${cr}%;height:5px;border-radius:3px"></div>
          </div>
          <span style="font-weight:700">${cr}</span>
        </div>
      </td>
      <td><span class="chip ${isF ? 'chip-red' : 'chip-green'}">${isF ? 'FRAUD' : 'LEGIT'}</span></td>
    </tr>`;
  }).join('');
}

// ── Filter ────────────────────────────────────────────────────────────
function filterCsvTable() {
  const q       = (document.getElementById('fa-search')?.value || '').toLowerCase();
  const verdict = document.getElementById('fa-filter-verdict')?.value || '';
  const risk    = document.getElementById('fa-filter-risk')?.value    || '';

  _csvFiltered = _csvResults.filter(r => {
    const cr = r.trust_scores.combined_risk;
    const matchQ = !q || r.campaign_id.toLowerCase().includes(q) || r.title.toLowerCase().includes(q);
    const matchV = !verdict ||
      (verdict === 'fraud' && r.is_fraud) ||
      (verdict === 'legit' && !r.is_fraud);
    const matchR = !risk ||
      (risk === 'high'   && cr >= 70) ||
      (risk === 'medium' && cr >= 40 && cr < 70) ||
      (risk === 'low'    && cr < 40);
    return matchQ && matchV && matchR;
  });
  renderCsvTable(_csvFiltered);
}

// ── Campaign Detail Modal ─────────────────────────────────────────────
function openCsvModal(campaignId) {
  const r = _csvResults.find(x => x.campaign_id === campaignId);
  if (!r) return;

  const ts   = r.trust_scores;
  const cr   = ts.combined_risk;
  const prob = (r.fraud_probability * 100).toFixed(1);
  const isF  = r.is_fraud || cr >= 70;
  const crColor = cr >= 70 ? 'var(--danger)' : cr >= 40 ? 'var(--warning)' : 'var(--success)';
  const f    = r.features;
  const sdi  = r.sdi_details;

  document.getElementById('fa-modal-title').textContent = r.title || r.campaign_id;
  document.getElementById('fa-modal-sub').textContent   = r.campaign_id + ' · ' +
    (isF ? '⚠ Fraud Detected' : '✓ Legitimate');

  const body = document.getElementById('fa-modal-body');
  body.innerHTML = `
    <!-- Verdict + gauge -->
    <div class="modal-gauge-row">
      <div>
        <svg class="modal-gauge-svg" viewBox="0 0 180 110" width="180" height="110">
          <path d="M18,100 A80,80 0 0,1 162,100" fill="none" stroke="var(--border)" stroke-width="14" stroke-linecap="round"/>
          <path d="M18,100 A80,80 0 0,1 162,100" fill="none" stroke="${isF ? 'var(--danger)' : cr >= 40 ? 'var(--warning)' : 'var(--success)'}"
                stroke-width="14" stroke-linecap="round"
                stroke-dasharray="251" stroke-dashoffset="${251 - (r.fraud_probability * 251)}"/>
        </svg>
        <div class="modal-gauge-val" style="color:${isF ? 'var(--danger)' : 'var(--success)'}">${prob}%</div>
        <div class="modal-gauge-lbl">Fraud Probability</div>
      </div>
      <div>
        <div class="verdict-chip ${isF ? 'fraud' : 'legit'}" style="margin-bottom:12px;font-size:15px">
          ${isF ? '⚠ FRAUD DETECTED' : '✓ LEGITIMATE'}
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
          <div class="mfg-item"><div class="mfg-lbl">SDI</div><div class="mfg-val">${ts.sdi}</div><div class="mfg-sub">Semantic Drift</div></div>
          <div class="mfg-item"><div class="mfg-lbl">CTI</div><div class="mfg-val">${ts.cti}</div><div class="mfg-sub">Crowd Trust</div></div>
          <div class="mfg-item"><div class="mfg-lbl">FVRS</div><div class="mfg-val">${ts.fvrs}</div><div class="mfg-sub">Velocity Risk</div></div>
          <div class="mfg-item"><div class="mfg-lbl">CFIS</div><div class="mfg-val">${ts.cfis}</div><div class="mfg-sub">Collusion</div></div>
          <div class="mfg-item"><div class="mfg-lbl">CTCS</div><div class="mfg-val">${ts.ctcs}</div><div class="mfg-sub">Temporal</div></div>
          <div class="mfg-item" style="border-color:${crColor}"><div class="mfg-lbl">COMBINED</div><div class="mfg-val" style="color:${crColor}">${cr}</div><div class="mfg-sub">/100</div></div>
        </div>
      </div>
    </div>

    <!-- Combined risk bar -->
    <div class="card" style="padding:14px;margin-bottom:12px">
      <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-bottom:6px">
        <span>Combined Risk Score</span><span style="font-weight:700;color:${crColor}">${cr}/100 · Threshold 70</span>
      </div>
      <div style="height:10px;background:var(--border);border-radius:5px;overflow:hidden">
        <div style="height:10px;width:${cr}%;background:${crColor};border-radius:5px;transition:width .5s"></div>
      </div>
    </div>

    <!-- Sub-feature breakdown -->
    <div class="card" style="padding:14px;margin-bottom:12px">
      <div class="card-title" style="margin-bottom:10px">Sub-Feature Breakdown</div>
      <div class="modal-feat-grid">
        ${[
          ['FGR','fgr','Funding Growth'],['TBR','tbr','Burst Rate'],['NUR','nur','New User Rate'],
          ['FTR','ftr','Timing Regularity'],['RAR','rar','Round Amount'],
          ['WGD','wgd','Graph Density'],['CDS','cds','Community'],['COL','col','Collision'],
          ['INF','inf','Influence'],['FTS','fts','Time Similarity'],['FAS','fas','Amt Similarity'],
          ['CTCC','ctcc','Correlation'],['TS','ts','Sync Score'],['CTCS','ctcs','Temporal']
        ].map(([lbl,key,sub]) =>
          `<div class="mfg-item"><div class="mfg-lbl">${lbl}</div><div class="mfg-val">${(f[key] !== undefined ? f[key] : '--')}</div><div class="mfg-sub">${sub}</div></div>`
        ).join('')}
      </div>
    </div>

    <!-- SDI plagiarism details -->
    ${sdi && sdi.sdi_score !== undefined ? `
    <div class="card" style="padding:14px;margin-bottom:12px">
      <div class="card-title" style="margin-bottom:8px">SDI · Plagiarism Analysis</div>
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px">
        <span class="chip ${sdi.risk_label && sdi.risk_label.includes('HIGH') ? 'chip-red' : sdi.risk_label && sdi.risk_label.includes('MODERATE') ? 'chip-yellow' : 'chip-green'}">${sdi.risk_label || 'UNKNOWN'}</span>
        <span style="font-size:13px;font-weight:700">Score: ${f.sdi_score}</span>
      </div>
      ${sdi.top_match_1 ? `<div style="font-size:12px;color:var(--text2);margin-bottom:4px">Top match: <strong>${esc(sdi.top_match_1)}</strong> (similarity: ${sdi.top_match_1_sim})</div>` : ''}
      ${sdi.top_match_2 ? `<div style="font-size:12px;color:var(--muted)">2nd match: <strong>${esc(sdi.top_match_2)}</strong> (similarity: ${sdi.top_match_2_sim})</div>` : ''}
    </div>` : ''}

    <!-- CTI detail -->
    <div class="card" style="padding:14px">
      <div class="card-title" style="margin-bottom:8px">CTI · Trust Breakdown</div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px">
        <div class="mfg-item"><div class="mfg-lbl">Approval (A)</div><div class="mfg-val">${f.cti_score !== undefined ? f.cti_score : '--'}</div></div>
        <div class="mfg-item"><div class="mfg-lbl">CTI Score</div><div class="mfg-val">${f.cti_norm !== undefined ? f.cti_norm : '--'}</div></div>
        <div class="mfg-item"><div class="mfg-lbl">Combined</div><div class="mfg-val" style="color:var(--success)">${ts.cti}</div><div class="mfg-sub">/100</div></div>
      </div>
    </div>`;

  document.getElementById('fa-modal-backdrop').classList.remove('hidden');
  document.getElementById('fa-modal').classList.remove('hidden');
  document.body.style.overflow = 'hidden';
}

function closeCsvModal() {
  document.getElementById('fa-modal-backdrop').classList.add('hidden');
  document.getElementById('fa-modal').classList.add('hidden');
  document.body.style.overflow = '';
}

// ── Export ────────────────────────────────────────────────────────────
function exportCsvResults() {
  if (!_csvResults.length) { showToast('No results to export.', 'error'); return; }
  const header = 'campaign_id,title,fraud_probability,is_fraud,sdi,cti,fvrs,cfis,ctcs,combined_risk\n';
  const rows   = _csvResults.map(r => {
    const ts = r.trust_scores;
    return [r.campaign_id, '"'+r.title.replace(/"/g,'""')+'"',
      r.fraud_probability, r.is_fraud, ts.sdi, ts.cti, ts.fvrs, ts.cfis, ts.ctcs, ts.combined_risk
    ].join(',');
  }).join('\n');
  const blob = new Blob([header + rows], { type: 'text/csv' });
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = 'trustnet_results.csv';
  a.click();
  showToast('Exported trustnet_results.csv', 'success');
}

// ── Reset ─────────────────────────────────────────────────────────────
function resetCsvAnalysis() {
  _csvFile     = null;
  _csvResults  = [];
  _csvFiltered = [];

  const show = document.getElementById('fa-upload-state');
  const hide = document.getElementById('fa-results-state');
  const rst  = document.getElementById('fa-reset-btn');
  if (show) show.classList.remove('hidden');
  if (hide) hide.classList.add('hidden');
  if (rst)  rst.style.display = 'none';

  const analyseBtn = document.getElementById('fa-analyse-btn');
  if (analyseBtn) analyseBtn.style.display = 'none';
  document.getElementById('fa-file-info')?.classList.add('hidden');
  document.getElementById('fa-progress-wrap')?.classList.add('hidden');
  document.getElementById('fa-error-box')?.classList.add('hidden');

  const icon  = document.getElementById('fa-upload-icon');
  const title = document.getElementById('fa-upload-title');
  const sub   = document.getElementById('fa-upload-sub');
  const zone  = document.getElementById('fa-upload-zone');
  if (icon)  icon.textContent  = '☁';
  if (title) title.textContent = 'Drag & drop unified_dataset.csv';
  if (sub)   sub.textContent   = 'or click to browse';
  if (zone)  zone.style.borderColor = '';

  const badge = document.getElementById('fa-summary-badge');
  if (badge) badge.classList.add('hidden');

  const input = document.getElementById('fa-csv-input');
  if (input) input.value = '';

  STEPS.forEach(s => {
    const el = document.getElementById('fp-' + s);
    if (el) {
      el.className = 'fp-step';
      const st = el.querySelector('.fp-status');
      if (st) st.textContent = 'Waiting';
    }
  });
}
