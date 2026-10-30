/**
 * PHISHFINDER RADAR — Tactical Frontend Controller
 * DistilBERT Subword Self-Attention & 8-Head Neural Ensemble
 * BITS Pilani Project
 */

document.addEventListener("DOMContentLoaded", () => {
  // Global State
  let currentBatchData = [];
  let currentActiveFilter = "all";
  const HISTORY_KEY = "phishfinder_radar_history_v2";
  const SVG_CIRCUMFERENCE = 414.7; // 2 * PI * 66

  // Cache DOM Elements
  const modeBtns = document.querySelectorAll(".mode-btn");
  const viewPanels = document.querySelectorAll(".view-panel");
  const scanForm = document.getElementById("url-scan-form");
  const urlInput = document.getElementById("url-input");
  const btnAnalyze = document.getElementById("btn-analyze");
  const btnAnalyzeText = document.getElementById("btn-analyze-text");
  const btnPaste = document.getElementById("btn-paste-url");
  const sampleChips = document.querySelectorAll(".sample-chip");
  const scanResultsArea = document.getElementById("scan-results-area");
  const btnCopyUrl = document.getElementById("btn-copy-url");
  const copyStatusText = document.getElementById("copy-status-text");

  // Token Inspector HUD
  const hudSubword = document.getElementById("hud-subword-name");
  const hudWeight = document.getElementById("hud-subword-weight");
  const hudThreat = document.getElementById("hud-subword-threat");

  // Batch DOM
  const btnRun30Tests = document.getElementById("btn-run-30-notebook-tests");
  const btnRun20Presets = document.getElementById("btn-run-20-presets");
  const btnRunCustomBatch = document.getElementById("btn-run-custom-batch");
  const btnClearBatch = document.getElementById("btn-clear-batch");
  const batchTextarea = document.getElementById("batch-textarea");
  const batchFileInput = document.getElementById("batch-file-input");
  const btnUploadFile = document.getElementById("btn-upload-file");
  const btnExportCsv = document.getElementById("btn-export-csv");
  const batchSummaryCards = document.getElementById("batch-summary-cards");
  const batchResultsWrapper = document.getElementById("batch-results-wrapper");
  const batchTableBody = document.getElementById("batch-table-body");
  const filterPills = document.querySelectorAll(".filter-pill");

  // Toast
  const toastBox = document.getElementById("toast-notify");
  const toastMsg = document.getElementById("toast-msg");

  // =========================================================================
  // 1. Tactical View Switching (Live Radar vs 30-Site Test Suite)
  // =========================================================================
  function switchMode(targetViewId) {
    modeBtns.forEach(btn => {
      const match = btn.getAttribute("aria-controls") === targetViewId;
      btn.classList.toggle("active", match);
      btn.setAttribute("aria-selected", match ? "true" : "false");
    });

    viewPanels.forEach(panel => {
      const match = panel.id === targetViewId;
      panel.classList.toggle("active", match);
    });

    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  modeBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("aria-controls");
      if (target) switchMode(target);
    });
  });

  // =========================================================================
  // 2. Toast Notifications
  // =========================================================================
  let toastTimer = null;
  function showToast(message) {
    if (!toastBox || !toastMsg) return;
    toastMsg.textContent = message;
    toastBox.classList.remove("hidden");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toastBox.classList.add("hidden");
    }, 2800);
  }

  // =========================================================================
  // 3. System Status Initialization
  // =========================================================================
  async function checkEngineStatus() {
    renderHistory();
    try {
      const res = await fetch("/api/status");
      if (res.ok) {
        const data = await res.json();
        const modeEl = document.getElementById("engine-mode-text");
        if (modeEl) {
          modeEl.textContent = data.engine_mode ? `${data.engine_mode}` : "Neural Engine Online";
        }
      }
    } catch (e) {
      console.warn("Status check failed:", e);
    }
  }

  // =========================================================================
  // 4. Live URL Scanner Execution
  // =========================================================================
  if (scanForm) {
    scanForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const targetUrl = urlInput.value.trim();
      if (targetUrl) executeScan(targetUrl);
    });
  }

  if (btnPaste && urlInput) {
    btnPaste.addEventListener("click", async () => {
      try {
        const text = await navigator.clipboard.readText();
        if (text) {
          urlInput.value = text.trim();
          executeScan(urlInput.value);
        }
      } catch (err) {
        urlInput.focus();
        showToast("Clipboard access denied. Please paste manually.");
      }
    });
  }

  sampleChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const sampleUrl = chip.getAttribute("data-url");
      if (sampleUrl && urlInput) {
        urlInput.value = sampleUrl;
        switchMode("view-scanner");
        executeScan(sampleUrl);
      }
    });
  });

  if (btnCopyUrl) {
    btnCopyUrl.addEventListener("click", () => {
      const urlText = document.getElementById("verdict-url-display")?.textContent;
      if (urlText && urlText !== "https://...") {
        navigator.clipboard.writeText(urlText).then(() => {
          if (copyStatusText) copyStatusText.textContent = "Copied!";
          showToast("URL copied to clipboard");
          setTimeout(() => {
            if (copyStatusText) copyStatusText.textContent = "Copy";
          }, 2000);
        });
      }
    });
  }

  async function executeScan(url) {
    if (!url) return;
    btnAnalyze.classList.add("loading");
    btnAnalyze.disabled = true;
    if (btnAnalyzeText) btnAnalyzeText.textContent = "Scanning...";

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url })
      });

      const data = await response.json();

      if (!response.ok) {
        showToast(data.error || "Inference error occurred.");
        return;
      }

      renderLiveResults(data);
      saveScanToHistory(data);

    } catch (err) {
      console.error("Scan error:", err);
      showToast("Network error: Could not contact neural backend.");
    } finally {
      btnAnalyze.classList.remove("loading");
      btnAnalyze.disabled = false;
      if (btnAnalyzeText) btnAnalyzeText.textContent = "Scan URL";
    }
  }

  // =========================================================================
  // 5. Render Live Inspection Results
  // =========================================================================
  function renderLiveResults(res) {
    if (!scanResultsArea) return;
    scanResultsArea.classList.remove("hidden");

    const isPhish = res.is_phishing;
    const prob = res.phishing_probability;
    const pct = Math.round(prob * 100);

    // A. Master Verdict Scope Card
    const banner = document.getElementById("verdict-banner");
    const icon = document.getElementById("verdict-icon");
    const title = document.getElementById("verdict-title");
    const urlDisplay = document.getElementById("verdict-url-display");
    const meterPath = document.getElementById("gauge-meter-path");
    const scoreNum = document.getElementById("gauge-score");
    const scoreLabel = document.getElementById("gauge-label");
    const severityBadge = document.getElementById("risk-severity-badge");

    if (banner) {
      banner.className = `verdict-scope-card ${isPhish ? 'danger' : 'safe'}`;
    }

    if (icon) {
      icon.innerHTML = isPhish
        ? `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`
        : `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg>`;
    }

    if (title) {
      title.textContent = isPhish ? "Phishing Attack Intercepted" : "Verified Legitimate Domain";
    }

    if (urlDisplay) {
      urlDisplay.textContent = res.url;
    }

    if (scoreNum) {
      scoreNum.textContent = `${pct}%`;
    }

    if (scoreLabel) {
      scoreLabel.textContent = isPhish ? "Phishing Risk" : "Safety Score";
    }

    if (severityBadge) {
      severityBadge.textContent = res.risk_level || (isPhish ? "CRITICAL THREAT" : "CLEAN VERIFIED");
    }

    // Radial SVG Meter Animation
    if (meterPath) {
      const offset = SVG_CIRCUMFERENCE - (SVG_CIRCUMFERENCE * prob);
      meterPath.style.strokeDashoffset = offset;
    }

    // Advisory Flags
    const advisoryFlags = document.getElementById("advisory-flags");
    if (advisoryFlags) {
      const lex = res.lexical_features || {};
      const flags = [];

      if (isPhish) {
        if (lex.brand_spoofing_detected && lex.detected_brands?.length > 0) {
          flags.push(`Target Spoof: ${lex.detected_brands.join(", ").toUpperCase()}`);
        }
        if (lex.is_ip_address) {
          flags.push("Raw Numerical IP Host");
        }
        if (lex.has_suspicious_tld) {
          flags.push("Abuse-Prone TLD");
        }
        if (lex.hyphen_count >= 2) {
          flags.push(`Hyphen Keyword Stuffing (${lex.hyphen_count})`);
        }
        if (lex.num_subdomains >= 2) {
          flags.push(`Deep Subdomains (${lex.num_subdomains})`);
        }
        if (lex.suspicious_keywords?.length > 0) {
          flags.push(`Triggers: ${lex.suspicious_keywords.slice(0, 3).join(", ")}`);
        }
        if (flags.length === 0) {
          flags.push("Transformer Attention Anomaly");
        }

        advisoryFlags.innerHTML = flags.map(f => `<span class="threat-flag-pill flag-bad">${escapeHtml(f)}</span>`).join("");
      } else {
        advisoryFlags.innerHTML = `
          <span class="threat-flag-pill flag-ok">Official Domain Authority</span>
          <span class="threat-flag-pill flag-ok">Benign Subword Sequence</span>
          <span class="threat-flag-pill flag-ok">Consensus: All Heads Clean</span>
        `;
      }
    }

    // Telemetry Strip
    const consensusEl = document.getElementById("metric-heads-consensus");
    const latencyEl = document.getElementById("metric-latency");
    const tokensEl = document.getElementById("metric-tokens-count");
    const padSavingsEl = document.getElementById("metric-pad-savings");

    const headList = res.pipeline?.step4_ensemble_heads?.head_outputs || [];
    const phishHeads = headList.filter(h => (typeof h === 'object' ? h.prob : h) >= 0.50).length;

    if (consensusEl) {
      consensusEl.textContent = isPhish ? `${phishHeads}/8 Heads Flagged` : `${8 - phishHeads}/8 Heads Clear`;
    }
    if (latencyEl) latencyEl.textContent = `${res.inference_time_ms} ms`;
    if (tokensEl) tokensEl.textContent = res.pipeline?.step1_tokenization?.token_count || 0;
    if (padSavingsEl) padSavingsEl.textContent = `~${res.pipeline?.step1_tokenization?.computation_saved_pct || 89}%`;

    // B. Subword Attention Heatmap (Interactive Explainability)
    renderTokenHeatmap(res);

    // C. 8 Parallel Dense Classifier Heads
    renderEnsembleHeads(headList);

    // D. Structural URL Signal Indicators
    renderSignalsGrid(res);

    // Smooth scroll into results
    scanResultsArea.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // =========================================================================
  // 6. Subword Attention Heatmap Renderer
  // =========================================================================
  function renderTokenHeatmap(res) {
    const tokenRack = document.getElementById("token-chips-flow");
    if (!tokenRack) return;

    const tokens = res.pipeline?.step1_tokenization?.tokens || [];
    if (tokens.length === 0) {
      tokenRack.innerHTML = `<span class="text-muted">No subword tokens generated</span>`;
      return;
    }

    tokenRack.innerHTML = tokens.map((tObj, idx) => {
      const tokenStr = typeof tObj === 'object' ? tObj.token : tObj;
      const weight = typeof tObj === 'object' && tObj.attention_weight ? tObj.attention_weight : 0.1;
      
      let heatClass = "heat-low";
      if (weight >= 0.70) heatClass = "heat-high";
      else if (weight >= 0.30) heatClass = "heat-mid";

      return `
        <span class="xai-token ${heatClass}" 
              data-token="${escapeHtml(tokenStr)}" 
              data-weight="${weight.toFixed(4)}" 
              data-idx="${idx}">
          ${escapeHtml(tokenStr)}
          <span class="token-micro-w">${weight.toFixed(2)}</span>
        </span>
      `;
    }).join("");

    // Attach Interactive Hover Telemetry
    const tokenEls = tokenRack.querySelectorAll(".xai-token");
    tokenEls.forEach(el => {
      el.addEventListener("mouseenter", () => {
        const t = el.getAttribute("data-token");
        const w = parseFloat(el.getAttribute("data-weight"));
        
        if (hudSubword) hudSubword.textContent = `"${t}"`;
        if (hudWeight) hudWeight.textContent = `${w.toFixed(4)}`;
        if (hudThreat) {
          if (w >= 0.70) {
            hudThreat.innerHTML = `<span class="text-danger">High Attention Anomaly (Strong Indicator)</span>`;
          } else if (w >= 0.30) {
            hudThreat.innerHTML = `<span style="color:#f59e0b">Suspicious Keyword / Delimiter</span>`;
          } else {
            hudThreat.innerHTML = `<span style="color:#10b981">Standard Lexical Token</span>`;
          }
        }
      });
    });

    // Default HUD value
    if (hudSubword) hudSubword.textContent = "Hover any token above";
    if (hudWeight) hudWeight.textContent = "--";
    if (hudThreat) hudThreat.textContent = "--";
  }

  // =========================================================================
  // 7. 8-Head Ensemble Matrix Renderer
  // =========================================================================
  function renderEnsembleHeads(headList) {
    const container = document.getElementById("ensemble-matrix-container");
    if (!container) return;

    if (!headList || headList.length === 0) {
      container.innerHTML = `<p class="text-muted">Ensemble telemetry unavailable.</p>`;
      return;
    }

    container.innerHTML = headList.map((head, idx) => {
      const prob = typeof head === 'object' ? head.prob : head;
      const pct = Math.round(prob * 100);
      const isHeadPhish = prob >= 0.50;

      return `
        <div class="head-node-card">
          <div class="head-top-row">
            <span class="head-id">HEAD 0${idx + 1}</span>
            <span class="head-led-indicator ${isHeadPhish ? 'threat' : 'safe'}">
              ${isHeadPhish ? 'PHISH' : 'SAFE'} (${prob.toFixed(3)})
            </span>
          </div>
          <div class="head-meter-track">
            <div class="head-meter-fill ${isHeadPhish ? 'threat' : 'safe'}" style="width: ${pct}%"></div>
          </div>
        </div>
      `;
    }).join("");
  }

  // =========================================================================
  // 8. Heuristic / Lexical Signal Grid
  // =========================================================================
  function renderSignalsGrid(res) {
    const container = document.getElementById("signals-grid-container");
    if (!container) return;

    const lex = res.lexical_features || {};
    const signals = [
      {
        label: "HOST PROTOCOL & FORMAT",
        val: lex.is_ip_address ? "Raw Numerical IP" : "Standard Domain",
        status: lex.is_ip_address ? "bad" : "ok",
        note: lex.is_ip_address ? "Bypasses DNS resolution" : "Standard DNS resolved"
      },
      {
        label: "BRAND IMPERSONATION",
        val: lex.detected_brands && lex.detected_brands.length > 0 ? lex.detected_brands.join(", ").toUpperCase() : "None Detected",
        status: lex.brand_spoofing_detected ? "bad" : "ok",
        note: lex.brand_spoofing_detected ? "Spoofed enterprise target" : "Legitimate domain name"
      },
      {
        label: "SUBDOMAIN DEPTH",
        val: `${lex.num_subdomains || 0} Subdomains`,
        status: (lex.num_subdomains || 0) > 2 ? "bad" : ((lex.num_subdomains || 0) === 2 ? "warn" : "ok"),
        note: (lex.num_subdomains || 0) > 2 ? "Excessive subdomain hierarchy" : "Standard structure"
      },
      {
        label: "DELIMITER STUFFING",
        val: `${lex.hyphen_count || 0} Hyphens`,
        status: (lex.hyphen_count || 0) >= 2 ? "bad" : "ok",
        note: (lex.hyphen_count || 0) >= 2 ? "Keyword obfuscation technique" : "Normal hyphenation"
      },
      {
        label: "ABUSE-PRONE TLD CHECK",
        val: lex.has_suspicious_tld ? "High-Risk TLD" : "Standard TLD",
        status: lex.has_suspicious_tld ? "bad" : "ok",
        note: lex.has_suspicious_tld ? "Commonly abused TLD" : "Established registrar"
      },
      {
        label: "SHANNON ENTROPY",
        val: `${lex.entropy || 3.4} bits`,
        status: (lex.entropy || 3.4) > 4.5 ? "warn" : "ok",
        note: (lex.entropy || 3.4) > 4.5 ? "High character randomness" : "Natural language pattern"
      }
    ];

    container.innerHTML = signals.map(s => `
      <div class="signal-tile">
        <span class="signal-lbl">${escapeHtml(s.label)}</span>
        <span class="signal-val">${escapeHtml(s.val)}</span>
        <span class="signal-status-pill ${s.status}">${escapeHtml(s.note)}</span>
      </div>
    `).join("");
  }

  // =========================================================================
  // 9. Session History Management
  // =========================================================================
  function getHistory() {
    try {
      return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    } catch {
      return [];
    }
  }

  function saveScanToHistory(res) {
    const history = getHistory();
    const entry = {
      url: res.url,
      is_phishing: res.is_phishing,
      prob: res.phishing_probability,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    };

    const deduped = history.filter(h => h.url !== res.url);
    deduped.unshift(entry);
    const trimmed = deduped.slice(0, 10);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed));
    renderHistory();
  }

  function renderHistory() {
    const historyList = document.getElementById("history-list");
    const emptyMsg = document.getElementById("empty-history-msg");
    const history = getHistory();

    if (!historyList) return;

    if (history.length === 0) {
      if (emptyMsg) emptyMsg.classList.remove("hidden");
      historyList.innerHTML = `<p class="empty-log-msg">No scans performed in this browser session yet.</p>`;
      return;
    }

    historyList.innerHTML = history.map(item => `
      <div class="history-item-row">
        <div class="history-item-left">
          <span class="history-badge ${item.is_phishing ? 'threat' : 'safe'}">
            ${item.is_phishing ? 'PHISH' : 'CLEAN'}
          </span>
          <span class="history-url" title="${escapeHtml(item.url)}">${escapeHtml(item.url)}</span>
        </div>
        <div class="history-actions">
          <span class="history-score ${item.is_phishing ? 'text-danger' : 'text-safe'}">${Math.round(item.prob * 100)}%</span>
          <button type="button" class="btn-inspect-small" data-retest="${escapeHtml(item.url)}">Inspect</button>
        </div>
      </div>
    `).join("");

    historyList.querySelectorAll("[data-retest]").forEach(btn => {
      btn.addEventListener("click", () => {
        const targetUrl = btn.getAttribute("data-retest");
        if (targetUrl && urlInput) {
          urlInput.value = targetUrl;
          switchMode("view-scanner");
          executeScan(targetUrl);
        }
      });
    });
  }

  const btnClearHistory = document.getElementById("btn-clear-history");
  if (btnClearHistory) {
    btnClearHistory.addEventListener("click", () => {
      localStorage.removeItem(HISTORY_KEY);
      renderHistory();
      showToast("Session audit trail cleared.");
    });
  }

  // =========================================================================
  // 10. 30-Site Test Suite & Batch Radar
  // =========================================================================
  if (btnRun30Tests) {
    btnRun30Tests.addEventListener("click", async () => {
      btnRun30Tests.disabled = true;
      btnRun30Tests.innerHTML = `<span>Loading 30 Notebook Sites...</span>`;
      try {
        const res = await fetch("/api/custom-tests");
        const data = await res.json();
        if (data.urls && data.urls.length > 0) {
          await runBatchAnalysis(data.urls, "Notebook 30 Test Websites");
        } else {
          showToast("Failed to load notebook test URLs.");
        }
      } catch (err) {
        showToast("Error retrieving notebook URLs.");
      } finally {
        btnRun30Tests.disabled = false;
        btnRun30Tests.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          <span>Run 30 Notebook Test Websites</span>
        `;
      }
    });
  }

  if (btnRun20Presets) {
    btnRun20Presets.addEventListener("click", async () => {
      btnRun20Presets.disabled = true;
      try {
        const res = await fetch("/api/presets");
        const data = await res.json();
        if (data.presets) {
          await runBatchAnalysis(data.presets, "20 Benchmark Preset URLs");
        }
      } catch {
        showToast("Could not load presets.");
      } finally {
        btnRun20Presets.disabled = false;
      }
    });
  }

  if (btnRunCustomBatch && batchTextarea) {
    btnRunCustomBatch.addEventListener("click", () => {
      const raw = batchTextarea.value.trim();
      if (!raw) {
        showToast("Please enter at least one URL to scan.");
        return;
      }
      const urls = raw.split("\n").map(u => u.trim()).filter(u => u.length > 0);
      runBatchAnalysis(urls, "Custom URL Batch");
    });
  }

  if (btnClearBatch && batchTextarea) {
    btnClearBatch.addEventListener("click", () => {
      batchTextarea.value = "";
    });
  }

  if (btnUploadFile && batchFileInput) {
    btnUploadFile.addEventListener("click", () => batchFileInput.click());
    batchFileInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (evt) => {
        const content = evt.target.result;
        batchTextarea.value = content;
        showToast(`Imported ${file.name}`);
      };
      reader.readAsText(file);
    });
  }

  async function runBatchAnalysis(urls, batchTitle = "Batch Results") {
    showToast(`Executing neural analysis on ${urls.length} URLs...`);

    try {
      const res = await fetch("/api/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls: urls })
      });

      const data = await res.json();

      if (!res.ok) {
        showToast(data.error || "Batch execution error.");
        return;
      }

      currentBatchData = data.results || [];
      renderBatchSummary(data, batchTitle);
      renderBatchTable();

    } catch (err) {
      console.error("Batch error:", err);
      showToast("Network failure during batch analysis.");
    }
  }

  function renderBatchSummary(data, batchTitle) {
    if (batchSummaryCards) batchSummaryCards.classList.remove("hidden");
    if (batchResultsWrapper) batchResultsWrapper.classList.remove("hidden");

    const totalEl = document.getElementById("batch-total-count");
    const legitEl = document.getElementById("batch-legit-count");
    const phishEl = document.getElementById("batch-phish-count");
    const latencyEl = document.getElementById("batch-avg-latency");
    const titleEl = document.getElementById("batch-table-title");

    if (totalEl) totalEl.textContent = data.total_scanned;
    if (legitEl) legitEl.textContent = data.legitimate_count;
    if (phishEl) phishEl.textContent = data.phishing_count;
    if (latencyEl) latencyEl.textContent = `${data.average_latency_ms} ms`;
    if (titleEl) titleEl.textContent = `${batchTitle} (${data.total_scanned} Targets)`;

    // Update Pill Counts
    const countAll = document.getElementById("count-all");
    const countPhish = document.getElementById("count-phish");
    const countLegit = document.getElementById("count-legit");

    if (countAll) countAll.textContent = data.total_scanned;
    if (countPhish) countPhish.textContent = data.phishing_count;
    if (countLegit) countLegit.textContent = data.legitimate_count;

    batchSummaryCards.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // Filter Pills Handling
  filterPills.forEach(pill => {
    pill.addEventListener("click", () => {
      filterPills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      currentActiveFilter = pill.getAttribute("data-filter") || "all";
      renderBatchTable();
    });
  });

  function renderBatchTable() {
    if (!batchTableBody) return;

    let filtered = currentBatchData;
    if (currentActiveFilter === "phish") {
      filtered = currentBatchData.filter(item => item.is_phishing);
    } else if (currentActiveFilter === "legit") {
      filtered = currentBatchData.filter(item => !item.is_phishing);
    }

    if (filtered.length === 0) {
      batchTableBody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding: 2rem; color: var(--text-muted);">No records match filter.</td></tr>`;
      return;
    }

    batchTableBody.innerHTML = filtered.map((row, idx) => {
      const isPhish = row.is_phishing;
      const prob = row.phishing_probability;
      const pct = Math.round(prob * 100);

      return `
        <tr>
          <td class="col-num">${idx + 1}</td>
          <td class="col-url" title="${escapeHtml(row.url)}">${escapeHtml(row.url)}</td>
          <td class="col-verdict">
            <span class="table-verdict-tag ${isPhish ? 'threat' : 'safe'}">
              ${isPhish ? 'Phishing' : 'Legitimate'}
            </span>
          </td>
          <td class="col-risk">
            <div class="table-risk-bar-wrap">
              <div class="table-risk-track">
                <div class="table-risk-fill ${isPhish ? 'threat' : 'safe'}" style="width: ${pct}%"></div>
              </div>
              <span class="table-risk-pct ${isPhish ? 'text-danger' : 'text-safe'}">${pct}%</span>
            </div>
          </td>
          <td class="col-time mono">${row.inference_time_ms} ms</td>
          <td class="col-action">
            <button type="button" class="btn-inspect-small" data-table-inspect="${escapeHtml(row.url)}">
              Inspect
            </button>
          </td>
        </tr>
      `;
    }).join("");

    batchTableBody.querySelectorAll("[data-table-inspect]").forEach(btn => {
      btn.addEventListener("click", () => {
        const u = btn.getAttribute("data-table-inspect");
        if (u && urlInput) {
          urlInput.value = u;
          switchMode("view-scanner");
          executeScan(u);
        }
      });
    });
  }

  // CSV Export
  if (btnExportCsv) {
    btnExportCsv.addEventListener("click", () => {
      if (!currentBatchData || currentBatchData.length === 0) {
        showToast("No batch results to export.");
        return;
      }

      const headers = ["Index", "URL", "Classification", "Phishing_Probability", "Risk_Level", "Latency_ms"];
      const rows = currentBatchData.map((r, i) => [
        i + 1,
        `"${r.url.replace(/"/g, '""')}"`,
        r.is_phishing ? "Phishing" : "Legitimate",
        r.phishing_probability,
        r.risk_level,
        r.inference_time_ms
      ]);

      const csvContent = [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute("download", `phishfinder_audit_${Date.now()}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      showToast("CSV file exported successfully.");
    });
  }

  // =========================================================================
  // Utility: HTML Escaping
  // =========================================================================
  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // =========================================================================
  // Initial Boot: Check Status & Auto-Scan Initial Vector
  // =========================================================================
  checkEngineStatus();

  // Auto-scan the high-profile PayPal Spoof vector on startup so the UI is immediately alive
  setTimeout(() => {
    if (urlInput) {
      urlInput.value = "http://paypal-security-update-verification.com/login.php?client=auth";
      executeScan(urlInput.value);
    }
  }, 250);
});
