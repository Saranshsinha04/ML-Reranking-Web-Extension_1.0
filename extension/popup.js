/**
 * popup.js
 * Reads interaction data from chrome.storage.local and renders:
 *  - Summary stats (searches, clicks, avg dwell)
 *  - Recent queries with click/skip/dwell breakdown
 *  - Export button (downloads interactions.json for fine-tuning)
 *  - Clear button (wipes all stored data)
 */

const STORAGE_KEY = "reranker_interactions";

// ─── Boot ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadAndRender();

  document.getElementById("btn-export").addEventListener("click", exportData);
  document.getElementById("btn-clear").addEventListener("click", clearData);
});

// ─── Load & Render ────────────────────────────────────────────────────────────
function loadAndRender() {
  chrome.storage.local.get([STORAGE_KEY], result => {
    const records = result[STORAGE_KEY] || [];
    renderStats(records);
    renderQueryList(records);
    updateLastUpdated(records);
    document.getElementById("finetune-tip").style.display =
      records.length >= 10 ? "block" : "none";
  });
}

// ─── Stats ────────────────────────────────────────────────────────────────────
function renderStats(records) {
  // Unique searches = distinct (query, timestamp-rounded-to-session) combos
  const sessions = new Set(records.map(r => `${r.query}__${Math.floor(r.timestamp / 60000)}`));
  const clicks   = records.filter(r => r.clicked);
  const dwells   = clicks.filter(r => r.dwellMs != null).map(r => r.dwellMs);
  const avgDwell = dwells.length
    ? Math.round(dwells.reduce((a, b) => a + b, 0) / dwells.length / 1000)
    : null;

  document.getElementById("stat-searches").textContent = sessions.size;
  document.getElementById("stat-clicks").textContent   = clicks.length;
  document.getElementById("stat-avg-dwell").textContent =
    avgDwell != null ? `${avgDwell}s` : "—";
}

// ─── Query List ───────────────────────────────────────────────────────────────
function renderQueryList(records) {
  const list = document.getElementById("query-list");

  if (records.length === 0) {
    list.innerHTML = `<div class="empty">No data yet.<br>Search on Google to start collecting.</div>`;
    return;
  }

  // Group records by query, take the 8 most recent unique queries
  const byQuery = new Map();
  for (const r of [...records].reverse()) {
    if (!byQuery.has(r.query)) byQuery.set(r.query, []);
    byQuery.get(r.query).push(r);
  }

  const recentQueries = [...byQuery.entries()].slice(0, 8);

  list.innerHTML = recentQueries.map(([query, recs]) => {
    const clickCount = recs.filter(r => r.clicked).length;
    const skipCount  = recs.filter(r => r.skipped).length;
    const dwells     = recs.filter(r => r.dwellMs != null).map(r => r.dwellMs);
    const avgDwell   = dwells.length
      ? `${Math.round(dwells.reduce((a, b) => a + b, 0) / dwells.length / 1000)}s`
      : null;

    const pills = [
      clickCount ? `<span class="pill pill-click">${clickCount} click${clickCount > 1 ? "s" : ""}</span>` : "",
      skipCount  ? `<span class="pill pill-skip">${skipCount} skip${skipCount > 1 ? "s" : ""}</span>` : "",
      avgDwell   ? `<span class="pill pill-dwell">${avgDwell} dwell</span>` : "",
    ].filter(Boolean).join("");

    return `
      <div class="query-row">
        <span class="query-text" title="${escHtml(query)}">${escHtml(query)}</span>
        <span class="query-meta">${pills}</span>
      </div>`;
  }).join("");
}

// ─── Last Updated ─────────────────────────────────────────────────────────────
function updateLastUpdated(records) {
  if (records.length === 0) return;
  const latest = Math.max(...records.map(r => r.timestamp));
  const diff   = Date.now() - latest;
  const label  = diff < 60000       ? "just now"
               : diff < 3600000     ? `${Math.floor(diff / 60000)}m ago`
               : diff < 86400000    ? `${Math.floor(diff / 3600000)}h ago`
               : `${Math.floor(diff / 86400000)}d ago`;
  document.getElementById("last-updated").textContent = label;
}

// ─── Export ───────────────────────────────────────────────────────────────────
/**
 * Downloads all stored interactions as a JSON file.
 * The format is directly consumable by finetune.py.
 */
function exportData() {
  chrome.storage.local.get([STORAGE_KEY], result => {
    const records = result[STORAGE_KEY] || [];

    if (records.length === 0) {
      showStatus("No data to export yet.", "error");
      return;
    }

    const blob = new Blob([JSON.stringify(records, null, 2)], {
      type: "application/json",
    });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `interactions_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);

    showStatus(`Exported ${records.length} records.`, "success");
  });
}

// ─── Clear ────────────────────────────────────────────────────────────────────
function clearData() {
  if (!confirm("Delete all collected interaction data? This cannot be undone.")) return;

  chrome.storage.local.remove([STORAGE_KEY], () => {
    showStatus("All data cleared.", "success");
    loadAndRender();
  });
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function showStatus(msg, type) {
  const el = document.getElementById("status-msg");
  el.textContent  = msg;
  el.className    = `status ${type}`;
  setTimeout(() => { el.className = "status"; }, 3000);
}

function escHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
