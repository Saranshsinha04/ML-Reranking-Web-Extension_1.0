/**
 * Search Re-Ranker – content.js (v2.1 - fixed selectors)
 * Works with Google's current DOM structure (2024-2025)
 */

const API_URL       = "http://localhost:8000/rerank";
const STARTUP_DELAY = 3000;
const MAX_RESULTS   = 10;
const STORAGE_KEY   = "reranker_interactions";

let currentQuery = "";
let shownResults = [];
let didClick     = false;

// ─── Confirm script is running ────────────────────────────────────────────────
console.log("[Re-Ranker] content.js v2.1 loaded on:", window.location.href);

// Check for dwell time from previous click before anything else
checkReturnDwell();

// Wait for page to fully render then run
setTimeout(main, STARTUP_DELAY);

// ─── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  console.log("[Re-Ranker] Running main()...");

  try {
    currentQuery = extractQuery();
    if (!currentQuery) {
      console.log("[Re-Ranker] No query found in URL, stopping.");
      return;
    }
    console.log("[Re-Ranker] Query:", currentQuery);

    const results = extractResults();
    console.log("[Re-Ranker] Results extracted:", results.length);

    if (results.length === 0) {
      console.log("[Re-Ranker] No results found. Check selectors.");
      debugSelectors(); // Log what selectors DO work
      return;
    }

    const ranked = await fetchRankedResults(currentQuery, results);
    if (!ranked || ranked.length === 0) {
      console.log("[Re-Ranker] No ranked results returned from backend.");
      // Still track even if reranking failed
      setupTracking(results.map((r, i) => ({ ...r, rank: i + 1 })));
      return;
    }

    reorderDOM(results, ranked);

    shownResults = ranked.map((r, i) => ({
      title:   r.title,
      snippet: r.snippet,
      url:     r.url,
      rank:    i + 1,
    }));

    setupTracking(shownResults);
    console.log("[Re-Ranker] Done. Tracking", shownResults.length, "results.");

  } catch (err) {
    console.warn("[Re-Ranker] Error in main():", err.message, err.stack);
  }
}

// ─── Selector Debug Helper ────────────────────────────────────────────────────
function debugSelectors() {
  const toTry = [
    'div.g', '.MjjYud', 'div.tF2Cxc', '.yuRUbf', '#rso > div',
    '[data-sokoban-container]', '[data-hveid]', '.hlcw0c',
    'div[data-ved]', '.rc', 'div.kvH3mc', 'div.Gx5Zad'
  ];
  console.log("[Re-Ranker] Selector debug:");
  toTry.forEach(sel => {
    const count = document.querySelectorAll(sel).length;
    if (count > 0) console.log(`  ${sel} → ${count} elements`);
  });
}

// ─── Extract Query ────────────────────────────────────────────────────────────
function extractQuery() {
  const params = new URLSearchParams(window.location.search);
  return (params.get("q") || "").trim();
}

// ─── Extract Results (multi-selector approach) ────────────────────────────────
/**
 * Tries multiple known Google result container selectors in order.
 * Google frequently changes their class names — this makes us resilient.
 */
function extractResults() {
  // Ordered from most specific/stable to most generic
  const containerSelectors = [
    '#rso .g',
    'div.g',
    'div.tF2Cxc',
    '.MjjYud > div',
    '#rso > div > div',
    '.hlcw0c',
    'div.kvH3mc',
  ];

  let blocks = [];
  for (const sel of containerSelectors) {
    const found = document.querySelectorAll(sel);
    if (found.length >= 3) {
      console.log(`[Re-Ranker] Using selector: "${sel}" (${found.length} blocks)`);
      blocks = Array.from(found);
      break;
    }
  }

  if (blocks.length === 0) {
    console.warn("[Re-Ranker] No result blocks found with any selector.");
    return [];
  }

  const results = [];

  for (const block of blocks) {
    if (results.length >= MAX_RESULTS) break;

    // ── Title: must have an h3 ──
    const h3 = block.querySelector("h3");
    if (!h3) continue;
    const title = h3.innerText.trim();
    if (!title || title.length < 3) continue;

    // ── URL: first real external link ──
    const anchor = block.querySelector("a[href]");
    if (!anchor) continue;
    const url = anchor.href.trim();
    if (!url.startsWith("http")) continue;
    if (url.includes("google.com/search")) continue;
    if (url.includes("google.com/aclk")) continue; // skip ads

    // ── Snippet ──
    const snippet = extractSnippet(block, title);
    // Allow empty snippet — still track the result

    results.push({ title, snippet: snippet || "", url, element: block });
  }

  return results;
}

function extractSnippet(block, title) {
  // Try known snippet selectors
  const selectors = [
    '.VwiC3b',
    '[data-sncf]',
    '.lEBKkf',
    '.ITZIwc',
    '[style="-webkit-line-clamp:2"]',
    'div[data-ved] span',
    '.s3v9rd',
    '.st',
  ];

  for (const sel of selectors) {
    const el   = block.querySelector(sel);
    const text = el?.innerText?.trim();
    if (text && text.length > 20 && text !== title) return text;
  }

  // Fallback: grab all text, strip title, take what's left
  const full = block.innerText.trim();
  const rest = full.replace(title, "").trim();
  return rest.length > 20 ? rest.slice(0, 300) : "";
}

// ─── Fetch from Backend ───────────────────────────────────────────────────────
async function fetchRankedResults(query, results) {
  const payload = {
    query,
    results: results.map(({ title, snippet, url }) => ({ title, snippet, url })),
  };

  try {
    const response = await fetch(API_URL, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    if (!response.ok) {
      console.warn("[Re-Ranker] Backend returned", response.status);
      return null;
    }

    const data = await response.json();
    console.log("[Re-Ranker] Backend returned", data.ranked?.length, "ranked results.");
    return data.ranked || null;

  } catch (err) {
    console.warn("[Re-Ranker] Backend unreachable:", err.message);
    console.warn("[Re-Ranker] Make sure uvicorn is running on port 8000.");
    return null;
  }
}

// ─── Reorder DOM ──────────────────────────────────────────────────────────────
function reorderDOM(originalResults, rankedResults) {
  const elementMap = new Map(
    originalResults.map(r => [normalizeUrl(r.url), r.element])
  );

  const firstEl = originalResults[0]?.element;
  if (!firstEl) return;
  const container = firstEl.parentElement;
  if (!container) return;

  const managed  = originalResults.map(r => r.element);
  const newOrder = [];

  for (const r of rankedResults) {
    const el = elementMap.get(normalizeUrl(r.url));
    if (el) { newOrder.push(el); elementMap.delete(normalizeUrl(r.url)); }
  }
  for (const el of elementMap.values()) newOrder.push(el);

  const anchor = document.createComment("re-ranker-anchor");
  container.insertBefore(anchor, managed[0]);
  managed.forEach(el => el.remove());
  for (let i = newOrder.length - 1; i >= 0; i--) anchor.after(newOrder[i]);
  anchor.remove();

  newOrder.forEach((el, idx) => addRankBadge(el, idx + 1));
}

// ─── Click & Dwell Tracking ───────────────────────────────────────────────────
function setupTracking(results) {
  shownResults = results;

  for (const result of shownResults) {
    const normTarget = normalizeUrl(result.url);
    document.querySelectorAll("a[href]").forEach(anchor => {
      if (normalizeUrl(anchor.href) !== normTarget) return;
      anchor.addEventListener("click", () => {
        if (didClick) return;
        didClick = true;
        sessionStorage.setItem("reranker_pending_dwell", JSON.stringify({
          query:     currentQuery,
          url:       result.url,
          clickTime: Date.now(),
        }));
        saveInteractions(result.url);
        console.log("[Re-Ranker] Click tracked:", result.url);
      }, { once: true });
    });
  }

  window.addEventListener("pagehide", () => {
    if (!didClick && shownResults.length > 0) {
      saveInteractions(null);
      console.log("[Re-Ranker] Page left without click — all results marked as skipped.");
    }
  });
}

function checkReturnDwell() {
  const raw = sessionStorage.getItem("reranker_pending_dwell");
  if (!raw) return;
  try {
    const { query, url, clickTime } = JSON.parse(raw);
    const dwellMs = Date.now() - clickTime;
    sessionStorage.removeItem("reranker_pending_dwell");
    updateDwellTime(query, url, dwellMs);
    console.log(`[Re-Ranker] Dwell recorded: ${(dwellMs / 1000).toFixed(1)}s on ${url}`);
  } catch {
    sessionStorage.removeItem("reranker_pending_dwell");
  }
}

// ─── Storage ──────────────────────────────────────────────────────────────────
async function saveInteractions(clickedUrl) {
  const timestamp = Date.now();
  const records = shownResults.map(r => {
    const wasClicked = clickedUrl
      ? normalizeUrl(r.url) === normalizeUrl(clickedUrl)
      : false;
    return {
      query: currentQuery, url: r.url, title: r.title,
      snippet: r.snippet,  rank: r.rank,
      clicked: wasClicked, skipped: !wasClicked,
      dwellMs: null, timestamp,
    };
  });

  try {
    const existing = await chromeGet(STORAGE_KEY) || [];
    await chromeSet(STORAGE_KEY, [...existing, ...records]);
    console.log(`[Re-Ranker] Saved ${records.length} records. Total: ${existing.length + records.length}`);
  } catch (e) {
    console.warn("[Re-Ranker] Storage write failed:", e.message);
  }
}

async function updateDwellTime(query, url, dwellMs) {
  try {
    const stored  = await chromeGet(STORAGE_KEY) || [];
    const normUrl = normalizeUrl(url);
    for (let i = stored.length - 1; i >= 0; i--) {
      if (stored[i].query === query && normalizeUrl(stored[i].url) === normUrl && stored[i].clicked) {
        stored[i].dwellMs = dwellMs;
        break;
      }
    }
    await chromeSet(STORAGE_KEY, stored);
  } catch (e) {
    console.warn("[Re-Ranker] Dwell update failed:", e.message);
  }
}

// ─── Chrome Storage Wrappers ──────────────────────────────────────────────────
function chromeGet(key) {
  return new Promise((resolve, reject) => {
    chrome.storage.local.get([key], result => {
      if (chrome.runtime.lastError) reject(new Error(chrome.runtime.lastError.message));
      else resolve(result[key]);
    });
  });
}

function chromeSet(key, value) {
  return new Promise((resolve, reject) => {
    chrome.storage.local.set({ [key]: value }, () => {
      if (chrome.runtime.lastError) reject(new Error(chrome.runtime.lastError.message));
      else resolve();
    });
  });
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function normalizeUrl(url) {
  try {
    const u = new URL(url);
    return (u.origin + u.pathname).replace(/\/$/, "").toLowerCase() + u.search;
  } catch { return url.toLowerCase(); }
}

function addRankBadge(element, rank) {
  element.querySelector(".reranker-badge")?.remove();
  const badge = document.createElement("span");
  badge.className = "reranker-badge";
  badge.textContent = `#${rank} semantically`;
  badge.style.cssText = `
    display:inline-block;margin-left:8px;padding:1px 7px;
    font-size:11px;font-family:monospace;background:#e8f0fe;
    color:#1a73e8;border:1px solid #c5d8fd;border-radius:10px;
    vertical-align:middle;letter-spacing:0.3px;
  `;
  element.querySelector("h3")?.appendChild(badge);
}
