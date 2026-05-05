# 🔍 Personalized Search Re-Ranker

A full-stack ML system that **semantically re-ranks Google search results in real time** and **learns from your personal click behaviour** to get smarter over time.

Built with a Chrome Extension (Manifest V3), a FastAPI backend, and Sentence-BERT (`all-MiniLM-L6-v2`).

---

## ✨ Features

- **Semantic re-ranking** — results are reordered by meaning, not just keyword match
- **Personal data collection** — tracks clicks, skips, and dwell time silently in the background
- **Fine-tuning pipeline** — periodically train the model on your own behaviour
- **No page reload** — pure DOM manipulation, Google's UI is preserved
- **Fully local** — all data stays on your machine, nothing is sent externally
- **Popup dashboard** — see your collected data, export it, and clear it anytime

---

## 🏗️ Architecture

```
User searches on Google
        │
        ▼
┌─────────────────────┐
│   Chrome Extension  │  Manifest V3 · content.js
│                     │
│  1. Wait 2.5s       │
│  2. Extract query   │
│  3. Extract top 10  │  ← title, snippet, url from organic results
│     results         │
└────────┬────────────┘
         │  POST /rerank
         ▼
┌─────────────────────┐
│   FastAPI Backend   │  localhost:8000 · main.py
│                     │
│  1. Encode query    │  ← all-MiniLM-L6-v2 (or fine-tuned model)
│  2. Encode docs     │
│  3. Cosine sim      │  ← sklearn
│  4. Sort & return   │
└────────┬────────────┘
         │  ranked[] with scores
         ▼
┌─────────────────────┐
│   Chrome Extension  │
│                     │
│  Reorder DOM        │  ← no reload, match by URL
│  Track clicks       │  ← positive signal
│  Track skips        │  ← negative signal
│  Measure dwell time │  ← signal strength
│  Save to storage    │  ← chrome.storage.local
└─────────────────────┘
         │  (periodically)
         ▼
┌─────────────────────┐
│   Fine-tuning       │  finetune.py
│                     │
│  Export JSON        │  ← from extension popup
│  Build pairs        │  ← click=positive, skip=negative
│  PyTorch training   │  ← MSE loss on cosine similarity
│  Save model         │  → ./model_finetuned/
└─────────────────────┘
```

---

## 📁 Project Structure

```
search-reranker/
├── backend/
│   ├── main.py              ← FastAPI server + SBERT re-ranking
│   ├── finetune.py          ← Fine-tuning script (plain PyTorch)
│   └── requirements.txt     ← All Python dependencies
├── extension/
│   ├── manifest.json        ← Chrome Extension Manifest V3
│   ├── content.js           ← Result extraction, reranking, tracking
│   ├── popup.html           ← Extension popup UI
│   └── popup.js             ← Popup logic (stats, export, clear)
└── README.md
```

> ⚠️ **Critical:** The `backend/` and `extension/` folders must stay separate. Do NOT put all files in one flat folder.

---

## ⚙️ Setup Instructions

### Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| Python | 3.10 or higher | `python --version` |
| Google Chrome | Any recent version | — |
| pip | Comes with Python | `pip --version` |

---

### Step 1 — Create the folder structure

```bash
mkdir search-reranker
cd search-reranker
mkdir backend extension
```

Download all files from this repo and place them in the correct folders:
- `main.py`, `finetune.py`, `requirements.txt` → `backend/`
- `manifest.json`, `content.js`, `popup.html`, `popup.js` → `extension/`

---

### Step 2 — Set up the Python backend

#### Create and activate a virtual environment

**Mac / Linux:**
```bash
cd backend
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
cd backend
python -m venv venv
venv\Scripts\activate
```

You'll know it's active when your prompt shows `(venv)` at the start.

> **Windows note:** If you get a security error running the activate script, run this once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

#### Install dependencies

```bash
pip install -r requirements.txt
```

> The first install downloads `all-MiniLM-L6-v2` (~90 MB from Hugging Face). This only happens once.

---

### Step 3 — Start the backend server

**Mac / Linux:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Windows (PowerShell):**
```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

✅ **Verify it's running:** Open http://localhost:8000/health in your browser. You should see:
```json
{ "status": "ok", "model_loaded": true }
```

> **Keep this terminal open.** The server must stay running while you browse.

#### Create a startup script so you don't have to type this every time

**Mac / Linux** — save as `backend/start.sh`:
```bash
#!/bin/bash
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```
Run with: `bash start.sh`

**Windows** — save as `backend/start.bat`:
```bat
call venv\Scripts\activate
uvicorn main:app --host 0.0.0.0 --port 8000
```
Run by double-clicking `start.bat`.

---

### Step 4 — Load the Chrome Extension

1. Open Chrome and go to `chrome://extensions`
2. Toggle **Developer mode** ON (top-right corner)
3. Click **"Load unpacked"**
4. Select your `extension/` folder — **not** the root folder, not `backend/`
5. "Search Re-Ranker 2.0.0" should appear in the list

> The orange badge on the extension icon is Chrome's normal developer mode indicator — not an error.

---

### Step 5 — Test it

1. Make sure the backend server is running (Step 3)
2. Go to [google.com](https://google.com) and search anything
3. Wait ~3 seconds
4. Results will reorder and show `#1 semantically` badges next to titles
5. Click the extension icon in your toolbar to open the dashboard

**To see live logs:** Press `F12` on the Google results page → Console tab. You should see:
```
[Re-Ranker] content.js v2.1 loaded on: https://...
[Re-Ranker] Query: "your search"
[Re-Ranker] Using selector: "#rso .g" (10 blocks)
[Re-Ranker] Results extracted: 10
[Re-Ranker] Backend returned 10 ranked results.
[Re-Ranker] Done. Tracking 10 results.
```

---

## 🧠 Personalization & Fine-Tuning

The system collects three signals as you browse:

| Signal | What it means |
|--------|--------------|
| **Click** | You found this result relevant |
| **Skip** | You saw it but didn't click — less relevant |
| **Dwell time** | How long you stayed — longer = more relevant |

### How to fine-tune

**1. Collect data** — Just use Google normally for a few days. Aim for 20–30+ searches with real clicks.

**2. Export your data** — Click the extension icon → click **Export JSON**. Save the file into your `backend/` folder.

**3. Run fine-tuning** (venv must be active):
```bash
python finetune.py --data interactions_2026-05-05.json
```

You'll see output like:
```
Found 20 distinct search sessions.
Built 180 training pairs.
Epoch 1/3 | train_loss=0.0842 | eval_loss=0.0761
Epoch 2/3 | train_loss=0.0634 | eval_loss=0.0598
Epoch 3/3 | train_loss=0.0521 | eval_loss=0.0489
✓ Model saved to: ./model_finetuned
```

**4. Restart the backend with your personalized model:**

Mac / Linux:
```bash
MODEL_PATH=./model_finetuned uvicorn main:app --port 8000
```

Windows (PowerShell — two separate commands):
```powershell
$env:MODEL_PATH="./model_finetuned"
uvicorn main:app --port 8000
```

✅ Confirm it loaded: the first log line should say `Loading SentenceTransformer model from: ./model_finetuned`

**5. Re-finetune periodically** — export fresh data every week or two and repeat. More data = better personalization.

---

## 🔌 API Reference

### `POST /rerank`

**Request:**
```json
{
  "query": "machine learning frameworks",
  "results": [
    {
      "title": "TensorFlow Guide",
      "snippet": "Open source ML platform by Google",
      "url": "https://tensorflow.org"
    }
  ]
}
```

**Response:**
```json
{
  "ranked": [
    {
      "title": "TensorFlow Guide",
      "snippet": "Open source ML platform by Google",
      "url": "https://tensorflow.org",
      "score": 0.8431
    }
  ]
}
```

### `GET /health`
```json
{ "status": "ok", "model_loaded": true }
```

---

## 🐛 Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Extension does nothing | Content script not injecting | Check `chrome://extensions` for errors. Make sure you're on `google.com` or `google.co.in` |
| `[Re-Ranker] Results extracted: 0` | Google changed their DOM | The selector fallback should handle this — open an issue if it persists |
| Backend unreachable error in console | Server not running | Start uvicorn, check http://localhost:8000/health |
| `ModuleNotFoundError: sentence_transformers` | Running without venv | Activate venv first: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux) |
| `No module named accelerate` | Missing dependency | Run `pip install accelerate` |
| PowerShell `&&` error | Wrong shell syntax | Use two separate commands — `$env:MODEL_PATH=...` then `uvicorn ...` |
| Popup shows 0 searches | No data collected yet | Search on Google and click a result, then check popup again |
| Fine-tuned model not loading | Env var not set | Run `$env:MODEL_PATH` commands in the same terminal as uvicorn |

---

## 📦 Dependencies

### Backend (`requirements.txt`)

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | 0.111.0 | REST API framework |
| `uvicorn` | 0.29.0 | ASGI server |
| `sentence-transformers` | 3.0.1 | SBERT model loading and encoding |
| `scikit-learn` | 1.4.2 | Cosine similarity computation |
| `numpy` | 1.26.4 | Numerical operations |
| `pydantic` | 2.7.1 | Request/response validation |
| `torch` | ≥2.0.0 | PyTorch — model training |
| `transformers` | 4.41.2 | HuggingFace transformers (pinned for compatibility) |
| `accelerate` | ≥0.26.0 | Training acceleration |
| `datasets` | ≥2.14.0 | Dataset utilities |

### Extension
No external dependencies — pure vanilla JS with Chrome Extension APIs only.

---

## 🔒 Privacy

- All interaction data is stored in `chrome.storage.local` — **it never leaves your browser** unless you explicitly export it
- The fine-tuning runs entirely on your local machine
- The backend runs on `localhost` — no external API calls
- You can wipe all data anytime via the extension popup → **Clear data**

---

## 🚀 Extending the System

- **Popup toggle** — add an on/off switch to enable/disable reranking per session
- **Domain boosting** — manually boost domains you always trust (e.g. always prefer GitHub results)
- **GPU fine-tuning** — add `.to("cuda")` in `finetune.py` for 10x faster training if you have a GPU
- **Scheduled fine-tuning** — use Windows Task Scheduler or cron to auto-finetune weekly
- **Multi-profile** — support different user profiles with separate storage keys

---

## 📄 License

MIT — free to use, modify, and distribute.
