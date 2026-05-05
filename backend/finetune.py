"""
finetune.py – Fine-tune all-MiniLM-L6-v2 on personal click interaction data.
Uses the modern sentence-transformers >= 3.0 SentenceTransformerTrainer API.

Usage:
    python finetune.py --data interactions_2026-05-05.json

Flow:
  1. Load exported interaction JSON from the Chrome extension
  2. Build training pairs from click/skip/dwell signals
  3. Fine-tune using CoSENTLoss (works with soft labels 0.0–1.0)
  4. Save model to ./model_finetuned/
  5. Restart backend: MODEL_PATH=./model_finetuned uvicorn main:app --port 8000
"""

import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_MODEL   = "all-MiniLM-L6-v2"
OUTPUT_DIR   = "./model_finetuned"
EPOCHS       = 3
BATCH_SIZE   = 16
WARMUP_RATIO = 0.1
MAX_DWELL_MS = 120_000   # cap at 2 minutes
MIN_DWELL_MS = 3_000     # under 3s = bounce

# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune SBERT on personal search interactions.")
    p.add_argument("--data",       required=True,          help="Path to exported interactions JSON.")
    p.add_argument("--epochs",     type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--output",     default=OUTPUT_DIR)
    return p.parse_args()

# ─── Load Data ────────────────────────────────────────────────────────────────
def load_interactions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    logger.info(f"Loaded {len(records)} raw interaction records.")
    required = {"query", "url", "title", "snippet", "clicked", "skipped"}
    valid = [r for r in records if required.issubset(r.keys()) and r.get("query")]
    logger.info(f"{len(valid)} records passed validation.")
    return valid

# ─── Build Training Pairs ─────────────────────────────────────────────────────
def build_dataset(records: list[dict]) -> Dataset:
    """
    Returns a HuggingFace Dataset with columns:
        sentence1 (query), sentence2 (doc text), label (float 0.0–1.0)

    Label mapping:
        clicked + long dwell  → up to 1.0  (strong positive)
        clicked + bounce      → 0.70       (weak positive)
        clicked + no dwell    → 0.75       (positive, dwell unknown)
        skipped               → 0.0–0.1   (negative, weighted by rank)
    """
    sentence1, sentence2, labels = [], [], []

    # Group records into search sessions by (query, 5-min timestamp bucket)
    sessions = defaultdict(list)
    for r in records:
        bucket = (r["query"], r["timestamp"] // 300_000)
        sessions[bucket].append(r)

    logger.info(f"Found {len(sessions)} distinct search sessions.")
    skipped_sessions = 0

    for (query, _), session in sessions.items():
        clicked = [r for r in session if r.get("clicked")]
        skipped = [r for r in session if r.get("skipped")]

        if not clicked and not skipped:
            skipped_sessions += 1
            continue

        # Positive pairs
        for r in clicked:
            sentence1.append(query)
            sentence2.append(f"{r['title']}. {r['snippet']}")
            labels.append(compute_positive_label(r.get("dwellMs")))

        # Negative pairs
        for r in skipped:
            rank_factor = max(0.0, 1.0 - (r.get("rank", 5) - 1) * 0.05)
            sentence1.append(query)
            sentence2.append(f"{r['title']}. {r['snippet']}")
            labels.append(round(0.1 * rank_factor, 3))

    logger.info(
        f"Built {len(sentence1)} training pairs. "
        f"({skipped_sessions} sessions skipped — no signal.)"
    )

    if len(sentence1) == 0:
        raise ValueError(
            "No training pairs could be built. "
            "Make sure your JSON contains clicked/skipped records."
        )

    return Dataset.from_dict({
        "sentence1": sentence1,
        "sentence2": sentence2,
        "label":     labels,
    })

def compute_positive_label(dwell_ms) -> float:
    if dwell_ms is None:
        return 0.75
    dwell_ms = min(dwell_ms, MAX_DWELL_MS)
    if dwell_ms < MIN_DWELL_MS:
        return 0.70
    t = (dwell_ms - MIN_DWELL_MS) / (MAX_DWELL_MS - MIN_DWELL_MS)
    return round(0.75 + 0.25 * t, 3)

# ─── Fine-Tune ────────────────────────────────────────────────────────────────
def finetune(args):
    records = load_interactions(args.data)
    dataset = build_dataset(records)

    # Split off 10% for evaluation
    split      = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds   = split["train"]
    eval_ds    = split["test"]

    logger.info(f"Train: {len(train_ds)} pairs | Eval: {len(eval_ds)} pairs")

    logger.info(f"Loading base model: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL)

    # CoSENTLoss supports soft float labels — perfect for our dwell-weighted signals
    loss = losses.CoSENTLoss(model)

    total_steps  = (len(train_ds) // args.batch_size) * args.epochs
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

    training_args = SentenceTransformerTrainingArguments(
        output_dir              = args.output,
        num_train_epochs        = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        warmup_steps            = warmup_steps,
        eval_strategy           = "epoch",   # renamed in newer transformers
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        logging_steps           = 5,
        fp16                    = False,   # CPU-safe
        bf16                    = False,
    )

    trainer = SentenceTransformerTrainer(
        model     = model,
        args      = training_args,
        train_dataset = train_ds,
        eval_dataset  = eval_ds,
        loss          = loss,
    )

    logger.info("Starting fine-tuning...")
    trainer.train()

    # Save the final model
    model.save_pretrained(args.output)
    logger.info(f"\n✓ Fine-tuned model saved to: {args.output}")
    logger.info(
        f"✓ Restart the backend with:\n"
        f"  MODEL_PATH={args.output} uvicorn main:app --port 8000\n"
        f"  (Windows: set MODEL_PATH={args.output} && uvicorn main:app --port 8000)"
    )

# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    finetune(args)
