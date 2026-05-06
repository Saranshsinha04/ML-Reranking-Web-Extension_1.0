"""
compare_models.py — Compare base vs fine-tuned SBERT model performance.

Runs 5 metrics on both models using your real interaction data:
  1. Ranking accuracy   — did the model rank clicked results above skipped ones?
  2. Mean Reciprocal Rank (MRR) — how high did clicked results rank on average?
  3. NDCG@10           — normalized quality of the full ranking
  4. Average similarity scores — how the models score clicked vs skipped
  5. Speed benchmark   — inference time per query

Usage:
    python compare_models.py --data interactions_2026-05-05.json
"""

import json
import time
import argparse
import math
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL      = "all-MiniLM-L6-v2"
FINETUNED_MODEL = "./model_finetuned"

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Compare base vs fine-tuned model.")
    p.add_argument("--data",      required=True, help="Path to interactions JSON")
    p.add_argument("--base",      default=BASE_MODEL,      help="Base model path/name")
    p.add_argument("--finetuned", default=FINETUNED_MODEL, help="Fine-tuned model path")
    return p.parse_args()

# ── Load Data ─────────────────────────────────────────────────────────────────
def load_sessions(path):
    """
    Load interactions and group into sessions.
    Each session = one search query with its shown results and click signals.
    Returns list of dicts: { query, results: [{title,snippet,url,clicked,rank}] }
    """
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    sessions = defaultdict(list)
    for r in records:
        if not r.get("query"):
            continue
        bucket = (r["query"], r["timestamp"] // 300_000)
        sessions[bucket].append(r)

    result = []
    for (query, _), recs in sessions.items():
        # Only include sessions that have at least one click and one skip
        has_click = any(r.get("clicked") for r in recs)
        has_skip  = any(r.get("skipped") for r in recs)
        if has_click and has_skip:
            result.append({
                "query":   query,
                "results": [
                    {
                        "title":   r["title"],
                        "snippet": r["snippet"],
                        "url":     r["url"],
                        "clicked": bool(r.get("clicked")),
                        "rank":    r.get("rank", 5),
                    }
                    for r in recs
                ],
            })

    return result

# ── Rank Sessions Using a Model ───────────────────────────────────────────────
def rank_sessions(model, sessions):
    """
    For each session, re-rank results using the model and return the rank
    assigned to each clicked result.
    Returns list of per-session dicts with ranking info.
    """
    ranked_sessions = []

    for session in sessions:
        query    = session["query"]
        results  = session["results"]
        docs     = [f"{r['title']}. {r['snippet']}" for r in results]

        all_texts  = [query] + docs
        embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)

        query_emb = embeddings[0:1]
        doc_embs  = embeddings[1:]
        scores    = cosine_similarity(query_emb, doc_embs)[0]

        # Attach scores and sort
        scored = sorted(
            [(results[i], float(scores[i])) for i in range(len(results))],
            key=lambda x: x[1], reverse=True
        )

        # Find where clicked results ended up in the new ranking
        clicked_ranks  = []
        clicked_scores = []
        skipped_scores = []

        for new_rank, (result, score) in enumerate(scored, start=1):
            if result["clicked"]:
                clicked_ranks.append(new_rank)
                clicked_scores.append(score)
            else:
                skipped_scores.append(score)

        ranked_sessions.append({
            "query":          query,
            "clicked_ranks":  clicked_ranks,
            "clicked_scores": clicked_scores,
            "skipped_scores": skipped_scores,
            "n_results":      len(results),
            "scored":         scored,
            "original":       results,
        })

    return ranked_sessions

# ── Metrics ───────────────────────────────────────────────────────────────────
def mean_reciprocal_rank(ranked_sessions):
    """
    MRR = average of 1/rank for each clicked result across all sessions.
    MRR of 1.0 = clicked result always ranked #1.
    MRR of 0.5 = clicked result ranked #2 on average.
    """
    reciprocals = []
    for s in ranked_sessions:
        if s["clicked_ranks"]:
            best_rank = min(s["clicked_ranks"])
            reciprocals.append(1.0 / best_rank)
    return float(np.mean(reciprocals)) if reciprocals else 0.0

def ranking_accuracy(ranked_sessions):
    """
    Proportion of sessions where the top-ranked result was a clicked result.
    (Precision@1)
    """
    correct = 0
    total   = 0
    for s in ranked_sessions:
        if s["scored"]:
            top_result = s["scored"][0][0]
            if top_result["clicked"]:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0

def ndcg_at_k(ranked_sessions, k=10):
    """
    NDCG@K — Normalized Discounted Cumulative Gain.
    Measures quality of the full ranking, not just top-1.
    Clicked results contribute gain; higher rank = more gain.
    Perfect NDCG = 1.0.
    """
    ndcg_scores = []
    for s in ranked_sessions:
        n = min(k, len(s["scored"]))

        # Actual DCG: gain from the model's ranking
        dcg = 0.0
        for rank, (result, _) in enumerate(s["scored"][:n], start=1):
            if result["clicked"]:
                dcg += 1.0 / math.log2(rank + 1)

        # Ideal DCG: if all clicked results were at the top
        n_clicked = sum(1 for r in s["original"] if r["clicked"])
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n_clicked, n)))

        if idcg > 0:
            ndcg_scores.append(dcg / idcg)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

def avg_score_gap(ranked_sessions):
    """
    Average difference between clicked result scores and skipped result scores.
    Larger gap = model better discriminates relevant from irrelevant.
    """
    gaps = []
    for s in ranked_sessions:
        if s["clicked_scores"] and s["skipped_scores"]:
            avg_clicked = np.mean(s["clicked_scores"])
            avg_skipped = np.mean(s["skipped_scores"])
            gaps.append(avg_clicked - avg_skipped)
    return float(np.mean(gaps)) if gaps else 0.0

def speed_benchmark(model, sessions, n_runs=20):
    """
    Average inference time per query in milliseconds.
    Uses the first n_runs sessions (or all if fewer).
    """
    test_sessions = sessions[:n_runs]
    times = []
    for session in test_sessions:
        docs      = [f"{r['title']}. {r['snippet']}" for r in session["results"]]
        all_texts = [session["query"]] + docs

        start = time.perf_counter()
        model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return float(np.mean(times)), float(np.std(times))

# ── Report ────────────────────────────────────────────────────────────────────
def print_report(name, metrics):
    print(f"\n  {'Metric':<35} {'Score':>10}")
    print(f"  {'-'*35} {'-'*10}")
    for label, value, note in metrics:
        print(f"  {label:<35} {value:>10}   {note}")

def compare(args):
    print("\n" + "="*60)
    print("  SEARCH RE-RANKER — MODEL COMPARISON REPORT")
    print("="*60)

    # Load data
    print(f"\n  Loading interaction data: {args.data}")
    sessions = load_sessions(args.data)
    if not sessions:
        print("\n  ERROR: No sessions with both clicks and skips found.")
        print("  Make sure your data has clicked=true and skipped=true records.")
        return
    print(f"  Evaluating on {len(sessions)} sessions with click+skip data.\n")

    # Load models
    print("  Loading base model...")
    base_model = SentenceTransformer(args.base)
    print("  Loading fine-tuned model...")
    try:
        ft_model = SentenceTransformer(args.finetuned)
    except Exception as e:
        print(f"\n  ERROR loading fine-tuned model: {e}")
        print(f"  Make sure {args.finetuned} exists (run finetune.py first).")
        return

    # Run rankings
    print("\n  Running base model rankings...")
    base_ranked = rank_sessions(base_model, sessions)
    print("  Running fine-tuned model rankings...")
    ft_ranked   = rank_sessions(ft_model, sessions)

    # Compute metrics
    base_mrr     = mean_reciprocal_rank(base_ranked)
    ft_mrr       = mean_reciprocal_rank(ft_ranked)

    base_acc     = ranking_accuracy(base_ranked)
    ft_acc       = ranking_accuracy(ft_ranked)

    base_ndcg    = ndcg_at_k(base_ranked)
    ft_ndcg      = ndcg_at_k(ft_ranked)

    base_gap     = avg_score_gap(base_ranked)
    ft_gap       = avg_score_gap(ft_ranked)

    base_ms, base_std = speed_benchmark(base_model, sessions)
    ft_ms,   ft_std   = speed_benchmark(ft_model,   sessions)

    # Print comparison table
    print("\n" + "="*60)
    print("  BASE MODEL RESULTS")
    print("="*60)
    print_report("Base", [
        ("Ranking Accuracy (Precision@1)", f"{base_acc:.1%}",  "clicked result was ranked #1"),
        ("Mean Reciprocal Rank (MRR)",     f"{base_mrr:.4f}",  "1.0 = always #1, 0.5 = avg #2"),
        ("NDCG@10",                        f"{base_ndcg:.4f}", "1.0 = perfect ranking"),
        ("Avg Score Gap (click-skip)",     f"{base_gap:.4f}",  "higher = better discrimination"),
        ("Inference speed",                f"{base_ms:.1f}ms", f"± {base_std:.1f}ms per query"),
    ])

    print("\n" + "="*60)
    print("  FINE-TUNED MODEL RESULTS")
    print("="*60)
    print_report("Fine-tuned", [
        ("Ranking Accuracy (Precision@1)", f"{ft_acc:.1%}",  "clicked result was ranked #1"),
        ("Mean Reciprocal Rank (MRR)",     f"{ft_mrr:.4f}",  "1.0 = always #1, 0.5 = avg #2"),
        ("NDCG@10",                        f"{ft_ndcg:.4f}", "1.0 = perfect ranking"),
        ("Avg Score Gap (click-skip)",     f"{ft_gap:.4f}",  "higher = better discrimination"),
        ("Inference speed",                f"{ft_ms:.1f}ms", f"± {ft_std:.1f}ms per query"),
    ])

    # Delta summary
    print("\n" + "="*60)
    print("  IMPROVEMENT SUMMARY  (fine-tuned vs base)")
    print("="*60)
    print(f"\n  {'Metric':<35} {'Base':>8} {'Fine-tuned':>12} {'Change':>10}")
    print(f"  {'-'*35} {'-'*8} {'-'*12} {'-'*10}")

    rows = [
        ("Ranking Accuracy",   base_acc,  ft_acc,  ".1%"),
        ("MRR",                base_mrr,  ft_mrr,  ".4f"),
        ("NDCG@10",            base_ndcg, ft_ndcg, ".4f"),
        ("Score Gap",          base_gap,  ft_gap,  ".4f"),
        ("Inference (ms)",     base_ms,   ft_ms,   ".1f"),
    ]

    for label, b, f, fmt in rows:
        delta  = f - b
        sign   = "+" if delta >= 0 else ""
        better = "✓" if (delta > 0 and label != "Inference (ms)") or (delta < 0 and label == "Inference (ms)") else ""
        bstr   = format(b, fmt)
        fstr   = format(f, fmt)
        dstr   = f"{sign}{delta:{fmt}}"
        print(f"  {label:<35} {bstr:>8} {fstr:>12} {dstr:>10}  {better}")

    print("\n" + "="*60)
    print("  INTERPRETATION GUIDE")
    print("="*60)
    print("""
  Ranking Accuracy: Did the most relevant result (the one you
    clicked) end up at #1? Higher is better.

  MRR (Mean Reciprocal Rank): On average, how high was your
    clicked result? MRR=1.0 means always at #1. MRR=0.5 means
    at #2 on average. MRR=0.33 means at #3 on average.

  NDCG@10: Quality of the full top-10 ranking. Rewards putting
    clicked results near the top. 1.0 = perfect.

  Score Gap: Average cosine similarity difference between clicked
    results and skipped results. Larger gap means the model more
    confidently separates relevant from irrelevant content.

  Inference speed: Time to encode one query + 10 results. Should
    be similar between models since architecture is unchanged.

  NOTE: With 20-30 sessions, improvements will be modest but real.
  Collect more data and re-finetune for stronger personalization.
""")

if __name__ == "__main__":
    args = parse_args()
    compare(args)
