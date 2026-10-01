#!/usr/bin/env python3
"""Two baselines on BBC News:
1. LLM summarization baseline (Gemini compress-to-length rewriting)
2. Entropy-based token dropping (GPT-2 surprisal-based deletion + Gemini reconstruction)

Outputs results as JSON (baseline_results.json) and prints a summary table. The paper's Summ. column of Table 3
uses this summarization prompt in analysis/run_table3_wordfreq_summ.py.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import openpyxl
import torch
from bert_score import score as bert_score_fn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent.parent / "data" / "BBC_news_latest.xlsx"
RESULTS_PATH = Path(__file__).parent / "baseline_results.json"
CHUNK_SIZE = 512
N_TEST = 100
RETENTION_RATES = [0.9, 0.7, 0.5, 0.3, 0.1]

# Load API key
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.env"))
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]  # set your own key

if not GOOGLE_API_KEY:
    sys.exit("GOOGLE_API_KEY not found")

import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash")  # default decoder; the published runs used gemini-2.0-flash, since retired


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_test_chunks():
    """Load articles from BBC dataset, chunk to 512 chars, return first N_TEST."""
    wb = openpyxl.load_workbook(str(DATA_PATH), read_only=True)
    ws = wb["datasource"]

    chunks = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        article = str(row[2]) if row[2] else ""
        if not article or len(article) < 100:
            continue
        # Chunk article into ~512-char pieces
        for i in range(0, len(article), CHUNK_SIZE):
            chunk = article[i:i + CHUNK_SIZE]
            if len(chunk) >= 100:  # skip tiny trailing chunks
                chunks.append(chunk)
            if len(chunks) >= N_TEST:
                break
        if len(chunks) >= N_TEST:
            break

    wb.close()
    return chunks[:N_TEST]


# ---------------------------------------------------------------------------
# Gemini API call with retry
# ---------------------------------------------------------------------------
def call_gemini(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = gemini.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"  Gemini attempt {attempt+1} failed: {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return ""


# ---------------------------------------------------------------------------
# Baseline 1: LLM Summarization
# ---------------------------------------------------------------------------
def _save_partial(all_results):
    """Save partial results to disk."""
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)


def run_summarization_baseline(chunks, all_results):
    """Prompt Gemini to summarize each chunk to target character length."""
    results = all_results.get("summarization", {})

    for rkeep in RETENTION_RATES:
        if str(rkeep) in results:
            print(f"  rkeep={rkeep} already done, skipping.", flush=True)
            continue
        print(f"\n=== Summarization baseline rkeep={rkeep} ===", flush=True)
        reconstructions = []

        for i, chunk in enumerate(chunks):
            target_len = int(len(chunk) * rkeep)
            prompt = (
                f"Compress the following news text into exactly {target_len} characters "
                f"(±15%). Preserve all key facts, names, numbers, and events. "
                f"Do not add any new information. Return only the compressed text.\n\n"
                f"Text:\n{chunk}"
            )
            result = call_gemini(prompt)
            reconstructions.append(result)

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(chunks)} done", flush=True)
            time.sleep(0.1)  # rate limit

        # Compute BERTScore
        print("  Computing BERTScore...", flush=True)
        P, R, F1 = bert_score_fn(reconstructions, chunks, model_type="distilroberta-base", lang="en", verbose=False, device="cpu")
        mean_f1 = F1.mean().item()
        std_f1 = F1.std().item()
        print(f"  BERTScore F1: {mean_f1:.4f} ± {std_f1:.4f}", flush=True)

        results[str(rkeep)] = {
            "mean_f1": round(mean_f1, 4),
            "std_f1": round(std_f1, 4),
            "n": len(chunks),
        }
        # Save after each retention rate
        all_results["summarization"] = results
        _save_partial(all_results)

    return results


# ---------------------------------------------------------------------------
# Baseline 2: Entropy-based token dropping
# ---------------------------------------------------------------------------
def load_gpt2():
    """Load GPT-2 for surprisal computation."""
    print("Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return tokenizer, model


def compute_token_surprisal(text, tokenizer, model):
    """Compute per-token surprisal (self-information) using GPT-2."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :]
    shift_labels = inputs["input_ids"][:, 1:]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    surprisal = -token_log_probs.squeeze(0).cpu().numpy()  # higher = more surprising/informative

    # First token has no prediction, assign median surprisal
    median_s = float(np.median(surprisal))
    surprisal = np.concatenate([[median_s], surprisal])

    tokens = inputs["input_ids"].squeeze(0).cpu().tolist()
    token_strs = [tokenizer.decode([t]) for t in tokens]

    return token_strs, surprisal


def entropy_based_deletion(text, rkeep, tokenizer, model):
    """Drop tokens with lowest surprisal (most predictable) to reach target retention."""
    token_strs, surprisal = compute_token_surprisal(text, tokenizer, model)

    n_tokens = len(token_strs)
    n_keep = max(1, int(n_tokens * rkeep))

    # Keep tokens with highest surprisal (most informative)
    keep_indices = set(np.argsort(surprisal)[-n_keep:])

    kept = []
    for i, tok in enumerate(token_strs):
        if i in keep_indices:
            kept.append(tok)

    return "".join(kept)


def run_entropy_baseline(chunks, all_results):
    """Run entropy-based token dropping + Gemini reconstruction."""
    tokenizer, model = load_gpt2()
    results = all_results.get("entropy", {})

    for rkeep in RETENTION_RATES:
        if str(rkeep) in results:
            print(f"  rkeep={rkeep} already done, skipping.", flush=True)
            continue
        print(f"\n=== Entropy baseline rkeep={rkeep} ===", flush=True)
        reconstructions = []

        for i, chunk in enumerate(chunks):
            # Step 1: Entropy-based deletion
            degraded = entropy_based_deletion(chunk, rkeep, tokenizer, model)

            # Step 2: Gemini reconstruction
            prompt = (
                "You are a text reconstruction assistant. The input "
                "is an English news article from which some tokens have "
                "been deleted based on their predictability. Your task is to "
                "reconstruct the original article as faithfully as possible. "
                "Do not add new facts, names, or events not suggested by the input. "
                "The output should be approximately the same length as the original "
                f"article (~{len(chunk)} characters). Return only the reconstructed text.\n\n"
                f"Degraded text:\n{degraded}"
            )
            result = call_gemini(prompt)
            reconstructions.append(result)

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(chunks)} done", flush=True)
            time.sleep(0.1)

        # Compute BERTScore
        print("  Computing BERTScore...", flush=True)
        P, R, F1 = bert_score_fn(reconstructions, chunks, model_type="distilroberta-base", lang="en", verbose=False, device="cpu")
        mean_f1 = F1.mean().item()
        std_f1 = F1.std().item()
        print(f"  BERTScore F1: {mean_f1:.4f} ± {std_f1:.4f}", flush=True)

        results[str(rkeep)] = {
            "mean_f1": round(mean_f1, 4),
            "std_f1": round(std_f1, 4),
            "n": len(chunks),
        }
        all_results["entropy"] = results
        _save_partial(all_results)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading test chunks...")
    chunks = load_test_chunks()
    print(f"Loaded {len(chunks)} test chunks (avg {np.mean([len(c) for c in chunks]):.0f} chars)")

    # Load any existing partial results
    all_results = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        print(f"Loaded existing partial results from {RESULTS_PATH}")

    print("\n" + "=" * 60)
    print("BASELINE 1: LLM Summarization")
    print("=" * 60)
    all_results["summarization"] = run_summarization_baseline(chunks, all_results)

    print("\n" + "=" * 60)
    print("BASELINE 2: Entropy-Based Token Dropping")
    print("=" * 60)
    all_results["entropy"] = run_entropy_baseline(chunks, all_results)

    _save_partial(all_results)
    print(f"\nAll results saved to {RESULTS_PATH}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'rkeep':<8} {'Summarization':<20} {'Entropy+Gemini':<20}")
    print("-" * 48)

    for rkeep in RETENTION_RATES:
        k = str(rkeep)
        summ = all_results["summarization"].get(k, {}).get("mean_f1", "---")
        entr = all_results["entropy"].get(k, {}).get("mean_f1", "---")
        print(f"{rkeep:<8} {summ:<20} {entr:<20}")


if __name__ == "__main__":
    main()
