#!/usr/bin/env python3
"""Hybrid compression experiments on BBC News (the original runs of the semantic and hybrid strategies).

Option 1 (Entropy-LP):
    GPT-2 surprisal -> 3 surprisal buckets (tertiles) -> calibrate B_k^full ->
    LP budget allocation -> delete lowest-surprisal tokens within each bucket.

Option 2 (Entropy-in-FreqBuckets):
    Zipf frequency -> 3 frequency buckets (as in WordFreq) -> calibrate B_k^full ->
    LP budget allocation -> within each bucket delete the lowest-surprisal tokens.

Option 3 (Hybrid-alpha):
    Combined score = alpha * rank_norm(zipf_freq) + (1-alpha) * rank_norm(1/surprisal)
    Delete tokens with the highest combined score; alpha in {0.3, 0.5, 0.7}.

All run on n=100 test chunks of 512 characters at rkeep in {0.9, 0.7, 0.5, 0.3, 0.1}, with results saved
incrementally to hybrid_results.json (resumable). The shipped hybrid_results.json holds only the bucket calibration
of the published run, which used Gemini 2.0 Flash (since retired; the default decoder is now gemini-2.5-flash).
The paper's BBC semantic rows come from the re-run with per-sample records, analysis/run_e0_bbc_hybrid.py.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import openpyxl
import torch
from bert_score import score as bert_score_fn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from wordfreq import zipf_frequency

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent.parent / "data" / "BBC_news_latest.xlsx"
RESULTS_PATH = Path(__file__).parent / "hybrid_results.json"
CHUNK_SIZE = 512
N_TEST = 100
N_CALIB = 50
RETENTION_RATES = [0.9, 0.7, 0.5, 0.3, 0.1]
ALPHA_VALUES = [0.3, 0.5, 0.7]

FREQ_HIGH = 5.0
FREQ_MID  = 3.0

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.env"))
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]  # set your own key

import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash")  # default decoder; the published runs used gemini-2.0-flash, since retired


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def load_results() -> dict:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_results(all_results: dict):
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_chunks(n: int) -> list:
    wb = openpyxl.load_workbook(str(DATA_PATH), read_only=True)
    ws = wb["datasource"]
    chunks = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        article = str(row[2]) if row[2] else ""
        if not article or len(article) < 100:
            continue
        for i in range(0, len(article), CHUNK_SIZE):
            chunk = article[i:i + CHUNK_SIZE]
            if len(chunk) >= 100:
                chunks.append(chunk)
            if len(chunks) >= n:
                break
        if len(chunks) >= n:
            break
    wb.close()
    return chunks[:n]


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
def call_gemini(prompt: str, max_retries: int = 3) -> str:
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


def reconstruct_prompt(degraded: str, original_len: int) -> str:
    return (
        "You are a text reconstruction assistant. The input "
        "is an English news article from which some tokens have "
        "been deleted based on their predictability. Your task is to "
        "reconstruct the original article as faithfully as possible. "
        "Do not add new facts, names, or events not suggested by the input. "
        "The output should be approximately the same length as the original "
        f"article (~{original_len} characters). Return only the reconstructed text.\n\n"
        f"Degraded text:\n{degraded}"
    )


# ---------------------------------------------------------------------------
# GPT-2 surprisal
# ---------------------------------------------------------------------------
def load_gpt2():
    print("Loading GPT-2...", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return tokenizer, model


def compute_token_surprisal(text: str, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = inputs["input_ids"][:, 1:]
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    surprisal = -token_log_probs.squeeze(0).cpu().numpy()
    median_s = float(np.median(surprisal))
    surprisal = np.concatenate([[median_s], surprisal])
    tokens = inputs["input_ids"].squeeze(0).cpu().tolist()
    token_strs = [tokenizer.decode([t]) for t in tokens]
    return token_strs, surprisal


# ---------------------------------------------------------------------------
# Zipf helpers
# ---------------------------------------------------------------------------
def get_zipf_scores(token_strs: list) -> np.ndarray:
    scores = []
    for tok in token_strs:
        word = tok.strip()
        if not word or not word[0].isalpha():
            scores.append(0.0)
        else:
            scores.append(zipf_frequency(word.lower(), "en"))
    return np.array(scores, dtype=float)


def assign_freq_bucket(zipf_scores: np.ndarray) -> np.ndarray:
    buckets = np.zeros(len(zipf_scores), dtype=int)
    buckets[zipf_scores >= FREQ_HIGH] = 2
    buckets[(zipf_scores >= FREQ_MID) & (zipf_scores < FREQ_HIGH)] = 1
    return buckets


# ---------------------------------------------------------------------------
# Rank normalisation
# ---------------------------------------------------------------------------
def rank_normalize(arr: np.ndarray) -> np.ndarray:
    n = len(arr)
    if n <= 1:
        return np.zeros(n)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    return ranks / (n - 1)


# ---------------------------------------------------------------------------
# LP solver (greedy closed-form)
# ---------------------------------------------------------------------------
def solve_lp(b_full: np.ndarray, bucket_sizes: np.ndarray, n_delete_total: int) -> np.ndarray:
    """
    Greedy LP: sort buckets by (1 - B_k^full) ascending (cheapest first),
    fill greedily until deletion budget is met.
    Returns w_k in [0,1] per bucket.
    """
    cost = 1.0 - b_full
    order = np.argsort(cost)
    w = np.zeros(len(b_full))
    remaining = n_delete_total
    for k in order:
        cap = bucket_sizes[k]
        if cap <= 0:
            continue
        if remaining >= cap:
            w[k] = 1.0
            remaining -= cap
        else:
            w[k] = remaining / cap
            remaining = 0
            break
    return w


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibrate_surprisal_buckets(calib_chunks: list, tokenizer, model, all_results: dict) -> tuple:
    if "calib_surprisal" in all_results:
        print("  Surprisal calibration already exists, skipping.", flush=True)
        d = all_results["calib_surprisal"]
        return np.array(d["b_full"]), d["thresholds"]

    print(f"\n--- Calibrating surprisal buckets (n={len(calib_chunks)}) ---", flush=True)

    # Compute global tertile thresholds across calibration corpus
    all_surp = []
    for chunk in calib_chunks:
        _, surp = compute_token_surprisal(chunk, tokenizer, model)
        all_surp.extend(surp.tolist())
    t33 = float(np.percentile(all_surp, 33.33))
    t67 = float(np.percentile(all_surp, 66.67))
    print(f"  Surprisal thresholds: 33%={t33:.3f}, 67%={t67:.3f}", flush=True)

    bucket_recs = [[], [], []]
    bucket_refs = [[], [], []]

    for i, chunk in enumerate(calib_chunks):
        token_strs, surprisal = compute_token_surprisal(chunk, tokenizer, model)
        for bk in range(3):
            if bk == 0:
                mask = surprisal <= t33
            elif bk == 1:
                mask = (surprisal > t33) & (surprisal <= t67)
            else:
                mask = surprisal > t67
            degraded = "".join(tok for j, tok in enumerate(token_strs) if not mask[j])
            rec = call_gemini(reconstruct_prompt(degraded, len(chunk)))
            bucket_recs[bk].append(rec)
            bucket_refs[bk].append(chunk)
            time.sleep(0.1)
        if (i + 1) % 10 == 0:
            print(f"  calibration {i+1}/{len(calib_chunks)}", flush=True)

    b_full = np.zeros(3)
    for bk in range(3):
        _, _, F1 = bert_score_fn(bucket_recs[bk], bucket_refs[bk], model_type="distilroberta-base", lang="en", verbose=False, device="cpu")
        b_full[bk] = F1.mean().item()
        print(f"  Surprisal bucket {bk}: B_k^full={b_full[bk]:.4f}", flush=True)

    all_results["calib_surprisal"] = {"b_full": b_full.tolist(), "thresholds": [t33, t67]}
    save_results(all_results)
    return b_full, [t33, t67]


def calibrate_freq_buckets(calib_chunks: list, tokenizer, model, all_results: dict) -> np.ndarray:
    if "calib_frequency" in all_results:
        print("  Frequency calibration already exists, skipping.", flush=True)
        return np.array(all_results["calib_frequency"]["b_full"])

    print(f"\n--- Calibrating frequency buckets (n={len(calib_chunks)}) ---", flush=True)

    bucket_recs = [[], [], []]
    bucket_refs = [[], [], []]

    for i, chunk in enumerate(calib_chunks):
        token_strs, _ = compute_token_surprisal(chunk, tokenizer, model)
        zipf_scores = get_zipf_scores(token_strs)
        bucket_ids = assign_freq_bucket(zipf_scores)
        for bk in range(3):
            mask = bucket_ids == bk
            degraded = "".join(tok for j, tok in enumerate(token_strs) if not mask[j])
            rec = call_gemini(reconstruct_prompt(degraded, len(chunk)))
            bucket_recs[bk].append(rec)
            bucket_refs[bk].append(chunk)
            time.sleep(0.1)
        if (i + 1) % 10 == 0:
            print(f"  calibration {i+1}/{len(calib_chunks)}", flush=True)

    b_full = np.zeros(3)
    for bk in range(3):
        _, _, F1 = bert_score_fn(bucket_recs[bk], bucket_refs[bk], model_type="distilroberta-base", lang="en", verbose=False, device="cpu")
        b_full[bk] = F1.mean().item()
        print(f"  Frequency bucket {bk}: B_k^full={b_full[bk]:.4f}", flush=True)

    all_results["calib_frequency"] = {"b_full": b_full.tolist()}
    save_results(all_results)
    return b_full


# ---------------------------------------------------------------------------
# Option 1: Entropy-LP
# ---------------------------------------------------------------------------
def run_entropy_lp(chunks: list, tokenizer, model, b_full: np.ndarray,
                   thresholds: list, all_results: dict):
    results = all_results.get("entropy_lp", {})
    t33, t67 = thresholds

    for rkeep in RETENTION_RATES:
        key = str(rkeep)
        if key in results:
            print(f"  rkeep={rkeep} already done, skipping.", flush=True)
            continue
        print(f"\n=== Option 1 (Entropy-LP) rkeep={rkeep} ===", flush=True)
        reconstructions = []

        for i, chunk in enumerate(chunks):
            token_strs, surprisal = compute_token_surprisal(chunk, tokenizer, model)
            n = len(token_strs)
            n_del = max(0, int(n * (1 - rkeep)))

            # Assign surprisal buckets
            bk_ids = np.zeros(n, dtype=int)
            bk_ids[(surprisal > t33) & (surprisal <= t67)] = 1
            bk_ids[surprisal > t67] = 2
            bucket_sizes = np.array([np.sum(bk_ids == k) for k in range(3)])

            w = solve_lp(b_full, bucket_sizes, n_del)

            delete_mask = np.zeros(n, dtype=bool)
            for bk in range(3):
                idx = np.where(bk_ids == bk)[0]
                n_del_bk = int(round(w[bk] * len(idx)))
                if n_del_bk > 0:
                    cheapest = idx[np.argsort(surprisal[idx])][:n_del_bk]
                    delete_mask[cheapest] = True

            degraded = "".join(t for j, t in enumerate(token_strs) if not delete_mask[j])
            rec = call_gemini(reconstruct_prompt(degraded, len(chunk)))
            reconstructions.append(rec)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(chunks)} done", flush=True)
            time.sleep(0.1)

        print("  Computing BERTScore...", flush=True)
        _, _, F1 = bert_score_fn(reconstructions, chunks, model_type="distilroberta-base", lang="en", verbose=False, device="cpu")
        mf1, sf1 = F1.mean().item(), F1.std().item()
        print(f"  BERTScore F1: {mf1:.4f} +/- {sf1:.4f}", flush=True)
        results[key] = {"mean_f1": round(mf1, 4), "std_f1": round(sf1, 4), "n": len(chunks)}
        all_results["entropy_lp"] = results
        save_results(all_results)

    return results


# ---------------------------------------------------------------------------
# Option 2: Entropy-in-FreqBuckets
# ---------------------------------------------------------------------------
def run_entropy_in_freqbuckets(chunks: list, tokenizer, model,
                                b_full: np.ndarray, all_results: dict):
    results = all_results.get("entropy_in_freqbuckets", {})

    for rkeep in RETENTION_RATES:
        key = str(rkeep)
        if key in results:
            print(f"  rkeep={rkeep} already done, skipping.", flush=True)
            continue
        print(f"\n=== Option 2 (Entropy-in-FreqBuckets) rkeep={rkeep} ===", flush=True)
        reconstructions = []

        for i, chunk in enumerate(chunks):
            token_strs, surprisal = compute_token_surprisal(chunk, tokenizer, model)
            n = len(token_strs)
            n_del = max(0, int(n * (1 - rkeep)))

            zipf_scores = get_zipf_scores(token_strs)
            bk_ids = assign_freq_bucket(zipf_scores)
            bucket_sizes = np.array([np.sum(bk_ids == k) for k in range(3)])

            w = solve_lp(b_full, bucket_sizes, n_del)

            delete_mask = np.zeros(n, dtype=bool)
            for bk in range(3):
                idx = np.where(bk_ids == bk)[0]
                n_del_bk = int(round(w[bk] * len(idx)))
                if n_del_bk > 0:
                    cheapest = idx[np.argsort(surprisal[idx])][:n_del_bk]
                    delete_mask[cheapest] = True

            degraded = "".join(t for j, t in enumerate(token_strs) if not delete_mask[j])
            rec = call_gemini(reconstruct_prompt(degraded, len(chunk)))
            reconstructions.append(rec)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(chunks)} done", flush=True)
            time.sleep(0.1)

        print("  Computing BERTScore...", flush=True)
        _, _, F1 = bert_score_fn(reconstructions, chunks, model_type="distilroberta-base", lang="en", verbose=False, device="cpu")
        mf1, sf1 = F1.mean().item(), F1.std().item()
        print(f"  BERTScore F1: {mf1:.4f} +/- {sf1:.4f}", flush=True)
        results[key] = {"mean_f1": round(mf1, 4), "std_f1": round(sf1, 4), "n": len(chunks)}
        all_results["entropy_in_freqbuckets"] = results
        save_results(all_results)

    return results


# ---------------------------------------------------------------------------
# Option 3: Hybrid-alpha
# ---------------------------------------------------------------------------
def run_hybrid_alpha(chunks: list, tokenizer, model,
                     alpha: float, all_results: dict):
    result_key = f"hybrid_alpha_{alpha}"
    results = all_results.get(result_key, {})

    for rkeep in RETENTION_RATES:
        key = str(rkeep)
        if key in results:
            print(f"  alpha={alpha} rkeep={rkeep} already done, skipping.", flush=True)
            continue
        print(f"\n=== Option 3 (Hybrid alpha={alpha}) rkeep={rkeep} ===", flush=True)
        reconstructions = []

        for i, chunk in enumerate(chunks):
            token_strs, surprisal = compute_token_surprisal(chunk, tokenizer, model)
            n = len(token_strs)
            n_keep = max(1, int(n * rkeep))

            zipf_scores = get_zipf_scores(token_strs)
            rank_freq = rank_normalize(zipf_scores)  # high = more common = more deletable
            inv_surp = np.where(surprisal > 0, 1.0 / surprisal, 0.0)
            rank_pred = rank_normalize(inv_surp)     # high = low surprisal = more deletable

            combined = alpha * rank_freq + (1.0 - alpha) * rank_pred
            keep_indices = set(np.argsort(combined)[:n_keep])
            degraded = "".join(t for j, t in enumerate(token_strs) if j in keep_indices)

            rec = call_gemini(reconstruct_prompt(degraded, len(chunk)))
            reconstructions.append(rec)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(chunks)} done", flush=True)
            time.sleep(0.1)

        print("  Computing BERTScore...", flush=True)
        _, _, F1 = bert_score_fn(reconstructions, chunks, model_type="distilroberta-base", lang="en", verbose=False, device="cpu")
        mf1, sf1 = F1.mean().item(), F1.std().item()
        print(f"  BERTScore F1: {mf1:.4f} +/- {sf1:.4f}", flush=True)
        results[key] = {"mean_f1": round(mf1, 4), "std_f1": round(sf1, 4), "n": len(chunks)}
        all_results[result_key] = results
        save_results(all_results)

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(all_results: dict):
    print("\n" + "=" * 90, flush=True)
    print("SUMMARY (BERTScore F1)", flush=True)
    print("=" * 90, flush=True)

    cols = [
        ("Entropy-LP",      all_results.get("entropy_lp", {})),
        ("Ent.-FreqBkt",    all_results.get("entropy_in_freqbuckets", {})),
        ("Hybrid-0.3",      all_results.get("hybrid_alpha_0.3", {})),
        ("Hybrid-0.5",      all_results.get("hybrid_alpha_0.5", {})),
        ("Hybrid-0.7",      all_results.get("hybrid_alpha_0.7", {})),
    ]

    header = f"{'rkeep':<8}" + "".join(f"{lbl:<16}" for lbl, _ in cols)
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for rkeep in RETENTION_RATES:
        k = str(rkeep)
        row = f"{rkeep:<8}"
        for lbl, data in cols:
            if isinstance(data, dict) and k in data:
                v = data[k]
                val = v if isinstance(v, float) else v.get("mean_f1", "---")
            else:
                val = "---"
            row += f"{str(val):<16}"
        print(row, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading test chunks...", flush=True)
    all_chunks = load_chunks(N_TEST)
    calib_chunks = all_chunks[:N_CALIB]
    print(f"Loaded {len(all_chunks)} chunks (avg {np.mean([len(c) for c in all_chunks]):.0f} chars)", flush=True)

    all_results = load_results()
    if all_results:
        print(f"Loaded existing partial results from {RESULTS_PATH}", flush=True)

    tokenizer, model = load_gpt2()

    print("\n" + "=" * 60, flush=True)
    print("CALIBRATION", flush=True)
    print("=" * 60, flush=True)
    b_full_surp, thresholds = calibrate_surprisal_buckets(calib_chunks, tokenizer, model, all_results)
    b_full_freq = calibrate_freq_buckets(calib_chunks, tokenizer, model, all_results)

    print("\n" + "=" * 60, flush=True)
    print("OPTION 1: Entropy-LP", flush=True)
    print("=" * 60, flush=True)
    run_entropy_lp(all_chunks, tokenizer, model, b_full_surp, thresholds, all_results)

    print("\n" + "=" * 60, flush=True)
    print("OPTION 2: Entropy-in-FreqBuckets", flush=True)
    print("=" * 60, flush=True)
    run_entropy_in_freqbuckets(all_chunks, tokenizer, model, b_full_freq, all_results)

    print("\n" + "=" * 60, flush=True)
    print("OPTION 3: Hybrid-alpha", flush=True)
    print("=" * 60, flush=True)
    for alpha in ALPHA_VALUES:
        run_hybrid_alpha(all_chunks, tokenizer, model, alpha, all_results)

    save_results(all_results)
    print(f"\nAll results saved to {RESULTS_PATH}", flush=True)
    print_summary(all_results)


if __name__ == "__main__":
    main()
