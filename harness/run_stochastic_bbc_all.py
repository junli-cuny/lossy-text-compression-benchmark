#!/usr/bin/env python3
"""BBC stochastic character-deletion baselines (Gaussian, Bernoulli, Poisson) for all retention rates.

  - Data: BBC_news_latest.xlsx / datasource / Articles
  - Chunking: first 100 chunks of length up to 512 characters
  - Reconstruction: Gemini 2.0 Flash in the original runs (since retired; the default is now gemini-2.5-flash)
  - Metric: BERTScore F1 with distilroberta-base

Results are saved incrementally to `stochastic_bbc_results.json`. Table 1 itself comes from
analysis/run_table1_charlevel.py, which applies these deletion rules (same seeds) and the Step rule to the chunks
of Table 3 with one decoder and stores every sample.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bert_score import score as bert_score_fn

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

DATA_PATH = Path(__file__).parent.parent / "data" / "BBC_news_latest.xlsx"
RESULTS_PATH = Path(__file__).parent / "stochastic_bbc_results.json"
CHUNK_SIZE = 512
N_TEST = 100
RETENTION_RATES = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
SEED = 42

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]  # set your own key

import google.generativeai as genai

genai.configure(api_key=GOOGLE_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash")  # default decoder; the published runs used gemini-2.0-flash, since retired


def load_test_chunks():
    df = pd.read_excel(DATA_PATH, sheet_name="datasource")
    chunks = []
    for article in df["Articles"].dropna():
        text = str(article).strip()
        for i in range(0, len(text), CHUNK_SIZE):
            chunk = text[i:i + CHUNK_SIZE]
            if len(chunk) >= 100:
                chunks.append(chunk)
            if len(chunks) >= N_TEST:
                return chunks[:N_TEST]
    return chunks[:N_TEST]


def bernoulli_delete(text, rkeep, rng):
    mask = rng.random(len(text)) < rkeep
    if not mask.any():
        mask[rng.integers(0, len(text))] = True
    return "".join(ch for ch, keep in zip(text, mask) if keep)


def gaussian_delete(text, rkeep, rng):
    n = len(text)
    n_keep = max(1, round(n * rkeep))
    if n_keep >= n:
        return text

    positions = np.arange(n)
    center = (n - 1) / 2.0
    sigma = max(n / 4.0, 1.0)
    deletion_weights = np.exp(-0.5 * ((positions - center) / sigma) ** 2)
    deletion_weights += rng.random(n) * 1e-6
    keep_idx = np.argsort(deletion_weights)[:n_keep]
    keep_mask = np.zeros(n, dtype=bool)
    keep_mask[keep_idx] = True
    return "".join(ch for ch, keep in zip(text, keep_mask) if keep)


def poisson_delete(text, rkeep, rng):
    n = len(text)
    if rkeep >= 1.0:
        return text

    lam = (1.0 - rkeep) / rkeep
    kept = []
    i = 0
    while i < n:
        kept.append(text[i])
        skip = int(rng.poisson(lam))
        i += skip + 1
    return "".join(kept)


def call_gemini(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = gemini.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"  Gemini attempt {attempt + 1} failed: {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2)
    return ""


def reconstruct(degraded, original_len):
    prompt = (
        "You are a text reconstruction assistant. The input is an English news "
        "article from which some characters have been deleted. Your task is to "
        "reconstruct the original article as faithfully as possible. "
        "Do not add new facts, names, or events not suggested by the input. "
        f"The output should be approximately {original_len} characters long. "
        "Return only the reconstructed text.\n\n"
        f"Degraded text:\n{degraded}"
    )
    return call_gemini(prompt)


def compute_bertscore(candidates, references):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Computing BERTScore on {device}...", flush=True)
    _, _, f1 = bert_score_fn(
        candidates,
        references,
        model_type="distilroberta-base",
        lang="en",
        verbose=False,
        device=device,
        batch_size=8,
    )
    return f1.mean().item(), f1.std().item()


def load_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def run_method(method_name, delete_fn, chunks, results):
    method_results = results.setdefault(method_name, {})

    for rkeep in RETENTION_RATES:
        key = f"{rkeep:.1f}"
        if key in method_results:
            print(f"{method_name} rkeep={key} already done, skipping.", flush=True)
            continue

        print(f"\n=== {method_name} rkeep={key} ===", flush=True)
        reconstructions = []
        originals = []
        actual_keeps = []

        for i, chunk in enumerate(chunks):
            rng = np.random.default_rng(SEED + i + int(rkeep * 1000) * 1000)
            degraded = delete_fn(chunk, rkeep, rng)
            actual_keeps.append(len(degraded) / len(chunk))
            recon = reconstruct(degraded, len(chunk))
            reconstructions.append(recon)
            originals.append(chunk)

            if (i + 1) % 10 == 0:
                avg_keep = sum(actual_keeps) / len(actual_keeps)
                print(f"  {i + 1}/{len(chunks)} done (avg keep={avg_keep:.3f})", flush=True)
            time.sleep(0.1)

        mean_f1, std_f1 = compute_bertscore(reconstructions, originals)
        avg_keep = sum(actual_keeps) / len(actual_keeps)
        print(
            f"  BERTScore F1: {mean_f1:.4f} +/- {std_f1:.4f} "
            f"(avg keep={avg_keep:.4f})",
            flush=True,
        )
        method_results[key] = {
            "mean_f1": round(mean_f1, 4),
            "std_f1": round(std_f1, 4),
            "avg_keep": round(avg_keep, 4),
            "n": len(chunks),
        }
        save_results(results)


def main():
    chunks = load_test_chunks()
    print(f"Loaded {len(chunks)} BBC chunks", flush=True)
    results = load_results()

    run_method("Gaussian", gaussian_delete, chunks, results)
    run_method("Bernoulli", bernoulli_delete, chunks, results)
    run_method("Poisson", poisson_delete, chunks, results)

    print("\n=== FINAL RESULTS ===", flush=True)
    for method, method_results in results.items():
        print(method, flush=True)
        for key in sorted(method_results.keys(), reverse=True):
            row = method_results[key]
            print(
                f"  rkeep={key}: {row['mean_f1']:.4f} +/- {row['std_f1']:.4f} "
                f"(avg keep={row['avg_keep']:.4f})",
                flush=True,
            )


if __name__ == "__main__":
    main()
