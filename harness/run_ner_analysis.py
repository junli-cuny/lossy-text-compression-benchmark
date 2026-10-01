#!/usr/bin/env python3
"""Named-entity survival in the compressed skeleton, before reconstruction (Table 11).

For each method and rkeep in {0.9, 0.7, 0.5, 0.3, 0.1}, on the first 100 BBC test chunks of 512 characters, reports
  - entity_preservation: fraction of the original's spaCy entity mentions found in the skeleton (case-insensitive
    substring, whitespace-stripped fallback; label ignored)
  - total_entities, preserved, and per-type rates.

Methods: Step, WordLen, WordFreq, LP, Entropy, Entropy-LP, Entropy-in-FreqBuckets, Hybrid-alpha (0.3, 0.5, 0.7).
The semantic and hybrid deletions are those of run_hybrid_experiments.py (bucket calibration read from
hybrid_results.json). The Step, WordLen, WordFreq and LP rows are approximations of the word-level pipeline's rules
on the same chunks, as used for Table 11: Step keeps evenly spaced characters, WordLen applies staged
whitespace / vowel / short-word / long-word-tail / punctuation deletion at character level, WordFreq keeps the
lowest-Zipf GPT-2 tokens, and LP allocates the budget over three Zipf buckets of GPT-2 tokens.

Results are saved to results/ner_results.json.
"""

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import openpyxl
import spacy
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from wordfreq import zipf_frequency

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent.parent / "data" / "BBC_news_latest.xlsx"
RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "ner_results.json"
HYBRID_RESULTS_PATH = Path(__file__).parent / "hybrid_results.json"
CHUNK_SIZE = 512
N_TEST = 100
RETENTION_RATES = [0.9, 0.7, 0.5, 0.3, 0.1]

FREQ_HIGH = 5.0
FREQ_MID = 3.0

# ---------------------------------------------------------------------------
# Data loading (same as other scripts)
# ---------------------------------------------------------------------------
def load_test_chunks():
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
            if len(chunks) >= N_TEST:
                break
        if len(chunks) >= N_TEST:
            break
    wb.close()
    return chunks[:N_TEST]


def load_hybrid_calibration():
    if not HYBRID_RESULTS_PATH.exists():
        raise FileNotFoundError(f"{HYBRID_RESULTS_PATH} not found")
    with open(HYBRID_RESULTS_PATH) as f:
        data = json.load(f)
    calib_surprisal = data.get("calib_surprisal")
    calib_frequency = data.get("calib_frequency")
    if not calib_surprisal or not calib_frequency:
        raise ValueError("hybrid_results.json missing calibration entries")
    return (
        np.array(calib_surprisal["b_full"], dtype=float),
        calib_surprisal["thresholds"],
        np.array(calib_frequency["b_full"], dtype=float),
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


def compute_token_surprisal(text, tokenizer, model):
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
# WordFreq deletion: delete tokens with HIGHEST Zipf frequency
# ---------------------------------------------------------------------------
def wordfreq_deletion(text, rkeep, tokenizer, model):
    """
    Tokenize with GPT-2, compute Zipf frequency for each token,
    delete highest-frequency tokens to reach target rkeep.
    Returns the compressed text.
    """
    token_strs, _ = compute_token_surprisal(text, tokenizer, model)
    n_tokens = len(token_strs)
    n_keep = max(1, int(n_tokens * rkeep))

    # Compute Zipf frequency for each token
    zipf_scores = []
    for tok in token_strs:
        word = tok.strip()
        if not word or not word[0].isalpha():
            zipf_scores.append(0.0)  # punctuation/numbers: low score = keep
        else:
            zipf_scores.append(zipf_frequency(word.lower(), "en"))
    zipf_scores = np.array(zipf_scores)

    # Keep tokens with LOWEST frequency (most informative)
    keep_indices = set(np.argsort(zipf_scores)[:n_keep])
    kept = [tok for i, tok in enumerate(token_strs) if i in keep_indices]
    return "".join(kept)


# ---------------------------------------------------------------------------
# Entropy deletion: delete tokens with LOWEST surprisal
# ---------------------------------------------------------------------------
def entropy_deletion(text, rkeep, tokenizer, model):
    """
    Delete most predictable (lowest surprisal) tokens.
    """
    token_strs, surprisal = compute_token_surprisal(text, tokenizer, model)
    n_tokens = len(token_strs)
    n_keep = max(1, int(n_tokens * rkeep))

    # Keep tokens with HIGHEST surprisal (most informative)
    keep_indices = set(np.argsort(surprisal)[-n_keep:])
    kept = [tok for i, tok in enumerate(token_strs) if i in keep_indices]
    return "".join(kept)


def step_deletion(text, rkeep):
    n = len(text)
    n_keep = max(1, int(n * rkeep))
    if n_keep >= n:
        return text
    keep_idx = set(np.linspace(0, n - 1, n_keep, dtype=int).tolist())
    return "".join(ch for i, ch in enumerate(text) if i in keep_idx)


def _choose_evenly_spaced_indices(indices, k):
    if k <= 0 or not indices:
        return []
    if k >= len(indices):
        return list(indices)
    pick_idx = np.linspace(0, len(indices) - 1, k, dtype=int).tolist()
    return [indices[i] for i in pick_idx]


def _apply_stage_deletions(candidates, remove_set, target_remove):
    remaining = [idx for idx in candidates if idx not in remove_set]
    need = target_remove - len(remove_set)
    if need <= 0 or not remaining:
        return
    for idx in _choose_evenly_spaced_indices(remaining, min(need, len(remaining))):
        remove_set.add(idx)


def wordlen_deletion(text, rkeep):
    n = len(text)
    n_keep = max(1, int(n * rkeep))
    target_remove = max(0, n - n_keep)
    if target_remove == 0:
        return text

    remove_set = set()
    token_pattern = re.compile(r"\s+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]+")
    vowels = set("aeiouAEIOU")

    whitespace_positions = []
    vowel_positions = []
    short_word_positions = []
    long_word_tail_positions = []
    punct_digit_positions = []

    for match in token_pattern.finditer(text):
        tok = match.group(0)
        start, end = match.span()
        positions = list(range(start, end))
        if tok.isspace():
            whitespace_positions.extend(positions)
        elif tok.isalpha():
            if len(tok) >= 3:
                for off, ch in enumerate(tok):
                    if off > 0 and ch in vowels:
                        vowel_positions.append(start + off)
            if len(tok) <= 2:
                short_word_positions.extend(positions)
            if len(tok) > 6:
                long_word_tail_positions.extend(range(start + 4, end))
        elif tok.isdigit() or re.fullmatch(r"[^\w\s]+", tok):
            punct_digit_positions.extend(positions)

    _apply_stage_deletions(whitespace_positions, remove_set, target_remove)
    _apply_stage_deletions(vowel_positions, remove_set, target_remove)
    _apply_stage_deletions(short_word_positions, remove_set, target_remove)
    _apply_stage_deletions(long_word_tail_positions, remove_set, target_remove)
    _apply_stage_deletions(punct_digit_positions, remove_set, target_remove)

    if len(remove_set) < target_remove:
        remaining = [idx for idx in range(n) if idx not in remove_set]
        for idx in _choose_evenly_spaced_indices(remaining, target_remove - len(remove_set)):
            remove_set.add(idx)

    return "".join(ch for i, ch in enumerate(text) if i not in remove_set)


def assign_freq_bucket(zipf_scores):
    buckets = np.zeros(len(zipf_scores), dtype=int)
    buckets[zipf_scores >= FREQ_HIGH] = 2
    buckets[(zipf_scores >= FREQ_MID) & (zipf_scores < FREQ_HIGH)] = 1
    return buckets


def rank_normalize(arr):
    n = len(arr)
    if n <= 1:
        return np.zeros(n)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    return ranks / (n - 1)


def solve_lp(b_full, bucket_sizes, n_delete_total):
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


def entropy_lp_deletion(text, rkeep, tokenizer, model, b_full, thresholds):
    token_strs, surprisal = compute_token_surprisal(text, tokenizer, model)
    n = len(token_strs)
    n_del = max(0, int(n * (1 - rkeep)))
    t33, t67 = thresholds
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
    return "".join(tok for i, tok in enumerate(token_strs) if not delete_mask[i])


def entropy_in_freqbuckets_deletion(text, rkeep, tokenizer, model, b_full):
    token_strs, surprisal = compute_token_surprisal(text, tokenizer, model)
    n = len(token_strs)
    n_del = max(0, int(n * (1 - rkeep)))
    zipf_scores = []
    for tok in token_strs:
        word = tok.strip()
        if not word or not word[0].isalpha():
            zipf_scores.append(0.0)
        else:
            zipf_scores.append(zipf_frequency(word.lower(), "en"))
    zipf_scores = np.array(zipf_scores)
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
    return "".join(tok for i, tok in enumerate(token_strs) if not delete_mask[i])


def opt_deletion(text, rkeep, tokenizer, model, b_full):
    token_strs, _ = compute_token_surprisal(text, tokenizer, model)
    n = len(token_strs)
    n_del = max(0, int(n * (1 - rkeep)))
    zipf_scores = []
    for tok in token_strs:
        word = tok.strip()
        if not word or not word[0].isalpha():
            zipf_scores.append(0.0)
        else:
            zipf_scores.append(zipf_frequency(word.lower(), "en"))
    zipf_scores = np.array(zipf_scores)
    bk_ids = assign_freq_bucket(zipf_scores)
    bucket_sizes = np.array([np.sum(bk_ids == k) for k in range(3)])
    w = solve_lp(b_full, bucket_sizes, n_del)
    delete_mask = np.zeros(n, dtype=bool)
    for bk in range(3):
        idx = np.where(bk_ids == bk)[0]
        n_del_bk = int(round(w[bk] * len(idx)))
        if n_del_bk > 0:
            most_common = idx[np.argsort(zipf_scores[idx])[::-1]][:n_del_bk]
            delete_mask[most_common] = True
    return "".join(tok for i, tok in enumerate(token_strs) if not delete_mask[i])


def hybrid_alpha_deletion(text, rkeep, tokenizer, model, alpha):
    token_strs, surprisal = compute_token_surprisal(text, tokenizer, model)
    n = len(token_strs)
    n_keep = max(1, int(n * rkeep))
    zipf_scores = []
    for tok in token_strs:
        word = tok.strip()
        if not word or not word[0].isalpha():
            zipf_scores.append(0.0)
        else:
            zipf_scores.append(zipf_frequency(word.lower(), "en"))
    zipf_scores = np.array(zipf_scores)
    rank_freq = rank_normalize(zipf_scores)
    inv_surp = np.where(surprisal > 0, 1.0 / surprisal, 0.0)
    rank_pred = rank_normalize(inv_surp)
    combined = alpha * rank_freq + (1.0 - alpha) * rank_pred
    keep_indices = set(np.argsort(combined)[:n_keep])
    return "".join(tok for i, tok in enumerate(token_strs) if i in keep_indices)


# ---------------------------------------------------------------------------
# NER analysis
# ---------------------------------------------------------------------------
def extract_entities(nlp, text):
    """Extract named entities from text using spaCy. Returns list of (text, label) tuples."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def check_entity_preserved(entity_text, compressed_text):
    """
    Check if an entity survives in the compressed text.
    Uses case-insensitive substring matching, allowing for minor whitespace changes.
    """
    # Normalize whitespace
    entity_norm = " ".join(entity_text.lower().split())
    compressed_norm = compressed_text.lower()

    # Direct substring match
    if entity_norm in compressed_norm:
        return True

    # Try without spaces (tokens may be concatenated after deletion)
    entity_nospace = entity_norm.replace(" ", "")
    compressed_nospace = compressed_norm.replace(" ", "")
    if entity_nospace in compressed_nospace:
        return True

    return False


def run_ner_analysis():
    print("Loading spaCy model...", flush=True)
    nlp = spacy.load("en_core_web_sm")

    print("Loading test chunks...", flush=True)
    chunks = load_test_chunks()
    print(f"Loaded {len(chunks)} test chunks", flush=True)

    tokenizer, model = load_gpt2()
    b_full_surp, thresholds, b_full_freq = load_hybrid_calibration()

    # First extract all entities from original texts
    print("\nExtracting entities from original texts...", flush=True)
    chunk_entities = []
    total_ent_count = 0
    for i, chunk in enumerate(chunks):
        ents = extract_entities(nlp, chunk)
        chunk_entities.append(ents)
        total_ent_count += len(ents)
    print(f"Total entities across {len(chunks)} chunks: {total_ent_count}", flush=True)

    # Show entity type distribution
    all_labels = [label for ents in chunk_entities for _, label in ents]
    label_counts = Counter(all_labels)
    print("Entity type distribution:", flush=True)
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}", flush=True)

    results = {"methods": {}}

    method_specs = [
        ("step", "Step", lambda chunk, rk: step_deletion(chunk, rk)),
        ("wordlen", "WordLen", lambda chunk, rk: wordlen_deletion(chunk, rk)),
        ("wordfreq", "WordFreq", lambda chunk, rk: wordfreq_deletion(chunk, rk, tokenizer, model)),
        ("opt", "LP", lambda chunk, rk: opt_deletion(chunk, rk, tokenizer, model, b_full_freq)),
        ("entropy", "Entropy", lambda chunk, rk: entropy_deletion(chunk, rk, tokenizer, model)),
        (
            "entropy_lp",
            "Entropy-LP",
            lambda chunk, rk: entropy_lp_deletion(chunk, rk, tokenizer, model, b_full_surp, thresholds),
        ),
        (
            "entropy_in_freqbuckets",
            "Ent.-FreqBkt",
            lambda chunk, rk: entropy_in_freqbuckets_deletion(chunk, rk, tokenizer, model, b_full_freq),
        ),
        (
            "hybrid_alpha_0.3",
            "Hybrid-0.3",
            lambda chunk, rk: hybrid_alpha_deletion(chunk, rk, tokenizer, model, 0.3),
        ),
        (
            "hybrid_alpha_0.5",
            "Hybrid-0.5",
            lambda chunk, rk: hybrid_alpha_deletion(chunk, rk, tokenizer, model, 0.5),
        ),
        (
            "hybrid_alpha_0.7",
            "Hybrid-0.7",
            lambda chunk, rk: hybrid_alpha_deletion(chunk, rk, tokenizer, model, 0.7),
        ),
    ]

    for method_key, method_label, compressor in method_specs:
        print("\n" + "=" * 60, flush=True)
        print(f"{method_label} - NER Preservation", flush=True)
        print("=" * 60, flush=True)

        method_results = {}
        for rkeep in RETENTION_RATES:
            print(f"\n  rkeep={rkeep}", flush=True)
            preserved_count = 0
            total_count = 0
            per_type_preserved = Counter()
            per_type_total = Counter()

            for i, chunk in enumerate(chunks):
                ents = chunk_entities[i]
                if not ents:
                    continue
                compressed = compressor(chunk, rkeep)
                for ent_text, ent_label in ents:
                    total_count += 1
                    per_type_total[ent_label] += 1
                    if check_entity_preserved(ent_text, compressed):
                        preserved_count += 1
                        per_type_preserved[ent_label] += 1

                if (i + 1) % 25 == 0:
                    print(f"    {i+1}/{len(chunks)} done", flush=True)

            preservation_rate = preserved_count / total_count if total_count > 0 else 0.0
            print(f"  Preservation: {preserved_count}/{total_count} = {preservation_rate:.4f}", flush=True)

            per_type_rates = {}
            for label in per_type_total:
                rate = per_type_preserved[label] / per_type_total[label] if per_type_total[label] > 0 else 0.0
                per_type_rates[label] = {
                    "preserved": per_type_preserved[label],
                    "total": per_type_total[label],
                    "rate": round(rate, 4),
                }

            method_results[str(rkeep)] = {
                "entity_preservation": round(preservation_rate, 4),
                "total_entities": total_count,
                "preserved": preserved_count,
                "per_type": per_type_rates,
            }

        results["methods"][method_key] = method_results

    # --- Summary table ---
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY: Entity Preservation Rate (skeleton, before reconstruction)", flush=True)
    print("=" * 60, flush=True)
    summary_order = [
        ("step", "Step"),
        ("wordlen", "WordLen"),
        ("wordfreq", "WordFreq"),
        ("opt", "LP"),
        ("entropy", "Entropy"),
        ("entropy_lp", "Entropy-LP"),
        ("entropy_in_freqbuckets", "Ent.-FreqBkt"),
        ("hybrid_alpha_0.3", "Hybrid-0.3"),
        ("hybrid_alpha_0.5", "Hybrid-0.5"),
        ("hybrid_alpha_0.7", "Hybrid-0.7"),
    ]
    print(f"{'rkeep':<8}" + "".join(f"{label:<14}" for _, label in summary_order), flush=True)
    print("-" * (8 + 14 * len(summary_order)), flush=True)
    for rkeep in RETENTION_RATES:
        k = str(rkeep)
        row = f"{rkeep:<8}"
        for method_key, _ in summary_order:
            row += f"{results['methods'][method_key][k]['entity_preservation']:<14.4f}"
        print(row, flush=True)

    # Also create the simplified "rkeep" format requested
    results["rkeep"] = {}
    for rkeep in RETENTION_RATES:
        k = str(rkeep)
        results["rkeep"][k] = {method_key: results["methods"][method_key][k] for method_key, _ in summary_order}

    # Save
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    run_ner_analysis()
