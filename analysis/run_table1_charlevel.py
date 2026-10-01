"""Table 1 on one decoder and one sample: the four character-level deletion rules (Step, Gaussian, Bernoulli,
Poisson) on the same 100 BBC chunks as Table 3, reconstructed with gemini-2.5-flash.

Deletion rules: Step = step_deletion.delete_chars_uniform (the Step pipeline's rule, verbatim);
Gaussian / Bernoulli / Poisson = the functions of harness/run_stochastic_bbc_all.py, with the same per-chunk
seeds. Prompt: the character-level reconstruction prompt of that script, verbatim.
Per-sample records -> table1_charlevel_samples.jsonl (resumable). Scoring: score_table1.py.
"""
import json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for line in ((Path.home() / ".env").read_text().splitlines() if (Path.home() / ".env").exists() else []):
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
import google.generativeai as genai
import step_deletion as S

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
MODEL = "gemini-2.5-flash"
gemini = genai.GenerativeModel(MODEL)
OUT = HERE.parent / "samples" / "table1_charlevel_samples.jsonl"
RATES = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
SEED = 42
WORKERS = 20

chunks = {}
for l in open(HERE.parent / "samples" / "bbc_hybrid_samples.jsonl"):
    r = json.loads(l)
    chunks.setdefault(r["sample_idx"], r["reference"])
chunks = [chunks[i] for i in sorted(chunks)]
assert len(chunks) == 100


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


def step_delete(text, rkeep, rng):
    return S.delete_chars_uniform(text, 1.0 - rkeep)[0]


METHODS = {"Step": step_delete, "Gaussian": gaussian_delete, "Bernoulli": bernoulli_delete, "Poisson": poisson_delete}


def prompt_for(degraded, original_len):
    return ("You are a text reconstruction assistant. The input is an English news "
            "article from which some characters have been deleted. Your task is to "
            "reconstruct the original article as faithfully as possible. "
            "Do not add new facts, names, or events not suggested by the input. "
            f"The output should be approximately {original_len} characters long. "
            "Return only the reconstructed text.\n\n"
            f"Degraded text:\n{degraded}")


def call(prompt, tries=6):
    for a in range(tries):
        try:
            return gemini.generate_content(prompt, request_options={"timeout": 120}).text.strip()
        except Exception as e:
            print(f"  attempt {a+1}: {str(e)[:120]}", flush=True)
            time.sleep(min(30, 3 + 5 * a))
    return ""


done = set()
if OUT.exists():
    for l in OUT.open():
        r = json.loads(l)
        if r.get("reconstruction"):
            done.add((r["method"], r["rkeep"], r["sample_idx"]))
jobs = []
for m, fn in METHODS.items():
    for rk in RATES:
        for i, ch in enumerate(chunks):
            if (m, str(rk), i) in done:
                continue
            rng = np.random.default_rng(SEED + i + int(rk * 1000) * 1000)
            degraded = fn(ch, rk, rng)
            jobs.append((m, str(rk), i, ch, degraded))
print(f"{len(done)} done, {len(jobs)} to run", flush=True)
lock = threading.Lock(); n = [0]


def work(j):
    m, rk, i, ch, degraded = j
    out = call(prompt_for(degraded, len(ch)))
    rec = dict(dataset="bbc", language="en", decoder_model=MODEL, method=m, rkeep=rk, sample_idx=i,
               reference=ch, degraded=degraded, reconstruction=out)
    with lock:
        with OUT.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n[0] += 1
        if n[0] % 100 == 0:
            print(f"  {n[0]}/{len(jobs)}  {time.strftime('%H:%M:%S')}", flush=True)


with ThreadPoolExecutor(WORKERS) as pool:
    list(pool.map(work, jobs))
print("DONE", flush=True)
