"""Table 3 on one decoder and one scorer: WordFreq-3 and the LLM summarization baseline on the same 100
BBC chunks as the semantic/hybrid methods (bbc_hybrid_samples.jsonl), decoded with gemini-2.5-flash.

WordFreq-3: skeleton from wf3_deletion.delete_zipf_strict (the word-level pipeline's deletion rule),
            reconstructed with the SAME prompt the semantic/hybrid methods use (harness reconstruct_prompt).
Summ.:      the summarization baseline's compress-to-length prompt (harness/run_baselines.py).
Per-sample records -> table3_wordfreq_summ_samples.jsonl (resumable). Scoring: score_table3.py.
"""
import json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for line in ((Path.home() / ".env").read_text().splitlines() if (Path.home() / ".env").exists() else []):
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
import google.generativeai as genai
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'harness'))
import wf3_deletion as W

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
MODEL = "gemini-2.5-flash"
gemini = genai.GenerativeModel(MODEL)
OUT = HERE.parent / "samples" / "table3_wordfreq_summ_samples.jsonl"
RATES = [0.9, 0.7, 0.5, 0.3, 0.1]
WORKERS = 16

chunks = {}
for l in open(HERE.parent / "samples" / "bbc_hybrid_samples.jsonl"):
    r = json.loads(l)
    chunks.setdefault(r["sample_idx"], r["reference"])
chunks = [chunks[i] for i in sorted(chunks)]
assert len(chunks) == 100, len(chunks)


def reconstruct_prompt(degraded, original_len):  # identical to the harness prompt for English
    return ("You are a text reconstruction assistant. The input is a BBC news article excerpt "
            "from which some tokens have been deleted. Reconstruct the original text as faithfully as possible. "
            "Do not add new facts, names, or events not suggested by the input. "
            f"The output should be approximately the same length (~{original_len} characters). "
            "Return only the reconstructed text.\n\n"
            f"Degraded text:\n{degraded}")


def summ_prompt(chunk, rkeep):  # identical to harness/run_baselines.py
    target_len = int(len(chunk) * rkeep)
    return (f"Compress the following news text into exactly {target_len} characters "
            f"(±15%). Preserve all key facts, names, numbers, and events. "
            f"Do not add any new information. Return only the compressed text.\n\n"
            f"Text:\n{chunk}")


def call(prompt, tries=5):
    for a in range(tries):
        try:
            r = gemini.generate_content(prompt, request_options={"timeout": 120})
            return r.text.strip()
        except Exception as e:
            print(f"  attempt {a+1}: {str(e)[:120]}", flush=True)
            time.sleep(min(20, 3 + 4 * a))
    return ""


done = set()
if OUT.exists():
    for l in OUT.open():
        r = json.loads(l)
        if r.get("reconstruction"):
            done.add((r["method"], r["rkeep"], r["sample_idx"]))
jobs = []
for rk in RATES:
    for i, ch in enumerate(chunks):
        if ("wordfreq3", str(rk), i) not in done:
            skel = W.delete_zipf_strict(ch, keep_ratio=rk)[0]
            jobs.append(("wordfreq3", str(rk), i, ch, skel, reconstruct_prompt(skel, len(ch))))
        if ("summarization", str(rk), i) not in done:
            jobs.append(("summarization", str(rk), i, ch, None, summ_prompt(ch, rk)))
print(f"{len(done)} done, {len(jobs)} to run", flush=True)
lock = threading.Lock(); n = [0]


def work(j):
    method, rk, i, ch, skel, prompt = j
    out = call(prompt)
    rec = dict(dataset="bbc", language="en", decoder_model=MODEL, method=method, rkeep=rk, sample_idx=i,
               reference=ch, degraded=skel, reconstruction=out)
    with lock:
        with OUT.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n[0] += 1
        if n[0] % 50 == 0:
            print(f"  {n[0]}/{len(jobs)}", flush=True)


with ThreadPoolExecutor(WORKERS) as pool:
    list(pool.map(work, jobs))
print("DONE", flush=True)
