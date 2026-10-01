"""Table 22, WordFreq-3 row: WordFreq-3 at r_keep = 0.05, and 0.1 as the control, with the word-level pipeline.

Deletion: delete_zipf_strict (harness/wf3_deletion.py, the notebook's rule). Decoding: the notebook's two passes,
the restoration prompt (notebooks/bbc_word_level_pipeline_v1.ipynb, code cell 441 counting code cells only,
'category-based deletion' wording) and then the second-pass prompt (notebooks/bbc_word_level_pipeline_v2.ipynb,
code cell 168), both read from the notebooks at run time; temperature 0.1. The notebook used max_output_tokens=1024
with gemini-2.0-flash; gemini-2.5-flash spends that budget on thinking and truncates its output, so the cap is 8192
here. Decoder: gemini-2.5-flash (2.0 Flash is retired). Output: samples/bbc_wordfreq3_samples_r005_r01.jsonl
(resumable, threaded). Scoring: score_r005_final.py.
"""
import argparse, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'harness'))
import wf3_deletion as W

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
XLSX = ROOT / "data" / "BBC_news_latest.xlsx"; OUT = Path(__file__).resolve().parents[1] / "samples" / "bbc_wordfreq3_samples_r005_r01.jsonl"
DECODER = "gemini-2.5-flash"; GEN = {"temperature": 0.1, "max_output_tokens": 8192, "top_p": 0.9, "top_k": 40}

def notebook_prompt(path, cell, name):
    nb = json.load(open(path)); cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    src = cells[cell]; s = src.index(f'def {name}('); e = src.find('\ndef ', s + 10); ns = {}
    exec(src[s:(e if e > 0 else len(src))], ns); return ns[name]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.05]); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--n", type=int)
    args = ap.parse_args()
    restore_prompt = notebook_prompt(ROOT / "notebooks/bbc_word_level_pipeline_v1.ipynb", 441, "build_prompt")
    polish_prompt = notebook_prompt(ROOT / "notebooks/bbc_word_level_pipeline_v2.ipynb", 168, "build_prompt")
    assert "category-based deletion" in restore_prompt("x") and "Polished Output" in polish_prompt("x")
    df = pd.read_excel(XLSX, sheet_name="wf_3classes").dropna(subset=["Articles", "Category"]).reset_index(drop=True)
    if args.n: df = df.head(args.n)
    for line in ((Path.home() / ".env").read_text().splitlines() if (Path.home() / ".env").exists() else []):
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"]); model = genai.GenerativeModel(DECODER)
    def call(prompt):
        for attempt in range(5):
            try:
                r = model.generate_content(prompt, generation_config=GEN, request_options={"timeout": 120})
                return r.text.strip() if r and r.text else ""
            except Exception as e:
                print(f"  attempt {attempt+1} failed: {str(e)[:80]}", flush=True); time.sleep(min(10, 2 + 2 * attempt))
        return ""
    done = set()
    if OUT.exists():
        for l in OUT.open(): d = json.loads(l); done.add((d["rkeep"], d["sample_idx"]))
    lock = threading.Lock()
    for r in args.rates:
        random.seed(42)                         # the deletion rule draws from the global random module
        items = []
        for idx, row in df.iterrows():
            skel, keep, stats = W.delete_zipf_strict(row["Articles"], keep_ratio=r, tol=0.02)
            if (str(r), idx) not in done: items.append((idx, row, skel, keep, stats))
        print(f"r={r}: {len(items)} articles to decode", flush=True)
        def work(it):
            idx, row, skel, keep, stats = it
            restored = call(restore_prompt(skel)); polished = call(polish_prompt(restored)) if restored else ""
            with lock, OUT.open("a") as f:
                f.write(json.dumps(dict(strategy="wordfreq3", rkeep=str(r), sample_idx=int(idx), category=row["Category"], decoder_model=DECODER,
                                        word_keep_ratio=round(keep, 4), zipf_stats=stats, original=row["Articles"], skeleton=skel,
                                        restored=restored, reconstruction=polished)) + "\n")
        with ThreadPoolExecutor(args.workers) as pool: list(pool.map(work, items))
        print(f"r={r} complete", flush=True)

if __name__ == "__main__":
    main()
