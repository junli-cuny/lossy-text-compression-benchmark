"""Decoder version check, word-level track (Table 16).

Feeds the workbook's own WordFreq-3 skeletons (sheet wf_3classes, columns del_zipf_{90,70,50,30,10}; the inputs
behind the published Gemini 2.0 Flash outputs in polished_*) to gemini-2.5-flash with the semantic methods'
reconstruction prompt, so the two decoders are compared on identical inputs.
Output: samples/bbc_wordfreq3_decoder_check_samples.jsonl (one record per article x rate; resumable, threaded).
Scoring: score_decoder_check.py.
"""
import argparse, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
XLSX = ROOT / "data" / "BBC_news_latest.xlsx"
OUT = Path(__file__).resolve().parents[1] / "samples" / "bbc_wordfreq3_decoder_check_samples.jsonl"
DECODER = "gemini-2.5-flash"
RATES = [90, 70, 50, 30, 10]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--n", type=int)
    args = ap.parse_args()
    df = pd.read_excel(XLSX, sheet_name="wf_3classes").dropna(subset=["Articles", "Category"]).reset_index(drop=True)
    if args.n: df = df.head(args.n)
    for line in ((Path.home() / ".env").read_text().splitlines() if (Path.home() / ".env").exists() else []):
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    sys.path.insert(0, str(ROOT / "harness"))
    import run_appendix_semantic_hybrid as base
    base.genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = base.genai.GenerativeModel(DECODER)
    cfg = {"language": "en", "domain_label": "BBC news article"}
    done = set()
    if OUT.exists():
        for l in OUT.open():
            d = json.loads(l); done.add((d["rate"], d["sample_idx"]))
    lock = threading.Lock()
    for r in RATES:
        items = [(i, row) for i, row in df.iterrows() if (r, i) not in done and isinstance(row[f"del_zipf_{r}"], str)]
        print(f"r=0.{r//10 if r>=10 else r}: {len(items)} articles to decode", flush=True)
        def work(item):
            i, row = item
            skel = row[f"del_zipf_{r}"]
            prompt = base.reconstruct_prompt(cfg, skel, len(row["Articles"]))
            rec = ""
            for attempt in range(5):
                try:
                    rec = model.generate_content(prompt, request_options={"timeout": 120}).text.strip(); break
                except Exception as e:
                    print(f"  attempt {attempt+1} failed: {str(e)[:80]}", flush=True); time.sleep(min(10, 2 + 2*attempt))
            with lock, OUT.open("a") as f:
                f.write(json.dumps(dict(strategy="wordfreq3", rate=r, rkeep=r/100, sample_idx=int(i), decoder_model=DECODER,
                                        original=row["Articles"], skeleton=skel, reconstruction=rec,
                                        published_polished=row[f"polished_{r}"], published_restored=row[f"restored_{r}"])) + "\n")
        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(work, items))
        print(f"r={r} complete", flush=True)

if __name__ == "__main__":
    main()
