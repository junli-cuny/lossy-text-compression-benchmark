"""E3: LLMLingua-2 as an encoder on the BBC benchmark (Appendix E; Tables 3, 18 and 19).

Same 100 BBC 512-character chunks as E0 (run_e0_bbc_hybrid.py), retention rates 0.7 / 0.5 / 0.3
(run_e3_llmlingua_extra_rates.py adds 0.9 and 0.1):
  A. the LLMLingua-2 output is the skeleton, reconstructed with gemini-2.5-flash and the semantic methods' prompt;
  B. the LLMLingua-2 output is scored directly against the original, without reconstruction.
Stage 1 (--skeletons-only): local compression, no API calls -> samples/llmlingua2_skeletons.jsonl.
Stage 2 (default): Gemini reconstruction with resume -> samples/llmlingua2_samples.jsonl (same schema as
samples/*_hybrid_samples.jsonl, method 'llmlingua2'); aggregates -> results/e3_llmlingua_results.json.
All-rate scoring: score_e3_llmlingua_all_rates.py and ner_e3_llmlingua_extra_rates.py (Table 19).
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent / "harness"
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(HERE))

RATES = [0.7, 0.5, 0.3]
SAMPLES_PATH = Path(__file__).resolve().parents[1] / "samples" / "llmlingua2_samples.jsonl"
RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "e3_llmlingua_results.json"
SKELETONS_PATH = Path(__file__).resolve().parents[1] / "samples" / "llmlingua2_skeletons.jsonl"


def load_bbc_chunks(n=100, size=512):
    import openpyxl
    wb = openpyxl.load_workbook(
        str(HERE.parent / "data" / "BBC_news_latest.xlsx"), read_only=True)
    ws = wb["datasource"]
    chunks = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        article = str(row[2]) if row[2] else ""
        if not article or len(article) < 100:
            continue
        for i in range(0, len(article), size):
            c = article[i:i + size]
            if len(c) >= 100:
                chunks.append(c)
            if len(chunks) >= n:
                break
        if len(chunks) >= n:
            break
    wb.close()
    return chunks


def build_skeletons():
    from llmlingua import PromptCompressor
    print("Loading LLMLingua-2 compressor...", flush=True)
    comp = PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2=True,
        device_map="cpu",
    )
    chunks = load_bbc_chunks()
    done = set()
    if SKELETONS_PATH.exists():
        with SKELETONS_PATH.open() as f:
            for line in f:
                r = json.loads(line)
                done.add((r["rkeep"], r["sample_idx"]))
    t0 = time.time()
    with SKELETONS_PATH.open("a") as f:
        for rate in RATES:
            for i, chunk in enumerate(chunks):
                if (str(rate), i) in done:
                    continue
                res = comp.compress_prompt(chunk, rate=rate, force_tokens=["\n"])
                skel = res["compressed_prompt"]
                f.write(json.dumps({
                    "method": "llmlingua2", "rkeep": str(rate), "sample_idx": i,
                    "reference": chunk, "degraded": skel,
                    "actual_ratio": round(len(skel) / len(chunk), 4),
                }) + "\n")
                f.flush()
            print(f"rate {rate} done ({time.time()-t0:.0f}s)", flush=True)


def reconstruct(sleep_s=1.0):
    import run_appendix_semantic_hybrid as base
    base.GEMINI_MODEL_NAME = "gemini-2.5-flash"
    base.gemini = base.genai.GenerativeModel(base.GEMINI_MODEL_NAME)
    cfg = {"language": "en", "domain_label": "BBC news article excerpt"}

    done = {}
    if SAMPLES_PATH.exists():
        with SAMPLES_PATH.open() as f:
            for line in f:
                r = json.loads(line)
                done[(r["rkeep"], r["sample_idx"])] = r
    skels = [json.loads(l) for l in SKELETONS_PATH.open()]
    print(f"{len(skels)} skeletons, {len(done)} already reconstructed", flush=True)
    with SAMPLES_PATH.open("a") as f:
        for n, s in enumerate(skels):
            if (s["rkeep"], s["sample_idx"]) in done:
                continue
            prompt = base.reconstruct_prompt(cfg, s["degraded"], len(s["reference"]))
            rec = base.call_gemini(prompt)
            out = dict(s)
            out.update({"dataset": "bbc", "language": "en",
                        "decoder_model": base.GEMINI_MODEL_NAME,
                        "reconstruction": rec})
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            if (n + 1) % 20 == 0:
                print(f"  {n+1}/{len(skels)}", flush=True)
            time.sleep(sleep_s)


def evaluate():
    """Aggregate BERTScore for mode A (reconstruction) and mode B (skeleton as-is)."""
    from bert_score import BERTScorer
    scorer = BERTScorer(model_type="distilroberta-base", lang="en", rescale_with_baseline=False)
    samples = [json.loads(l) for l in SAMPLES_PATH.open()]
    results = {}
    for rate in RATES:
        sub = [s for s in samples if s["rkeep"] == str(rate)]
        if not sub:
            continue
        refs = [s["reference"] for s in sub]
        entry = {"n": len(sub),
                 "actual_ratio_mean": round(sum(s["actual_ratio"] for s in sub) / len(sub), 4)}
        for mode, key in (("A_reconstructed", "reconstruction"), ("B_skeleton_direct", "degraded")):
            hyps = [s[key] for s in sub]
            _, _, F = scorer.score(hyps, refs, batch_size=32)
            entry[f"bert_f1_{mode}"] = round(float(F.mean()), 4)
        results[str(rate)] = entry
        print(rate, entry, flush=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeletons-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()
    if args.eval_only:
        evaluate()
    else:
        build_skeletons()
        if not args.skeletons_only:
            reconstruct(args.sleep)
            evaluate()
    print("DONE")
