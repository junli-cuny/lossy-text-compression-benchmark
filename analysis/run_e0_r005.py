"""E0 at r_keep = 0.05 (Table 22, semantic rows).

Same pipeline, chunks, prompt and decoder (gemini-2.5-flash) as run_e0_bbc_hybrid.py, with the decoder calls spread
over threads. Writes samples/bbc_hybrid_samples_r005.jsonl and results/bbc_hybrid_results_r005.json and leaves the
r_keep >= 0.1 records untouched; the bucket calibration is seeded from harness/hybrid_results.json. Reads
GOOGLE_API_KEY from the environment or ~/.env. Scoring: score_r005_final.py.
"""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
for line in ((Path.home() / ".env").read_text().splitlines() if (Path.home() / ".env").exists() else []):
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
import run_e0_bbc_hybrid as e0          # registers the 'bbc' config, 512-char chunking, 2.5-flash
base = e0.base
base.genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
base.gemini = base.genai.GenerativeModel(base.GEMINI_MODEL_NAME)
base.DATASETS["bbc"]["results_path"] = Path(__file__).resolve().parents[1] / "results" / "bbc_hybrid_results_r005.json"
base.DATASETS["bbc"]["samples_path"] = Path(__file__).resolve().parents[1] / "samples" / "bbc_hybrid_samples_r005.jsonl"
if not base.DATASETS["bbc"]["results_path"].exists():  # the bucket calibration of the r_keep >= 0.1 run
    import json
    _src = json.loads((e0.HARNESS / "hybrid_results.json").read_text())
    base.DATASETS["bbc"]["results_path"].write_text(json.dumps({k: _src[k] for k in ("calib_surprisal", "calib_frequency")}, indent=2))
base.DATASETS["bbc"]["bert_model"] = "distilroberta-base"  # Table 22 is scored by score_r005_final.py
import threading, time
from concurrent.futures import ThreadPoolExecutor
WORKERS = 4

def parallel_run_method(cfg, method_key, display_name, examples, tokenizer, model, results_path,
                        samples_path, all_results, sample_records, rates, builder_fn):
    """base.run_method with the decoder calls fanned out over WORKERS threads; same records, same files."""
    results = all_results.get(method_key, {})
    for rkeep in rates:
        rate_key = str(rkeep)
        print(f"\n=== {display_name} rkeep={rkeep} ===", flush=True)
        todo = [(i, ex, builder_fn(ex, rkeep)) for i, ex in enumerate(examples)
                if sample_records.get((method_key, rate_key, i)) is None]
        lock = threading.Lock()
        def work(item):
            i, ex, degraded = item
            rec = base.call_gemini(base.reconstruct_prompt(cfg, degraded, len(ex["raw_text"])))
            record = base.sample_record_base(cfg, method_key, rkeep, ex, i, degraded, rec)
            with lock:
                base.append_sample_record(samples_path, record)
                sample_records[(method_key, rate_key, i)] = record
                if len(sample_records) % 20 == 0:
                    print(f"  {len(todo)} to do, records so far {len(sample_records)}", flush=True)
        with ThreadPoolExecutor(WORKERS) as pool:
            list(pool.map(work, todo))
        preds = [sample_records[(method_key, rate_key, i)]["reconstruction"] for i in range(len(examples))]
        refs = [ex["raw_text"] for ex in examples]
        mean_f1, std_f1 = base.evaluate_reconstructions(cfg, preds, refs)
        print(f"  BERTScore F1: {mean_f1:.4f} +/- {std_f1:.4f}", flush=True)
        results[rate_key] = {"mean_f1": round(mean_f1, 4), "std_f1": round(std_f1, 4), "n": len(examples)}
        all_results[method_key] = results
        base.save_results(results_path, all_results)
    return results

base.run_method = parallel_run_method

if __name__ == "__main__":
    sys.argv = ["run_e0_r005", "--dataset", "bbc", "--rates", "0.05"] + sys.argv[1:]
    base.main()
