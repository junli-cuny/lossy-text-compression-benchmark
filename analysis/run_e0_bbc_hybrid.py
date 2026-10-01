"""E0: the BBC semantic and hybrid grid with per-sample records.

Reuses harness/run_appendix_semantic_hybrid.py (per-sample JSONL records with resume) with a 'bbc' dataset config
that follows harness/run_hybrid_experiments.py: datasource sheet of data/BBC_news_latest.xlsx, the first 100 chunks
of 512 characters, and the bucket calibration of harness/hybrid_results.json (not recomputed). Decoder:
gemini-2.5-flash. These samples are the semantic rows of Tables 3, 5, 10, 18 and 22 and the semantic block of
Table 15 (scored by score_table3.py, run_ext_merge.py, run_downstream_full.py, score_r005_final.py).
Outputs: samples/bbc_hybrid_samples.jsonl (per chunk), results/bbc_hybrid_results_v2.json (aggregates).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent / "harness"
sys.path.insert(0, str(HARNESS))

import run_appendix_semantic_hybrid as base  # noqa: E402

# gemini-2.0-flash, the decoder of the original runs, has been retired; its successor is recorded in every
# sample record (decoder_model). Compare strategies within this run, not against gemini-2.0-flash aggregates.
base.GEMINI_MODEL_NAME = "gemini-2.5-flash"
base.gemini = base.genai.GenerativeModel(base.GEMINI_MODEL_NAME)

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "bbc_hybrid_results_v2.json"
SAMPLES_PATH = Path(__file__).resolve().parents[1] / "samples" / "bbc_hybrid_samples.jsonl"

# Seed calibration from the paper's run so bucket models are identical.
if not RESULTS_PATH.exists():
    src = json.loads((HARNESS / "hybrid_results.json").read_text())
    seed = {k: src[k] for k in ("calib_surprisal", "calib_frequency") if k in src}
    RESULTS_PATH.write_text(json.dumps(seed, indent=2))
    print(f"Seeded calibration into {RESULTS_PATH}")

base.DATASETS["bbc"] = {
    "data_path": base.DATA_ROOT / "BBC_news_latest.xlsx",
    "sheet": "datasource",
    "text_field": "Articles",
    "language": "en",
    "domain_label": "BBC news article excerpt",
    "bert_model": "distilroberta-base",
    "bert_lang": "en",
    "n_test": 100,
    "n_calib": 50,
    "results_path": RESULTS_PATH,
    "samples_path": SAMPLES_PATH,
    "chunk_size": 512,
}

_orig_load = base.load_examples


def load_examples(cfg, n):
    """512-char chunking, replicating run_hybrid_experiments.load_chunks."""
    if "chunk_size" not in cfg:
        return _orig_load(cfg, n)
    import openpyxl
    wb = openpyxl.load_workbook(str(cfg["data_path"]), read_only=True)
    ws = wb[cfg["sheet"]]
    size = cfg["chunk_size"]
    examples = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        article = str(row[2]) if row[2] else ""
        if not article or len(article) < 100:
            continue
        for i in range(0, len(article), size):
            chunk = article[i:i + size]
            if len(chunk) >= 100:
                examples.append({"id": len(examples), "raw_text": chunk, "compact_text": chunk})
            if len(examples) >= n:
                break
        if len(examples) >= n:
            break
    wb.close()
    return examples


base.load_examples = load_examples

if __name__ == "__main__":
    sys.argv = ["run_e0_bbc_hybrid", "--dataset", "bbc", "--use-cuda"] + sys.argv[1:]
    base.main()
