"""BERTScore F1 of the semantic and hybrid rows of the cross-domain tables, recomputed from the per-sample records.

  wiki     samples/wiki_hybrid_samples.jsonl      Table 25 (Wikipedia)       distilroberta-base
  reddit   samples/reddit_hybrid_samples.jsonl    Table 26 (Reddit)          distilroberta-base
  news_zh  samples/news_zh_hybrid_samples.jsonl   Table 29 (Chinese news)    bert-base-chinese
  wiki_zh  samples/wiki_zh_hybrid_samples.jsonl   Table 30 (Chinese Wiki)    bert-base-chinese

Scoring follows harness/run_appendix_semantic_hybrid.py (bert-score defaults for the model, no baseline rescaling,
mean over all records of a cell). The Zhihu semantic rows (Table 31) have no per-sample records; their aggregates
are results/zhihu_hybrid_results.json (harness/run_zhihu_hybrid_experiments.py).

Usage: python analysis/score_semantic_samples.py [wiki reddit news_zh wiki_zh]
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
from bert_score import BERTScorer

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"wiki": ("wiki_hybrid_samples.jsonl", "distilroberta-base", "en"),
            "reddit": ("reddit_hybrid_samples.jsonl", "distilroberta-base", "en"),
            "news_zh": ("news_zh_hybrid_samples.jsonl", "bert-base-chinese", "zh"),
            "wiki_zh": ("wiki_zh_hybrid_samples.jsonl", "bert-base-chinese", "zh")}
METHODS = ["entropy", "entropy_lp", "entropy_in_freqbuckets", "hybrid_alpha_0.3", "hybrid_alpha_0.5", "hybrid_alpha_0.7"]
LABEL = {"entropy": "Entropy", "entropy_lp": "Entropy-LP", "entropy_in_freqbuckets": "Entropy-in-FreqBuckets",
         "hybrid_alpha_0.3": "Hybrid-0.3", "hybrid_alpha_0.5": "Hybrid-0.5", "hybrid_alpha_0.7": "Hybrid-0.7"}
RATES = ["0.9", "0.7", "0.5", "0.3", "0.1"]


def main(names):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    for name in names:
        fn, model, lang = DATASETS[name]
        recs = [json.loads(l) for l in (ROOT / "samples" / fn).open()]
        scorer = BERTScorer(model_type=model, lang=lang, rescale_with_baseline=False, device=device)
        _, _, f1 = scorer.score([r["reconstruction"] or "" for r in recs], [r["reference"] for r in recs], batch_size=32)
        cells, empty = defaultdict(list), defaultdict(int)
        for r, f in zip(recs, f1.tolist()):
            cells[(r["method"], r["rkeep"])].append(f)
            empty[(r["method"], r["rkeep"])] += not (r["reconstruction"] or "").strip()
        print(f"\n{name} ({fn}, {model}): mean BERTScore F1 per cell (n; empty outputs)")
        print(f"{'method':<24}" + "".join(f"{'r=' + r:>18}" for r in RATES))
        for m in METHODS:
            row = []
            for r in RATES:
                v = cells.get((m, r))
                row.append(f"{statistics.mean(v):.4f} ({len(v)};{empty[(m, r)]})" if v else "-")
            print(f"{LABEL[m]:<24}" + "".join(f"{c:>18}" for c in row))


if __name__ == "__main__":
    main(sys.argv[1:] or list(DATASETS))
