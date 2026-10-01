"""Reddit cells of Tables 26-28 that are not stored as aggregates in the workbooks.

  Table 26, WordFreq-3 (char.) row: BERTScore F1 (distilroberta-base) of the second-pass reconstructions in sheet
      wf_3classes_dp (columns polished_char_<r>) of data/Reddit_Dataset.xlsx against the original text (first
      column of the sheet), on rows 100-199 of the 200 test rows, the rows the published cells were computed on.
  Tables 27 and 28, word-level rows: ROUGE-L F1 (rouge-score, no stemming) and character error rate (Levenshtein
      distance / length of the original) of the second-pass reconstructions in the workbook sheets, on rows
      100-199 of the 200 test rows.
  Tables 27 and 28, semantic and hybrid rows: the same CER and ROUGE-L with stemming, on the 100 rows of
      samples/reddit_hybrid_samples.jsonl (test rows 0-99).

The other cells of Table 26 are stored in the workbook (sheets *_stats_result, row F1_chunk_mean) and, for the
semantic rows, recomputed by analysis/score_semantic_samples.py. Every value printed here is also given for all
200 test rows (word-level) for comparison.
"""
import json
import statistics
from pathlib import Path

import Levenshtein
import openpyxl
import torch
from bert_score import BERTScorer
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "data" / "Reddit_Dataset.xlsx"
SAMPLES = ROOT / "samples" / "reddit_hybrid_samples.jsonl"
RATES = (90, 70, 50, 30, 10)
SHEETS = {  # paper row -> (sheet, reconstruction column pattern)
    "Step": ("benchmark", "polished_char_{r}"),
    "WordLen": ("adaptive_length_deletion", "polished_char_{r}"),
    "WordFreq-3": ("wf_3classes", "polished_{r}"),
    "WordFreq-3 (char.)": ("wf_3classes_dp", "polished_char_{r}"),
    "WordFreq-6": ("wf_6classes", "polished_char_{r}"),
    "LP": ("wf_optimal", "polished_optimal_{r}"),
}
SEMANTIC = {"Entropy": "entropy", "Entropy-LP": "entropy_lp", "Entropy-in-FreqBuckets": "entropy_in_freqbuckets",
            "Hybrid-0.3": "hybrid_alpha_0.3", "Hybrid-0.5": "hybrid_alpha_0.5", "Hybrid-0.7": "hybrid_alpha_0.7"}


def cer(ref, hyp):
    return Levenshtein.distance(ref, hyp) / len(ref)


def main():
    rouge_plain = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_stem = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    wb = openpyxl.load_workbook(XLSX, read_only=True)
    pairs = {}
    for name, (sheet, pat) in SHEETS.items():
        rows = list(wb[sheet].iter_rows(values_only=True))
        hdr = [str(h) for h in rows[0]]
        data = [r for r in rows[1:] if r and r[0]]
        assert len(data) == 200, (sheet, len(data))
        for r in RATES:
            ci = hdr.index(pat.format(r=r))
            pairs[(name, r)] = [(str(x[0]).strip(), str(x[ci]).strip() if x[ci] else "") for x in data]

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    scorer = BERTScorer(model_type="distilroberta-base", lang="en", rescale_with_baseline=False, device=device)
    print("Table 26, WordFreq-3 (char.), BERTScore F1: rows 100-199 (published) | all 200 rows")
    for r in RATES:
        p = pairs[("WordFreq-3 (char.)", r)]
        _, _, f1 = scorer.score([h for _, h in p], [o for o, _ in p], batch_size=32)
        f1 = f1.tolist()
        print(f"  r={r / 100}: {statistics.mean(f1[100:]):.4f} | {statistics.mean(f1):.4f}")

    print("\nTables 27 / 28, word-level rows: ROUGE-L (no stemming) / CER on rows 100-199 (published) | all 200 rows")
    for name in SHEETS:
        cells = []
        for r in RATES:
            p = pairs[(name, r)]
            rl = [rouge_plain.score(o, h)["rougeL"].fmeasure for o, h in p]
            ce = [cer(o, h) for o, h in p]
            cells.append(f"{statistics.mean(rl[100:]):.4f}/{statistics.mean(ce[100:]):.4f} | "
                         f"{statistics.mean(rl):.4f}/{statistics.mean(ce):.4f}")
        print(f"  {name:<20}" + "   ".join(f"r={r / 100}: {c}" for r, c in zip(RATES, cells)))

    agg = {}
    for line in SAMPLES.open():
        x = json.loads(line)
        o, h = x["reference"].strip(), (x["reconstruction"] or "").strip()
        agg.setdefault((x["method"], x["rkeep"]), []).append((rouge_stem.score(o, h)["rougeL"].fmeasure, cer(o, h)))
    print("\nTables 27 / 28, semantic and hybrid rows: ROUGE-L (stemming) / CER on samples/reddit_hybrid_samples.jsonl")
    for name, key in SEMANTIC.items():
        cells = []
        for r in RATES:
            v = agg[(key, str(r / 100))]
            cells.append(f"{statistics.mean(a for a, _ in v):.4f}/{statistics.mean(b for _, b in v):.4f}")
        print(f"  {name:<24}" + "   ".join(f"r={r / 100}: {c}" for r, c in zip(RATES, cells)))


if __name__ == "__main__":
    main()
