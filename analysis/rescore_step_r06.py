"""Re-score the Step (uniform) r_keep = 0.6 reconstructions stored in the BBC workbook.

The r = 0.6 Step reconstructions (`polished_char_60`, sheet `benchmark`) were regenerated after the workbook's
per-row score column (`polished_char_bert_chunk`) had been filled, so that column held the score of the earlier
outputs (mean 0.9457). The shipped workbook stores the re-scored value: BERTScore F1 with distilroberta-base (the
scorer of every word-level table) of `polished_char_60` against `Articles`, mean over the 200 rows -> 0.8829,
matching the `benchmark_stats_result` sheet. The script also prints the CER of the same texts (Levenshtein distance
over the original length, the workbook's convention): 0.6313 at r = 0.6, the value of Table 8; the CER cell of
`benchmark_stats_result` for r = 0.6 still holds the earlier outputs' value.

Usage: python analysis/rescore_step_r06.py [path/to/BBC_news_latest.xlsx]
"""
import sys
from pathlib import Path

import Levenshtein
import openpyxl
from bert_score import BERTScorer

xlsx = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "BBC_news_latest.xlsx")
ws = openpyxl.load_workbook(xlsx, read_only=True)["benchmark"]
hdr = [str(c) for c in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
rows = list(ws.iter_rows(min_row=2, values_only=True))[:200]
refs = [str(r[hdr.index("Articles")]) for r in rows]
scorer = BERTScorer(model_type="distilroberta-base", lang="en", rescale_with_baseline=False)
for col in ("polished_char_70", "polished_char_60"):
    hyp = [str(r[hdr.index(col)]) for r in rows]
    _, _, f1 = scorer.score(hyp, refs)
    cer = sum(Levenshtein.distance(o, h) / len(o) for o, h in zip(refs, hyp)) / len(hyp)
    print(f"{col}: distilroberta-base BERTScore F1 mean = {f1.mean().item():.4f}, CER = {cer:.4f}  (n={len(hyp)})")
print("expected: polished_char_70 = 0.9583 / CER 0.1952 (both as stored), polished_char_60 = 0.8829 / CER 0.6313 "
      "(the CER is the Step cell of Table 8 at r = 0.6; the stored CER cell of benchmark_stats_result for r = 0.6, 0.9537, "
      "predates the regeneration)")
