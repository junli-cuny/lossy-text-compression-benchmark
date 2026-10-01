"""Item sheet of the human fact-level study.

Samples 54 reconstructions stratified over {wordfreq6 (WordFreq-6), opt_prepolish (LP)} x rkeep {0.5, 0.3, 0.1}
(9 per cell, seeded) from the BBC workbook's second-pass outputs. Rows are shuffled and strategy / rate are kept in a
separate key sheet so annotators stay blind to condition.

Output: e5_annotation_sheet_rebuilt.xlsx (sheets README, annotate, key); the sheet used in the study is
e5_annotation_sheet.xlsx.
"""
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from common_data import iter_bbc_records

HERE = Path(__file__).parent
STRATEGIES = ("wordfreq6", "opt_prepolish")
RATES = (0.5, 0.3, 0.1)
PER_CELL = 9  # 2 x 3 x 9 = 54 items
SEED = 20260708


def main():
    pool = {}
    for r in iter_bbc_records(outputs=("polished",)):
        if r["strategy"] in STRATEGIES and r["rkeep"] in RATES:
            pool.setdefault((r["strategy"], r["rkeep"]), []).append(r)

    rng = random.Random(SEED)
    items = []
    for key, recs in sorted(pool.items()):
        items.extend(rng.sample(recs, min(PER_CELL, len(recs))))
    rng.shuffle(items)

    rows, key_rows = [], []
    for i, r in enumerate(items, 1):
        rows.append({
            "item_id": i,
            "original_text": r["original"],
            "reconstruction": r["reconstruction"],
            "fact_1": "", "fact_1_judgment": "",
            "fact_2": "", "fact_2_judgment": "",
            "fact_3": "", "fact_3_judgment": "",
            "fact_4": "", "fact_4_judgment": "",
            "fact_5": "", "fact_5_judgment": "",
            "hallucinated_claim_present": "",
            "notes": "",
        })
        key_rows.append({"item_id": i, "strategy": r["strategy"],
                         "rkeep": r["rkeep"], "chunk_id": r["chunk_id"]})

    out = HERE / "e5_annotation_sheet_rebuilt.xlsx"  # the shipped sheet is not overwritten
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        readme = pd.DataFrame({"instructions": [
            "For each item: read original_text, identify up to 5 atomic factual statements",
            "(who/what/when/where/quantity). Write each in fact_N.",
            "Then judge each fact against the reconstruction:",
            "  P = preserved (fact recoverable, meaning intact)",
            "  D = distorted (fact present but wrong entity/number/relation)",
            "  M = missing (fact absent from reconstruction)",
            "hallucinated_claim_present: Y if the reconstruction asserts a specific",
            "factual claim with no basis in the original, else N.",
            "Do NOT look at the 'key' sheet until annotation is complete.",
        ]})
        readme.to_excel(w, sheet_name="README", index=False)
        pd.DataFrame(rows).to_excel(w, sheet_name="annotate", index=False)
        pd.DataFrame(key_rows).to_excel(w, sheet_name="key", index=False)
    print(f"Wrote {out} with {len(rows)} items "
          f"({len(STRATEGIES)} strategies x {len(RATES)} rates x {PER_CELL})")


if __name__ == "__main__":
    main()
