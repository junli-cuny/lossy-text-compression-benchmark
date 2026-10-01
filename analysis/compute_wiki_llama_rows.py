"""Wikipedia Llama base / Llama SFT rows (BERTScore F1, distilroberta-base) from the per-row generations in
data/SFT_qlora_Wiki_Text.xlsx: sheet `base_sft_generation_result` (base model) and sheet
`sft_qlora_generation_stats_resu` (QLoRA-tuned), columns <prefix>_prediction_<r> vs <prefix>_ground_truth_<r>.
Expected (Table 25): base 0.9708 / 0.9344 / 0.8288 / 0.8627 / 0.8313 and SFT 0.9862 / 0.9797 / 0.9514 / 0.9093 / 0.8756
at r = 0.9 / 0.7 / 0.5 / 0.3 / 0.1 (the workbook-stored values; a fresh scoring run agrees within 0.0001)."""
from pathlib import Path
import pandas as pd
from bert_score import BERTScorer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
XLSX = ROOT / "data" / "SFT_qlora_Wiki_Text.xlsx"
RATES = ["90", "70", "50", "30", "10"]
SHEETS = {"Llama base": ("base_sft_generation_result", "llama_base"),
          "Llama SFT": ("sft_qlora_generation_stats_resu", "llama")}

scorer = BERTScorer(model_type="distilroberta-base", lang="en")
for label, (sheet, prefix) in SHEETS.items():
    df = pd.read_excel(XLSX, sheet_name=sheet, dtype=str).fillna("")
    cells = []
    for r in RATES:
        pairs = [(p.strip(), g.strip()) for p, g in zip(df[f"{prefix}_prediction_{r}"], df[f"{prefix}_ground_truth_{r}"])
                 if p.strip() and g.strip()]
        _, _, f1 = scorer.score([p for p, _ in pairs], [g for _, g in pairs], batch_size=32)
        cells.append(f"{f1.mean().item():.4f} (n={len(pairs)})")
    print(f"{label:10s} " + "  ".join(f"r={int(r) / 100}: {c}" for r, c in zip(RATES, cells)))
