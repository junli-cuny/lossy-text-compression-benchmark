"""E2a: encoder design and decoder adaptation on the same axis (Table 20).

Reads results/e4_per_chunk.csv.gz and reports
  1. the decoder lever: paired per-chunk BERTScore gain of the QLoRA-tuned Llama-3.2-3B-Instruct over the base
     model on identical skeletons (sources bbc_sft_qlora vs bbc_sft_base; at each rate the skeletons of the
     strategy whose fine-tuned decoder performs best: LP at r_keep 0.9 and 0.5, WordLen at 0.7, WordFreq-3 at 0.3
     and 0.1), with a Wilcoxon signed-rank test;
  2. the encoder lever: the best-worst spread across the word-level deletion strategies at the fixed Gemini decoder
     (source bbc, second-pass outputs; strategies wordlen, wordfreq3, wordfreq6, opt_prepolish and opt; Step
     excluded);
  3. the two levers side by side.
Output: results/e2a_decoder_lever.json.
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).parent


def main():
    with gzip.open(Path(__file__).resolve().parents[1] / "results" / "e4_per_chunk.csv.gz", "rt") as f:
        df = pd.read_csv(f)

    out = {}

    # --- 1. decoder lever: paired qlora - base ------------------------------
    sft = df[df.source.isin(["bbc_sft_qlora", "bbc_sft_base"])]
    piv = sft.pivot_table(index=["rkeep", "chunk_id"], columns="source",
                          values=["bert_f1", "chrfpp", "meteor"])
    decoder_rows = []
    for rk, g in piv.groupby(level="rkeep"):
        row = {"rkeep": rk, "n_pairs": len(g)}
        for m in ["bert_f1", "chrfpp", "meteor"]:
            q = g[(m, "bbc_sft_qlora")].dropna()
            b = g[(m, "bbc_sft_base")].dropna()
            common = q.index.intersection(b.index)
            diff = (q.loc[common] - b.loc[common])
            row[f"{m}_qlora_mean"] = round(float(q.mean()), 4)
            row[f"{m}_base_mean"] = round(float(b.mean()), 4)
            row[f"{m}_gain"] = round(float(diff.mean()), 4)
            if len(diff) > 10 and not np.allclose(diff, 0):
                row[f"{m}_wilcoxon_p"] = float(f"{wilcoxon(diff).pvalue:.2e}")
        decoder_rows.append(row)
    out["decoder_lever_qlora_vs_base"] = decoder_rows

    # --- 2. encoder lever: strategy spread at fixed gemini decoder ----------
    bbc = df[(df.source == "bbc") & (df.output_kind == "polished")]
    encoder_rows = []
    for rk, g in bbc.groupby("rkeep"):
        means = g.groupby("strategy")["bert_f1"].mean().sort_values()
        encoder_rows.append({
            "rkeep": rk,
            "worst_strategy": means.index[0], "worst_bert_f1": round(float(means.iloc[0]), 4),
            "best_strategy": means.index[-1], "best_bert_f1": round(float(means.iloc[-1]), 4),
            "encoder_spread": round(float(means.iloc[-1] - means.iloc[0]), 4),
        })
    out["encoder_lever_strategy_spread"] = encoder_rows

    # word-level variant (the reported one): character-level Step excluded at all rates
    wl = bbc[~bbc.strategy.str.startswith("step")]
    wl_rows = []
    for rk, g in wl.groupby("rkeep"):
        means = g.groupby("strategy")["bert_f1"].mean().sort_values()
        wl_rows.append({
            "rkeep": rk,
            "worst_strategy": means.index[0], "worst_bert_f1": round(float(means.iloc[0]), 4),
            "best_strategy": means.index[-1], "best_bert_f1": round(float(means.iloc[-1]), 4),
            "encoder_spread": round(float(means.iloc[-1] - means.iloc[0]), 4),
        })
    out["encoder_lever_strategy_spread_wordlevel"] = wl_rows
    encoder_rows = wl_rows

    # --- 3. summary comparison ----------------------------------------------
    comp = []
    dec = {r["rkeep"]: r.get("bert_f1_gain") for r in decoder_rows}
    enc = {r["rkeep"]: r["encoder_spread"] for r in encoder_rows}
    for rk in sorted(set(dec) & set(enc)):
        comp.append({"rkeep": rk, "decoder_gain_bert": dec[rk], "encoder_spread_bert": enc[rk]})
    out["lever_comparison"] = comp

    with open(Path(__file__).resolve().parents[1] / "results" / "e2a_decoder_lever.json", "w") as f:
        json.dump(out, f, indent=1)

    print(json.dumps(out["lever_comparison"], indent=1))
    print("\nDecoder lever detail:")
    for r in decoder_rows:
        print(f"  rkeep={r['rkeep']}: base={r['bert_f1_base_mean']} -> qlora={r['bert_f1_qlora_mean']} "
              f"(gain {r['bert_f1_gain']}, p={r.get('bert_f1_wilcoxon_p')})")
    print("Encoder lever detail:")
    for r in encoder_rows:
        print(f"  rkeep={r['rkeep']}: {r['worst_strategy']}={r['worst_bert_f1']} .. "
              f"{r['best_strategy']}={r['best_bert_f1']} (spread {r['encoder_spread']})")
    print("DONE")


if __name__ == "__main__":
    main()
