#!/usr/bin/env python3
"""Table 19 (Recon., Skel., realised ratio): LLMLingua-2 at all five retention rates, BERTScore F1 with
distilroberta-base.

Mode A = reconstruction vs original (the framework's objective).
Mode B = the LLMLingua-2 output scored directly against the original, without reconstruction (how prompt
         compression consumes it).
Reads samples/llmlingua2_samples.jsonl (r = 0.7/0.5/0.3) and samples/llmlingua2_samples_extra_rates.jsonl
(r = 0.9/0.1) and prints the published Table 19 values beside the recomputed ones as a regression check.
Writes results/e3_llmlingua_results_all_rates.json.
"""
import json, statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(__file__).resolve().parents[1] / "results" / "e3_llmlingua_results_all_rates.json"
PUBLISHED = {"0.9": (0.9857, 0.9864), "0.7": (0.9709, 0.9503), "0.5": (0.9462, 0.9134),
             "0.3": (0.9121, 0.8862), "0.1": (0.8805, 0.8614)}  # Table 19 (Recon., Skel.)

rows = []
for f in (Path(__file__).resolve().parents[1] / "samples" / "llmlingua2_samples.jsonl", Path(__file__).resolve().parents[1] / "samples" / "llmlingua2_samples_extra_rates.jsonl"):
    rows += [json.loads(l) for l in f.open()]
by_rate = defaultdict(list)
for r in rows:
    by_rate[str(float(r["rkeep"]))].append(r)
print({k: len(v) for k, v in sorted(by_rate.items())})

from bert_score import BERTScorer
scorer = BERTScorer(model_type="distilroberta-base", lang="en", rescale_with_baseline=False)

results = {}
for rate in sorted(by_rate, key=float, reverse=True):
    v = sorted(by_rate[rate], key=lambda x: x["sample_idx"])
    refs = [str(x["reference"]) for x in v]
    recs = [str(x.get("reconstruction") or "") for x in v]
    skels = [str(x["degraded"]) for x in v]
    fa = scorer.score(recs, refs)[2]
    fb = scorer.score(skels, refs)[2]
    results[rate] = {
        "n": len(v),
        "actual_ratio_mean": round(st.mean(len(s) / len(r) for s, r in zip(skels, refs)), 4),
        "bert_f1_A_reconstructed": round(float(fa.mean()), 4),
        "bert_f1_A_std": round(float(fa.std()), 4),
        "bert_f1_B_skeleton_direct": round(float(fb.mean()), 4),
        "bert_f1_B_std": round(float(fb.std()), 4),
        "decoder": v[0].get("decoder_model", "gemini-2.5-flash"),
    }
    line = f"r={rate}: A {results[rate]['bert_f1_A_reconstructed']:.4f}  B {results[rate]['bert_f1_B_skeleton_direct']:.4f}  ratio {results[rate]['actual_ratio_mean']:.3f}"
    if rate in PUBLISHED:
        pa, pb = PUBLISHED[rate]
        da = results[rate]["bert_f1_A_reconstructed"] - pa
        db = results[rate]["bert_f1_B_skeleton_direct"] - pb
        line += f"   [published {pa}/{pb}, delta {da:+.4f}/{db:+.4f}]"
    print(line, flush=True)

json.dump(results, OUT.open("w"), indent=1)
print(f"wrote {OUT}")
