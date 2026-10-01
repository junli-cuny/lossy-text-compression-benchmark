"""Score Table 1 (samples/table1_charlevel_samples.jsonl): BERTScore F1 (distilroberta-base) mean and std, ROUGE-L,
CER (Levenshtein distance between the whitespace-stripped texts over the stripped original length; the workbook's
CER columns use the unstripped text), realised keep ratio and output length, per (method, rkeep). Empty outputs are
excluded and counted. Writes results/table1_scores.json.
"""
import json, statistics as st
from collections import defaultdict
from pathlib import Path
import Levenshtein
from bert_score import BERTScorer
from rouge_score import rouge_scorer

HERE = Path(__file__).resolve().parent
uniq = {}
for l in (HERE.parent / "samples" / "table1_charlevel_samples.jsonl").open():
    r = json.loads(l)
    uniq[(r["method"], r["rkeep"], r["sample_idx"])] = r
recs = list(uniq.values())
empty = defaultdict(int)
for r in recs:
    if not r["reconstruction"].strip():
        empty[(r["method"], r["rkeep"])] += 1
recs = [r for r in recs if r["reconstruction"].strip()]
sc = BERTScorer(model_type="distilroberta-base", lang="en", rescale_with_baseline=False)
_, _, F = sc.score([r["reconstruction"] for r in recs], [r["reference"] for r in recs], batch_size=32)
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
strip = lambda t: "".join(t.split())
agg = defaultdict(lambda: defaultdict(list))
for r, f in zip(recs, F.tolist()):
    a = agg[(r["method"], r["rkeep"])]
    o, h = r["reference"], r["reconstruction"]
    a["bert"].append(f)
    a["rougeL"].append(rouge.score(o, h)["rougeL"].fmeasure)
    a["cer"].append(Levenshtein.distance(strip(o), strip(h)) / max(1, len(strip(o))))
    a["keep"].append(len(r["degraded"]) / len(o))
    a["len"].append(len(h) / len(o))
rows = []
for k in sorted(agg, key=lambda k: (k[0], -float(k[1]))):
    a = agg[k]
    rows.append(dict(method=k[0], rkeep=k[1], n=len(a["bert"]), n_empty=empty.get(k, 0),
                     bert_mean=round(st.mean(a["bert"]), 4), bert_std=round(st.pstdev(a["bert"]), 4),
                     rougeL=round(st.mean(a["rougeL"]), 4), cer=round(st.mean(a["cer"]), 4),
                     keep=round(st.mean(a["keep"]), 3), output_len=round(st.mean(a["len"]), 3)))
(HERE.parent / "results" / "table1_scores.json").write_text(json.dumps(rows, indent=1))
for r in rows:
    print(r)
