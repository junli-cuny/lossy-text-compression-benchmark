"""Score the LP objective variants of run_lp_objective_compare.py (Appendix P). Per (objective, rate): BERTScore F1
(distilroberta-base), ROUGE-L, entity recall, unsupported-entity rate, skeleton length ratio; then paired per-article
differences of sum_nop (the linear program of Appendix P) and prod (the objective of Sec. 3) against sum_p, with
bootstrap 95% CIs and win rates. Writes results/lp_objective_compare_scores.json.
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np, spacy
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from entity_match import check_preserved, norm_pair

HERE = Path(__file__).resolve().parent
recs = [json.loads(l) for l in (Path(__file__).resolve().parents[1] / "samples" / "bbc_lp_objective_compare_samples.jsonl").open()]
recs = [r for r in recs if r["reconstruction"].strip()]
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
origs = list({r["original"]: None for r in recs}); ents = {}
for doc, k in zip(nlp.pipe(origs, batch_size=64), origs): ents[k] = [e.text for e in doc.ents]
rents = [[e.text for e in d.ents] for d in nlp.pipe([r["reconstruction"] for r in recs], batch_size=64)]
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
F = {}
for name, mt in (("bert_dr", "distilroberta-base"),):
    _, _, f = BERTScorer(model_type=mt, lang="en").score([r["reconstruction"] for r in recs], [r["original"] for r in recs], batch_size=32); F[name] = f.numpy()
per = {}   # (obj, rate, idx) -> metrics
for i, (r, re_) in enumerate(zip(recs, rents)):
    hn, hs = norm_pair(r["reconstruction"]); on, os_ = norm_pair(r["original"]); E = ents[r["original"]]
    rec = sum(check_preserved(e, hn, hs) for e in E) / max(1, len(E)); un = sum(not check_preserved(e, on, os_) for e in re_) / max(1, len(re_))
    per[(r["obj"], r["rate"], r["sample_idx"])] = dict(bert_dr=float(F["bert_dr"][i]),
        rougeL=rouge.score(r["original"], r["reconstruction"])["rougeL"].fmeasure, entity_recall=rec, unsupported=un,
        skel_len=len(r["skeleton"]) / len(r["original"]), skeleton=r["skeleton"])
objs = ["sum_p", "sum_nop", "prod"]; rates = ["0.9", "0.7", "0.5", "0.3", "0.1"]; metrics = ["bert_dr", "rougeL", "entity_recall", "unsupported", "skel_len"]
rows = []
print("| rate | objective | n | BERT | ROUGE-L | ent.recall | unsupported | skel/orig |"); print("|---|---|---|---|---|---|---|---|")
for rate in rates:
    for obj in objs:
        vals = [v for (o, rt, _), v in per.items() if o == obj and rt == rate]
        if not vals: continue
        row = dict(rate=rate, objective=obj + (" (= prod_p)" if obj == "prod" else ""), n=len(vals), **{m: round(float(np.mean([v[m] for v in vals])), 4) for m in metrics}); rows.append(row)
        print(f"| {rate} | {row['objective']} | {row['n']} | {row['bert_dr']:.4f} | {row['rougeL']:.3f} | {row['entity_recall']:.3f} | {row['unsupported']:.3f} | {row['skel_len']:.3f} |")
print("\nPaired against 'sum_p' (same article, same rate): mean diff [bootstrap 95% CI], win rate, share of identical skeletons")
rng = np.random.default_rng(0); paired = []
for rate in rates + ["all"]:
    for obj in ("sum_nop", "prod"):
        keys = [(rt, i) for (o, rt, i) in per if o == "sum_p" and (rate == "all" or rt == rate) and (obj, rt, i) in per]
        if not keys: continue
        same = np.mean([per[(obj, rt, i)]["skeleton"] == per[("sum_p", rt, i)]["skeleton"] for rt, i in keys])
        line = f"  r={rate:>4} {obj:8s} n={len(keys):4d} identical-skeleton {same:.2f} |"
        for m in ("bert_dr", "rougeL", "entity_recall", "unsupported"):
            d = np.array([per[(obj, rt, i)][m] - per[("sum_p", rt, i)][m] for rt, i in keys])
            bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]; lo, hi = np.percentile(bs, [2.5, 97.5])
            line += f" {m} {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] win {np.mean(d > 0):.2f} |"
            paired.append(dict(rate=rate, objective=obj, metric=m, mean_diff=float(d.mean()), ci=[float(lo), float(hi)], win_rate=float(np.mean(d > 0)), n=len(d)))
        print(line)
(Path(__file__).resolve().parents[1] / "results" / "lp_objective_compare_scores.json").write_text(json.dumps(dict(per_cell=rows, paired=paired), indent=1))
