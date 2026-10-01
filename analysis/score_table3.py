"""Score every Table 3 column on the same 100 BBC chunks, one decoder (gemini-2.5-flash), one scorer
(distilroberta-base): WordFreq-3 and Summ. from samples/table3_wordfreq_summ_samples.jsonl, Entropy / Entropy-LP /
Hybrid-0.5 (and the other hybrids) from samples/bbc_hybrid_samples.jsonl, LLMLingua-2 from
samples/llmlingua2_samples*.jsonl. Also reports ROUGE-L, entity recall, unsupported-entity rate (matching rule:
entity_match.py) and the realised character ratio of the compressed text. Empty outputs (API refusals) are excluded
and counted. Writes results/table3_scores.json.
"""
import json, statistics as st
from collections import defaultdict
from pathlib import Path
import spacy
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from entity_match import check_preserved, norm_pair

HERE = Path(__file__).resolve().parent
recs = []
for fn in ("table3_wordfreq_summ_samples.jsonl", "bbc_hybrid_samples.jsonl",
           "llmlingua2_samples.jsonl", "llmlingua2_samples_extra_rates.jsonl"):
    for l in (HERE.parent / "samples" / fn).open():
        r = json.loads(l)
        recs.append(dict(method=r["method"], rkeep=str(float(r["rkeep"])), idx=r["sample_idx"],
                         ref=r["reference"], hyp=r.get("reconstruction") or "", skel=r.get("degraded")))
# keep the last record per (method, rkeep, idx)
uniq = {}
for r in recs:
    uniq[(r["method"], r["rkeep"], r["idx"])] = r
recs = list(uniq.values())
empty = defaultdict(int)
for r in recs:
    if not r["hyp"].strip():
        empty[(r["method"], r["rkeep"])] += 1
recs = [r for r in recs if r["hyp"].strip()]
print(len(recs), "scored records; empty:", dict(empty), flush=True)

sc = BERTScorer(model_type="distilroberta-base", lang="en", rescale_with_baseline=False)
_, _, F = sc.score([r["hyp"] for r in recs], [r["ref"] for r in recs], batch_size=32)
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
refs = list({r["ref"]: None for r in recs}); ents = {}
for d, k in zip(nlp.pipe(refs, batch_size=64), refs):
    ents[k] = [e.text for e in d.ents]
rents = [[e.text for e in d.ents] for d in nlp.pipe([r["hyp"] for r in recs], batch_size=64)]

agg = defaultdict(lambda: defaultdict(list)); cnt = defaultdict(lambda: dict(o=0, rec=0, re=0, un=0))
for r, f, re_ in zip(recs, F.tolist(), rents):
    k = (r["method"], r["rkeep"]); a = agg[k]; c = cnt[k]
    a["bert"].append(f)
    a["rougeL"].append(rouge.score(r["ref"], r["hyp"])["rougeL"].fmeasure)
    a["len"].append(len(r["hyp"]) / len(r["ref"]))
    if r["skel"] is not None:
        a["ratio"].append(len(r["skel"]) / len(r["ref"]))
    hn, hs = norm_pair(r["hyp"]); on, os_ = norm_pair(r["ref"])
    for e in ents[r["ref"]]:
        c["o"] += 1; c["rec"] += check_preserved(e, hn, hs)
    for e in re_:
        c["re"] += 1; c["un"] += not check_preserved(e, on, os_)
rows = []
for k in sorted(agg, key=lambda k: (k[0], -float(k[1]))):
    a, c = agg[k], cnt[k]
    rows.append(dict(method=k[0], rkeep=k[1], n=len(a["bert"]), n_empty=empty.get(k, 0),
                     bert_mean=round(st.mean(a["bert"]), 4), bert_std=round(st.pstdev(a["bert"]), 4),
                     rougeL=round(st.mean(a["rougeL"]), 4),
                     entity_recall=round(c["rec"] / c["o"], 4), unsupported_rate=round(c["un"] / c["re"], 4) if c["re"] else None,
                     realised_ratio=round(st.mean(a["ratio"]), 3) if a["ratio"] else None,
                     output_len=round(st.mean(a["len"]), 3)))
(HERE.parent / "results" / "table3_scores.json").write_text(json.dumps(rows, indent=1))
for r in rows:
    print(r)
