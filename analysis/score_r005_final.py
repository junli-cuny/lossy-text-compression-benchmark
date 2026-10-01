"""Table 22: score the r_keep = 0.05 stress point and the r_keep = 0.1 rows it is compared with.

Sources:
  samples/bbc_hybrid_samples_r005.jsonl         six semantic methods at r = 0.05 (E0 pipeline, gemini-2.5-flash)
  samples/bbc_hybrid_samples.jsonl              the same methods at r = 0.1 (E0)
  samples/bbc_wordfreq3_samples_r005_r01.jsonl  WordFreq-3 with the word-level pipeline at r = 0.1 and 0.05
Metrics per (strategy, rkeep): BERTScore F1 (distilroberta-base), ROUGE-L F, entity recall, unsupported-entity rate,
skeleton entity survival, recovery rate, skeleton/original and reconstruction/original length ratios.
Entity matching: entity_match.py. Writes results/r005_scores.json.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import spacy
from rouge_score import rouge_scorer
from bert_score import BERTScorer

HERE = Path(__file__).resolve().parent

from entity_match import check_preserved, norm_pair

def load():
    recs = []
    for f, rates in ((Path(__file__).resolve().parents[1] / "samples" / "bbc_hybrid_samples_r005.jsonl", {"0.05"}), (Path(__file__).resolve().parents[1] / "samples" / "bbc_hybrid_samples.jsonl", {"0.1"})):
        if f.exists():
            for l in f.open():
                d = json.loads(l)
                if d["rkeep"] in rates:
                    recs.append(dict(strategy=d["method"], rkeep=d["rkeep"], original=d["reference"],
                                     skeleton=d["degraded"], reconstruction=d["reconstruction"], src="semantic"))
    f = Path(__file__).resolve().parents[1] / "samples" / "bbc_wordfreq3_samples_r005_r01.jsonl"
    if f.exists():
        for l in f.open():
            d = json.loads(l)
            recs.append(dict(strategy="wordfreq3", rkeep=d["rkeep"], original=d["original"],
                             skeleton=d["skeleton"], reconstruction=d["reconstruction"], src="wordlevel"))
    return [r for r in recs if r["reconstruction"].strip()]

def main():
    recs = load()
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
    ents = {}
    originals = list({r["original"]: None for r in recs})
    for doc, k in zip(nlp.pipe(originals, batch_size=64), originals):
        ents[k] = [e.text for e in doc.ents]
    recon_ents = [[e.text for e in doc.ents] for doc in nlp.pipe([r["reconstruction"] for r in recs], batch_size=64)]
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scorers = {"bert_dr": BERTScorer(model_type="distilroberta-base", lang="en")}
    F = {}
    for name, sc in scorers.items():
        _, _, f = sc.score([r["reconstruction"] for r in recs], [r["original"] for r in recs], batch_size=32)
        F[name] = f.numpy()
    agg = defaultdict(lambda: defaultdict(list))
    cnt = defaultdict(lambda: dict(orig=0, recall=0, skel=0, deleted=0, recov=0, rents=0, unsup=0))
    for i, (r, rents) in enumerate(zip(recs, recon_ents)):
        k = (r["strategy"], r["rkeep"]); a = agg[k]; c = cnt[k]
        a["bert_dr"].append(float(F["bert_dr"][i]))
        a["rougeL"].append(rouge.score(r["original"], r["reconstruction"])["rougeL"].fmeasure)
        a["skel_len"].append(len(r["skeleton"]) / len(r["original"]))
        a["recon_len"].append(len(r["reconstruction"]) / len(r["original"]))
        rn, rs = norm_pair(r["reconstruction"]); on, os_ = norm_pair(r["original"]); sn, ss = norm_pair(r["skeleton"])
        for e in ents[r["original"]]:
            c["orig"] += 1; in_rec = check_preserved(e, rn, rs); c["recall"] += in_rec
            if check_preserved(e, sn, ss): c["skel"] += 1
            else:
                c["deleted"] += 1; c["recov"] += in_rec
        for e in rents:
            c["rents"] += 1; c["unsup"] += not check_preserved(e, on, os_)
    rows = []
    for k in sorted(agg):
        a, c = agg[k], cnt[k]
        row = dict(strategy=k[0], rkeep=k[1], n=len(a["bert_dr"]),
                   **{m: round(sum(v) / len(v), 4) for m, v in a.items()},
                   entity_recall=round(c["recall"] / c["orig"], 4), unsupported_rate=round(c["unsup"] / c["rents"], 4) if c["rents"] else None,
                   skeleton_survival=round(c["skel"] / c["orig"], 4), recovery_rate=round(c["recov"] / c["deleted"], 4) if c["deleted"] else None,
                   n_orig_entities=c["orig"], n_recon_entities=c["rents"])
        rows.append(row)
    (Path(__file__).resolve().parents[1] / "results" / "r005_scores.json").write_text(json.dumps(rows, indent=1))
    hdr = ["strategy", "rkeep", "n", "bert_dr", "rougeL", "entity_recall", "unsupported_rate", "skeleton_survival", "recovery_rate", "skel_len", "recon_len"]
    print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
    for r in rows: print("| " + " | ".join(str(r[h]) for h in hdr) + " |")

if __name__ == "__main__":
    main()
