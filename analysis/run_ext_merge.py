"""E1 and E4 for the BBC semantic, hybrid and LLMLingua-2 samples (Table 5 semantic rows, Table 10).

Applies the entity metrics of run_e1_ner.py and the lexical metrics of run_e4_metrics.py to the E0 samples
(samples/bbc_hybrid_samples.jsonl: six semantic and hybrid methods, five rates) and the LLMLingua-2 samples
(samples/llmlingua2_samples.jsonl: r_keep 0.7 / 0.5 / 0.3), all on the same 100 BBC chunks.
Outputs:
  results/e1_ner_results_ext.json  entity recall, unsupported-entity rate, skeleton survival, recovery rate
  results/e4_metrics_ext.json      means of BERTScore F1 (distilroberta-base), chrF++, METEOR and ROUGE-L
  printout                         combined comparison table
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import spacy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from entity_match import check_preserved, norm_pair


def load_records():
    recs = []
    for line in (Path(__file__).resolve().parents[1] / "samples" / "bbc_hybrid_samples.jsonl").open():
        r = json.loads(line)
        if not r["reconstruction"].strip():
            continue
        recs.append(dict(source="bbc_hybrid", strategy=r["method"], rkeep=float(r["rkeep"]),
                         chunk_id=r["sample_idx"], original=r["reference"],
                         skeleton=r["degraded"], reconstruction=r["reconstruction"]))
    for line in (Path(__file__).resolve().parents[1] / "samples" / "llmlingua2_samples.jsonl").open():
        r = json.loads(line)
        if not r["reconstruction"].strip():
            continue
        recs.append(dict(source="bbc_llmlingua", strategy="llmlingua2", rkeep=float(r["rkeep"]),
                         chunk_id=r["sample_idx"], original=r["reference"],
                         skeleton=r["degraded"], reconstruction=r["reconstruction"]))
    return recs


def ner_metrics(recs):
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
    uniq = {}
    for r in recs:
        uniq.setdefault(r["original"], None)
    keys = list(uniq)
    for doc, k in zip(nlp.pipe(keys, batch_size=64), keys):
        uniq[k] = [(e.text, e.label_) for e in doc.ents]
    recon_ents = []
    texts = [r["reconstruction"] for r in recs]
    for doc in nlp.pipe(texts, batch_size=64):
        recon_ents.append([(e.text, e.label_) for e in doc.ents])

    agg = defaultdict(lambda: dict(n=0, orig=0, recall=0, skel=0, deleted=0, recov=0,
                                   rents=0, unsupported=0))
    for r, rents in zip(recs, recon_ents):
        a = agg[(r["source"], r["strategy"], r["rkeep"])]
        a["n"] += 1
        rec_n, rec_s = norm_pair(r["reconstruction"])
        orig_n, orig_s = norm_pair(r["original"])
        skel_n, skel_s = norm_pair(r["skeleton"])
        for e, _ in uniq[r["original"]]:
            a["orig"] += 1
            in_rec = check_preserved(e, rec_n, rec_s)
            if in_rec:
                a["recall"] += 1
            if check_preserved(e, skel_n, skel_s):
                a["skel"] += 1
            else:
                a["deleted"] += 1
                if in_rec:
                    a["recov"] += 1
        for e, _ in rents:
            a["rents"] += 1
            if not check_preserved(e, orig_n, orig_s):
                a["unsupported"] += 1

    rows = []
    for (src, strat, rk), a in sorted(agg.items()):
        rows.append(dict(
            source=src, strategy=strat, rkeep=rk, n_chunks=a["n"],
            entity_recall=round(a["recall"] / a["orig"], 4) if a["orig"] else None,
            unsupported_entity_rate=round(a["unsupported"] / a["rents"], 4) if a["rents"] else None,
            skeleton_survival=round(a["skel"] / a["orig"], 4) if a["orig"] else None,
            recovery_rate=round(a["recov"] / a["deleted"], 4) if a["deleted"] else None,
        ))
    return rows


def lexical_metrics(recs):
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    from sacrebleu.metrics import CHRF
    from bert_score import BERTScorer
    chrf = CHRF(word_order=2)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scorer = BERTScorer(model_type="distilroberta-base", lang="en", rescale_with_baseline=False)

    agg = defaultdict(lambda: defaultdict(list))
    for i in range(0, len(recs), 512):
        batch = recs[i:i + 512]
        refs = [r["original"] for r in batch]
        hyps = [r["reconstruction"] for r in batch]
        _, _, F = scorer.score(hyps, refs, batch_size=64)
        for r, f in zip(batch, F.numpy()):
            k = (r["source"], r["strategy"], r["rkeep"])
            agg[k]["bert_f1"].append(float(f))
            agg[k]["chrfpp"].append(chrf.sentence_score(r["reconstruction"], [r["original"]]).score / 100)
            agg[k]["meteor"].append(meteor_score([r["original"].split()], r["reconstruction"].split()))
            agg[k]["rougeL"].append(rouge.score(r["original"], r["reconstruction"])["rougeL"].fmeasure)
        print(f"  lexical+bert {min(i+512, len(recs))}/{len(recs)}", flush=True)

    rows = []
    for (src, strat, rk), m in sorted(agg.items()):
        row = dict(source=src, strategy=strat, rkeep=rk, n=len(m["bert_f1"]))
        for name, vals in m.items():
            row[name + "_mean"] = round(sum(vals) / len(vals), 4)
        rows.append(row)
    return rows


def main():
    recs = load_records()
    print(f"{len(recs)} extension records", flush=True)
    ner_rows = ner_metrics(recs)
    (Path(__file__).resolve().parents[1] / "results" / "e1_ner_results_ext.json").write_text(json.dumps(ner_rows, indent=1))
    lex_rows = lexical_metrics(recs)
    (Path(__file__).resolve().parents[1] / "results" / "e4_metrics_ext.json").write_text(json.dumps(lex_rows, indent=1))

    print("\n=== BBC same-decoder comparison (entity metrics) ===")
    print(f"{'strategy':<24}{'rkeep':>6}{'bert':>8}{'chrf++':>8}{'ent_recall':>11}{'unsupp':>8}{'recovery':>9}")
    lex_idx = {(r["strategy"], r["rkeep"]): r for r in lex_rows}
    for r in ner_rows:
        lx = lex_idx.get((r["strategy"], r["rkeep"]), {})
        print(f"{r['strategy']:<24}{r['rkeep']:>6}{lx.get('bert_f1_mean',''):>8}"
              f"{lx.get('chrfpp_mean',''):>8}{r['entity_recall']:>11}"
              f"{r['unsupported_entity_rate']:>8}{str(r['recovery_rate']):>9}")
    print("DONE")


if __name__ == "__main__":
    main()
