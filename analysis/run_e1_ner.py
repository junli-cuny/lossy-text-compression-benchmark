"""E1: entity metrics computed on the reconstructions (Table 2, entity-recall block; Table 5, word-level rows).

Extends the skeleton-side analysis of harness/run_ner_analysis.py to the reconstructed outputs. Per (source,
strategy, rkeep):
  entity_recall            original entities present in the reconstruction / original entities
  unsupported_entity_rate  reconstruction entities absent from the original / reconstruction entities
  recovery_rate            original entities absent from the skeleton but present in the reconstruction /
                           original entities absent from the skeleton
  skeleton_survival        original entities present in the skeleton / original entities
Entities: spaCy en_core_web_sm; matching rule: entity_match.py. Sources: the BBC workbook (second-pass outputs),
samples/wiki_hybrid_samples.jsonl, samples/reddit_hybrid_samples.jsonl, and the base / QLoRA Llama outputs.
Output: results/e1_ner_results.json.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import spacy

sys.path.insert(0, str(Path(__file__).parent))
from common_data import iter_bbc_records, iter_jsonl_records, iter_sft_records
from entity_match import check_preserved, norm_pair

OUT = Path(__file__).parent


def main():
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
    # 'ner' needs tok2vec; keep defaults otherwise.

    print("Collecting records...", flush=True)
    recs = []
    for r in iter_bbc_records(outputs=("polished",)):
        r["source"] = "bbc"; recs.append(r)
    for r in iter_jsonl_records(datasets=("wiki", "reddit")):
        r["source"] = r.pop("dataset"); recs.append(r)
    for r in iter_sft_records():
        r["source"] = f"bbc_sft_{r.pop('decoder')}"; recs.append(r)
    print(f"{len(recs)} records", flush=True)

    # NER once per unique original and per reconstruction
    uniq_orig = {}
    for r in recs:
        uniq_orig.setdefault(r["original"], None)
    print(f"{len(uniq_orig)} unique originals; extracting entities...", flush=True)
    t0 = time.time()
    keys = list(uniq_orig)
    for doc, key in zip(nlp.pipe(keys, batch_size=64), keys):
        uniq_orig[key] = [(e.text, e.label_) for e in doc.ents]
    print(f"  originals done ({time.time()-t0:.0f}s)", flush=True)

    print("Extracting entities from reconstructions...", flush=True)
    t0 = time.time()
    recon_texts = [r["reconstruction"] for r in recs]
    recon_ents = [None] * len(recs)
    for i, doc in enumerate(nlp.pipe(recon_texts, batch_size=64)):
        recon_ents[i] = [(e.text, e.label_) for e in doc.ents]
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(recs)} ({time.time()-t0:.0f}s)", flush=True)

    # aggregate
    agg = defaultdict(lambda: dict(
        n=0, orig_ents=0, recalled=0, skel_survived=0,
        deleted_from_skel=0, recovered=0, recon_ents=0, unsupported=0,
        n_with_skel=0))

    for r, r_ents in zip(recs, recon_ents):
        o_ents = uniq_orig[r["original"]]
        key = (r["source"], r["strategy"], r["rkeep"])
        a = agg[key]
        a["n"] += 1
        rec_norm, rec_nosp = norm_pair(r["reconstruction"])
        orig_norm, orig_nosp = norm_pair(r["original"])
        skel = r.get("skeleton")
        if skel:
            skel_norm, skel_nosp = norm_pair(skel)
            a["n_with_skel"] += 1

        for etext, _ in o_ents:
            a["orig_ents"] += 1
            in_recon = check_preserved(etext, rec_norm, rec_nosp)
            if in_recon:
                a["recalled"] += 1
            if skel:
                in_skel = check_preserved(etext, skel_norm, skel_nosp)
                if in_skel:
                    a["skel_survived"] += 1
                else:
                    a["deleted_from_skel"] += 1
                    if in_recon:
                        a["recovered"] += 1

        for etext, _ in r_ents:
            a["recon_ents"] += 1
            if not check_preserved(etext, orig_norm, orig_nosp):
                a["unsupported"] += 1

    results = []
    for (source, strategy, rkeep), a in sorted(agg.items()):
        results.append(dict(
            source=source, strategy=strategy, rkeep=rkeep, n_chunks=a["n"],
            entity_recall=round(a["recalled"] / a["orig_ents"], 4) if a["orig_ents"] else None,
            unsupported_entity_rate=round(a["unsupported"] / a["recon_ents"], 4) if a["recon_ents"] else None,
            skeleton_survival=round(a["skel_survived"] / (a["skel_survived"] + a["deleted_from_skel"]), 4)
                if (a["skel_survived"] + a["deleted_from_skel"]) else None,
            recovery_rate=round(a["recovered"] / a["deleted_from_skel"], 4) if a["deleted_from_skel"] else None,
            orig_entities=a["orig_ents"], recon_entities=a["recon_ents"],
            deleted_entities=a["deleted_from_skel"],
        ))

    with open(Path(__file__).resolve().parents[1] / "results" / "e1_ner_results.json", "w") as f:
        json.dump(results, f, indent=1)

    print("\n=== E1 summary (source/strategy/rkeep: recall | unsupported | recovery) ===")
    for r in results:
        print(f"{r['source']}/{r['strategy']}@{r['rkeep']}: "
              f"recall={r['entity_recall']} unsupported={r['unsupported_entity_rate']} "
              f"skel_surv={r['skeleton_survival']} recov={r['recovery_rate']}")
    print("DONE")


if __name__ == "__main__":
    main()
