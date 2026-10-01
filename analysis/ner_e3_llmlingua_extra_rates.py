#!/usr/bin/env python3
"""Table 19 (Ent., Uns.): entity recall and unsupported-entity rate of LLMLingua-2 at all five rates.

Metric definitions as in run_e1_ner.py (matching rule: entity_match.py), so the values are comparable with
results/e1_ner_results_ext.json; the r = 0.7/0.5/0.3 cells double as a regression check against that file.
Writes results/e1_ner_llmlingua_all_rates.json.
"""
import json
from collections import defaultdict
from pathlib import Path
import spacy

HERE = Path(__file__).resolve().parent
PUBLISHED = {0.7: 0.9638, 0.5: 0.8564, 0.3: 0.6429}   # entity_recall from e1_ner_results_ext.json

from entity_match import check_preserved, norm_pair


rows = []
for f in (Path(__file__).resolve().parents[1] / "samples" / "llmlingua2_samples.jsonl", Path(__file__).resolve().parents[1] / "samples" / "llmlingua2_samples_extra_rates.jsonl"):
    for line in f.open():
        r = json.loads(line)
        rows.append({"strategy": "llmlingua2", "rkeep": float(r["rkeep"]),
                     "original": str(r["reference"]), "skeleton": str(r["degraded"]),
                     "reconstruction": str(r.get("reconstruction") or "")})
print(f"{len(rows)} records")

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
uniq = {r["original"]: None for r in rows}
keys = list(uniq)
for doc, k in zip(nlp.pipe(keys, batch_size=64), keys):
    uniq[k] = [(e.text, e.label_) for e in doc.ents]
recon_ents = [[(e.text, e.label_) for e in d.ents]
              for d in nlp.pipe([r["reconstruction"] for r in rows], batch_size=64)]

agg = defaultdict(lambda: dict(n=0, orig_ents=0, recalled=0, skel_survived=0,
                               deleted_from_skel=0, recovered=0, recon_ents=0, unsupported=0))
for r, r_ents in zip(rows, recon_ents):
    a = agg[r["rkeep"]]
    a["n"] += 1
    rec_norm, rec_nosp = norm_pair(r["reconstruction"])
    orig_norm, orig_nosp = norm_pair(r["original"])
    skel_norm, skel_nosp = norm_pair(r["skeleton"])
    for etext, _ in uniq[r["original"]]:
        a["orig_ents"] += 1
        in_recon = check_preserved(etext, rec_norm, rec_nosp)
        a["recalled"] += in_recon
        if check_preserved(etext, skel_norm, skel_nosp):
            a["skel_survived"] += 1
        else:
            a["deleted_from_skel"] += 1
            a["recovered"] += in_recon
    for etext, _ in r_ents:
        a["recon_ents"] += 1
        if not check_preserved(etext, orig_norm, orig_nosp):
            a["unsupported"] += 1

out = {}
for rk in sorted(agg, reverse=True):
    a = agg[rk]
    out[str(rk)] = dict(
        n_chunks=a["n"],
        entity_recall=round(a["recalled"] / a["orig_ents"], 4),
        unsupported_entity_rate=round(a["unsupported"] / a["recon_ents"], 4) if a["recon_ents"] else None,
        skeleton_survival=round(a["skel_survived"] / (a["skel_survived"] + a["deleted_from_skel"]), 4),
        recovery_rate=round(a["recovered"] / a["deleted_from_skel"], 4) if a["deleted_from_skel"] else None)
    chk = ""
    if rk in PUBLISHED:
        chk = f"   [published recall {PUBLISHED[rk]}, delta {out[str(rk)]['entity_recall'] - PUBLISHED[rk]:+.4f}]"
    print(f"r={rk}: recall {out[str(rk)]['entity_recall']:.4f}  unsupported "
          f"{out[str(rk)]['unsupported_entity_rate']:.4f}  survival {out[str(rk)]['skeleton_survival']:.4f}{chk}")

json.dump(out, (Path(__file__).resolve().parents[1] / "results" / "e1_ner_llmlingua_all_rates.json").open("w"), indent=1)
print("wrote e1_ner_llmlingua_all_rates.json")
