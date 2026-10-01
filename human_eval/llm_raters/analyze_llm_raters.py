#!/usr/bin/env python3
"""Aggregate the LLM-rater judgments (Table 15): fact preservation by strategy x rate, invented-claim rate,
inter-rater agreement.

Reads annotations_<rater>/batch_*.json (grid batches only; batch_000 is the calibration set, see
calibrate_llm_raters.py) and mapping.json. Writes llm_rater_results.json and prints a compact table.
"""
import json, sys, statistics as st
from collections import defaultdict, Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
JUDGES = [j for j in ("claude", "gpt") if (HERE / f"annotations_{j}").exists()]
mapping = json.load(open(HERE / "mapping.json"))


def load(judge):
    out = {}
    for f in sorted((HERE / f"annotations_{judge}").glob("batch_*.json")):
        if f.name == "batch_000.json" or ".v1" in f.name or f.name.endswith(".errors.json"):
            continue
        for j in json.load(open(f)):
            if j.get("unresolved") or "facts" not in j:
                continue
            out[j["item_id"]] = j
    return out


def kappa(a, b, labels):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


ann = {j: load(j) for j in JUDGES}
common = set.intersection(*(set(a) for a in ann.values())) if len(ann) > 1 else set(next(iter(ann.values())))
print(f"judges: {JUDGES}; items judged: {{ {', '.join(f'{j}: {len(a)}' for j, a in ann.items())} }}; common: {len(common)}")

res = {"per_judge": {}, "consensus": {}, "agreement": {}}
for judge, A in ann.items():
    cells = defaultdict(lambda: {"P": 0, "D": 0, "M": 0, "items": 0, "halluc": 0})
    for iid, j in A.items():
        m = mapping.get(iid)
        if not m or m["set"] != "grid":
            continue
        c = cells[(m["strategy"], m["rkeep"])]
        for f in j["facts"]:
            c[f["label"]] += 1
        c["items"] += 1
        c["halluc"] += j["hallucinated_claim_present"] == "Y"
    tab = {}
    for (s, r), c in sorted(cells.items()):
        n = c["P"] + c["D"] + c["M"]
        tab[f"{s}@{r}"] = {"n_items": c["items"], "n_facts": n, "preserved": round(c["P"] / n, 4), "distorted": round(c["D"] / n, 4),
                            "missing": round(c["M"] / n, 4), "halluc_rate": round(c["halluc"] / c["items"], 4)}
    res["per_judge"][judge] = tab

# consensus (both judges agree P vs not-P per fact) and agreement stats
if len(ann) == 2:
    a, b = [ann[j] for j in JUDGES]
    la, lb, ha, hb = [], [], [], []
    cons = defaultdict(lambda: {"P_both": 0, "P_any": 0, "n": 0, "items": 0, "halluc_both": 0})
    for iid in common:
        m = mapping.get(iid)
        if not m or m["set"] != "grid":
            continue
        ja, jb = a[iid], b[iid]
        c = cons[(m["strategy"], m["rkeep"])]
        for fa, fb in zip(ja["facts"], jb["facts"]):
            la.append(fa["label"]); lb.append(fb["label"])
            c["n"] += 1; c["P_both"] += fa["label"] == "P" == fb["label"]; c["P_any"] += (fa["label"] == "P") or (fb["label"] == "P")
        c["items"] += 1; c["halluc_both"] += ja["hallucinated_claim_present"] == "Y" == jb["hallucinated_claim_present"]
        ha.append(ja["hallucinated_claim_present"]); hb.append(jb["hallucinated_claim_present"])
    binA = ["P" if x == "P" else "N" for x in la]; binB = ["P" if x == "P" else "N" for x in lb]
    res["agreement"] = {"n_facts": len(la), "raw": round(sum(x == y for x, y in zip(la, lb)) / len(la), 4),
                        "kappa_pdm": round(kappa(la, lb, ["P", "D", "M"]), 4), "kappa_binary": round(kappa(binA, binB, ["P", "N"]), 4),
                        "halluc_agreement": round(sum(x == y for x, y in zip(ha, hb)) / len(ha), 4)}
    res["consensus"] = {f"{s}@{r}": {"n_items": c["items"], "preserved_both": round(c["P_both"] / c["n"], 4),
                                     "preserved_any": round(c["P_any"] / c["n"], 4), "halluc_both": round(c["halluc_both"] / c["items"], 4)}
                        for (s, r), c in sorted(cons.items())}
    print("inter-judge agreement:", res["agreement"])

json.dump(res, open(HERE / "llm_rater_results.json", "w"), indent=1)
# compact table: preserved share per judge (and consensus) by strategy x rate
strategies = sorted({k.split("@")[0] for t in res["per_judge"].values() for k in t})
rates = sorted({float(k.split("@")[1]) for t in res["per_judge"].values() for k in t}, reverse=True)
for judge, tab in res["per_judge"].items():
    print(f"\n[{judge}] fact preservation rate (P share); halluc rate in parentheses")
    print(f"{'strategy':12s}" + "".join(f"{r:>14}" for r in rates))
    for s in strategies:
        row = []
        for r in rates:
            c = tab.get(f"{s}@{r}")
            row.append(f"{c['preserved']:.2f} ({c['halluc_rate']:.2f})" if c else "-")
        print(f"{s:12s}" + "".join(f"{x:>14}" for x in row))
