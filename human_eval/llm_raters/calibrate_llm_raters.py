#!/usr/bin/env python3
"""Calibration of the two LLM raters against the four human annotators (paper Appendix C, "Calibration").

No API calls. Reads each rater's judgments of the 54 human-study items (annotations_<rater>/batch_000.json,
item ids e5_NNN = item NNN of the human study) and the human judgments of the same items
(../e5_annotations_annotator{1,2,3,4}.json). Facts are aligned by position within an item (both sides judge
the same fact list, ../e5_facts.json).

Prints, per rater and per human annotator: number of aligned facts, raw agreement, Cohen's kappa on the
three-way P/D/M labels and on the binary preserved-vs-not collapse, and agreement on the invented-claim flag;
the number of items each rater flags as containing an invented claim and how many of the four adjudicated
items (../e5_halluc_adjudicated.json) it catches; and the human-human kappa ranges for comparison.
"""
import itertools
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
HUMAN_DIR = HERE.parent
RATERS = ("claude", "gpt")
HUMANS = ("annotator1", "annotator2", "annotator3", "annotator4")


def kappa(a, b, labels):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def binary(labels):
    return ["P" if x == "P" else "N" for x in labels]


def load_human(name):
    d = json.load(open(HUMAN_DIR / f"e5_annotations_{name}.json"))
    ann = d["annotations"]
    pairs = ann.items() if isinstance(ann, dict) else ((str(x["item_id"]), x) for x in ann)
    out = {}
    for k, v in pairs:
        labs = [str(f.get("judgment") or "").upper()[:1] for f in v["facts"]]
        hal = str(v.get("hallucinated_claim_present") or v.get("halluc") or "").upper()[:1]
        out[int(k)] = {"facts": labs, "hal": hal}
    return out


def main():
    humans = {h: load_human(h) for h in HUMANS}
    adjudicated = {x["item_id"] for x in json.load(open(HUMAN_DIR / "e5_halluc_adjudicated.json"))["items_with_invented_claim"]}
    summary = {}
    for rater in RATERS:
        mine = {int(j["item_id"].split("_")[1]): j for j in json.load(open(HERE / f"annotations_{rater}" / "batch_000.json"))}
        k3s, k2s = [], []
        print(f"[{rater}] vs human: n_facts, raw agreement, kappa P/D/M, kappa binary, invented-flag agreement")
        for h in HUMANS:
            H = humans[h]
            a, b, ha, hb = [], [], [], []
            for iid, j in mine.items():
                if iid not in H or not H[iid]["facts"]:
                    continue
                for f, hl in zip(j["facts"], H[iid]["facts"]):
                    if hl in ("P", "D", "M"):
                        a.append(f["label"]); b.append(hl)
                ha.append(j["hallucinated_claim_present"]); hb.append(H[iid]["hal"])
            k3, k2 = kappa(a, b, ["P", "D", "M"]), kappa(binary(a), binary(b), ["P", "N"])
            k3s.append(k3); k2s.append(k2)
            print(f"   {h}: {len(a)}, {sum(x == y for x, y in zip(a, b)) / len(a):.3f}, {k3:.3f}, {k2:.3f}, "
                  f"{sum(x == y for x, y in zip(ha, hb)) / len(ha):.3f}")
        flagged = sorted(i for i, j in mine.items() if j["hallucinated_claim_present"] == "Y")
        print(f"   kappa P/D/M {min(k3s):.2f}-{max(k3s):.2f}; binary {min(k2s):.2f}-{max(k2s):.2f}")
        print(f"   invented-claim flags: {len(flagged)} of {len(mine)} items {flagged}; "
              f"adjudicated items caught: {sorted(adjudicated & set(flagged))} of {sorted(adjudicated)}")
        summary[rater] = {"kappa_pdm": [round(min(k3s), 3), round(max(k3s), 3)],
                          "kappa_binary": [round(min(k2s), 3), round(max(k2s), 3)], "n_flagged": len(flagged)}

    k3s, k2s = [], []
    for x, y in itertools.combinations(HUMANS, 2):
        a, b = [], []
        for k in humans[x]:
            if k in humans[y]:
                for p, q in zip(humans[x][k]["facts"], humans[y][k]["facts"]):
                    if p in ("P", "D", "M") and q in ("P", "D", "M"):
                        a.append(p); b.append(q)
        k3s.append(kappa(a, b, ["P", "D", "M"])); k2s.append(kappa(binary(a), binary(b), ["P", "N"]))
    print(f"[human-human] kappa P/D/M {min(k3s):.2f}-{max(k3s):.2f}; binary {min(k2s):.2f}-{max(k2s):.2f}")
    n_judgments = sum(sum(1 for f in it["facts"] if f in ("P", "D", "M")) for H in humans.values() for it in H.values())
    print(f"human fact judgments: {n_judgments}; label distribution: "
          f"{dict(Counter(f for H in humans.values() for it in H.values() for f in it['facts']))}")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
