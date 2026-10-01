"""Human fact-level study (paper Appendix B, Table 14): fact-preservation rates, inter-annotator agreement,
and the correlation of fact preservation with BERTScore.

Inputs (this directory): e5_annotations_annotator{1,2,3,4}.json (the four human annotators; both the
annotation tool's export format and its raw store format are read), and the condition key in
e5_annotation_sheet.xlsx (sheet `key`: item_id -> strategy, rkeep, chunk_id). BERTScore and ROUGE-L of the
same reconstructions come from ../results/e4_per_chunk.csv.gz (source `bbc`, output_kind `polished`).

Strategy keys: wordfreq6 = WordFreq-6, opt_prepolish = LP (see the naming map in the top-level README).

Output: e5_analysis.json and a printed summary. `--table14` prints Table 14 only: the share of facts judged
preserved (P), distorted (D) or missing (M) per condition, averaged over the four annotators.
"""
import argparse
import gzip
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ANNOTATORS = ["annotator1", "annotator2", "annotator3", "annotator4"]
PRIMARY = "annotator1"  # used for the per-rate breakdown and the BERTScore correlation
LABEL = {"wordfreq6": "WordFreq-6", "opt_prepolish": "LP"}


def load_annotator(path):
    d = json.loads(Path(path).read_text())
    out = {}
    ann = d["annotations"]
    if isinstance(ann, dict):  # raw store format {id: {facts, halluc, notes}}
        for k, a in ann.items():
            facts = [(f["text"].strip(), f["judgment"]) for f in a["facts"] if f["text"].strip()]
            out[int(k)] = {"facts": facts, "halluc": a["halluc"]}
    else:  # export format
        for a in ann:
            facts = [(f["text"].strip(), f["judgment"]) for f in a["facts"] if f["text"].strip()]
            out[a["item_id"]] = {"facts": facts, "halluc": a["hallucinated_claim_present"]}
    return out


def table14(per_cond):
    """Mean over annotators of each annotator's P/D/M share, per (strategy, rkeep)."""
    rows = []
    for (strat, rk), by_ann in sorted(per_cond.items(), key=lambda kv: (kv[0][0] != "wordfreq6", -kv[0][1])):
        shares = {j: sum(c[j] / sum(c.values()) for c in by_ann.values()) / len(by_ann) for j in "PDM"}
        rows.append({"strategy": strat, "rkeep": rk, "n_annotators": len(by_ann),
                     **{j: round(v, 4) for j, v in shares.items()}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table14", action="store_true", help="print Table 14 only")
    args = ap.parse_args()

    annotators = {name: load_annotator(HERE / f"e5_annotations_{name}.json") for name in ANNOTATORS}
    key = pd.read_excel(HERE / "e5_annotation_sheet.xlsx", sheet_name="key")
    cond = {r.item_id: (r.strategy, float(r.rkeep), int(r.chunk_id)) for r in key.itertuples()}

    # per-condition label counts, per annotator
    per_cond = defaultdict(lambda: defaultdict(Counter))
    halluc_cond = defaultdict(lambda: defaultdict(list))
    for name, ann in annotators.items():
        for i, a in ann.items():
            strat, rk, _ = cond[i]
            for _, j in a["facts"]:
                per_cond[(strat, rk)][name][j] += 1
            halluc_cond[(strat, rk)][name].append(a["halluc"] == "Y")

    t14 = table14(per_cond)
    if args.table14:
        print("Table 14: share of facts judged P / D / M, averaged over the four annotators")
        print(f"{'strategy':<12}{'rkeep':>6}{'P':>7}{'D':>7}{'M':>7}")
        for r in t14:
            print(f"{LABEL[r['strategy']]:<12}{r['rkeep']:>6}{r['P']:>7.2f}{r['D']:>7.2f}{r['M']:>7.2f}")
        return

    results = {"annotators": ANNOTATORS,
               "n_judgments": sum(len(a["facts"]) for ann in annotators.values() for a in ann.values())}

    # --- pairwise agreement (facts aligned by text within an item) -----------
    agr = {}
    for (n1, a1), (n2, a2) in combinations(annotators.items(), 2):
        x, y = [], []
        for i in set(a1) & set(a2):
            f2 = dict(a2[i]["facts"])
            for text, j1 in a1[i]["facts"]:
                if text in f2:
                    x.append(j1); y.append(f2[text])
        common = sorted(set(a1) & set(a2))
        hx = [a1[i]["halluc"] for i in common]
        hy = [a2[i]["halluc"] for i in common]
        agr[f"{n1}|{n2}"] = {
            "n_facts": len(x),
            "raw_agreement": round(sum(1 for a, b in zip(x, y) if a == b) / len(x), 4),
            "kappa_pdm": round(float(cohen_kappa_score(x, y)), 3),
            "kappa_binary": round(float(cohen_kappa_score([v == "P" for v in x], [v == "P" for v in y])), 3),
            "halluc_agreement": round(sum(1 for a, b in zip(hx, hy) if a == b) / len(hx), 4),
        }
    results["pairwise_agreement"] = agr

    # --- per-condition rates, each annotator -------------------------------
    rows = []
    for (strat, rk), by_ann in sorted(per_cond.items()):
        row = {"strategy": strat, "rkeep": rk}
        for name, c in by_ann.items():
            n = sum(c.values())
            for j in "PDM":
                row[f"{name}_{j}"] = round(c[j] / n, 3)
            flags = halluc_cond[(strat, rk)][name]
            row[f"{name}_inventedY"] = round(sum(flags) / len(flags), 3)
        rows.append(row)
    results["per_condition"] = rows
    results["table14_mean_over_annotators"] = t14

    # --- invented-claim flags ---------------------------------------------
    flags = Counter()
    for ann in annotators.values():
        for i, a in ann.items():
            flags[i] += a["halluc"] == "Y"
    results["invented_claim_flags"] = {
        "per_annotator": {n: sum(a["halluc"] == "Y" for a in ann.values()) for n, ann in annotators.items()},
        "items_flagged_by_two_or_more": sorted(i for i, c in flags.items() if c >= 2),
    }

    # --- rate-level breakdown, primary annotator ---------------------------
    pooled = defaultdict(Counter)
    for i, a in annotators[PRIMARY].items():
        _, rk, _ = cond[i]
        for _, j in a["facts"]:
            pooled[rk][j] += 1
    results["primary_by_rate"] = {
        str(rk): {j: round(c[j] / sum(c.values()), 3) for j in "PDM"}
        for rk, c in sorted(pooled.items())
    }

    # --- correlation: per-item preservation (primary) vs BERTScore -----------
    with gzip.open(ROOT / "results" / "e4_per_chunk.csv.gz", "rt") as f:
        e4 = pd.read_csv(f)
    e4 = e4[(e4.source == "bbc") & (e4.output_kind == "polished")]
    scores = {(r.strategy, r.rkeep, r.chunk_id): (r.bert_f1, r.rougeL) for r in e4.itertuples()}
    xs, ys = [], []
    for i, a in annotators[PRIMARY].items():
        s = scores.get(cond[i])
        if s is not None and a["facts"]:
            xs.append(sum(1 for _, j in a["facts"] if j == "P") / len(a["facts"]))
            ys.append(s[0])
    sp, sp_p = spearmanr(xs, ys)
    pe, pe_p = pearsonr(xs, ys)
    results["preservation_vs_bertscore"] = {
        "n_items": len(xs), "spearman": round(float(sp), 3), "spearman_p": float(f"{sp_p:.1e}"),
        "pearson": round(float(pe), 3), "pearson_p": float(f"{pe_p:.1e}"),
    }
    by_rate = defaultdict(list)
    for i in cond:
        if cond[i] in scores:
            by_rate[cond[i][1]].append(scores[cond[i]])
    results["items_mean_scores_by_rate"] = {
        str(rk): {"n_items": len(v), "bert_f1": round(sum(b for b, _ in v) / len(v), 4),
                  "rougeL": round(sum(r for _, r in v) / len(v), 4)}
        for rk, v in sorted(by_rate.items(), reverse=True)
    }

    (HERE / "e5_analysis.json").write_text(json.dumps(results, indent=1))
    print("pairwise agreement:", json.dumps(agr, indent=1))
    print("\nTable 14 (mean over annotators):")
    for r in t14:
        print(f"  {LABEL[r['strategy']]}@{r['rkeep']}: P={r['P']:.2f} D={r['D']:.2f} M={r['M']:.2f}")
    print("\ninvented-claim flags:", results["invented_claim_flags"])
    print("preservation vs BERTScore:", results["preservation_vs_bertscore"])
    print("mean scores of the rated items by rate:", results["items_mean_scores_by_rate"])


if __name__ == "__main__":
    main()
