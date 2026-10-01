"""E4: re-score the saved reconstructions with chrF++, METEOR, ROUGE-L and BERTScore (distilroberta-base), and test
whether the strategy ranking depends on the metric (Tables 12 and 13; the per-record scores also feed
run_e2a_decoder_lever.py and human_eval/run_e5_analysis.py).

Outputs:
  results/e4_per_chunk.csv.gz     per-record scores, all metrics
  results/e4_summary.json         means per (source, strategy, rkeep, output_kind)
  results/e4_rank_stability.json  Spearman rho between each metric's strategy ranking and BERTScore's per
                                  (source, rkeep); the entries bbc_wordlevel4@<rkeep> are Table 13 (the four
                                  word-level configurations WordLen, WordFreq-3, WordFreq-6 and LP)
"""
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sacrebleu.metrics import CHRF
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from common_data import iter_bbc_records, iter_jsonl_records, iter_sft_records

OUT = Path(__file__).parent
CHRF_PP = CHRF(word_order=2)  # chrF++
ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
BERT_MODEL = "distilroberta-base"  # BERTScore backbone of the word-level tables
# the four word-level configurations of the rank-stability table (second-pass outputs)
WORDLEVEL4_STRATEGIES = ["wordlen", "wordfreq3", "wordfreq6", "opt_prepolish"]


def collect_records():
    recs = []
    for r in iter_bbc_records(outputs=("polished", "restored")):
        r["source"] = "bbc"
        recs.append(r)
    for r in iter_jsonl_records(datasets=("wiki", "reddit")):
        r["source"] = r.pop("dataset")
        recs.append(r)
    for r in iter_sft_records():
        r["source"] = f"bbc_sft_{r.pop('decoder')}"
        recs.append(r)
    return recs


def lexical_scores(recs):
    rows = []
    t0 = time.time()
    for i, r in enumerate(recs):
        ref, hyp = r["original"], r["reconstruction"]
        chrf = CHRF_PP.sentence_score(hyp, [ref]).score / 100.0
        met = meteor_score([ref.split()], hyp.split())
        rl = ROUGE.score(ref, hyp)["rougeL"].fmeasure
        rows.append(dict(
            source=r["source"], strategy=r["strategy"], rkeep=r["rkeep"],
            chunk_id=r["chunk_id"], output_kind=r["output_kind"],
            chrfpp=chrf, meteor=met, rougeL=rl,
        ))
        if (i + 1) % 5000 == 0:
            print(f"  lexical {i+1}/{len(recs)} ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def bert_scores(recs, batch_size=64):
    import torch
    from bert_score import BERTScorer
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    scorer = BERTScorer(model_type=BERT_MODEL, lang="en", rescale_with_baseline=False, device=device)
    f1s = np.zeros(len(recs))
    t0 = time.time()
    for i in range(0, len(recs), 1024):
        chunk = recs[i:i + 1024]
        refs = [r["original"] for r in chunk]
        hyps = [r["reconstruction"] for r in chunk]
        _, _, F = scorer.score(hyps, refs, batch_size=batch_size)
        f1s[i:i + len(chunk)] = F.numpy()
        print(f"  bertscore {min(i+1024,len(recs))}/{len(recs)} ({time.time()-t0:.0f}s)", flush=True)
    return f1s


def rank_stability(df):
    """Spearman rho between metric rankings of strategies, per (source, rkeep)."""
    out = {}
    metrics = ["bert_f1", "chrfpp", "meteor", "rougeL"]
    sub = df[df.output_kind.isin(["polished", "reconstruction"])]
    for (source, rkeep), g in sub.groupby(["source", "rkeep"]):
        means = g.groupby("strategy")[metrics].mean()
        if len(means) < 3:
            continue
        key = f"{source}@{rkeep}"
        out[key] = {"n_strategies": len(means), "strategy_means": means.round(4).to_dict()}
        for m in metrics[1:]:
            rho, p = spearmanr(means["bert_f1"], means[m])
            out[key][f"spearman_bert_vs_{m}"] = round(float(rho), 3)
    # rank-stability table: the four word-level configurations on BBC News
    bbc = sub[(sub.source == "bbc") & (sub.strategy.isin(WORDLEVEL4_STRATEGIES))]
    for rkeep, g in bbc.groupby("rkeep"):
        means = g.groupby("strategy")[metrics].mean()
        key = f"bbc_wordlevel4@{rkeep}"
        out[key] = {"n_strategies": len(means), "strategy_means": means.round(4).to_dict()}
        for m in metrics[1:]:
            rho, p = spearmanr(means["bert_f1"], means[m])
            out[key][f"spearman_bert_vs_{m}"] = round(float(rho), 3)
    return out


def main():
    print("Collecting records...", flush=True)
    recs = collect_records()
    print(f"{len(recs)} (ref, hyp) pairs", flush=True)

    df = lexical_scores(recs)
    print(f"Lexical metrics done. Running BERTScore ({BERT_MODEL})...", flush=True)
    df["bert_f1"] = bert_scores(recs)

    with gzip.open(Path(__file__).resolve().parents[1] / "results" / "e4_per_chunk.csv.gz", "wt") as f:
        df.to_csv(f, index=False)

    summary = (
        df.groupby(["source", "strategy", "rkeep", "output_kind"])
        [["bert_f1", "chrfpp", "meteor", "rougeL"]]
        .agg(["mean", "std", "count"]).round(4)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_json(Path(__file__).resolve().parents[1] / "results" / "e4_summary.json", orient="records", indent=1)

    stab = rank_stability(df)
    with open(Path(__file__).resolve().parents[1] / "results" / "e4_rank_stability.json", "w") as f:
        json.dump(stab, f, indent=1)

    print("\n=== Rank stability (Spearman rho vs BERTScore) ===")
    for k, v in sorted(stab.items()):
        rhos = {m.split("_vs_")[1]: r for m, r in v.items() if m.startswith("spearman")}
        print(f"{k}: {rhos}")
    print("DONE")


if __name__ == "__main__":
    main()
