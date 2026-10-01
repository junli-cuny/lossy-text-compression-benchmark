#!/usr/bin/env python3
"""Downstream task evaluation on BBC News (Tables 17 and 18).

Tasks
  topic      : 5-way topic classification. A classifier is trained once on original articles from `datasource`
               that are not in the test set, then applied to each variant.
  retrieval  : the test row's Summary is the query and the 200 variant texts are the corpus; Recall@1 / @5 (the
               correct document is the variant of the same article).
Backends
  tfidf      : TF-IDF + LinearSVC (topic) / TF-IDF cosine (retrieval)
  neural     : sentence-transformers embeddings + LinearSVC (topic) / cosine (retrieval)

Variants per strategy: original, skeleton@r, reconstructed@r for every available rate. The word-level strategies
come from data/BBC_news_latest.xlsx (Table 17). The semantic, hybrid and LLMLingua-2 samples (--semantic) are
512-character chunks from 18 politics articles, so they are reported separately and only as single-class topic
recall (Table 18); samples/bbc_chunk_labels.json identifies their source articles, which are held out of training.
Outputs: results/downstream_tfidf.json, results/downstream_neural.json (macro-F1 and Recall@5 included).

Usage:
  python analysis/run_downstream_full.py --backend tfidf --semantic
  python analysis/run_downstream_full.py --backend neural --semantic --st-model sentence-transformers/all-MiniLM-L6-v2
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
XLSX = REPO / "data" / "BBC_news_latest.xlsx"
RATES = [90, 80, 70, 60, 50, 40, 30, 20, 10]

# strategy -> (sheet, skeleton column fn, reconstruction column fn)
# Step keeps its skeletons inside the df_char dict rather than in plain columns.
SHEETS = {
    "Step":       ("benchmark", None, lambda r: f"polished_char_{r}"),
    "WordLen":    ("adpative_word_length_deletion", None, lambda r: f"polished_{r}"),
    "WordFreq-3": ("wf_3classes", lambda r: f"del_zipf_{r}", lambda r: f"polished_{r}"),
    "WordFreq-6": ("wf_6classes", lambda r: f"del_zipf_{r}", lambda r: f"polished_{r}"),
    "LP":         ("wf_optimal", lambda r: f"optimal_keep_{r}", lambda r: f"polished_optimal_keep_{r}"),
}


class Backend:
    """Two interchangeable text representations for both tasks."""

    def __init__(self, kind, st_model=None):
        self.kind = kind
        if kind == "neural":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(st_model)

    def fit_transform(self, texts):
        if self.kind == "tfidf":
            self.vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
            return self.vec.fit_transform(texts)
        return self.model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    def transform(self, texts):
        if self.kind == "tfidf":
            return self.vec.transform(texts)
        return self.model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    def fit_transform_retrieval(self, corpus):
        if self.kind == "tfidf":
            self.rvec = TfidfVectorizer(max_features=50000, sublinear_tf=True)
            return self.rvec.fit_transform(corpus)
        return self.model.encode(corpus, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    def transform_retrieval(self, queries):
        if self.kind == "tfidf":
            return self.rvec.transform(queries)
        return self.model.encode(queries, batch_size=64, show_progress_bar=False, normalize_embeddings=True)


def topic_task(be, Xtr, train_y, variant_texts, test_y):
    clf = LinearSVC()
    clf.fit(Xtr, train_y)
    pred = clf.predict(be.transform(variant_texts))
    return dict(acc=round(float(accuracy_score(test_y, pred)), 4),
                macro_f1=round(float(f1_score(test_y, pred, average="macro")), 4))


def retrieval_task(be, queries, corpus, gold=None):
    C = be.fit_transform_retrieval(corpus)
    Q = be.transform_retrieval(queries)
    sim = cosine_similarity(Q, C)
    ranks = (-sim).argsort(axis=1)
    gold = gold if gold is not None else list(range(len(queries)))
    out = {}
    for k in (1, 5):
        hit = sum(1 for i in range(len(queries)) if gold[i] in ranks[i, :k])
        out[f"recall@{k}"] = round(hit / len(queries), 4)
    return out


def step_skeletons(df, rate):
    """Step stores its skeletons in the df_char dict column."""
    import ast
    out = []
    for v in df["df_char"]:
        try:
            out.append(str(ast.literal_eval(v)[f"deleted_char_{rate}"]))
        except Exception:
            out.append("")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tfidf", "neural"], default="tfidf")
    ap.add_argument("--st-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--semantic", action="store_true", help="also score the semantic/hybrid + LLMLingua chunk samples (single-class topic recall)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    be = Backend(args.backend, args.st_model)
    xl = pd.ExcelFile(XLSX)
    src = xl.parse("datasource").dropna(subset=["Articles", "Category"])
    results = {"backend": args.backend, "st_model": args.st_model if args.backend == "neural" else None,
               "workbook": XLSX.name, "strategies": {}}

    for strat, (sheet, skel_fn, rec_fn) in SHEETS.items():
        df = xl.parse(sheet).dropna(subset=["Articles", "Category"])
        test_articles = set(df["Articles"].astype(str))
        train = src[~src["Articles"].astype(str).isin(test_articles)]
        Xtr = be.fit_transform(train["Articles"].astype(str).tolist())
        train_y = train["Category"].astype(str).tolist()
        test_y = df["Category"].astype(str).tolist()
        queries = df["Summaries"].astype(str).tolist()
        orig = df["Articles"].astype(str).tolist()
        entry = {"n_test": len(df), "n_train": len(train),
                 "original": {"topic": topic_task(be, Xtr, train_y, orig, test_y),
                              "retrieval": retrieval_task(be, queries, orig)}}
        print(f"\n=== {strat} ({args.backend}, n_test={len(df)}, n_train={len(train)}) ===", flush=True)
        print(f"  original            topic acc {entry['original']['topic']['acc']:.4f}  R@1 {entry['original']['retrieval']['recall@1']:.4f}", flush=True)
        for r in RATES:
            for kind in ("skeleton", "reconstructed"):
                if kind == "skeleton":
                    if strat == "Step":
                        texts = step_skeletons(df, r)
                    elif skel_fn is None or skel_fn(r) not in df.columns:
                        continue
                    else:
                        texts = df[skel_fn(r)].astype(str).tolist()
                else:
                    col = rec_fn(r)
                    if col not in df.columns:
                        continue
                    texts = df[col].astype(str).tolist()
                if not any(t.strip() for t in texts):
                    continue
                m = {"topic": topic_task(be, Xtr, train_y, texts, test_y),
                     "retrieval": retrieval_task(be, queries, texts)}
                entry[f"{kind}@0.{r // 10 if r >= 10 else r}"] = m
                print(f"  {kind:<14}@0.{r//10}  topic acc {m['topic']['acc']:.4f}  macro-F1 {m['topic']['macro_f1']:.4f}  R@1 {m['retrieval']['recall@1']:.4f}  R@5 {m['retrieval']['recall@5']:.4f}", flush=True)
        results["strategies"][strat] = entry

    if args.semantic:
        results["semantic_chunks"] = semantic_topic_eval(be, src)

    out = Path(args.out) if args.out else REPO / "results" / f"downstream_{args.backend}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")


def semantic_topic_eval(be, src):
    """Semantic/hybrid + LLMLingua-2 samples: 100 512-char chunks that all come from 18
    `politics` articles (load_chunks walks datasource from the top and stops at 100).

    Retrieval is not identifiable on this sample: several chunks share one source article, so a
    per-chunk gold caps the ORIGINAL chunks at R@1 = 0.18 while an article-level gold saturates
    every variant at 1.00 (both measured). We therefore report only single-class topic recall --
    the share of variants that a 5-way classifier, trained on the datasource articles OUTSIDE the
    18 sources, still labels `politics`. The original chunks score 0.93 (TF-IDF) and 0.95 (MiniLM), so the ceiling
    is real.
    """
    import collections
    samples = REPO / "samples"
    labels = {x["sample_idx"]: x for x in json.load(open(Path(__file__).resolve().parents[1] / "samples" / "bbc_chunk_labels.json"))}
    src_sums = src["Summaries"].astype(str).str[:200]
    held = set(x["summary"][:200] for x in labels.values())
    train = src[~src_sums.isin(held)]
    Xtr = be.fit_transform(train["Articles"].astype(str).tolist())
    clf = LinearSVC().fit(Xtr, train["Category"].astype(str).tolist())
    print(f"\n=== semantic chunks ({be.kind}, n_chunks=100 from 18 politics articles, "
          f"n_train={len(train)}) === single-class topic recall", flush=True)

    rows = []
    for f in (samples / "bbc_hybrid_samples.jsonl", samples / "llmlingua2_samples.jsonl",
              samples / "llmlingua2_samples_extra_rates.jsonl"):  # LLMLingua-2: r = 0.7/0.5/0.3, then 0.9/0.1
        if not f.exists():
            sys.exit(f"missing {f}")
        rows += [json.loads(l) for l in open(f)]
    cells = collections.defaultdict(list)
    for x in rows:
        cells[(x["method"], float(x["rkeep"]))].append(x)

    out = {"note": "100 chunks of 512 chars from 18 politics articles; single-class topic recall "
                   "(share still labelled politics); retrieval not identifiable on this sample",
           "n_train": len(train), "methods": {}}
    for (m, rk), v in sorted(cells.items()):
        for kind, key in (("original", "reference"), ("skeleton", "degraded"), ("reconstructed", "reconstruction")):
            texts = [str(x.get(key) or "") for x in v]
            if not any(t.strip() for t in texts):
                continue
            pred = clf.predict(be.transform(texts))
            rec = round(float(sum(p == "politics" for p in pred)) / len(pred), 4)
            out["methods"].setdefault(m, {})[f"{kind}@{rk}"] = {"n": len(texts), "recall": rec}
        d = out["methods"][m]
        print(f"  {m:24s} r={rk:<4} orig {d[f'original@{rk}']['recall']:.2f}  "
              f"skel {d.get(f'skeleton@{rk}', {}).get('recall', float('nan')):.2f}  "
              f"recon {d.get(f'reconstructed@{rk}', {}).get('recall', float('nan')):.2f}", flush=True)
    return out


if __name__ == "__main__":
    main()
