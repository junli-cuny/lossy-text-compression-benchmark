# Downstream sanity check on new compressors (`harness/run_downstream_eval.py`)

This is a self-contained tool for trying the two downstream tasks on compressors of your own. It did **not**
produce the paper's downstream tables: Tables 17-18 come from `analysis/run_downstream_full.py`, which scores the
paper's stored skeletons and reconstructions (`results/downstream_tfidf.json`, `results/downstream_neural.json`).
This tool instead builds its own approximate skeletons (truncation, random, an evenly spaced Step, a character-level
WordLen approximation, GPT-2 entropy, and GPT-2-token word frequency), uses a random train/test split of the BBC
`datasource` sheet, and truncates articles to `--max-article-chars`.

Tasks:

1. Topic classification. With the default `--backend sklearn` this is a supervised TF-IDF (1-2 grams) + linear SVM
   classifier trained on the original training articles; with `--backend hf` it is zero-shot NLI classification. The
   results record which one ran (`topic_method`).
2. Retrieval: the article summaries are the queries and the variants of the test articles are the corpus
   (Recall@K); TF-IDF cosine with `sklearn`, sentence-transformer embeddings with `hf`.

Each task runs on the original text, the skeleton, and, if a decoder is chosen, the reconstruction.

## Install

```bash
pip install -r harness/downstream_eval_requirements.txt
```

## Run

Skeleton only (no decoder):

```bash
python harness/run_downstream_eval.py --compressor wordlen --decoder none
```

Gemini reconstruction (the default model is gemini-2.5-flash):

```bash
export GOOGLE_API_KEY=...
python harness/run_downstream_eval.py --compressor entropy --decoder gemini
```

Local Hugging Face reconstruction:

```bash
python harness/run_downstream_eval.py --compressor wordlen --decoder hf --hf-model gpt2
```

Zero-shot topic classification and sentence-transformer retrieval (heavier dependencies):

```bash
python harness/run_downstream_eval.py --backend hf --compressor wordlen --decoder none
```

Results are written to `harness/downstream_eval_results.json` (override with `--out`). Start with a small
`--n-test` (e.g. 20), and compare methods on the same `--rkeeps` grid.
