"""Shared loaders for the analysis scripts.

Yields one dict per record (original / skeleton / reconstruction) from
  1. data/BBC_news_latest.xlsx: the word-level strategies at the nine retention rates, and the base vs QLoRA
     Llama decoder outputs of Table 6;
  2. samples/*_hybrid_samples.jsonl: the semantic and hybrid strategies of the cross-domain datasets (five rates).
Workbook strategy keys: step_char / step_token = Step on characters / tokens, wordlen = WordLen, wordfreq3 /
wordfreq6 = WordFreq-3 / -6, opt_prepolish = LP as reported, opt = an LP variant (sheet
wf_optimal_without_prepolish) not in the paper's tables. output_kind: restored = first decoder pass, polished = the
second decoder pass, whose outputs are the reconstructions the paper scores.
"""
import ast
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # release root
BBC_XLSX = ROOT / "data" / "BBC_news_latest.xlsx"
HARNESS = ROOT / "harness"

RATES = [90, 80, 70, 60, 50, 40, 30, 20, 10]


def _parse_df_dict(cell):
    """benchmark sheet stores skeletons as stringified dicts."""
    if not isinstance(cell, str) or not cell.startswith("{"):
        return {}
    try:
        return ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return {}


def iter_bbc_records(outputs=("polished", "restored")):
    """Yield dicts: strategy, rkeep, chunk_id, original, skeleton, output_kind, reconstruction."""
    xl = pd.ExcelFile(BBC_XLSX)

    # --- STEP (benchmark sheet): char and token variants -------------------
    bench = xl.parse("benchmark")
    for variant in ("char", "token"):
        for _, row in bench.iterrows():
            orig = row.get("Articles")
            if not isinstance(orig, str) or len(orig) < 50:
                continue
            hi = _parse_df_dict(row.get(f"df_{variant}"))
            lo = _parse_df_dict(row.get(f"df_{variant}_50minus"))
            skel_map = {}
            for k, v in {**hi, **lo}.items():
                # keys like 'deleted_char_90'
                tail = k.rsplit("_", 1)[-1]
                if tail.isdigit():
                    skel_map[int(tail)] = v
            for r in RATES:
                for kind in outputs:
                    rec = row.get(f"{kind}_{variant}_{r}")
                    if isinstance(rec, str) and rec.strip():
                        yield dict(
                            strategy=f"step_{variant}", rkeep=r / 100.0,
                            chunk_id=int(row.name), original=orig,
                            skeleton=skel_map.get(r), output_kind=kind,
                            reconstruction=rec,
                        )

    # --- simple sheets: skeleton column pattern + restored/polished --------
    sheet_cfg = [
        ("adpative_word_length_deletion", "wordlen",
         lambda r: f"ratio_{r}_text", lambda k, r: f"{k}_{r}"),
        ("wf_3classes", "wordfreq3",
         lambda r: f"del_zipf_{r}", lambda k, r: f"{k}_{r}"),
        ("wf_6classes", "wordfreq6",
         lambda r: f"del_zipf_{r}", lambda k, r: f"{k}_{r}"),
        ("wf_optimal", "opt_prepolish",
         lambda r: f"optimal_keep_{r}", lambda k, r: f"{k}_optimal_keep_{r}"),
        ("wf_optimal_without_prepolish", "opt",
         lambda r: f"optimal_keep_fine_{r}", lambda k, r: f"{k}_optimal_keep_fine_{r}"),
    ]
    for sheet, strat, skel_col, rec_col in sheet_cfg:
        df = xl.parse(sheet)
        for _, row in df.iterrows():
            orig = row.get("Articles")
            if not isinstance(orig, str) or len(orig) < 50:
                continue
            for r in RATES:
                skel = row.get(skel_col(r))
                if not isinstance(skel, str):
                    skel = None
                for kind in outputs:
                    rec = row.get(rec_col(kind, r))
                    if isinstance(rec, str) and rec.strip():
                        yield dict(
                            strategy=strat, rkeep=r / 100.0,
                            chunk_id=int(row.name), original=orig,
                            skeleton=skel, output_kind=kind,
                            reconstruction=rec,
                        )


def iter_sft_records():
    """Base vs QLoRA Llama-3.2-3B-Instruct decoder outputs on BBC News (Table 6). At each rate the skeletons are those
    of the strategy whose fine-tuned decoder performs best: LP at r_keep 0.9 and 0.5, WordLen at 0.7, WordFreq-3 at
    0.3 and 0.1. The record key strategy="wordfreq_opt_sft" is kept for compatibility with the shipped results."""
    xl = pd.ExcelFile(BBC_XLSX)
    cfgs = [
        ("sft_generation_result", "qlora", "llama"),
        ("base_sft_generation_result", "base", "llama_base"),
    ]
    for sheet, decoder, prefix in cfgs:
        df = xl.parse(sheet)
        for _, row in df.iterrows():
            for r in (90, 70, 50, 30, 10):
                gt = row.get(f"{prefix}_ground_truth_{r}")
                pred = row.get(f"{prefix}_prediction_{r}")
                cor = row.get(f"{prefix}_corrupted_text_{r}")
                if isinstance(gt, str) and isinstance(pred, str) and gt.strip() and pred.strip():
                    yield dict(
                        strategy="wordfreq_opt_sft", decoder=decoder,
                        rkeep=r / 100.0, chunk_id=int(row.name),
                        original=gt, skeleton=cor if isinstance(cor, str) else None,
                        output_kind="prediction", reconstruction=pred,
                    )


def iter_jsonl_records(datasets=("wiki", "reddit")):
    """Semantic and hybrid samples of the cross-domain datasets (Gemini decoder); wiki and reddit are English."""
    paths = {
        "wiki": Path(__file__).resolve().parents[1] / "samples" / "wiki_hybrid_samples.jsonl",
        "reddit": Path(__file__).resolve().parents[1] / "samples" / "reddit_hybrid_samples.jsonl",
        "news_zh": Path(__file__).resolve().parents[1] / "samples" / "news_zh_hybrid_samples.jsonl",
        "wiki_zh": Path(__file__).resolve().parents[1] / "samples" / "wiki_zh_hybrid_samples.jsonl",
    }
    for name in datasets:
        with open(paths[name]) as fh:
            for line in fh:
                r = json.loads(line)
                yield dict(
                    dataset=name, strategy=r["method"], rkeep=float(r["rkeep"]),
                    chunk_id=r["sample_idx"], original=r["reference"],
                    skeleton=r["degraded"], output_kind="reconstruction",
                    reconstruction=r["reconstruction"],
                )


if __name__ == "__main__":
    import collections
    cov = collections.Counter()
    for rec in iter_bbc_records():
        cov[(rec["strategy"], rec["output_kind"])] += 1
    for k in sorted(cov):
        print("bbc", k, cov[k])
    cov = collections.Counter()
    for rec in iter_sft_records():
        cov[(rec["decoder"], rec["rkeep"])] += 1
    print("sft cells:", dict(cov))
