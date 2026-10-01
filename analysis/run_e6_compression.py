"""E6: three-tier storage accounting on BBC News (Tables 23 and 24).

Per word-level strategy and r_keep (means over articles):
  tier1                len(skeleton) / len(original)
  tier2_<codec>        zlib | bz2 | lzma (skeleton) / len(original): the skeleton is text, so a lossless codec
                       composes with it
  lossless baseline    zlib | bz2 | lzma (original) / len(original)
Tier 3 (system cost) is analytic; its side-information sizes are printed at the end.
Output: results/e6_compression_accounting.json.
"""
import bz2
import json
import lzma
import sys
import zlib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common_data import iter_bbc_records

OUT = Path(__file__).parent

CODECS = {
    "zlib": lambda b: zlib.compress(b, 9),
    "bz2": lambda b: bz2.compress(b, 9),
    "lzma": lambda b: lzma.compress(b, preset=6),
}


def main():
    seen = set()
    agg = defaultdict(lambda: defaultdict(list))
    orig_stats = defaultdict(list)

    for r in iter_bbc_records(outputs=("polished",)):
        key = (r["strategy"], r["rkeep"], r["chunk_id"])
        if key in seen or not r.get("skeleton"):
            continue
        seen.add(key)
        o = r["original"].encode("utf-8")
        s = r["skeleton"].encode("utf-8")
        a = agg[(r["strategy"], r["rkeep"])]
        a["tier1"].append(len(s) / len(o))
        for name, fn in CODECS.items():
            a[f"tier2_{name}"].append(len(fn(s)) / len(o))
        if (r["chunk_id"], "orig") not in seen:
            seen.add((r["chunk_id"], "orig"))
            for name, fn in CODECS.items():
                orig_stats[name].append(len(fn(o)) / len(o))

    results = []
    for (strategy, rkeep), a in sorted(agg.items()):
        row = dict(strategy=strategy, rkeep=rkeep, n=len(a["tier1"]))
        for k, v in a.items():
            row[k + "_mean"] = round(sum(v) / len(v), 4)
        results.append(row)

    baseline = {name: round(sum(v) / len(v), 4) for name, v in orig_stats.items()}

    out = {"lossless_baseline_on_original": baseline, "per_strategy": results}
    with open(Path(__file__).resolve().parents[1] / "results" / "e6_compression_accounting.json", "w") as f:
        json.dump(out, f, indent=1)

    print("Lossless baselines (original):", baseline)
    print(f"\n{'strategy':<16}{'rkeep':>6}{'tier1':>8}{'t2 zlib':>9}{'t2 bz2':>9}{'t2 lzma':>9}")
    for row in results:
        print(f"{row['strategy']:<16}{row['rkeep']:>6}{row['tier1_mean']:>8}"
              f"{row.get('tier2_zlib_mean',''):>9}{row.get('tier2_bz2_mean',''):>9}"
              f"{row.get('tier2_lzma_mean',''):>9}")

    print("\nTier-3 side information (one-time, amortized over corpus):")
    print("  Llama-3.2-3B fp16 weights ~6.4 GB; QLoRA adapter 194.6 MB as trained here (LoRA rank 32 on all projections);")
    print("  reconstruction prompt ~200-400 bytes/chunk; Gemini decoder = 0 local bytes.")
    print("DONE")


if __name__ == "__main__":
    main()
