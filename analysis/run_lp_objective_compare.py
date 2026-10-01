"""LP objective variants (Appendix P): which bucket objective gives the best reconstruction?

Three objectives for the bucket deletion ratios w_k, all under the pipeline's budget constraint
sum_k p_k (1 - w_k) <= T (keep at most T) and 0 <= w_k <= 1, with B_k(w_k) = 1 - w_k (1 - b_k):
  prod     max prod_k B_k(w_k)       the objective of Sec. 3, which the notebook's solve_max_product_cvxpy solves;
                                     prod_k p_k B_k(w_k) has the same argmax and is reported as its alias prod_p
  sum_nop  max sum_k B_k(w_k)        its first-order expansion, the linear program of Appendix P
  sum_p    max sum_k p_k B_k(w_k)    the additive variant the Entropy-LP and Entropy-in-FreqBuckets hybrids use
Inputs: workbook sheet wf_optimal (stored b_zipf_stats and p_zipf_stats per article), the first --n articles,
rates 0.9/0.7/0.5/0.3/0.1. Deletion replicates the notebook's delete_by_w_length_budget_with_target (segment
level, per class, random within class, whitespace top-up to the band, fallback below the band; classes via _SEG_RE
and classify_word_zipf with z <= 3 / <= 5 thresholds). Decoder: gemini-2.5-flash with the semantic methods' prompt.
--verify: regenerate skeletons from the recorded w and compare per-class deleted counts with the stored
optimal_keep_* columns (the notebook's seeds hash a str and are not reproducible across processes, so byte
identity cannot be checked; counts can).
--agreement: the allocation agreement of Appendix P (no API): share of the 200 articles x 9 rates where sum_nop
reproduces the allocation of prod, with no w_k differing by more than 0.02.
Output: samples/bbc_lp_objective_compare_samples.jsonl (resumable; identical skeletons decoded once).
Scoring: score_lp_objective_compare.py.
"""
import argparse, ast, json, os, random, re, sys, threading, time, unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import linprog, minimize
from wordfreq import zipf_frequency

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
XLSX = ROOT / "data" / "BBC_news_latest.xlsx"
OUT = Path(__file__).resolve().parents[1] / "samples" / "bbc_lp_objective_compare_samples.jsonl"
CLASSES = ["low", "mid", "high", "punct", "num_other", "whitespace"]
RATES = ["0.9", "0.7", "0.5", "0.3", "0.1"]
OBJS = ["sum_p", "sum_nop", "prod"]          # prod_p is reported as an alias of prod
_SEG_RE = re.compile(r"\s+|\w+|[^\w\s]", flags=re.UNICODE)

def classify_word_zipf(w):
    z = zipf_frequency(w.lower(), "en"); return "low" if z <= 3 else ("mid" if z <= 5 else "high")
def is_punct_token(tok):
    return bool(tok) and unicodedata.category(tok[0])[0] in ("P", "S") and all(unicodedata.category(ch)[0] in ("P", "S") for ch in tok)
def segment_class(seg):
    if seg.isspace(): return "whitespace"
    if re.fullmatch(r"\w+", seg): return classify_word_zipf(seg) if seg.isalpha() else "num_other"
    return "punct" if is_punct_token(seg) else "num_other"

def delete_by_w(text, w_dict, T, seed, lower=None):
    """Replica of the notebook's delete_by_w_length_budget_with_target (segment level, the routine that
    produced optimal_keep_*): per class, delete whole segments in random order until w * class length is
    reached; if the result keeps more than T, drop whitespace chars from the end until T; if it keeps less
    than `lower`, fall back to the pre-tweak text."""
    rng = random.Random(seed); segs = _SEG_RE.findall(text)
    items = defaultdict(list)
    for idx, seg in enumerate(segs): items[segment_class(seg)].append((idx, seg, len(seg)))
    dele = set()
    for c, its in items.items():
        w = max(0.0, min(1.0, float(w_dict.get(c, 0.0))))
        if w == 0.0: continue
        target = w * sum(l for _, _, l in its); its = its[:]; rng.shuffle(its); acc = 0
        for idx, seg, l in its:
            if acc >= target: break
            dele.add(idx); acc += l
    before = "".join(seg for i, seg in enumerate(segs) if i not in dele); out = before
    if len(out) / len(text) > T + 1e-6:
        need = len(out) - int(round(T * len(text)))
        if need > 0:
            chars = list(out); removed = 0
            for j in range(len(chars) - 1, -1, -1):
                if removed >= need: break
                if chars[j].isspace(): chars[j] = ""; removed += 1
            out = "".join(chars)
    if lower is not None and len(out) / len(text) < lower - 1e-6: return before
    return out

def kept_segments(text, skel):
    """Greedy alignment of original segments to a skeleton built by concatenating surviving segments."""
    segs = _SEG_RE.findall(text); pos = 0; kept = []
    for seg in segs:
        if skel.startswith(seg, pos): kept.append(True); pos += len(seg)
        else: kept.append(False)
    return segs, kept, pos == len(skel)

def solve(obj, b, p, T):
    a = 1.0 - b
    if obj == "sum_p":   return linprog(-(p * -a), A_ub=[-p], b_ub=[T - p.sum()], bounds=[(0, 1)] * 6, method="highs").x
    if obj == "sum_nop": return linprog(-(-a),     A_ub=[-p], b_ub=[T - p.sum()], bounds=[(0, 1)] * 6, method="highs").x
    f = lambda w: -np.sum(np.log(np.clip(1 - a * w, 1e-12, None)))
    best = None
    for x0 in (np.full(6, 0.5), np.full(6, 0.1), np.full(6, 0.9)):
        r = minimize(f, x0, bounds=[(0, 1)] * 6, constraints=[{"type": "ineq", "fun": lambda w: T - np.sum(p * (1 - w))}],
                     method="SLSQP", options={"ftol": 1e-12, "maxiter": 1000})
        if best is None or r.fun < best.fun: best = r
    return best.x

def band_T(rate):
    T = float(rate); return min(1.0, T + 0.02) if rate in ("0.3", "0.2", "0.1") else T
def band_lower(rate): return max(0.0, float(rate) - 0.02)

def load(n):
    df = pd.read_excel(XLSX, sheet_name="wf_optimal").dropna(subset=["Articles"]).reset_index(drop=True)
    return df.head(n) if n else df

def verify(df):
    ok = tot = aligned = 0; rel = []
    for i, row in df.iterrows():
        w_all = ast.literal_eval(str(row["w_optimal_cvx"]))
        for rate in RATES:
            col = f"optimal_keep_{int(round(float(rate) * 100))}"
            if not isinstance(row.get(col), str): continue
            segs, kept, full = kept_segments(row["Articles"], row[col]); tot += 1; aligned += full
            w = dict(zip(CLASSES, w_all[rate])); worst = 0.0
            for c in CLASSES:
                L = sum(len(s_) for s_ in segs if segment_class(s_) == c)
                K = sum(len(s_) for s_, k in zip(segs, kept) if k and segment_class(s_) == c)
                if L: worst = max(worst, abs(K / L - (1 - w[c])) if c != "whitespace" else 0.0)
            rel.append(worst); ok += worst <= 0.05
    print(f"verify: {aligned}/{tot} stored skeletons align exactly as concatenations of surviving segments; "
          f"{ok}/{tot} have every non-whitespace class kept within 5% of (1-w_k); median worst gap {np.median(rel):.3f}")

def agreement(df, tol=0.02):
    """Appendix P: share of (article, rate) cells where the LP (sum_nop) reproduces the product's allocation (prod),
    i.e. no w_k differs by more than tol; all articles of the sheet, nine rates. No API calls."""
    ok = tot = 0
    for _, row in df.iterrows():
        bd, pdict = ast.literal_eval(str(row["b_zipf_stats"])), ast.literal_eval(str(row["p_zipf_stats"]))
        b = np.array([bd[c] for c in CLASSES]); p = np.array([pdict[c] for c in CLASSES])
        for rate in ("0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1"):
            tot += 1; ok += np.max(np.abs(solve("sum_nop", b, p, float(rate)) - solve("prod", b, p, float(rate)))) <= tol
    print(f"LP (sum_nop) reproduces the product allocation (prod) within {tol} in {ok}/{tot} cells = {ok / tot:.1%} "
          f"({len(df)} articles x 9 rates)")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--agreement", action="store_true", help="Appendix P allocation agreement of the LP with the product objective (no API)")
    ap.add_argument("--workers", type=int, default=4); args = ap.parse_args()
    df = load(args.n)
    if args.verify: return verify(df)
    if args.agreement: return agreement(load(0))
    for line in ((Path.home() / ".env").read_text().splitlines() if (Path.home() / ".env").exists() else []):
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    sys.path.insert(0, str(ROOT / "harness")); import run_appendix_semantic_hybrid as base
    base.genai.configure(api_key=os.environ["GOOGLE_API_KEY"]); model = base.genai.GenerativeModel("gemini-2.5-flash")
    cfg = {"language": "en", "domain_label": "BBC news article"}
    done = {}
    if OUT.exists():
        for l in OUT.open():
            d = json.loads(l); done[(d["obj"], d["rate"], d["sample_idx"])] = d
    cache = {d["skeleton"]: d["reconstruction"] for d in done.values() if d["reconstruction"].strip()}
    items = []
    for i, row in df.iterrows():
        bd, pdict = ast.literal_eval(str(row["b_zipf_stats"])), ast.literal_eval(str(row["p_zipf_stats"]))
        b = np.array([bd[c] for c in CLASSES]); p = np.array([pdict[c] for c in CLASSES])
        for rate in RATES:
            for obj in OBJS:
                if (obj, rate, i) in done: continue
                w = solve(obj, b, p, float(rate))
                skel = delete_by_w(row["Articles"], dict(zip(CLASSES, w)), band_T(rate), seed=(123 * 1000003 + int(i) * 1009 + int(float(rate) * 100)), lower=band_lower(rate))
                items.append(dict(obj=obj, rate=rate, sample_idx=int(i), w=[round(float(x), 4) for x in w], original=row["Articles"], skeleton=skel))
    print(f"{len(items)} cells to decode ({len({it['skeleton'] for it in items})} distinct skeletons)", flush=True)
    lock = threading.Lock()
    def work(it):
        skel = it["skeleton"]
        with lock: rec = cache.get(skel)
        if rec is None:
            rec = ""
            for attempt in range(5):
                try: rec = model.generate_content(base.reconstruct_prompt(cfg, skel, len(it["original"])), request_options={"timeout": 120}).text.strip(); break
                except Exception as e: print(f"  attempt {attempt+1} failed: {str(e)[:80]}", flush=True); time.sleep(min(10, 2 + 2 * attempt))
            with lock: cache[skel] = rec
        with lock, OUT.open("a") as f:
            f.write(json.dumps(dict(**it, decoder_model="gemini-2.5-flash", reconstruction=rec)) + "\n")
    with ThreadPoolExecutor(args.workers) as pool: list(pool.map(work, items))
    print("done", flush=True)

if __name__ == "__main__":
    main()
