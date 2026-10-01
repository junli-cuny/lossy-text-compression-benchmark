#!/usr/bin/env python3
"""E3 at r_keep = 0.9 and 0.1, the two rates run_e3_llmlingua.py does not cover.

Same 100 chunks (checked byte-identical against samples/llmlingua2_samples.jsonl), compressor settings, prompt and
decoder (gemini-2.5-flash) as run_e3_llmlingua.py. Writes samples/llmlingua2_skeletons_extra_rates.jsonl and
samples/llmlingua2_samples_extra_rates.jsonl.

Usage:
  python analysis/run_e3_llmlingua_extra_rates.py --skeletons-only   # local, no API calls
  python analysis/run_e3_llmlingua_extra_rates.py                    # Gemini reconstruction, resumable
"""
import argparse, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RATES = [0.9, 0.1]
ORIG_SAMPLES = Path(__file__).resolve().parents[1] / "samples" / "llmlingua2_samples.jsonl"
SKELETONS_PATH = Path(__file__).resolve().parents[1] / "samples" / "llmlingua2_skeletons_extra_rates.jsonl"
SAMPLES_PATH = Path(__file__).resolve().parents[1] / "samples" / "llmlingua2_samples_extra_rates.jsonl"
DECODER = "gemini-2.5-flash"


def load_env():
    p = Path.home() / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_bbc_chunks(n=100, size=512):
    """Verbatim copy of run_e3_llmlingua.load_bbc_chunks (same workbook, same walk order)."""
    import openpyxl
    wb = openpyxl.load_workbook(str(REPO / "data" / "BBC_news_latest.xlsx"), read_only=True)
    ws = wb["datasource"]
    chunks = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        article = str(row[2]) if row[2] else ""
        if not article or len(article) < 100:
            continue
        for i in range(0, len(article), size):
            c = article[i:i + size]
            if len(c) >= 100:
                chunks.append(c)
            if len(chunks) >= n:
                break
        if len(chunks) >= n:
            break
    wb.close()
    return chunks


def verify_chunks(chunks):
    """Abort unless our chunks are byte-identical to the ones the original run used."""
    ref = {}
    with ORIG_SAMPLES.open() as f:
        for line in f:
            r = json.loads(line)
            ref.setdefault(r["sample_idx"], r["reference"])
    bad = [i for i, c in enumerate(chunks) if i in ref and ref[i] != c]
    print(f"chunk check: {len(chunks)} loaded, {len(ref)} in original samples, {len(bad)} mismatched")
    if bad:
        sys.exit(f"ABORT: chunks differ from the original run at indices {bad[:5]}")


def reconstruct_prompt(degraded_text, original_len):
    """Verbatim copy of run_appendix_semantic_hybrid.reconstruct_prompt (en branch)."""
    return (
        "You are a text reconstruction assistant. The input is a BBC news article excerpt "
        "from which some tokens have been deleted. Reconstruct the original text as faithfully as possible. "
        "Do not add new facts, names, or events not suggested by the input. "
        f"The output should be approximately the same length (~{original_len} characters). "
        "Return only the reconstructed text.\n\n"
        f"Degraded text:\n{degraded_text}"
    )


def build_skeletons():
    from llmlingua import PromptCompressor
    chunks = load_bbc_chunks()
    verify_chunks(chunks)
    print("Loading LLMLingua-2 compressor...", flush=True)
    comp = PromptCompressor(model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                            use_llmlingua2=True, device_map="cpu")
    done = set()
    if SKELETONS_PATH.exists():
        for line in SKELETONS_PATH.open():
            r = json.loads(line)
            done.add((r["rkeep"], r["sample_idx"]))
    t0 = time.time()
    with SKELETONS_PATH.open("a") as f:
        for rate in RATES:
            for i, chunk in enumerate(chunks):
                if (str(rate), i) in done:
                    continue
                res = comp.compress_prompt(chunk, rate=rate, force_tokens=["\n"])
                skel = res["compressed_prompt"]
                f.write(json.dumps({"method": "llmlingua2", "rkeep": str(rate), "sample_idx": i,
                                    "reference": chunk, "degraded": skel,
                                    "actual_ratio": round(len(skel) / len(chunk), 4)}) + "\n")
                f.flush()
            print(f"rate {rate} done ({time.time() - t0:.0f}s)", flush=True)


def reconstruct(sleep_s=0.6):
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(DECODER)

    def call(prompt, max_retries=5):
        for attempt in range(max_retries):
            try:
                return model.generate_content(prompt).text.strip()
            except Exception as e:
                print(f"  Gemini attempt {attempt + 1} failed: {e}", flush=True)
                if attempt < max_retries - 1:
                    time.sleep(min(10, 2 + 2 * attempt))
        return ""

    done = set()
    if SAMPLES_PATH.exists():
        for line in SAMPLES_PATH.open():
            r = json.loads(line)
            done.add((r["rkeep"], r["sample_idx"]))
    skels = [json.loads(l) for l in SKELETONS_PATH.open()]
    todo = [s for s in skels if (s["rkeep"], s["sample_idx"]) not in done]
    print(f"{len(skels)} skeletons, {len(done)} already reconstructed, {len(todo)} to go", flush=True)
    with SAMPLES_PATH.open("a") as f:
        for n, s in enumerate(todo, 1):
            rec = call(reconstruct_prompt(s["degraded"], len(s["reference"])))
            out = dict(s)
            out.update({"dataset": "bbc", "language": "en", "decoder_model": DECODER,
                        "reconstruction": rec})
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            if n % 20 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)
            time.sleep(sleep_s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeletons-only", action="store_true")
    a = ap.parse_args()
    load_env()
    build_skeletons()
    if not a.skeletons_only:
        reconstruct()
