#!/usr/bin/env python3
"""Evaluate QLoRA decompression adapters (Llama-3.2-3B-Instruct) on their held-out test splits.

For every directory under --adapter-root that contains adapter_config.json we
  * infer the retention ratio from the directory name (…_<ratio>_lr…),
  * locate the matching test JSON (messages format) in the same parent folder,
  * reconstruct the original text from the corrupted skeleton with base+adapter,
  * score with BERTScore F1 (distilroberta-base, as in the paper) and ROUGE-L.
Optionally (--include-base) the un-adapted base model is scored on the same inputs.

Outputs: <out_dir>/summary.json and <out_dir>/<adapter_name>[__base].jsonl per-sample records.
"""
import argparse, json, os, re, sys, time, glob
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_adapters(root):
    out = []
    for cfg in sorted(glob.glob(os.path.join(root, "**", "adapter_config.json"), recursive=True)):
        out.append(Path(cfg).parent)
    return out


def infer_ratio(name):
    m = re.search(r"_(\d{1,3})_lr", name)
    if m:
        return int(m.group(1))
    nums = re.findall(r"(\d{1,3})", name)
    return int(nums[-1]) if nums else None


def find_test_json(adapter_dir, ratio):
    parent = adapter_dir.parent
    cands = sorted(glob.glob(str(parent / f"*test*_{ratio}.json")))
    if not cands:
        cands = [p for p in glob.glob(str(parent / f"*_{ratio}.json"))
                 if not re.search(r"train|val", os.path.basename(p))]
    if not cands:
        cands = sorted(glob.glob(str(parent / "**" / f"*test*_{ratio}.json"), recursive=True))
    return cands[0] if cands else None


def load_pairs(path):
    data = json.load(open(path))
    pairs = []
    for ex in data:
        msgs = ex["messages"] if isinstance(ex, dict) and "messages" in ex else ex
        user = next(m["content"] for m in msgs if m["role"] == "user")
        ref = next(m["content"] for m in msgs if m["role"] == "assistant")
        pairs.append((user, ref))
    return pairs


def load_base(base, quant, dtype="float16"):
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt = getattr(torch, dtype)
    kw = dict(device_map="cuda", dtype=dt)
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dt, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base, **kw)
    model.eval()
    return tok, model


@torch.no_grad()
def _gen_batch(model, tok, chunk, refs_chunk, max_new_cap):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True) for p in chunk]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    ref_len = max(len(tok(r).input_ids) for r in refs_chunk)
    max_new = min(max_new_cap, int(ref_len * 1.3) + 32)
    gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                         pad_token_id=tok.pad_token_id)
    return [tok.decode(row, skip_special_tokens=True).strip() for row in gen[:, enc.input_ids.shape[1]:]]


@torch.no_grad()
def generate(model, tok, prompts, refs, bs, max_new_cap):
    """Batched greedy generation; on CUDA OOM the batch is split in half and retried."""
    outs = []
    i = 0
    while i < len(prompts):
        cur = bs
        while True:
            try:
                outs.extend(_gen_batch(model, tok, prompts[i:i + cur], refs[i:i + cur], max_new_cap))
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if cur == 1:
                    raise
                cur = max(1, cur // 2)
                print(f"    OOM -> retry with batch {cur}", flush=True)
        i += cur
        print(f"    {min(i, len(prompts))}/{len(prompts)}", flush=True)
    return outs


def score(preds, refs, device):
    from bert_score import score as bs
    from rouge_score import rouge_scorer
    P, R, F = bs(preds, refs, lang="en", model_type="distilroberta-base", device=device, verbose=False)
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl = [rs.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    return F.tolist(), rl


def run_one(tag, model, tok, pairs, args, out_dir):
    prompts = [p for p, _ in pairs]
    refs = [r for _, r in pairs]
    t0 = time.time()
    preds = generate(model, tok, prompts, refs, args.batch_size, args.max_new_tokens)
    gen_s = time.time() - t0
    f1, rl = score(preds, refs, "cuda")
    with open(out_dir / f"{tag}.jsonl", "w") as f:
        for p, r, y, a, b in zip(prompts, refs, preds, f1, rl):
            f.write(json.dumps({"prompt": p, "reference": r, "prediction": y,
                                "bertscore_f1": a, "rougeL": b}, ensure_ascii=False) + "\n")
    mean = lambda xs: sum(xs) / len(xs)
    std = lambda xs: (sum((x - mean(xs)) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5
    res = {"n": len(preds), "bertscore_f1_mean": mean(f1), "bertscore_f1_std": std(f1),
           "rougeL_mean": mean(rl), "rougeL_std": std(rl), "gen_seconds": gen_s}
    print(f"  {tag}: BERTScore F1 {res['bertscore_f1_mean']:.4f}  ROUGE-L {res['rougeL_mean']:.4f}  n={res['n']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-root", required=True)
    ap.add_argument("--base", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--quant", choices=["4bit", "bf16"], default="4bit", help="bf16 = unquantized base in --dtype")
    ap.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16", help="Turing GPUs lack native bf16; fp16 is much faster there")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-test", type=int, default=0, help="0 = all")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--include-base", action="store_true")
    ap.add_argument("--only", default="", help="substring filter on adapter dir name")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}

    adapters = [a for a in find_adapters(args.adapter_root) if args.only in a.name]
    jobs = []
    for a in adapters:
        ratio = infer_ratio(a.name)
        tj = find_test_json(a, ratio)
        if tj is None:
            print(f"!! no test json for {a} (ratio {ratio}); skipping", file=sys.stderr)
            continue
        jobs.append((a, ratio, tj))
    print(f"{len(jobs)} adapter jobs:")
    for a, r, tj in jobs:
        print(f"  {a.name}  ratio={r}  test={os.path.relpath(tj, args.adapter_root)}")

    tok, model = load_base(args.base, args.quant, args.dtype)
    from peft import PeftModel
    peft_model = None
    for a, ratio, tj in jobs:
        pairs = load_pairs(tj)
        if args.n_test:
            pairs = pairs[:args.n_test]
        key = a.name
        if key in summary and summary[key].get("n") == len(pairs):
            print(f"skip {key} (done)"); continue
        print(f"== {key} (ratio {ratio}, {len(pairs)} test items)", flush=True)
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, str(a), adapter_name=key)
        else:
            peft_model.load_adapter(str(a), adapter_name=key)
        peft_model.set_adapter(key)
        peft_model.eval()
        res = run_one(key, peft_model, tok, pairs, args, out_dir)
        res.update({"adapter_dir": str(a), "test_json": tj, "ratio": ratio, "quant": args.quant, "dtype": args.dtype,
                    "base": args.base, "decoder": "sft"})
        summary[key] = res
        json.dump(summary, open(summary_path, "w"), indent=2)

    if args.include_base:
        import contextlib
        ctx = peft_model.disable_adapter() if peft_model is not None else contextlib.nullcontext()
        gen_model = peft_model if peft_model is not None else model
        with ctx:
            for a, ratio, tj in jobs:
                pairs = load_pairs(tj)
                if args.n_test:
                    pairs = pairs[:args.n_test]
                key = a.name + "__base"
                if key in summary and summary[key].get("n") == len(pairs):
                    print(f"skip {key} (done)"); continue
                print(f"== {key} (base model, ratio {ratio}, {len(pairs)} items)", flush=True)
                res = run_one(key, gen_model, tok, pairs, args, out_dir)
                res.update({"test_json": tj, "ratio": ratio, "quant": args.quant, "dtype": args.dtype, "base": args.base, "decoder": "base"})
                summary[key] = res
                json.dump(summary, open(summary_path, "w"), indent=2)

    print("\nSummary:")
    for k, v in sorted(summary.items(), key=lambda kv: (kv[1].get("decoder"), kv[1].get("ratio") or 0)):
        print(f"  {k:60s} r={v.get('ratio')}  BERTScore {v['bertscore_f1_mean']:.4f}  ROUGE-L {v['rougeL_mean']:.4f}  n={v['n']}")


if __name__ == "__main__":
    main()
