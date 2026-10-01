#!/usr/bin/env python3
"""LLM-rater runner: Claude Opus 5 (Anthropic) and GPT-5.6 (OpenAI), same prompt, one request per item.

Subcommands
  pilot     --judge J [--batch 0]        synchronous run of one batch (default batch_000 = the 54 human-study
                                         items), writes annotations
  submit    --judge J [--batches 1 2 ..] submit pending batch files to the provider's batch API; state in state_J.json
  collect   --judge J                    poll submitted provider batches; write annotations_J/batch_NNN.json when complete
  calibrate --judge J                    agreement of annotations_J/batch_000 with the four human annotators
  status                                 what is done / pending per rater
Raters: claude (claude-opus-5), gpt (gpt-5.6-sol). Keys are read from the environment or ~/.env
(ANTHROPIC_API_KEY, OPENAI_API_KEY).
"""
import argparse, json, os, re, sys, time, itertools
from pathlib import Path

HERE = Path(__file__).resolve().parent
HUMAN_DIR = HERE.parent  # the human study (annotator1-4)
JUDGES = {
    "claude": {"provider": "anthropic", "model": "claude-opus-5"},
    "gpt": {"provider": "openai", "model": "gpt-5.6-sol"},
}

SYSTEM = """You are a careful human-like annotator performing a fact-level evaluation of reconstructed news text.
Work strictly blind: judge only from the two texts shown. Do not use outside knowledge about the news events,
and do not try to infer which system or setting produced the reconstruction.

You are given one item: an ORIGINAL news excerpt, a RECONSTRUCTION of it produced by a system, and 3-5 atomic
factual statements ("facts") taken from the original.

1. Judge each fact against the RECONSTRUCTION only:
   - "P" (preserved): the fact is recoverable from the reconstruction with meaning intact.
   - "D" (distorted): the fact appears in the reconstruction but with a wrong entity, number, or relation.
   - "M" (missing): the fact is absent from the reconstruction.
2. Decide "hallucinated_claim_present". This question is NARROW and most items are "N". Answer "Y" only if the
   reconstruction asserts a specific factual claim (a person, event, date, quantity, or place) that was INVENTED FROM
   NOTHING, with no anchor anywhere in the original. The test for every candidate claim: can you point to the sentence
   in the original that the claim came from? If yes, even with a wrong name, number, or relation, it is a DISTORTION
   (already captured by a "D" label), and the answer stays "N". A garbled or misspelled name, a changed number or date,
   a wrong attribution, a merged or reordered detail, an omitted qualifier, or a wrongly described role are all
   distortions, not hallucinations. Generic filler with no factual content never counts. Only when a claim has no
   source sentence at all (a new person, event, place, quantity, or date that the original never mentions in any form)
   answer "Y", and quote that claim in "hallucination_note". When in doubt, answer "N".
3. Judgments must be evidence-based: before assigning P, locate the supporting span in the reconstruction; before D,
   identify what changed; before M, verify the content is absent. Judge every fact independently; do not let one bad
   fact color the others.
4. If the reconstruction is empty, unreadable, or is just a copy of a corrupted skeleton, judge normally (facts will
   mostly be M) and set "notes" to "degenerate_output".

Return ONLY a JSON object, no prose and no code fences:
{"item_id": "<copied>", "facts": [{"fact_idx": 0, "label": "P|D|M", "evidence": "<=20 words: quoted span, or what changed, or 'absent'"}, ...],
 "hallucinated_claim_present": "Y|N", "hallucination_note": "<the invented claim, or ''>", "notes": ""}
Labels must be exactly P, D, or M; one entry per fact, in order; never add, drop, or reorder facts."""


def load_env():
    p = Path.home() / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def user_msg(item):
    facts = "\n".join(f"  [{i}] {f}" for i, f in enumerate(item["facts"]))
    return (f"item_id: {item['item_id']}\n\nORIGINAL:\n\"\"\"{item['original']}\"\"\"\n\n"
            f"RECONSTRUCTION:\n\"\"\"{item['reconstruction']}\"\"\"\n\nFACTS:\n{facts}\n\nReturn the JSON object now.")


def parse_judgment(text, item):
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.S)
    m = re.search(r"\{.*\}", raw, flags=re.S)
    obj = json.loads(m.group(0) if m else raw)
    obj["item_id"] = item["item_id"]
    facts = obj.get("facts", [])
    if len(facts) != len(item["facts"]):
        raise ValueError(f"expected {len(item['facts'])} facts, got {len(facts)}")
    for i, f in enumerate(facts):
        f["fact_idx"] = i
        lab = str(f.get("label", "")).strip().upper()[:1]
        if lab not in ("P", "D", "M"):
            raise ValueError(f"bad label {f.get('label')!r}")
        f["label"] = lab
    h = str(obj.get("hallucinated_claim_present", "")).strip().upper()[:1]
    if h not in ("Y", "N"):
        raise ValueError(f"bad halluc flag {obj.get('hallucinated_claim_present')!r}")
    obj["hallucinated_claim_present"] = h
    obj.setdefault("hallucination_note", "")
    obj.setdefault("notes", "")
    return obj


def batch_files():
    return sorted((HERE / "batches").glob("batch_*.json"))


def out_path(judge, bfile):
    d = HERE / f"annotations_{judge}"
    d.mkdir(exist_ok=True)
    return d / bfile.name


# ---------------- providers ----------------
def anthropic_params(model, item):
    return dict(model=model, max_tokens=4000, system=SYSTEM, output_config={"effort": "medium"},
                messages=[{"role": "user", "content": user_msg(item)}])


def openai_body(model, item):
    return dict(model=model, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(item)}],
                response_format={"type": "json_object"}, max_completion_tokens=4000)


def call_sync(judge, item, clients):
    cfg = JUDGES[judge]
    if cfg["provider"] == "anthropic":
        r = clients["anthropic"].messages.create(**anthropic_params(cfg["model"], item))
        if r.stop_reason == "refusal":
            raise RuntimeError("refusal")
        text = next(b.text for b in r.content if b.type == "text")
        usage = {"in": r.usage.input_tokens, "out": r.usage.output_tokens}
    else:
        r = clients["openai"].chat.completions.create(**openai_body(cfg["model"], item))
        text = r.choices[0].message.content
        usage = {"in": r.usage.prompt_tokens, "out": r.usage.completion_tokens}
    return parse_judgment(text, item), usage


def call_strict(judge, item, clients, n_prev):
    """Retry with an explicit reminder of the exact number of facts (used when the model split/merged facts or returned no JSON)."""
    cfg = JUDGES[judge]
    suffix = (f"\n\nIMPORTANT: there are exactly {len(item['facts'])} facts, indexed 0..{len(item['facts'])-1}. "
              f"Your previous answer had {n_prev} entries. Return exactly {len(item['facts'])} entries in the same order, "
              "one per input fact, and nothing else.")
    if cfg["provider"] == "anthropic":
        prm = anthropic_params(cfg["model"], item); prm["messages"][0]["content"] += suffix
        r = clients["anthropic"].messages.create(**prm)
        text = next(b.text for b in r.content if b.type == "text")
    else:
        body = openai_body(cfg["model"], item); body["messages"][1]["content"] += suffix
        r = clients["openai"].chat.completions.create(**body)
        text = r.choices[0].message.content or ""
    return parse_judgment(text, item)


def cmd_fix(args):
    """Resolve *.errors.json items with strict retries; write the batch, marking anything still failing as unresolved."""
    judge = args.judge; clients = make_clients(judge)
    for ef in sorted((HERE / f"annotations_{judge}").glob("batch_*.errors.json")):
        bname = ef.name.replace(".errors.json", ".json"); bfile = HERE / "batches" / bname
        if out_path(judge, bfile).exists():
            ef.unlink(); continue
        errors = json.load(open(ef)); items = {it["item_id"]: it for it in json.load(open(bfile))}
        # recover already-good judgments by re-collecting is expensive; instead re-judge only the failed ids and
        # re-run the normal collect afterwards (it re-reads provider results). Here: judge failed ids and store them in a side cache.
        side = HERE / f"annotations_{judge}" / f"{bfile.stem}.fixed.json"
        fixed = json.load(open(side)) if side.exists() else {}
        for cid, msg in errors.items():
            if cid in fixed: continue
            m = re.search(r"got (\d+)", msg); n_prev = int(m.group(1)) if m else 0
            for attempt in range(3):
                try:
                    fixed[cid] = call_strict(judge, items[cid], clients, n_prev); break
                except Exception as e:
                    msg = str(e); m = re.search(r"got (\d+)", msg); n_prev = int(m.group(1)) if m else n_prev
                    print(f"  {cid} strict attempt {attempt+1}: {e}")
            else:
                fixed[cid] = {"item_id": cid, "unresolved": True, "error": msg}
        json.dump(fixed, open(side, "w"), indent=1, ensure_ascii=False)
        print(f"{bname}: fixed {sum(1 for v in fixed.values() if not v.get('unresolved'))}/{len(errors)}; unresolved {sum(1 for v in fixed.values() if v.get('unresolved'))}")


def make_clients(judge):
    load_env()
    c = {}
    if JUDGES[judge]["provider"] == "anthropic":
        import anthropic
        c["anthropic"] = anthropic.Anthropic(max_retries=4)
    else:
        import openai
        c["openai"] = openai.OpenAI(max_retries=4)
    return c


# ---------------- commands ----------------
def cmd_pilot(args):
    judge = args.judge
    bfile = HERE / "batches" / f"batch_{args.batch:03d}.json"
    items = json.load(open(bfile))
    clients = make_clients(judge)
    out, fails, tin, tout = [], [], 0, 0
    t0 = time.time()
    for i, it in enumerate(items, 1):
        for attempt in range(3):
            try:
                j, u = call_sync(judge, it, clients)
                out.append(j); tin += u["in"]; tout += u["out"]
                break
            except Exception as e:
                print(f"  {it['item_id']} attempt {attempt+1}: {e}", file=sys.stderr)
                time.sleep(2)
        else:
            fails.append(it["item_id"])
        if i % 10 == 0:
            print(f"  {i}/{len(items)}  tokens in={tin} out={tout}", file=sys.stderr)
    json.dump(out, open(out_path(judge, bfile), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"judge": judge, "batch": bfile.name, "n": len(out), "failed": fails,
                      "tokens_in": tin, "tokens_out": tout, "per_item_in": round(tin / max(1, len(out))),
                      "per_item_out": round(tout / max(1, len(out))), "seconds": round(time.time() - t0)}, indent=1))


def state_path(judge):
    return HERE / f"state_{judge}.json"


def load_state(judge):
    return json.load(open(state_path(judge))) if state_path(judge).exists() else {}


def cmd_submit(args):
    judge = args.judge; cfg = JUDGES[judge]; clients = make_clients(judge)
    state = load_state(judge)
    todo = [b for b in batch_files() if not out_path(judge, b).exists() and b.name not in state]
    if args.batches:
        want = {f"batch_{int(x):03d}.json" for x in args.batches}
        todo = [b for b in todo if b.name in want]
    if not todo:
        print("nothing to submit"); return
    for b in todo:
        items = json.load(open(b))
        if cfg["provider"] == "anthropic":
            from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
            from anthropic.types.messages.batch_create_params import Request
            reqs = [Request(custom_id=it["item_id"], params=MessageCreateParamsNonStreaming(**anthropic_params(cfg["model"], it))) for it in items]
            mb = clients["anthropic"].messages.batches.create(requests=reqs)
            state[b.name] = {"provider_batch_id": mb.id, "n": len(items), "submitted": time.time()}
            print(f"submitted {b.name} -> anthropic {mb.id} ({len(items)} items)")
        else:
            jl = HERE / "tmp_openai" / f"{judge}_{b.stem}.jsonl"
            jl.parent.mkdir(exist_ok=True)
            with open(jl, "w") as f:
                for it in items:
                    f.write(json.dumps({"custom_id": it["item_id"], "method": "POST", "url": "/v1/chat/completions",
                                        "body": openai_body(cfg["model"], it)}) + "\n")
            up = clients["openai"].files.create(file=open(jl, "rb"), purpose="batch")
            ob = clients["openai"].batches.create(input_file_id=up.id, endpoint="/v1/chat/completions", completion_window="24h",
                                                  metadata={"batch": b.name, "judge": judge})
            state[b.name] = {"provider_batch_id": ob.id, "input_file_id": up.id, "n": len(items), "submitted": time.time()}
            print(f"submitted {b.name} -> openai {ob.id} ({len(items)} items)")
        json.dump(state, open(state_path(judge), "w"), indent=1)


def cmd_collect(args):
    judge = args.judge; cfg = JUDGES[judge]; clients = make_clients(judge)
    state = load_state(judge)
    for bname, st in list(state.items()):
        bfile = HERE / "batches" / bname
        if out_path(judge, bfile).exists():
            continue
        items = {it["item_id"]: it for it in json.load(open(bfile))}
        results, errors = {}, {}
        if cfg["provider"] == "anthropic":
            mb = clients["anthropic"].messages.batches.retrieve(st["provider_batch_id"])
            if mb.processing_status != "ended":
                print(f"{bname}: {mb.processing_status} ({mb.request_counts.processing} processing)"); continue
            for r in clients["anthropic"].messages.batches.results(st["provider_batch_id"]):
                it = items[r.custom_id]
                if r.result.type == "succeeded":
                    msg = r.result.message
                    try:
                        if msg.stop_reason == "refusal":
                            raise RuntimeError("refusal")
                        results[r.custom_id] = parse_judgment(next(b.text for b in msg.content if b.type == "text"), it)
                    except Exception as e:
                        errors[r.custom_id] = str(e)
                else:
                    errors[r.custom_id] = r.result.type
        else:
            ob = clients["openai"].batches.retrieve(st["provider_batch_id"])
            if ob.status != "completed":
                print(f"{bname}: {ob.status} ({ob.request_counts})");
                if ob.status in ("failed", "expired", "cancelled"):
                    print(f"  !! {bname} {ob.status}: {getattr(ob, 'errors', None)}; removing from state so it can be resubmitted")
                    del state[bname]; json.dump(state, open(state_path(judge), "w"), indent=1)
                continue
            content = clients["openai"].files.content(ob.output_file_id).text
            for line in content.splitlines():
                r = json.loads(line); cid = r["custom_id"]; it = items[cid]
                try:
                    if r.get("error"):
                        raise RuntimeError(str(r["error"]))
                    results[cid] = parse_judgment(r["response"]["body"]["choices"][0]["message"]["content"], it)
                except Exception as e:
                    errors[cid] = str(e)
            if ob.error_file_id:
                for line in clients["openai"].files.content(ob.error_file_id).text.splitlines():
                    r = json.loads(line); errors[r["custom_id"]] = str(r.get("error"))
        side = HERE / f"annotations_{judge}" / f"{bfile.stem}.fixed.json"
        if side.exists():
            for cid, v in json.load(open(side)).items():
                if cid in errors:
                    results[cid] = v; del errors[cid]
        if errors:
            print(f"{bname}: {len(results)} ok, {len(errors)} errors -> retrying those synchronously")
            for cid, e in list(errors.items()):
                try:
                    results[cid], _ = call_sync(judge, items[cid], clients); del errors[cid]
                except Exception as e2:
                    errors[cid] = str(e2)
        ordered = [results[it_id] for it_id in items if it_id in results]
        if errors:
            json.dump(errors, open(HERE / f"annotations_{judge}" / f"{bfile.stem}.errors.json", "w"), indent=1)
            print(f"{bname}: still {len(errors)} unresolved; partial output NOT written")
            continue
        json.dump(ordered, open(out_path(judge, bfile), "w"), indent=1, ensure_ascii=False)
        for extra in (HERE / f"annotations_{judge}" / f"{bfile.stem}.errors.json", side):
            if extra.exists(): extra.unlink()
        print(f"{bname}: written ({len(ordered)} items, {sum(1 for o in ordered if o.get('unresolved'))} unresolved)")


def cmd_status(args):
    bs = batch_files()
    for judge in JUDGES:
        done = sum(1 for b in bs if out_path(judge, b).exists())
        sub = len(load_state(judge))
        print(f"{judge}: {done}/{len(bs)} batches written; {sub} provider batches in state")


def kappa(a, b, labels):
    n = len(a)
    if n == 0: return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def cmd_calibrate(args):
    judge = args.judge
    mine = {j["item_id"]: j for j in json.load(open(HERE / f"annotations_{judge}" / "batch_000.json"))}
    humans = {f"annotator{i}": f"e5_annotations_annotator{i}.json" for i in (1, 2, 3, 4)}
    rows = []
    for name, fn in humans.items():
        h = json.load(open(HUMAN_DIR / fn))
        ann = h["annotations"] if isinstance(h, dict) and "annotations" in h else h
        # normalize: {item_id(str): {"facts":[labels], "hallucinated":Y/N}}
        H = {}
        for k, v in (ann.items() if isinstance(ann, dict) else ((str(x.get("item_id", x.get("id"))), x) for x in ann)):
            labs = v.get("facts") or v.get("labels") or v.get("fact_labels")
            if isinstance(labs, list) and labs and isinstance(labs[0], dict):
                labs = [str(x.get("label") or x.get("judgment") or "").upper()[:1] for x in labs]
            hal = str(v.get("hallucinated_claim_present") or v.get("hallucination") or v.get("halluc") or "").upper()[:1]
            H[str(k)] = {"facts": labs, "hal": hal}
        a, b, ha, hb = [], [], [], []
        for iid, j in mine.items():
            k = str(int(iid.split("_")[1]))
            if k not in H or not H[k]["facts"]: continue
            for f, hl in zip(j["facts"], H[k]["facts"]):
                if hl in ("P", "D", "M"):
                    a.append(f["label"]); b.append(hl)
            ha.append(j["hallucinated_claim_present"]); hb.append(H[k]["hal"])
        binA = ["P" if x == "P" else "N" for x in a]; binB = ["P" if x == "P" else "N" for x in b]
        rows.append((name, len(a), round(sum(x == y for x, y in zip(a, b)) / max(1, len(a)), 3), round(kappa(a, b, ["P", "D", "M"]), 3),
                     round(kappa(binA, binB, ["P", "N"]), 3), round(sum(x == y for x, y in zip(ha, hb)) / max(1, len(ha)), 3)))
    print(f"{judge} vs annotator: n_facts, raw agreement, kappa P/D/M, kappa binary(P vs not), halluc flag agreement")
    for r in rows: print("  ", r)
    from collections import Counter
    c = Counter(f["label"] for j in mine.values() for f in j["facts"])
    print("  label distribution:", dict(c), "| halluc Y:", sum(j["hallucinated_claim_present"] == "Y" for j in mine.values()), "/", len(mine))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pilot", "submit", "collect", "calibrate", "fix"):
        p = sub.add_parser(name); p.add_argument("--judge", required=True, choices=list(JUDGES))
        if name == "pilot": p.add_argument("--batch", type=int, default=0)
        if name == "submit": p.add_argument("--batches", nargs="*")
    sub.add_parser("status")
    args = ap.parse_args()
    {"pilot": cmd_pilot, "submit": cmd_submit, "collect": cmd_collect, "calibrate": cmd_calibrate, "status": cmd_status, "fix": cmd_fix}[args.cmd](args)
