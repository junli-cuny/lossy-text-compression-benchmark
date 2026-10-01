"""Draft atomic facts for the human-study items with gemini-2.5-flash.

Facts are extracted from the ORIGINAL text only (never the reconstruction), so pre-filling them cannot leak the
condition or bias the judgment. e5_facts.json holds the verified facts the annotators were shown.
Output: e5_facts_drafted.json {item_id: [fact, ...]} (3-5 facts per item).
"""
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
import run_appendix_semantic_hybrid as base  # reuse genai config + retry

base.GEMINI_MODEL_NAME = "gemini-2.5-flash"
base.gemini = base.genai.GenerativeModel(base.GEMINI_MODEL_NAME)

OUT = HERE / "e5_facts_drafted.json"  # e5_facts.json holds the verified facts

PROMPT = """Extract 3 to 5 atomic factual statements from the news excerpt below.

Rules:
- Each fact must be a single, self-contained claim (one subject, one predicate).
- Prefer facts anchored on named entities, numbers, dates, places, or quantities.
- Use only information stated in the excerpt; no outside knowledge, no inference.
- Phrase each fact as a short declarative sentence.
- Return ONLY a JSON array of strings, nothing else.

Excerpt:
{text}"""


def parse_facts(raw):
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return None
    try:
        facts = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    facts = [f.strip() for f in facts if isinstance(f, str) and f.strip()]
    return facts[:5] if len(facts) >= 3 else None


def main():
    items = json.loads((HERE / "e5_items.json").read_text())
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    for it in items:
        key = str(it["id"])
        if key in done:
            continue
        raw = base.call_gemini(PROMPT.format(text=it["original"]))
        facts = parse_facts(raw)
        if facts is None:
            print(f"item {key}: PARSE FAIL, retrying once", flush=True)
            facts = parse_facts(base.call_gemini(PROMPT.format(text=it["original"])))
        if facts is None:
            print(f"item {key}: FAILED", flush=True)
            continue
        done[key] = facts
        OUT.write_text(json.dumps(done, indent=1, ensure_ascii=False))
        print(f"item {key}: {len(facts)} facts", flush=True)
        time.sleep(1.0)
    missing = [str(it["id"]) for it in items if str(it["id"]) not in done]
    print(f"DONE. {len(done)}/{len(items)} items, missing: {missing}")


if __name__ == "__main__":
    main()
