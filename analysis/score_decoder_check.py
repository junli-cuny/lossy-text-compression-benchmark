"""Score the decoder version check (Table 16): the published Gemini 2.0 Flash outputs (workbook polished_*)
vs the gemini-2.5-flash re-decoding of the same WordFreq-3 skeletons (samples/bbc_wordfreq3_decoder_check_samples.jsonl).
Metrics per rate and decoder: BERTScore F1 (distilroberta-base), ROUGE-L F, chrF++, CER (Levenshtein distance between
the whitespace-stripped texts over the stripped original length; the workbook's CER columns use the unstripped
text), entity recall, unsupported-entity rate, reconstruction/original length ratio. Writes results/decoder_check_wordfreq3_scores.json and prints a markdown table.
"""
import json
from collections import defaultdict
from pathlib import Path
import Levenshtein, spacy
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from sacrebleu.metrics import CHRF
from entity_match import check_preserved, norm_pair

HERE = Path(__file__).resolve().parent
recs = [json.loads(l) for l in (Path(__file__).resolve().parents[1] / "samples" / "bbc_wordfreq3_decoder_check_samples.jsonl").open()]
recs = [r for r in recs if r["reconstruction"].strip() and isinstance(r["published_polished"], str)]
pairs = []   # (rate, decoder, original, reconstruction)
for r in recs:
    pairs.append((r["rate"], "2.0 Flash (published)", r["original"], r["published_polished"]))
    pairs.append((r["rate"], "2.5 Flash (re-run)", r["original"], r["reconstruction"]))
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
origs = list({p[2]: None for p in pairs}); ents = {}
for doc, k in zip(nlp.pipe(origs, batch_size=64), origs): ents[k] = [e.text for e in doc.ents]
rents = [[e.text for e in d.ents] for d in nlp.pipe([p[3] for p in pairs], batch_size=64)]
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True); chrf = CHRF(word_order=2)
F = {}
for name, mt in (("bert_dr", "distilroberta-base"),):
    _, _, f = BERTScorer(model_type=mt, lang="en").score([p[3] for p in pairs], [p[2] for p in pairs], batch_size=32); F[name] = f.numpy()
strip = lambda t: "".join(t.split())
agg = defaultdict(lambda: defaultdict(list)); cnt = defaultdict(lambda: dict(orig=0, rec=0, rents=0, un=0))
for i, ((rate, dec, o, h), re_) in enumerate(zip(pairs, rents)):
    k = (rate, dec); a = agg[k]; c = cnt[k]
    a["bert_dr"].append(float(F["bert_dr"][i]))
    a["rougeL"].append(rouge.score(o, h)["rougeL"].fmeasure); a["chrfpp"].append(chrf.sentence_score(h, [o]).score / 100)
    a["cer"].append(Levenshtein.distance(strip(o), strip(h)) / max(1, len(strip(o)))); a["len"].append(len(h) / len(o))
    hn, hs = norm_pair(h); on, os_ = norm_pair(o)
    for e in ents[o]: c["orig"] += 1; c["rec"] += check_preserved(e, hn, hs)
    for e in re_: c["rents"] += 1; c["un"] += not check_preserved(e, on, os_)
rows = []
for k in sorted(agg, key=lambda k: (-k[0], k[1])):
    a, c = agg[k], cnt[k]
    rows.append(dict(rkeep=k[0] / 100, decoder=k[1], n=len(a["bert_dr"]), **{m: round(sum(v) / len(v), 4) for m, v in a.items()},
                     entity_recall=round(c["rec"] / c["orig"], 4), unsupported_rate=round(c["un"] / c["rents"], 4)))
(Path(__file__).resolve().parents[1] / "results" / "decoder_check_wordfreq3_scores.json").write_text(json.dumps(rows, indent=1))
hdr = ["rkeep", "decoder", "n", "bert_dr", "rougeL", "chrfpp", "cer", "entity_recall", "unsupported_rate", "len"]
print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
for r in rows: print("| " + " | ".join(str(r[h]) for h in hdr) + " |")
