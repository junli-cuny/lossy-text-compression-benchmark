"""WordFreq-3 deletion rule of the word-level pipeline, copied verbatim from notebooks/bbc_word_level_pipeline_v1.ipynb
(counting code cells only: helpers from code cell 318, delete_zipf_strict from code cell 321, the version the
notebook's main loop in code cell 322 calls, returning (text, actual_keep, stats)). It produced the wf_3classes sheet
of the BBC workbook. Do not edit: analysis/run_r005_wordfreq3.py and analysis/run_table3_wordfreq_summ.py rely on it
reproducing that rule.
"""
import re, random, math
from wordfreq import zipf_frequency

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)

def classify_word_zipf(word):
    z = zipf_frequency(word.lower(), "en")
    if z <= 3:
        return "low"
    elif z <= 5:
        return "mid"
    else:
        return "high"


def delete_zipf_strict(text, keep_ratio=0.80, tol=0.02):
    tokens = tokenize(text)

    # Extract all words with classification
    indexed_words = []   # (token_index, word, class)
    for i, tok in enumerate(tokens):
        if tok.isalpha():
            cls = classify_word_zipf(tok)
            indexed_words.append((i, tok, cls))

    N = len(indexed_words)
    if N == 0:
        return text, 1.0, {"low":0, "mid":0, "high":0}

    # Zipf statistics (based on original text)
    count_low  = sum(1 for _,_,c in indexed_words if c == "low")
    count_mid  = sum(1 for _,_,c in indexed_words if c == "mid")
    count_high = sum(1 for _,_,c in indexed_words if c == "high")

    stats = {"low": count_low, "mid": count_mid, "high": count_high}

    # ---------------------------------------
    # Determine deletion count range
    # ---------------------------------------
    K = keep_ratio
    min_keep = max(0.0, K - tol)       # allowed lower bound for keep

    # Convert keep → delete count
    min_delete = math.ceil(N * (1 - K))           # cannot delete fewer than this
    max_delete = math.floor(N * (1 - min_keep))   # cannot delete more than this

    min_delete = max(0, min_delete)
    max_delete = min(N, max_delete)

    if min_delete > max_delete:
        delete_total = min_delete
    else:
        delete_total = random.randint(min_delete, max_delete)

    if delete_total <= 0:
        return text, 1.0, stats

    # ---------------------------------------
    # Assign deletion counts proportionally
    # ---------------------------------------
    desired_low  = delete_total * (count_low  / N)
    desired_mid  = delete_total * (count_mid  / N)
    desired_high = delete_total * (count_high / N)

    delete_counts = {}
    frac = {}

    def base_count(desired, cap):
        b = int(math.floor(desired))
        return min(b, cap), desired - b

    delete_counts["low"],  frac["low"]  = base_count(desired_low,  count_low)
    delete_counts["mid"],  frac["mid"]  = base_count(desired_mid,  count_mid)
    delete_counts["high"], frac["high"] = base_count(desired_high, count_high)

    # Distribute remaining quota
    assigned = sum(delete_counts.values())
    remaining = delete_total - assigned

    order = ["low", "mid", "high"]
    while remaining > 0:
        valid = [c for c in order if delete_counts[c] < stats[c]]
        if not valid:
            break
        best = max(valid, key=lambda x: frac[x])
        delete_counts[best] += 1
        remaining -= 1

    # ---------------------------------------
    # Actually delete words
    # ---------------------------------------
    low_idx  = [i for i,_,c in indexed_words if c == "low"]
    mid_idx  = [i for i,_,c in indexed_words if c == "mid"]
    high_idx = [i for i,_,c in indexed_words if c == "high"]

    delete_set = set()
    delete_set.update(random.sample(low_idx,  min(delete_counts["low"],  len(low_idx))))
    delete_set.update(random.sample(mid_idx,  min(delete_counts["mid"],  len(mid_idx))))
    delete_set.update(random.sample(high_idx, min(delete_counts["high"], len(high_idx))))

    # Rebuild text
    new_tokens = []
    for i, tok in enumerate(tokens):
        if i in delete_set:
            continue
        new_tokens.append(tok)

    # Detokenize
    out = []
    for i, tok in enumerate(new_tokens):
        if i == 0:
            out.append(tok)
        else:
            if re.match(r"[^\w\s]", tok):
                out.append(tok)
            else:
                out.append(" " + tok)

    actual_keep = (N - len(delete_set)) / N
    return "".join(out), actual_keep, stats
