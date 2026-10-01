"""Entity matching rule shared by every entity metric in the paper (entity recall, recovery rate,
unsupported-entity rate, skeleton survival).

An entity mention extracted by spaCy (en_core_web_sm) from one text counts as present in another text when
its lower-cased, whitespace-normalised string is a substring of the other text, or when the same string with
all spaces removed is a substring of the other text with spaces and newlines removed (deletion can glue
tokens together). The NER label is not used. Matching is lexical, so a distorted entity ("Kennedy" ->
"Kenedy") counts as absent and a short entity can match by accident inside a longer word.
"""


def norm_pair(text):
    """Lower-cased text and the same text with spaces and newlines removed."""
    low = text.lower()
    return low, low.replace(" ", "").replace("\n", "")


def check_preserved(entity_text, text_norm, text_nospace):
    """True if `entity_text` occurs in the text represented by norm_pair(text)."""
    e = " ".join(entity_text.lower().split())
    return e in text_norm or e.replace(" ", "") in text_nospace
