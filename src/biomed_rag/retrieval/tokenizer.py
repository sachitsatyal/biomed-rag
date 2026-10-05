"""Tokenizer for BM25: simple, predictable and biomedical-aware."""

import re
import unicodedata

# Papers write Greek letters; users type their names. Map both to the same token.
GREEK = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "κ": "kappa",
    "τ": "tau",
}

STOPWORDS = frozenset(
    """a an and are as at be by for from has have in into is it its of on or
    that the their this to was were which with we our these those than""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, spell out Greek letters, split on non-alphanumerics, drop stopwords."""
    text = unicodedata.normalize("NFKC", text).lower()
    for symbol, name in GREEK.items():
        text = text.replace(symbol, f" {name} ")
    return [token for token in _TOKEN.findall(text) if token not in STOPWORDS]
