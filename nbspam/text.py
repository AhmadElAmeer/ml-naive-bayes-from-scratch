"""Tokenizing and counting words. No libraries: a message becomes a
dict {token: count}, which is all Naive Bayes needs."""
from __future__ import annotations

import re
from collections import Counter

# Words, plus a few token types that matter for spam: phone numbers, prices,
# URLs. Numbers are grouped so "08452810075" and "09061701461" count as the
# same feature instead of two words seen once each.
_TOKEN = re.compile(r"https?://\S+|www\.\S+|£\d+(?:\.\d+)?|\$\d+(?:\.\d+)?|\d+|[a-z]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    out = []
    for tok in _TOKEN.findall(text.lower()):
        if tok.startswith(("http", "www.")):
            out.append("<url>")
        elif tok[0] in "£$":
            out.append("<money>")
        elif tok.isdigit():
            out.append("<longnum>" if len(tok) >= 5 else "<num>")
        else:
            out.append(tok)
    return out


_SIMPLE = re.compile(r"[a-z0-9]+")


def tokenize_simple(text: str) -> list[str]:
    """Baseline for comparison: plain lowercase words and digit runs."""
    return _SIMPLE.findall(text.lower())


def counts(text: str, simple: bool = False) -> Counter:
    return Counter(tokenize_simple(text) if simple else tokenize(text))


class Vocabulary:
    """Token -> column index, built from training data only."""

    def __init__(self, min_count: int = 1):
        if min_count < 1:
            raise ValueError("min_count must be at least 1")
        self.min_count = min_count
        self.index: dict[str, int] = {}

    def fit(self, docs: list[Counter]) -> "Vocabulary":
        total = Counter()
        for d in docs:
            total.update(d)
        words = sorted(w for w, c in total.items() if c >= self.min_count)
        self.index = {w: i for i, w in enumerate(words)}
        return self

    def __len__(self) -> int:
        return len(self.index)

    def filter(self, doc: Counter) -> dict[str, int]:
        """Drop tokens never seen in training (they carry no information)."""
        return {w: c for w, c in doc.items() if w in self.index}
