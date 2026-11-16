"""Multinomial and Bernoulli Naive Bayes, written from the formulas.

Both pick the class c that maximises

    log P(c) + sum over features of log P(feature | c)

Everything is done with log probabilities: multiplying hundreds of small
probabilities underflows to 0.0 in floating point, adding their logs does not.
"""
from __future__ import annotations

import math
from collections import Counter

from .text import Vocabulary


class NaiveBayes:
    """Shared fitting of class priors; subclasses supply the likelihoods."""

    def __init__(self, alpha: float = 1.0):
        if not alpha > 0:
            raise ValueError("alpha (smoothing) must be > 0")
        self.alpha = alpha
        self.classes: list[str] = []
        self.log_prior: dict[str, float] = {}
        self.vocab: Vocabulary | None = None

    def fit(self, docs: list[Counter], labels: list[str], vocab: Vocabulary) -> "NaiveBayes":
        if len(docs) != len(labels) or not docs:
            raise ValueError("need the same, non-zero number of documents and labels")
        self.vocab = vocab
        self.classes = sorted(set(labels))
        n = len(labels)
        class_count = Counter(labels)
        self.log_prior = {c: math.log(class_count[c] / n) for c in self.classes}
        self._fit_likelihoods([vocab.filter(d) for d in docs], labels)
        return self

    def joint_log_likelihood(self, doc: Counter) -> dict[str, float]:
        raise NotImplementedError

    def predict_one(self, doc: Counter) -> str:
        jll = self.joint_log_likelihood(doc)
        # ties broken by class name so results are deterministic
        return max(self.classes, key=lambda c: (jll[c], c))

    def predict(self, docs: list[Counter]) -> list[str]:
        return [self.predict_one(d) for d in docs]

    def predict_proba(self, doc: Counter) -> dict[str, float]:
        """Normalise with the log-sum-exp trick so huge negative logs don't underflow."""
        jll = self.joint_log_likelihood(doc)
        top = max(jll.values())
        total = sum(math.exp(v - top) for v in jll.values())
        return {c: math.exp(v - top) / total for c, v in jll.items()}


class MultinomialNB(NaiveBayes):
    """Features are word counts.

    P(w | c) = (count of w in class c + alpha) / (total words in c + alpha * |V|)
    """

    def _fit_likelihoods(self, docs, labels):
        V = len(self.vocab)
        word_counts = {c: Counter() for c in self.classes}
        for d, y in zip(docs, labels):
            word_counts[y].update(d)
        self.log_likelihood = {}
        self.log_unseen = {}
        for c in self.classes:
            denom = sum(word_counts[c].values()) + self.alpha * V
            self.log_likelihood[c] = {w: math.log((word_counts[c][w] + self.alpha) / denom)
                                      for w in self.vocab.index}
            self.log_unseen[c] = math.log(self.alpha / denom)

    def joint_log_likelihood(self, doc):
        doc = self.vocab.filter(doc)
        return {c: self.log_prior[c] + sum(n * self.log_likelihood[c][w] for w, n in doc.items())
                for c in self.classes}

    def top_tokens(self, cls: str, other: str, k: int = 15) -> list[tuple[str, float]]:
        """Tokens whose presence most favours `cls` over `other` (log ratio)."""
        ratio = {w: self.log_likelihood[cls][w] - self.log_likelihood[other][w] for w in self.vocab.index}
        return sorted(ratio.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


class BernoulliNB(NaiveBayes):
    """Features are presence/absence of each vocabulary word.

    P(w present | c) = (docs in c containing w + alpha) / (docs in c + 2 * alpha)
    Absent words count too: they contribute log(1 - P(w present | c)).
    """

    def _fit_likelihoods(self, docs, labels):
        doc_freq = {c: Counter() for c in self.classes}
        n_docs = Counter(labels)
        for d, y in zip(docs, labels):
            doc_freq[y].update(d.keys())
        self.log_p = {}
        self.log_not_p = {}
        self.sum_log_not_p = {}
        for c in self.classes:
            p = {w: (doc_freq[c][w] + self.alpha) / (n_docs[c] + 2 * self.alpha) for w in self.vocab.index}
            self.log_p[c] = {w: math.log(v) for w, v in p.items()}
            self.log_not_p[c] = {w: math.log(1 - v) for w, v in p.items()}
            self.sum_log_not_p[c] = sum(self.log_not_p[c].values())

    def joint_log_likelihood(self, doc):
        present = self.vocab.filter(doc).keys()
        # start from "every word absent", then switch the present ones
        return {c: self.log_prior[c] + self.sum_log_not_p[c]
                + sum(self.log_p[c][w] - self.log_not_p[c][w] for w in present)
                for c in self.classes}


MODELS = {"multinomial": MultinomialNB, "bernoulli": BernoulliNB}
