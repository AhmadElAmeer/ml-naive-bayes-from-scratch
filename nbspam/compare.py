"""Check the from-scratch models against scikit-learn on identical features.

This module is the only place scikit-learn is used, and only for checking.
"""
from __future__ import annotations

from collections import Counter

from .model import BernoulliNB, MultinomialNB
from .text import Vocabulary


def to_matrix(docs: list[Counter], vocab: Vocabulary):
    from scipy.sparse import csr_matrix
    rows, cols, vals = [], [], []
    for r, d in enumerate(docs):
        for w, n in vocab.filter(d).items():
            rows.append(r)
            cols.append(vocab.index[w])
            vals.append(n)
    return csr_matrix((vals, (rows, cols)), shape=(len(docs), len(vocab)), dtype=float)


def compare_with_sklearn(train_docs, train_labels, test_docs, vocab, alpha=1.0):
    """Return {model: (max |difference| in joint log-likelihood, predictions agree?)}."""
    from sklearn.naive_bayes import BernoulliNB as SkB, MultinomialNB as SkM
    Xtr, Xte = to_matrix(train_docs, vocab), to_matrix(test_docs, vocab)
    out = {}
    pairs = [("multinomial", MultinomialNB(alpha), SkM(alpha=alpha, force_alpha=True)),
             ("bernoulli", BernoulliNB(alpha), SkB(alpha=alpha, force_alpha=True, binarize=0.0))]
    for name, ours, theirs in pairs:
        ours.fit(train_docs, train_labels, vocab)
        theirs.fit(Xtr, train_labels)
        sk_jll = theirs.predict_joint_log_proba(Xte)
        worst = 0.0
        for i, d in enumerate(test_docs):
            jll = ours.joint_log_likelihood(d)
            for j, c in enumerate(theirs.classes_):
                worst = max(worst, abs(jll[c] - sk_jll[i, j]))
        agree = ours.predict(test_docs) == list(theirs.predict(Xte))
        out[name] = (worst, agree)
    return out
