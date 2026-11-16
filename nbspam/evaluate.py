"""Data loading, splitting and metrics, all from scratch."""
from __future__ import annotations

import csv
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

POSITIVE = "spam"


def load_tsv(path: str | Path) -> tuple[list[str], list[str]]:
    """Read `label<TAB>message` lines (the SMS Spam Collection format)."""
    texts, labels = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for lineno, row in enumerate(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE), 1):
            if not row or not "".join(row).strip():
                continue
            if len(row) < 2 or row[0] not in ("ham", "spam"):
                raise ValueError(f"{path}:{lineno}: expected 'ham' or 'spam', a tab, then the message")
            labels.append(row[0])
            texts.append("\t".join(row[1:]))
    if not texts:
        raise ValueError(f"{path}: no messages found")
    return texts, labels


def stratified_folds(labels: list[str], k: int, seed: int) -> list[list[int]]:
    """Split indices into k folds with the same class ratio in each."""
    if k < 2:
        raise ValueError("k must be at least 2")
    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in sorted(set(labels)):
        idx = [i for i, y in enumerate(labels) if y == cls]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(i)
    return [sorted(f) for f in folds]


def train_test_split(labels: list[str], test_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    k = round(1 / test_fraction)
    folds = stratified_folds(labels, k, seed)
    test = folds[0]
    train = sorted(i for f in folds[1:] for i in f)
    return train, test


@dataclass
class Scores:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def score(truth: list[str], predicted: list[str], positive: str = POSITIVE) -> Scores:
    if len(truth) != len(predicted):
        raise ValueError("truth and predictions differ in length")
    tp = sum(t == positive and p == positive for t, p in zip(truth, predicted))
    fp = sum(t != positive and p == positive for t, p in zip(truth, predicted))
    fn = sum(t == positive and p != positive for t, p in zip(truth, predicted))
    tn = len(truth) - tp - fp - fn
    return Scores(tp, fp, tn, fn)


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)
