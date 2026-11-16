"""Command-line interface.

    python -m nbspam evaluate [--model bernoulli] [--alpha 0.5] [--simple-tokens]
    python -m nbspam predict "WINNER!! Claim your prize now, call 09061701461"
    python -m nbspam report --out docs/results.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .evaluate import load_tsv, mean_std
from .experiments import K, cross_validate, report
from .model import MODELS
from .text import Vocabulary, counts

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "sms.tsv"


def positive_float(text: str) -> float:
    v = float(text)
    if not v > 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return v


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="nbspam", description="Naive Bayes spam filter written from scratch.")
    p.add_argument("--data", default=str(DEFAULT_DATA), help="label<TAB>message file (default: data/sms.tsv)")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("evaluate", help=f"{K}-fold cross-validation")
    e.add_argument("--model", choices=MODELS, default="multinomial")
    e.add_argument("--alpha", type=positive_float, default=1.0)
    e.add_argument("--simple-tokens", action="store_true", help="plain words instead of number/price/URL tokens")

    pr = sub.add_parser("predict", help="train on all data and classify messages")
    pr.add_argument("messages", nargs="+")
    pr.add_argument("--model", choices=MODELS, default="multinomial")
    pr.add_argument("--alpha", type=positive_float, default=1.0)

    r = sub.add_parser("report", help="run every experiment and write Markdown")
    r.add_argument("--out", help="also write the report to this file")

    a = p.parse_args(argv)
    try:
        texts, labels = load_tsv(a.data)
    except OSError as exc:
        print(f"error: cannot read {a.data}: {exc.strerror}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if a.cmd == "evaluate":
        folds, (tp, fp, tn, fn) = cross_validate(texts, labels, a.model, a.alpha, a.simple_tokens)
        print(f"{a.model} NB, alpha={a.alpha}, {K}-fold cross-validation on {len(texts):,} messages")
        for name in ("accuracy", "precision", "recall", "f1"):
            m, s = mean_std([getattr(f, name) for f in folds])
            print(f"  {name:<10} {m:.3f} ± {s:.3f}")
        print(f"  confusion  spam caught {tp}, spam missed {fn}, ham marked spam {fp}, ham passed {tn}")
    elif a.cmd == "predict":
        docs = [counts(t) for t in texts]
        model = MODELS[a.model](a.alpha).fit(docs, labels, Vocabulary().fit(docs))
        for msg in a.messages:
            proba = model.predict_proba(counts(msg))
            label = model.predict_one(counts(msg))
            print(f"{label:<4}  P(spam) = {proba['spam']:.4f}   {msg}")
    else:
        text = report(texts, labels)
        print(text)
        if a.out:
            Path(a.out).write_text(text + "\n", encoding="utf-8")
    return 0
