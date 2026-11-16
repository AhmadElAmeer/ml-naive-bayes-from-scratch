"""Experiments that produce docs/results.md."""
from __future__ import annotations

import math
import platform

from .compare import compare_with_sklearn
from .evaluate import POSITIVE, mean_std, score, stratified_folds, train_test_split
from .model import MODELS, MultinomialNB
from .text import Vocabulary, counts

K = 5
SEED = 1


def cross_validate(texts, labels, model="multinomial", alpha=1.0, simple=False, k=K, seed=SEED):
    docs = [counts(t, simple) for t in texts]
    folds = stratified_folds(labels, k, seed)
    per_fold, total = [], [0, 0, 0, 0]
    for f in range(k):
        test = folds[f]
        train = [i for g in range(k) if g != f for i in folds[g]]
        vocab = Vocabulary().fit([docs[i] for i in train])
        m = MODELS[model](alpha).fit([docs[i] for i in train], [labels[i] for i in train], vocab)
        s = score([labels[i] for i in test], m.predict([docs[i] for i in test]))
        per_fold.append(s)
        for j, v in enumerate((s.tp, s.fp, s.tn, s.fn)):
            total[j] += v
    return per_fold, total


def underflow_demo(texts, labels):
    """Multiply the raw probabilities for the longest message vs. add their logs."""
    docs = [counts(t) for t in texts]
    vocab = Vocabulary().fit(docs)
    m = MultinomialNB(1.0).fit(docs, labels, vocab)
    longest = max(range(len(docs)), key=lambda i: sum(vocab.filter(docs[i]).values()))
    doc = vocab.filter(docs[longest])
    n_tokens = sum(doc.values())
    product = math.exp(m.log_prior["ham"])
    for w, n in doc.items():
        for _ in range(n):
            product *= math.exp(m.log_likelihood["ham"][w])
    log_sum = m.joint_log_likelihood(docs[longest])["ham"]
    return n_tokens, product, log_sum


def report(texts, labels) -> str:
    import sklearn
    out = ["# Results", "",
           f"SMS Spam Collection: {len(texts):,} messages, {labels.count(POSITIVE):,} spam "
           f"({100 * labels.count(POSITIVE) / len(labels):.1f}%). "
           f"Python {platform.python_version()}, scikit-learn {sklearn.__version__} (used only for the comparison).", ""]

    # 1. equivalence with scikit-learn
    docs = [counts(t) for t in texts]
    tr, te = train_test_split(labels, 0.2, SEED)
    vocab = Vocabulary().fit([docs[i] for i in tr])
    out += ["## 1. Same answers as scikit-learn", "",
            f"Trained on {len(tr):,} messages, compared on the other {len(te):,}, using identical features "
            f"({len(vocab):,}-token vocabulary).", "",
            "| model | alpha | max difference in log-likelihood | identical predictions |", "|---|---:|---:|:---:|"]
    for alpha in (1.0, 0.1):
        res = compare_with_sklearn([docs[i] for i in tr], [labels[i] for i in tr], [docs[i] for i in te], vocab, alpha)
        for name, (diff, agree) in res.items():
            out.append(f"| {name} | {alpha} | {diff:.1e} | {'yes' if agree else '**no**'} |")
    out.append("")

    # 2. cross-validated quality
    out += [f"## 2. Quality ({K}-fold stratified cross-validation, alpha = 1)", "",
            "Mean ± standard deviation over the folds. Spam is the positive class.", "",
            "| model | tokenizer | accuracy | precision | recall | F1 | ham marked spam | spam missed |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    n_ham = labels.count("ham")
    out.append(f"| always say \"ham\" | — | {n_ham / len(labels):.3f} | — | 0.000 | 0.000 | 0 | {labels.count(POSITIVE)} |")
    for model in ("multinomial", "bernoulli"):
        for simple in (True, False):
            folds, (tp, fp, tn, fn) = cross_validate(texts, labels, model, 1.0, simple)
            cell = lambda attr: "{:.3f} ± {:.3f}".format(*mean_std([getattr(s, attr) for s in folds]))  # noqa: E731
            out.append(f"| {model} | {'plain words' if simple else 'words + number/price/URL tokens'} | "
                       f"{cell('accuracy')} | {cell('precision')} | {cell('recall')} | {cell('f1')} | {fp} | {fn} |")
    out.append("")

    # 3. smoothing
    out += ["## 3. Effect of the smoothing parameter alpha (multinomial, cross-validated)", "",
            "| alpha | precision | recall | F1 |", "|---:|---:|---:|---:|"]
    for alpha in (0.01, 0.1, 0.5, 1.0, 2.0, 5.0):
        folds, _ = cross_validate(texts, labels, "multinomial", alpha)
        out.append("| {} | {:.3f} | {:.3f} | {:.3f} |".format(
            alpha, *(mean_std([getattr(s, a) for s in folds])[0] for a in ("precision", "recall", "f1"))))
    out.append("")

    # 4. why logs
    n_tokens, product, log_sum = underflow_demo(texts, labels)
    out += ["## 4. Why the model works in log space", "",
            f"For the longest message ({n_tokens} known tokens), multiplying P(ham) by every P(token | ham) "
            f"gives `{product!r}` in 64-bit floating point. That is underflow, so every class would score zero. "
            f"Adding the logs instead gives `{log_sum:.1f}`, an ordinary number.", ""]

    # 5. what the model learned
    m = MultinomialNB(1.0).fit([docs[i] for i in tr], [labels[i] for i in tr], vocab)
    out += ["## 5. Tokens that most favour each class", "",
            "log P(token | class) − log P(token | other class), from the training split.", "",
            "| spam token | log ratio | ham token | log ratio |", "|---|---:|---|---:|"]
    spam_top, ham_top = m.top_tokens("spam", "ham", 12), m.top_tokens("ham", "spam", 12)
    for (ws, vs), (wh, vh) in zip(spam_top, ham_top):
        out.append(f"| `{ws}` | {vs:.2f} | `{wh}` | {vh:.2f} |")
    out.append("")

    # 6. mistakes
    pred = m.predict([docs[i] for i in te])
    false_pos = [texts[i] for i, p in zip(te, pred) if p == "spam" and labels[i] == "ham"]
    missed = [i for i, p in zip(te, pred) if p == "ham" and labels[i] == "spam"]
    strong = {w for w, _ in m.top_tokens("spam", "ham", 20)}
    no_strong = sum(1 for i in missed if not strong & set(vocab.filter(docs[i])))
    out += [f"## 6. Mistakes on the test split ({len(false_pos) + len(missed)} of {len(te):,})", "",
            f"**{len(false_pos)} real messages marked as spam** (shown in full):", ""]
    out += [f"- {t}" for t in false_pos]
    out += ["", f"**{len(missed)} spam messages missed.** {no_strong} of them contain none of the 20 tokens "
            "that most favour spam (no numbers, prices, URLs, 'claim', 'prize'...): they read like "
            "ordinary chat, which is the case a word-frequency model handles worst.", ""]
    return "\n".join(out)
