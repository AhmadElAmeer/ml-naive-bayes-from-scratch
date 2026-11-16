import math
from collections import Counter
from fractions import Fraction as F

import pytest

from nbspam.cli import main
from nbspam.evaluate import load_tsv, score, stratified_folds, train_test_split
from nbspam.model import BernoulliNB, MultinomialNB
from nbspam.text import Vocabulary, counts, tokenize, tokenize_simple

# Manning, Raghavan & Schütze, "Introduction to Information Retrieval",
# Example 13.1 (multinomial) and 13.2 (Bernoulli): the textbook gives the
# exact probabilities, so the implementation can be checked by hand.
IIR_TRAIN = ["Chinese Beijing Chinese", "Chinese Chinese Shanghai", "Chinese Macao", "Tokyo Japan Chinese"]
IIR_LABELS = ["c", "c", "c", "j"]
IIR_TEST = "Chinese Chinese Chinese Tokyo Japan"


def iir_model(cls):
    docs = [counts(t) for t in IIR_TRAIN]
    return cls(1.0).fit(docs, IIR_LABELS, Vocabulary().fit(docs))


def test_multinomial_matches_textbook_example():
    m = iir_model(MultinomialNB)
    ll = {c: {w: math.exp(v) for w, v in m.log_likelihood[c].items()} for c in "cj"}
    assert ll["c"]["chinese"] == pytest.approx(3 / 7)
    assert ll["c"]["tokyo"] == pytest.approx(1 / 14) == ll["c"]["japan"]
    assert ll["j"]["chinese"] == pytest.approx(2 / 9) == ll["j"]["tokyo"]
    jll = m.joint_log_likelihood(counts(IIR_TEST))
    expected_c = F(3, 4) * F(3, 7) ** 3 * F(1, 14) ** 2          # ≈ 0.0003
    expected_j = F(1, 4) * F(2, 9) ** 3 * F(2, 9) ** 2           # ≈ 0.0001
    assert math.exp(jll["c"]) == pytest.approx(float(expected_c))
    assert math.exp(jll["j"]) == pytest.approx(float(expected_j))
    assert m.predict_one(counts(IIR_TEST)) == "c"


def test_bernoulli_matches_textbook_example_and_disagrees():
    m = iir_model(BernoulliNB)
    p = {c: {w: math.exp(v) for w, v in m.log_p[c].items()} for c in "cj"}
    assert p["c"]["chinese"] == pytest.approx(4 / 5) and p["c"]["japan"] == pytest.approx(1 / 5)
    assert p["c"]["beijing"] == pytest.approx(2 / 5)
    assert p["j"]["chinese"] == pytest.approx(2 / 3) and p["j"]["beijing"] == pytest.approx(1 / 3)
    jll = m.joint_log_likelihood(counts(IIR_TEST))
    expected_c = F(3, 4) * F(4, 5) * F(1, 5) * F(1, 5) * (1 - F(2, 5)) ** 3   # ≈ 0.005
    expected_j = F(1, 4) * F(2, 3) ** 3 * (1 - F(1, 3)) ** 3                 # ≈ 0.022
    assert math.exp(jll["c"]) == pytest.approx(float(expected_c))
    assert math.exp(jll["j"]) == pytest.approx(float(expected_j))
    # The two models disagree on this document, exactly as the textbook says.
    assert m.predict_one(counts(IIR_TEST)) == "j"


# ---- model behaviour -----------------------------------------------------

def test_unseen_words_are_ignored_and_smoothing_avoids_zero():
    m = iir_model(MultinomialNB)
    assert m.joint_log_likelihood(counts("Chinese zebra")) == m.joint_log_likelihood(counts("Chinese"))
    # 'Japan' never appears in class c, but smoothing keeps its probability above zero
    assert math.isfinite(m.joint_log_likelihood(counts("Japan"))["c"])


def test_predict_proba_is_normalised_and_stable_for_long_messages():
    m = iir_model(MultinomialNB)
    proba = m.predict_proba(counts(" ".join(["Tokyo Japan"] * 2000)))
    assert sum(proba.values()) == pytest.approx(1.0)
    assert proba["j"] > 0.999 and all(math.isfinite(v) for v in proba.values())


def test_invalid_arguments():
    with pytest.raises(ValueError):
        MultinomialNB(alpha=0)
    with pytest.raises(ValueError):
        MultinomialNB().fit([], [], Vocabulary())
    with pytest.raises(ValueError):
        Vocabulary(min_count=0)


def test_vocabulary_min_count():
    docs = [Counter(a=3, b=1), Counter(a=1, c=2)]
    assert sorted(Vocabulary(min_count=2).fit(docs).index) == ["a", "c"]


# ---- tokenizer ------------------------------------------------------------

def test_tokenizer_groups_numbers_prices_and_urls():
    assert tokenize("Call 09061701461 NOW to win £1000! http://x.co/a or www.win.com, ref 42 I'll") == [
        "call", "<longnum>", "now", "to", "win", "<money>", "<url>", "or", "<url>", "ref", "<num>", "i'll"]
    assert tokenize_simple("Win £1000 now!") == ["win", "1000", "now"]
    assert tokenize("") == []


# ---- evaluation helpers -----------------------------------------------------

def test_scores_by_hand():
    # deliberately asymmetric (1 miss, 2 false alarms) so swapping fp and fn is caught
    s = score(["spam", "spam", "ham", "ham", "ham", "ham"], ["spam", "ham", "spam", "spam", "ham", "ham"])
    assert (s.tp, s.fn, s.fp, s.tn) == (1, 1, 2, 2)
    assert s.accuracy == pytest.approx(0.5)
    assert (s.precision, s.recall) == (pytest.approx(1 / 3), 0.5)
    assert s.f1 == pytest.approx(0.4)
    none = score(["ham", "ham"], ["ham", "ham"])
    assert none.precision == none.recall == none.f1 == 0.0
    with pytest.raises(ValueError):
        score(["ham"], [])


def test_stratified_folds_partition_and_keep_ratio():
    labels = ["spam"] * 50 + ["ham"] * 450
    folds = stratified_folds(labels, 5, seed=3)
    flat = [i for f in folds for i in f]
    assert sorted(flat) == list(range(500))                     # disjoint and complete
    for f in folds:
        assert sum(labels[i] == "spam" for i in f) == 10        # same 10% in every fold
    assert folds == stratified_folds(labels, 5, seed=3) != stratified_folds(labels, 5, seed=4)
    train, test = train_test_split(labels, 0.2, seed=3)
    assert not set(train) & set(test) and len(test) == 100


def test_load_tsv(tmp_path):
    good = tmp_path / "ok.tsv"
    good.write_text('ham\thello "there"\nspam\tWIN\tnow\n\n', encoding="utf-8")
    assert load_tsv(good) == (['hello "there"', "WIN\tnow"], ["ham", "spam"])
    bad = tmp_path / "bad.tsv"
    bad.write_text("maybe\thello\n")
    with pytest.raises(ValueError, match="bad.tsv:1"):
        load_tsv(bad)
    empty = tmp_path / "empty.tsv"
    empty.write_text("\n")
    with pytest.raises(ValueError, match="no messages"):
        load_tsv(empty)


# ---- against scikit-learn, on the real data --------------------------------

def test_matches_scikit_learn_on_sms_data():
    pytest.importorskip("sklearn")
    from nbspam.cli import DEFAULT_DATA
    from nbspam.compare import compare_with_sklearn
    texts, labels = load_tsv(DEFAULT_DATA)
    docs = [counts(t) for t in texts]
    tr, te = train_test_split(labels, 0.2, seed=7)
    vocab = Vocabulary().fit([docs[i] for i in tr])
    for alpha in (1.0, 0.05):
        res = compare_with_sklearn([docs[i] for i in tr], [labels[i] for i in tr], [docs[i] for i in te], vocab, alpha)
        for name, (diff, agree) in res.items():
            assert diff < 1e-9 and agree, name


def test_quality_floor_on_sms_data():
    """Guards against regressions that keep the code running but ruin the model."""
    from nbspam.cli import DEFAULT_DATA
    from nbspam.experiments import cross_validate
    texts, labels = load_tsv(DEFAULT_DATA)
    _, (tp, fp, tn, fn) = cross_validate(texts, labels, "multinomial", 1.0)
    assert tp / (tp + fn) > 0.93 and tp / (tp + fp) > 0.93


# ---- CLI ---------------------------------------------------------------------

def test_cli_predict_and_evaluate(capsys):
    assert main(["predict", "URGENT! You have won a £2000 prize. Call 09061701461 to claim", "see you at 7?"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("spam") and out[1].startswith("ham")
    assert main(["evaluate", "--model", "bernoulli", "--alpha", "0.5"]) == 0
    assert "precision" in capsys.readouterr().out


def test_cli_errors(tmp_path, capsys):
    assert main(["--data", str(tmp_path / "none.tsv"), "evaluate"]) == 2
    with pytest.raises(SystemExit):
        main(["evaluate", "--alpha", "0"])
