# Results

SMS Spam Collection: 5,574 messages, 747 spam (13.4%). Python 3.13.18, scikit-learn 1.9.1 (used only for the comparison).

## 1. Same answers as scikit-learn

Trained on 4,458 messages, compared on the other 1,116, using identical features (7,041-token vocabulary).

| model | alpha | max difference in log-likelihood | identical predictions |
|---|---:|---:|:---:|
| multinomial | 1.0 | 4.5e-13 | yes |
| bernoulli | 1.0 | 1.7e-13 | yes |
| multinomial | 0.1 | 6.8e-13 | yes |
| bernoulli | 0.1 | 1.7e-13 | yes |

## 2. Quality (5-fold stratified cross-validation, alpha = 1)

Mean ± standard deviation over the folds. Spam is the positive class.

| model | tokenizer | accuracy | precision | recall | F1 | ham marked spam | spam missed |
|---|---|---:|---:|---:|---:|---:|---:|
| always say "ham" | — | 0.866 | — | 0.000 | 0.000 | 0 | 747 |
| multinomial | plain words | 0.987 ± 0.005 | 0.972 ± 0.017 | 0.929 ± 0.028 | 0.950 ± 0.021 | 20 | 53 |
| multinomial | words + number/price/URL tokens | 0.988 ± 0.003 | 0.960 ± 0.019 | 0.950 ± 0.010 | 0.955 ± 0.010 | 30 | 37 |
| bernoulli | plain words | 0.979 ± 0.005 | 0.994 ± 0.008 | 0.846 ± 0.037 | 0.914 ± 0.021 | 4 | 115 |
| bernoulli | words + number/price/URL tokens | 0.985 ± 0.003 | 0.994 ± 0.008 | 0.893 ± 0.023 | 0.941 ± 0.013 | 4 | 80 |

## 3. Effect of the smoothing parameter alpha (multinomial, cross-validated)

| alpha | precision | recall | F1 |
|---:|---:|---:|---:|
| 0.01 | 0.969 | 0.944 | 0.956 |
| 0.1 | 0.967 | 0.952 | 0.960 |
| 0.5 | 0.964 | 0.952 | 0.958 |
| 1.0 | 0.960 | 0.950 | 0.955 |
| 2.0 | 0.966 | 0.940 | 0.952 |
| 5.0 | 0.983 | 0.905 | 0.942 |

## 4. Why the model works in log space

For the longest message (189 known tokens), multiplying P(ham) by every P(token | ham) gives `0.0` in 64-bit floating point. That is underflow, so every class would score zero. Adding the logs instead gives `-1182.1`, an ordinary number.

## 5. Tokens that most favour each class

log P(token | class) − log P(token | other class), from the training split.

| spam token | log ratio | ham token | log ratio |
|---|---:|---|---:|
| `<longnum>` | 6.03 | `gt` | 4.53 |
| `claim` | 5.55 | `lt` | 4.51 |
| `prize` | 5.32 | `he` | 4.03 |
| `won` | 5.11 | `i'll` | 3.90 |
| `tone` | 5.04 | `lor` | 3.89 |
| `guaranteed` | 4.68 | `she` | 3.68 |
| `ppm` | 4.65 | `later` | 3.66 |
| `cs` | 4.54 | `come` | 3.57 |
| `pobox` | 4.48 | `oh` | 3.53 |
| `<url>` | 4.40 | `happy` | 3.41 |
| `awarded` | 4.38 | `said` | 3.34 |
| `ringtone` | 4.35 | `my` | 3.33 |

## 6. Mistakes on the test split (10 of 1,116)

**2 real messages marked as spam** (shown in full):

- Wewa is 130. Iriver 255. All 128 mb.
- It's é only $140 ard...É rest all ard $180 at least...Which is é price 4 é 2 bedrm ($900)

**8 spam messages missed.** 6 of them contain none of the 20 tokens that most favour spam (no numbers, prices, URLs, 'claim', 'prize'...): they read like ordinary chat, which is the case a word-frequency model handles worst.

