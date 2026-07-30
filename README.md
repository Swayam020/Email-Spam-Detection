# Email Spam Detection and Classification using Support Vector Machines

Detects and classifies spam emails by training a Support Vector Machine on
TF-IDF word vectors extracted from the [SpamAssassin public corpus]
(https://spamassassin.apache.org/old/publiccorpus/) — ~6,000 real ham and
spam emails.

## Highlights

- **SVM classifier on word vectors** — emails are converted to TF-IDF
  vectors (unigrams + bigrams) and classified with a `LinearSVC`, with the
  regularization strength tuned by 5-fold cross-validated grid search.
- **Deliberate-misspelling detection** — spammers disguise trigger words to
  evade keyword filters (`fr3e`, `v1agra`, `c.a.s.h`, `freeeee`). The
  preprocessing pipeline:
  - normalizes leetspeak/symbol substitutions (`0→o`, `@→a`, `$→s`, ...),
  - collapses separator tricks (`f.r.e.e` → `free`) and character
    elongation (`freeeee` → `free`),
  - fuzzy-matches remaining tokens against a lexicon of common spam trigger
    words (Levenshtein-style similarity) and injects `obfuscated_<word>`
    marker features the SVM can learn from.
- **Real email parsing** — raw RFC-822 messages are parsed with Python's
  `email` package; HTML bodies are stripped with BeautifulSoup; URLs,
  email addresses, and long numbers are mapped to placeholder tokens.

## Project structure

```
src/
├── download_data.py   # fetch + extract the SpamAssassin corpus
├── preprocess.py      # email parsing, cleaning, misspelling normalization
├── train.py           # TF-IDF + LinearSVC training and evaluation
└── predict.py         # CLI: classify an .eml file or a text snippet
```

## Setup & usage

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python -m src.download_data     # download the corpus (~5 MB)
python -m src.train             # train + evaluate, saves models/spam_svm.joblib
python -m src.predict --text "Congratulations! You've w0n a FR3E prize, claim n0w!"
```

## Results

Held-out test set (20% stratified split of 6,046 emails — 1,896 spam / 4,150 ham):

| Metric | Score |
|---|---|
| Accuracy | 0.986 |
| Spam precision | 0.98 |
| Spam recall | 0.97 |
| F1 (spam) | 0.98 |

Best hyperparameters: `C=10.0` (5-fold cross-validated grid search, CV F1 = 0.982).

Example:

```
$ python -m src.predict --text "Congratulations! You have w0n a FR3E prize. Cl1ck here to claim your c.a.s.h now!!!"
prediction: spam  (decision score +0.462, positive = spam)
```
