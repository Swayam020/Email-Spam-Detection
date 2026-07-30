"""Train an SVM spam classifier on TF-IDF word vectors.

Usage:  python -m src.train
Saves the fitted pipeline to models/spam_svm.joblib and prints a full
evaluation report on a held-out test set.
"""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .preprocess import load_corpus

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "models" / "spam_svm.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            stop_words="english",
        )),
        ("svm", LinearSVC(class_weight="balanced")),
    ])


def main() -> None:
    print("Loading and preprocessing corpus ...")
    texts, labels = load_corpus(RAW_DIR)
    print(f"{len(texts)} emails loaded "
          f"({sum(labels)} spam / {len(labels) - sum(labels)} ham)")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42)

    search = GridSearchCV(
        build_pipeline(),
        param_grid={"svm__C": [0.1, 1.0, 10.0]},
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    print("Training SVM (5-fold CV over C) ...")
    search.fit(X_train, y_train)
    print(f"Best C: {search.best_params_['svm__C']}, "
          f"CV F1: {search.best_score_:.4f}")

    model = search.best_estimator_
    y_pred = model.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print("Confusion matrix (rows=truth, cols=pred):")
    print(confusion_matrix(y_test, y_pred))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
