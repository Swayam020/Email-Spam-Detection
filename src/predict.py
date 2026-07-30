"""Classify an email as spam or ham with the trained model.

Usage:
    python -m src.predict path/to/email.eml     # raw email file
    python -m src.predict --text "WIN A FR3E PRIZE NOW!!!"
"""

import argparse
from pathlib import Path

import joblib

from .preprocess import clean_text, extract_text

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "spam_svm.joblib"


def classify(text: str) -> tuple[str, float]:
    model = joblib.load(MODEL_PATH)
    cleaned = clean_text(text)
    label = model.predict([cleaned])[0]
    score = model.decision_function([cleaned])[0]
    return ("spam" if label == 1 else "ham"), float(score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eml", nargs="?", help="path to a raw email file")
    parser.add_argument("--text", help="classify a raw text snippet instead")
    args = parser.parse_args()

    if args.text:
        text = args.text
    elif args.eml:
        text = extract_text(Path(args.eml).read_bytes())
    else:
        parser.error("provide an email file or --text")

    label, score = classify(text)
    print(f"prediction: {label}  (decision score {score:+.3f}, "
          f"positive = spam)")


if __name__ == "__main__":
    main()
