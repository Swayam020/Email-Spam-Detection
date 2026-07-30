"""Email parsing and text normalization.

Includes deliberate-misspelling / obfuscation handling: spammers often
disguise trigger words ("fr3e", "v1agra", "c.a.s.h") to evade keyword
filters. We normalize leetspeak substitutions, collapse character
repetition and separator tricks, and fuzzy-match suspicious tokens
against a lexicon of common spam trigger words.
"""

import email
import email.policy
import re
from difflib import SequenceMatcher
from pathlib import Path

from bs4 import BeautifulSoup

# --- Obfuscation / deliberate misspelling handling -------------------------

# common leetspeak & symbol substitutions used to disguise words
LEET_MAP = str.maketrans({
    "0": "o",
    "1": "l",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
    "$": "s",
    "!": "i",
    "|": "l",
    "+": "t",
})

# words spammers frequently obfuscate; used for fuzzy matching
SPAM_LEXICON = [
    "free", "viagra", "cialis", "money", "cash", "winner", "prize",
    "lottery", "credit", "loan", "mortgage", "pharmacy", "pills",
    "sex", "porn", "casino", "investment", "guarantee", "unsubscribe",
    "offer", "discount", "cheap", "deal", "million", "urgent", "account",
    "password", "bank", "click", "buy", "order", "refinance", "insurance",
]

WORD_RE = re.compile(r"[a-z][a-z']*", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
NUMBER_RE = re.compile(r"\b\d{4,}\b")
# a token like f.r.e.e or f-r-e-e (single chars joined by separators)
SEPARATED_RE = re.compile(r"\b(?:\w[\.\-\*_ ]){2,}\w\b")


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def normalize_obfuscation(text: str) -> str:
    """Undo common spam obfuscation tricks so disguised words match
    their plain forms during vectorization."""
    # collapse "f.r.e.e"-style separator tricks
    def _collapse(match: re.Match) -> str:
        return re.sub(r"[\.\-\*_ ]", "", match.group(0))

    text = SEPARATED_RE.sub(_collapse, text)
    # leetspeak: only translate inside tokens that mix letters and symbols,
    # so genuine numbers are left alone
    tokens = text.split()
    out = []
    for tok in tokens:
        if re.search(r"[a-zA-Z]", tok) and re.search(r"[01345789@$!|+]", tok):
            tok = tok.translate(LEET_MAP)
        out.append(tok)
    text = " ".join(out)
    # collapse elongated characters: "freeeee" -> "free"
    text = re.sub(r"(\w)\1{2,}", r"\1\1", text)
    return text


def flag_misspelled_trigger_words(text: str) -> list[str]:
    """Fuzzy-match tokens against the spam lexicon to catch deliberate
    misspellings (e.g. 'vaigra', 'mony'). Returns marker tokens that get
    appended to the document so the classifier can learn from them."""
    flags = []
    for tok in WORD_RE.findall(text.lower()):
        if len(tok) < 4 or tok in SPAM_LEXICON:
            continue
        for word in SPAM_LEXICON:
            if abs(len(tok) - len(word)) <= 1 and _similar(tok, word) >= 0.8:
                flags.append(f"OBFUSCATED_{word.upper()}")
                break
    return flags


# --- Email parsing ----------------------------------------------------------

def extract_text(raw_bytes: bytes) -> str:
    """Parse a raw RFC-822 email and return subject + plain-text body."""
    msg = email.message_from_bytes(raw_bytes, policy=email.policy.default)
    subject = str(msg.get("Subject", ""))

    parts = []
    for part in msg.walk():
        ctype = part.get_content_type()
        if ctype not in ("text/plain", "text/html"):
            continue
        try:
            payload = part.get_content()
        except Exception:
            payload = part.get_payload(decode=False)
        if not isinstance(payload, str):
            continue
        if ctype == "text/html":
            payload = BeautifulSoup(payload, "html.parser").get_text(" ")
        parts.append(payload)
    return subject + "\n" + "\n".join(parts)


def clean_text(text: str) -> str:
    """Full normalization pipeline for one email's text."""
    text = URL_RE.sub(" URL ", text)
    text = EMAIL_RE.sub(" EMAILADDR ", text)
    text = normalize_obfuscation(text)
    text = NUMBER_RE.sub(" LONGNUM ", text)
    flags = flag_misspelled_trigger_words(text)
    text = text.lower()
    text = " ".join(WORD_RE.findall(text) + [f.lower() for f in flags])
    return text


def load_corpus(raw_dir: Path) -> tuple[list[str], list[int]]:
    """Read every extracted email under data/raw/{ham,spam} and return
    (cleaned_texts, labels) with spam=1, ham=0."""
    texts, labels = [], []
    for label_name, label in (("ham", 0), ("spam", 1)):
        for path in sorted((raw_dir / label_name).rglob("*")):
            # corpus files are named like 00001.7c53336b37003a9286aba55d2945844c
            if not path.is_file() or "." not in path.name:
                continue
            if path.name == "cmds":
                continue
            try:
                texts.append(clean_text(extract_text(path.read_bytes())))
                labels.append(label)
            except Exception as exc:  # skip rare malformed messages
                print(f"skipping {path}: {exc}")
    return texts, labels
