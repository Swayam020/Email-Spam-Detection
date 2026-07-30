"""Download and extract the SpamAssassin public corpus.

The corpus is a well-known, freely available collection of real ham and
spam emails published by the Apache SpamAssassin project:
https://spamassassin.apache.org/old/publiccorpus/
"""

import tarfile
import urllib.request
from pathlib import Path

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/"

# archive name -> label of the emails it contains
ARCHIVES = {
    "20030228_easy_ham.tar.bz2": "ham",
    "20030228_easy_ham_2.tar.bz2": "ham",
    "20030228_hard_ham.tar.bz2": "ham",
    "20030228_spam.tar.bz2": "spam",
    "20050311_spam_2.tar.bz2": "spam",
}

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def download_and_extract() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for archive, label in ARCHIVES.items():
        dest = RAW_DIR / archive
        if not dest.exists():
            url = BASE_URL + archive
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, dest)
        out_dir = RAW_DIR / label
        out_dir.mkdir(exist_ok=True)
        print(f"Extracting {archive} -> {out_dir}")
        with tarfile.open(dest, "r:bz2") as tar:
            tar.extractall(out_dir, filter="data")
    print("Done. Raw data in", RAW_DIR)


if __name__ == "__main__":
    download_and_extract()
