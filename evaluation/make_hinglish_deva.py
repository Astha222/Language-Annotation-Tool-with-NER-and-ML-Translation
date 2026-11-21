"""Make a Devanagari copy of the Hinglish test set by transliterating every word with Azure Translator (Latin -> Devanagari, language hi).

Each word is converted on its own, so the words and BIO tags stay aligned one to one. Punctuation and digits are left alone. The result is
evaluation/data/ner_hi/hinglish/deva_test.json (dataset name "hinglish_deva"); it is derived, so it is not frozen in MANIFEST.json.
Only the invented test sentences are sent to Azure. Needs AZURE_TRANSLATOR_KEY and AZURE_TRANSLATOR_REGION (backend/.env or the environment).

    python evaluation/make_hinglish_deva.py
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

from common import BACKEND, DATA, use_console_utf8

URL = "https://api.cognitive.microsofttranslator.com/transliterate"
SRC = DATA / "ner_hi" / "hinglish" / "hi_test.json"
OUT = DATA / "ner_hi" / "hinglish" / "deva_test.json"


def load_env():
    path = BACKEND / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def transliterate(words, key, region):
    """Azure accepts 10 texts per request."""
    out = []
    headers = {"Ocp-Apim-Subscription-Key": key, "Ocp-Apim-Subscription-Region": region, "Content-Type": "application/json"}
    for i in range(0, len(words), 10):
        batch = words[i:i + 10]
        for attempt in range(4):
            r = requests.post(URL, params={"api-version": "3.0", "language": "hi", "fromScript": "Latn", "toScript": "Deva"},
                              headers=headers, json=[{"Text": w} for w in batch], timeout=30)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        out.extend(x["text"] for x in r.json())
    return out


def main():
    use_console_utf8()
    load_env()
    key, region = os.getenv("AZURE_TRANSLATOR_KEY"), os.getenv("AZURE_TRANSLATOR_REGION")
    if not key or not region:
        sys.exit("AZURE_TRANSLATOR_KEY / AZURE_TRANSLATOR_REGION are not set")
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    vocab = sorted({w for r in rows for w in r["words"] if re.search(r"[A-Za-z]", w)})
    print(f"{len(vocab)} distinct words to transliterate")
    mapping = dict(zip(vocab, transliterate(vocab, key, region)))
    OUT.write_text("".join(json.dumps({"words": [mapping.get(w, w) for w in r["words"]], "ner": r["ner"]}, ensure_ascii=False) + chr(10)
                          for r in rows), encoding="utf-8")
    print(f"wrote {OUT}")
    for r in rows[:3]:
        print(" ".join(r["words"]), "->", " ".join(mapping.get(w, w) for w in r["words"]))


if __name__ == "__main__":
    main()
