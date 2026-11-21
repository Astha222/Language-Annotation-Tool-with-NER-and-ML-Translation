"""Test sets for Russian, Arabic, Japanese (and ordinary-text German and Chinese), frozen in MANIFEST.json by download_data.py.

    uner_ru / uner_de / uner_zh   Universal NER v1, Parallel UD test sets (1,000 sentences each, the same sentences translated; CC-BY-SA-4.0). Tags PER/ORG/LOC (OTH dropped).
    uner_en_ewt                   Universal NER v1, English Web Treebank test (2,077 sentences of emails, reviews, newsgroups, answers; CC-BY-SA-4.0).
    uner_zhs                      Universal NER v1, Chinese GSDSIMP test (Simplified Chinese; CC-BY-SA-4.0). uner_all was trained on its train split.
    anercorp                      ANERcorp, CAMeL Lab test split (25,008 words; CC-BY-SA-4.0). Taken from the Hugging Face mirror KFUPM-JRCAI/ANERcorp_experimental,
                                  which is sentence-split; the mirror asas-ai/ANERCorp has damaged text (the letters ha and fa are broken) and is NOT used.
                                  PERS -> PER, MISC dropped.
    stockmark_ja                  stockmark/ner-wikipedia-dataset (Japanese Wikipedia, 5,343 sentences, no official split, so all of it is a test set; CC-BY-SA-3.0).
                                  One token per character. Persons -> PER; companies, political and other organisations -> ORG; places AND facilities -> LOC
                                  (decided once, here); products and events dropped.
"""
import json
import time

from common import DATA, download

UNER = "https://raw.githubusercontent.com/UniversalNER/"
UNER_SETS = {"uner_ru": ("ru", "UNER_Russian-PUD/master/ru_pud-ud-test.iob2"),
             "uner_de": ("de", "UNER_German-PUD/master/de_pud-ud-test.iob2"),
             "uner_zh": ("zh", "UNER_Chinese-PUD/master/zh_pud-ud-test.iob2"),
             "uner_zhs": ("zh", "UNER_Chinese-GSDSIMP/master/zh_gsdsimp-ud-test.iob2"),
             # English web text (emails, reviews, newsgroups, answers): the fair English test E5; no current English model was trained on it
             "uner_en_ewt": ("en", "UNER_English-EWT/master/en_ewt-ud-test.iob2"),
             # Japanese PUD exists on GitHub although the Hugging Face loader script does not list it (found 2026-11-18)
             "uner_ja": ("ja", "UNER_Japanese-PUD/master/ja_pud-ud-test.iob2")}
ANER_ROWS = "https://datasets-server.huggingface.co/rows?dataset=KFUPM-JRCAI/ANERcorp_experimental&config=default&split=test"
STOCKMARK = "https://huggingface.co/datasets/stockmark/ner-wikipedia-dataset/resolve/main/ner.json"
JA_TYPES = {"人名": "PER", "法人名": "ORG", "政治的組織名": "ORG", "その他の組織名": "ORG", "地名": "LOC", "施設名": "LOC"}
ANER_TYPES = {"PERS": "PER", "PER": "PER", "ORG": "ORG", "LOC": "LOC"}


def parse_iob2(text):
    """UNER .iob2: '# comment' lines, then one 'index<TAB>token<TAB>tag<TAB>...' line per token, a blank line between sentences."""
    rows, words, tags = [], [], []
    for line in text.splitlines() + [""]:
        if not line.strip():
            if words:
                rows.append({"words": words, "ner": tags})
            words, tags = [], []
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        tag = cols[2] if len(cols) > 2 else "O"
        words.append(cols[1])
        tags.append(tag if tag[2:] in ("PER", "ORG", "LOC") else "O")
    return rows


def map_tags(tags, table):
    return [t if t == "O" else (f"{t[:2]}{table[t[2:]]}" if t[2:] in table else "O") for t in tags]


def char_bio(text, entities):
    """Japanese: one token per character, BIO tags from character spans (overlapping or unknown types are skipped)."""
    tags = ["O"] * len(text)
    for e in sorted(entities, key=lambda e: e["span"][0]):
        kind = JA_TYPES.get(e["type"])
        a, b = e["span"]
        if kind is None or any(t != "O" for t in tags[a:b]) or not 0 <= a < b <= len(text):
            continue
        tags[a] = "B-" + kind
        for i in range(a + 1, b):
            tags[i] = "I-" + kind
    return {"words": list(text), "ner": tags}


def _fetch_rows(url, total=None):
    import requests
    rows, offset = [], 0
    while total is None or offset < total:
        for attempt in range(5):
            r = requests.get(f"{url}&offset={offset}&length=100", timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(3 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        d = r.json()
        total = d["num_rows_total"]
        rows.extend(x["row"] for x in d["rows"])
        offset += 100
    return rows


def _write(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in rows if r["words"]), encoding="utf-8")


def foreign_entries(sha256):
    entries = []
    for name, (lang, rel) in UNER_SETS.items():
        out = DATA / "ner_foreign" / f"{name}_test.json"
        if not out.exists():
            raw = download(UNER + rel, DATA / "ner_foreign" / "raw" / rel.split("/")[-1])
            _write(parse_iob2(raw.read_text(encoding="utf-8")), out)
        entries.append((name, lang, out, UNER + rel + " (CC-BY-SA-4.0)"))
    out = DATA / "ner_foreign" / "anercorp_test.json"
    if not out.exists():
        rows = _fetch_rows(ANER_ROWS)
        _write([{"words": r["tokens"], "ner": map_tags(r["tags"], ANER_TYPES)} for r in rows], out)
    entries.append(("anercorp", "ar", out, "huggingface.co/datasets/KFUPM-JRCAI/ANERcorp_experimental (test; ANERcorp CAMeL Lab split, CC-BY-SA-4.0)"))
    out = DATA / "ner_foreign" / "stockmark_ja_test.json"
    if not out.exists():
        raw = download(STOCKMARK, DATA / "ner_foreign" / "raw" / "stockmark_ner.json")
        _write([char_bio(x["text"], x["entities"]) for x in json.loads(raw.read_text(encoding="utf-8"))], out)
    entries.append(("stockmark_ja", "ja", out, STOCKMARK + " (CC-BY-SA-3.0)"))
    result = []
    for name, lang, path, source in entries:
        n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        result.append({"dataset": name, "lang": lang, "split": "test", "role": "test", "sentences": n, "source": source,
                       "file": str(path.relative_to(DATA)).replace("\\", "/"), "sha256": sha256(path)})
    return result
