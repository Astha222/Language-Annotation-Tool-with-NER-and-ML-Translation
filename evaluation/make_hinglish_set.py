"""Turn evaluation/hinglish/hinglish_sentences.txt (entities marked as [text|TYPE]) into the frozen test set evaluation/data/ner_hi/hinglish/hi_test.json
(one {"words": [...], "ner": [...BIO...]} per line, the same shape as the other Hindi sets) and return its manifest entry.

    python evaluation/make_hinglish_set.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "hinglish" / "hinglish_sentences.txt"
OUT = HERE / "data" / "ner_hi" / "hinglish" / "hi_test.json"
ENTITY = re.compile(r"\[([^\]|]+)\|(PER|ORG|LOC)\]")


def parse_line(line):
    """'[Rahul|PER] ja raha hai [New Delhi|LOC]' -> (words, BIO tags)."""
    words, tags, pos = [], [], 0
    for m in ENTITY.finditer(line):
        for w in line[pos:m.start()].split():
            words.append(w)
            tags.append("O")
        for i, w in enumerate(m.group(1).split()):
            words.append(w)
            tags.append(("B-" if i == 0 else "I-") + m.group(2))
        pos = m.end()
    for w in line[pos:].split():
        words.append(w)
        tags.append("O")
    if "[" in "".join(words) or "|" in "".join(words):
        raise ValueError(f"unparsed markup in: {line}")
    return words, tags


def parse_file(path=SOURCE):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words, tags = parse_line(line)
            rows.append({"words": words, "ner": tags})
    return rows


def hinglish_entry():
    from download_data import sha256            # imported here: download_data imports this module for its manifest
    rows = parse_file()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in rows), encoding="utf-8")
    return {"dataset": "hinglish", "lang": "hi", "split": "test", "role": "test", "sentences": len(rows),
            "source": "evaluation/hinglish/hinglish_sentences.txt (written by the assistant, 2026-11-18; reviewed by a Hindi-speaking team member, 2026-11-22)",
            "file": "ner_hi/hinglish/hi_test.json", "sha256": sha256(OUT)}


def main():
    entry = hinglish_entry()
    ents = sum(t.startswith("B-") for r in parse_file() for t in r["ner"])
    print(f"{entry['sentences']} sentences, {ents} entities, sha256 {entry['sha256'][:12]}")
    manifest_path = HERE / "data" / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [e for e in manifest["files"] if e["dataset"] != "hinglish"] + [entry]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"MANIFEST.json updated ({len(manifest['files'])} files)")


if __name__ == "__main__":
    main()
