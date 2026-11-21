"""Download every NER test/dev split once and freeze it: write evaluation/data/MANIFEST.json with the sentence
count and sha256 of each file, so any later run can prove it is scoring exactly the same sentences.

    python evaluation/download_data.py            # download whatever is missing, then write the manifest
    python evaluation/download_data.py --verify   # check the files on disk against the manifest (exit 1 on mismatch)

Splits are always used in file order and never sampled at random. A run on a subset uses the first N sentences.
"""
import argparse
import hashlib
import json
import sys
from datetime import date

from common import DATA, download, use_console_utf8
from eval_ner import DATASETS

MANIFEST = DATA / "MANIFEST.json"
WIKIANN_LANGS = ["en", "hi", "fr", "es"]       # fr, es: cased (silver) tests for the de/fr/es model choice (2026-11-18)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wanted():
    """(dataset, lang, split name used by the source, role) for every split we freeze."""
    for dataset, (_, _, (dev, test), _) in DATASETS.items():
        for lang in (WIKIANN_LANGS if dataset == "wikiann" else ["en"]):
            yield dataset, lang, test, "test"
            if dataset != "conll2003":                   # CoNLL is only reported for reference; no dev split needed
                yield dataset, lang, dev, "dev"


def local_path(dataset, lang, split):
    return DATA / "ner" / dataset / f"{lang}_{split}.jsonl"


def hindi_entries():
    """Naamapadam Hindi (CC0): the human-annotated test split and the validation split. The 490 MB silver training split is not kept."""
    import zipfile
    from huggingface_hub import hf_hub_download
    target = DATA / "ner_hi" / "naamapadam"
    target.mkdir(parents=True, exist_ok=True)
    if not (target / "hi_test.json").exists():
        archive = hf_hub_download("ai4bharat/naamapadam", "data/hi_IndicNER_v1.0.zip", repo_type="dataset")
        with zipfile.ZipFile(archive) as z:
            for name in ("hi_test.json", "hi_val.json"):
                z.extract(name, target)
    out = []
    for split, role in (("test", "test"), ("val", "dev")):
        path = target / f"hi_{split}.json"
        n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        out.append({"dataset": "naamapadam", "lang": "hi", "split": split, "role": role, "sentences": n,
                    "source": "huggingface.co/datasets/ai4bharat/naamapadam (data/hi_IndicNER_v1.0.zip, CC0-1.0)",
                    "file": str(path.relative_to(DATA)).replace("\\", "/"), "sha256": sha256(path)})
    out.append(hiner_entry())
    return out + multiconer_entries() + multiconer_tune_entries() + indic_entries()


INDIC_LANGS = ["as", "bn", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te"]


def indic_entries():
    """Naamapadam human-annotated TEST splits for ten more Indian languages (ai4bharat, CC0-1.0). Only the test file is extracted from each zip."""
    import zipfile
    from huggingface_hub import hf_hub_download
    target = DATA / "ner_indic"
    target.mkdir(parents=True, exist_ok=True)
    out = []
    for lang in INDIC_LANGS:
        path = target / f"{lang}_test.json"
        if not path.exists():
            archive = hf_hub_download("ai4bharat/naamapadam", f"data/{lang}_IndicNER_v1.0.zip", repo_type="dataset")
            with zipfile.ZipFile(archive) as z:
                member = next(n for n in z.namelist() if n.endswith(f"{lang}_test.json"))
                path.write_bytes(z.read(member))
        n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        out.append({"dataset": f"nm_{lang}", "lang": lang, "split": "test", "role": "test", "sentences": n,
                    "source": f"huggingface.co/datasets/ai4bharat/naamapadam (data/{lang}_IndicNER_v1.0.zip, CC0-1.0)",
                    "file": f"ner_indic/{lang}_test.json", "sha256": sha256(path)})
    return out


MULTICONER_FOLDERS = {"de": "DE-German", "es": "ES-Spanish", "fr": "FR-French", "zh": "ZH-Chinese", "hi": "HI-Hindi"}
MC_PER = {"Scientist", "Artist", "Athlete", "Politician", "Cleric", "SportsManager", "OtherPER"}
MC_ORG = {"MusicalGRP", "PublicCORP", "PrivateCORP", "AerospaceManufacturer", "SportsGRP", "CarManufacturer", "ORG"}
MC_LOC = {"Facility", "OtherLOC", "HumanSettlement", "Station"}


def _mc_rows(raw, kinds):
    rows = []
    for block in open(raw, encoding="utf-8").read().split(chr(10) * 2):
        words, tags = [], []
        for line in block.splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            words.append(parts[0])
            tag = parts[-1]
            kind = kinds.get(tag[2:]) if tag != "O" else None
            tags.append(f"{tag[:2]}{kind}" if kind else "O")
        if words:
            rows.append({"words": words, "ner": tags})
    return rows


TUNE_SENTENCES = 600


def multiconer_tune_entries():
    """A fixed sample of the MultiCoNER v2 TRAIN split (de, es, fr; every k-th sentence, 600 each) used ONLY to choose the score threshold of zero-shot
    models (GLiNER, Otter). Our MultiCoNER scores use the dev split as the test set, so thresholds must never be tuned on dev."""
    from huggingface_hub import hf_hub_download
    kinds = {**{k: "PER" for k in MC_PER}, **{k: "ORG" for k in MC_ORG}, **{k: "LOC" for k in MC_LOC}}
    out = []
    for lang in ("de", "es", "fr"):
        path = DATA / "ner_multi" / f"mc_{lang}_tune.json"
        if not path.exists():
            raw = hf_hub_download("MultiCoNER/multiconer_v2", f"{MULTICONER_FOLDERS[lang]}/{lang}_train.conll", repo_type="dataset")
            rows = _mc_rows(raw, kinds)
            step = max(1, len(rows) // TUNE_SENTENCES)
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in rows[::step][:TUNE_SENTENCES]), encoding="utf-8")
        n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        out.append({"dataset": f"mc_{lang}_tune", "lang": lang, "split": "tune", "role": "tune", "sentences": n,
                    "source": f"huggingface.co/datasets/MultiCoNER/multiconer_v2 ({MULTICONER_FOLDERS[lang]}/{lang}_train.conll, every k-th sentence, CC-BY-4.0)",
                    "file": f"ner_multi/mc_{lang}_tune.json", "sha256": sha256(path)})
    return out


def multiconer_entries():
    """MultiCoNER v2 (CC-BY-4.0) dev splits for de, es, fr, zh, hi; every person subtype -> PER, group types -> ORG, location types -> LOC, the rest -> O."""
    from huggingface_hub import hf_hub_download
    target = DATA / "ner_multi"
    target.mkdir(parents=True, exist_ok=True)
    kinds = {**{k: "PER" for k in MC_PER}, **{k: "ORG" for k in MC_ORG}, **{k: "LOC" for k in MC_LOC}}
    out = []
    for lang, folder in MULTICONER_FOLDERS.items():
        path = target / f"mc_{lang}_dev.json"
        if not path.exists():
            raw = hf_hub_download("MultiCoNER/multiconer_v2", f"{folder}/{lang}_dev.conll", repo_type="dataset")
            with open(path, "w", encoding="utf-8") as f:
                for block in open(raw, encoding="utf-8").read().split(chr(10) * 2):
                    words, tags = [], []
                    for line in block.splitlines():
                        if not line.strip() or line.startswith("#"):
                            continue
                        parts = line.split()
                        words.append(parts[0])
                        tag = parts[-1]
                        kind = kinds.get(tag[2:]) if tag != "O" else None
                        tags.append(f"{tag[:2]}{kind}" if kind else "O")
                    if words:
                        f.write(json.dumps({"words": words, "ner": tags}, ensure_ascii=False) + chr(10))
        n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        out.append({"dataset": f"mc_{lang}", "lang": lang, "split": "dev", "role": "test", "sentences": n,
                    "source": f"huggingface.co/datasets/MultiCoNER/multiconer_v2 ({folder}/{lang}_dev.conll, CC-BY-4.0)",
                    "file": f"ner_multi/mc_{lang}_dev.json", "sha256": sha256(path)})
    return out


def hiner_entry():
    """HiNER-collapsed test (cfilt/HiNER-collapsed, CC-BY-SA-4.0): 21,657 sentences, a fixed subset of every 7th sentence (3,094) so one model takes minutes, not an hour."""
    import shutil
    from huggingface_hub import hf_hub_download
    target = DATA / "ner_hi" / "hiner"
    target.mkdir(parents=True, exist_ok=True)
    out_path = target / "hi_test.json"
    if not out_path.exists():
        raw = target / "test_full.json"
        if not raw.exists():
            shutil.copy(hf_hub_download("cfilt/HiNER-collapsed", "data/test.json", repo_type="dataset"), raw)
        labels = ["B-LOC", "B-ORG", "B-PER", "I-LOC", "I-ORG", "I-PER", "O"]          # order of cfilt/HiNER-collapsed data/label_list
        rows = json.loads(raw.read_text(encoding="utf-8"))
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows[::7]:
                f.write(json.dumps({"words": r["tokens"], "ner": [labels[t] for t in r["ner_tags"]]}, ensure_ascii=False) + chr(10))
        raw.unlink()
    n = sum(1 for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {"dataset": "hiner", "lang": "hi", "split": "test", "role": "test", "sentences": n,
            "source": "huggingface.co/datasets/cfilt/HiNER-collapsed (data/test.json, every 7th sentence of 21,657, CC-BY-SA-4.0)",
            "file": "ner_hi/hiner/hi_test.json", "sha256": sha256(out_path)}


def build():
    entries = []
    for dataset, lang, split, role in wanted():
        url_for, label_url = DATASETS[dataset][0], DATASETS[dataset][1]
        path = download(url_for(lang, split), local_path(dataset, lang, split))
        download(label_url, DATA / "ner" / dataset / "label.json")
        n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        entries.append({"dataset": dataset, "lang": lang, "split": split, "role": role, "sentences": n,
                        "source": url_for(lang, split), "file": str(path.relative_to(DATA)).replace("\\", "/"),
                        "sha256": sha256(path)})
    from make_hinglish_set import hinglish_entry
    from foreign_data import foreign_entries
    return entries + hindi_entries() + [hinglish_entry()] + foreign_entries(sha256)


def verify():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = 0
    for e in manifest["files"]:
        path = DATA / e["file"]
        ok = path.exists() and sha256(path) == e["sha256"]
        bad += not ok
        print(("ok      " if ok else "CHANGED ") + e["file"])
    return bad


def main():
    use_console_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        sys.exit(1 if verify() else 0)
    entries = build()
    MANIFEST.write_text(json.dumps({"created": date.today().isoformat(), "files": entries}, indent=2), encoding="utf-8")
    for e in entries:
        print(f"{e['dataset']:10s} {e['lang']} {e['role']:4s} {e['sentences']:6d} sentences  {e['sha256'][:12]}")
    print(f"\nwrote {MANIFEST}")


if __name__ == "__main__":
    main()
