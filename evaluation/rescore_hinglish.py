"""Re-attach the (corrected) gold labels of the Hinglish sets to prediction files produced on Colab, and save them in evaluation/runs/.

The Colab runs of 2026-11-18 (ws3e_runs) were made while ten lower-case sentences in the test set were missing their entity markup; the predictions do not depend on
the labels, so they are scored again against the corrected set instead of being recomputed.

    python evaluation/rescore_hinglish.py "<your folder>/ws3e_runs"
"""
import json
import sys
from pathlib import Path

import eval_ner
from common import use_console_utf8
from run_all import RUNS


def rescore(folder):
    out = []
    for f in sorted(Path(folder).glob("hinglish*.jsonl")):
        dataset = f.name.split("__")[0]
        sentences = eval_ner.load_split(dataset, "hi", "test", None)
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(rows) == len(sentences), (f.name, len(rows), len(sentences))
        for r, (tokens, tags) in zip(rows, sentences):
            text, offsets, gold = eval_ner.build(tokens, tags, "hi")
            assert text == r["text"], (f.name, text, r["text"])
            r["gold"] = sorted(gold)
        (RUNS / f.name).write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in rows), encoding="utf-8")
        out.append(f.name)
    return out


if __name__ == "__main__":
    use_console_utf8()
    for name in rescore(sys.argv[1]):
        print("rescored", name)
