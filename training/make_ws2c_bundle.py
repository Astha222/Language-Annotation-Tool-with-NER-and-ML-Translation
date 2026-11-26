"""Create the Colab package for option B of the CoNLL question: major/WS2_Colab/{WS2c_noCoNLL_Colab.ipynb, ws2c_bundle.zip}.

Kept apart from make_colab_bundle.py on purpose: running that script rebuilds the WS2 and WS2b notebooks and bundles that were actually run on Colab,
and those must stay as they were.

    python training/make_ws2c_bundle.py --out "C:/Users/muska/OneDrive - UPES/major/WS2_Colab"
(default --out: a WS2_Colab folder next to the repository, which is where the project folder used to keep the notebooks)
"""
import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "WS2_Colab"
RUN = "B_noCoNLL_plain"
SEEDS = (0, 1, 2)
DATASETS = "wnut2017,btc,fin,uner_en_ewt,conll2003"
EVAL_SCRIPTS = ["common.py", "eval_ner.py", "run_all.py", "bootstrap.py", "label_maps.py", "download_data.py", "gliner_adapter.py"]
EVAL_DATA = (["MANIFEST.json"] + [f"ner/{d}/{f}" for d in ("wnut2017", "btc", "fin", "conll2003") for f in ("en_test.jsonl", "label.json")]
             + ["ner_foreign/uner_en_ewt_test.json"])


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(True)}


def bundle(path: Path):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "training" / "distill.py", "distill.py")
        z.write(ROOT / "training" / "colab_runner.py", "colab_runner.py")
        z.write(ROOT / "evaluation" / "data" / "train_mix.jsonl", "train_mix.jsonl")
        z.write(ROOT / "evaluation" / "data" / "dev_mix.jsonl", "dev_mix.jsonl")
        z.write(ROOT / "backend" / "ner_clean.py", "backend/ner_clean.py")
        for f in EVAL_SCRIPTS:
            z.write(ROOT / "evaluation" / f, f"evaluation/{f}")
        for f in EVAL_DATA:
            z.write(ROOT / "evaluation" / "data" / f, f"evaluation/data/{f}")


def notebook():
    models = ",".join(f"{RUN}_s{s}=out/{RUN}_s{s}/best" for s in SEEDS)
    cells = [
        md("""
# WS2c: the English model trained WITHOUT CoNLL-2003 (backup for the licence question)

**Why.** The English model `B_noFIN` learned from three datasets. One of them, **CoNLL-2003, is Reuters news text** that is handed out for research use.
If the mentor or the UPES IPR cell decide that a model trained on it cannot be published under an open licence, we need a replacement. This notebook
trains the same recipe **only on WNUT-2017 and the Broad Twitter Corpus** (both CC-BY-4.0, 8,608 training sentences), three times with different random starts,
and scores the results, so the decision can be made with numbers.

| name | trained on |
|---|---|
| `B_noFIN_plain` (today's model, for comparison) | CoNLL-2003 + WNUT-2017 + BTC |
| `B_noCoNLL_plain_s0`, `_s1`, `_s2` (new) | WNUT-2017 + BTC only |

The best checkpoint is chosen on the same WNUT + BTC validation sentences as before. Then every new model is scored on five test sets, including the
CoNLL **test** set (news), only to see how much is lost on news text without CoNLL.

**How long.** About 20-30 minutes on the free GPU. Keep the tab open.

**Before you start:**
1. **Runtime > Change runtime type > T4 GPU > Save.**
2. Have `ws2c_bundle.zip` ready (it is next to this notebook).
3. Your Google Drive needs about **1 GB free**: the three trained models are copied to `MyDrive/ws2c_results` as soon as each finishes.
"""),
        code("!nvidia-smi"),
        code("""
# 1. Install the library version that matches the laptop (about 1 minute).
!pip install -q "transformers==4.57.6" sentencepiece
"""),
        code("""
# 2. Upload ws2c_bundle.zip when the file picker appears, then unpack it.
from google.colab import files
uploaded = files.upload()
!unzip -o -q ws2c_bundle.zip && ls
"""),
        code("""
# 3. Connect Google Drive (finished models are copied to MyDrive/ws2c_results).
from google.colab import drive
drive.mount('/content/drive')
"""),
        code("""
# 4. Validation file WITHOUT FIN sentences (the same one B_noFIN used), so the comparison is fair.
import json
rows = [l for l in open('dev_mix.jsonl', encoding='utf-8') if l.strip() and json.loads(l)['source'] != 'fin']
open('dev_mix_nofin.jsonl', 'w', encoding='utf-8').writelines(rows)
print(len(rows), 'validation sentences (WNUT + BTC)')
"""),
        md("**5. Train the three models** (progress lines show the loss and a validation F1 that should rise). The training log must say `8608` training sentences."),
        code(f"""
!python colab_runner.py --only {RUN} --seeds {",".join(map(str, SEEDS))} --dev dev_mix_nofin.jsonl --drive-dir /content/drive/MyDrive/ws2c_results
"""),
        md("**6. Score every model on the five test sets** (about 5-10 minutes). One line per model and test set."),
        code(f"""
!python evaluation/run_all.py --models "{models}" --datasets {DATASETS} --full 2>&1 | grep -E "strict F1|skipped|Error|Traceback"
"""),
        code("""
# 7. Pack the predictions (a few MB) and download them.
!cd evaluation && zip -q -r ../ws2c_runs.zip runs
from google.colab import files
files.download('ws2c_runs.zip')
"""),
        md("""
## When it is finished
Extract `ws2c_runs.zip` to `C:/Users/muska/major project/ws2c_runs` and tell Claude. The trained models stay in Google Drive (`ws2c_results`):
do not download them yet. Claude will compare them with today's model and say honestly what dropping CoNLL costs.

If a cell shows a red error, copy the last 15 lines of the error and send them to Claude.
"""),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT, help="folder for the notebook and the bundle")
    out = ap.parse_args().out
    out.mkdir(exist_ok=True)
    bundle(out / "ws2c_bundle.zip")
    (out / "WS2c_noCoNLL_Colab.ipynb").write_text(json.dumps(notebook(), indent=1), encoding="utf-8")
    for f in ("WS2c_noCoNLL_Colab.ipynb", "ws2c_bundle.zip"):
        print(f"wrote {out / f}  {(out / f).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
