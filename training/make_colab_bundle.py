"""Create the Colab package: major/WS2_Colab/{WS2_Distillation_Colab.ipynb, colab_bundle.zip}."""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "WS2_Colab"
OUT.mkdir(exist_ok=True)

with zipfile.ZipFile(OUT / "colab_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(ROOT / "training" / "distill.py", "distill.py")
    z.write(ROOT / "training" / "colab_runner.py", "colab_runner.py")
    z.write(ROOT / "evaluation" / "data" / "train_mix.jsonl", "train_mix.jsonl")
    z.write(ROOT / "evaluation" / "data" / "dev_mix.jsonl", "dev_mix.jsonl")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(True)}


cells = [
    md("""
# WS2: teach a small, fast NER model to copy a big, accurate one (distillation)

**What this does.** A big model (`roberta-large-ner-english`, the *teacher*) reads about 22,000 training sentences and says how sure it is
about each word (person / organization / location). A small, fast model (the *student*) is then trained to give the same answers.
The goal is a model that is about as accurate as the teacher but several times faster on a normal laptop.

**How long.** About 45-75 minutes on the free GPU. Keep this browser tab open and your computer awake while it runs.

**Before you start (once):**
1. Menu **Runtime > Change runtime type > T4 GPU > Save**.
2. Have the file `colab_bundle.zip` ready (it came with this notebook).

**Then:** run the cells below one at a time, top to bottom (click the play button on each, or use **Runtime > Run all** after step 1).
Nothing here uses private data: the training sentences come from four public datasets.
"""),
    code("""
# 1. Check that a GPU is attached. You should see a Tesla T4 below. If you see "failed", do the Runtime step above.
!nvidia-smi
"""),
    code("""
# 2. Install the library version that matches the laptop (about 1 minute).
!pip install -q "transformers==4.57.6" sentencepiece
"""),
    code("""
# 3. Upload colab_bundle.zip when the file picker appears, then unpack it.
from google.colab import files
uploaded = files.upload()
!unzip -o -q colab_bundle.zip && ls -la
"""),
    code("""
# 4. Connect Google Drive. Finished models are copied there as soon as each one is done, so nothing is lost if Colab disconnects.
from google.colab import drive
drive.mount('/content/drive')
"""),
    md("""
**5. Quick check (about 2 minutes).** Trains a tiny model on 256 sentences just to make sure everything works on this GPU.
If this cell ends with a *summary* table and no red error, continue. (It writes to a throw-away folder, not to your Drive.)
"""),
    code("""
!python colab_runner.py --only B_distilroberta_plain --out-dir quick_test --extra "--limit 256 --dev-limit 200 --epochs 1 --batch 16"
"""),
    md("""
**6. The real run (about 45-75 minutes).** Four experiments, one after another:

| name | what it is | why |
|---|---|---|
| A_distilroberta_distill | small model + teacher, all four datasets | **the main result** |
| B_distilroberta_plain | same, but *without* the teacher | shows how much the teacher helps |
| C_distilroberta_zeroshot | trained on news only | keeps Twitter / WNUT / finance *unseen*, for a fair comparison |
| D_deberta_distill | a different small model | the improvement plan's first choice |

You will see progress lines (loss and a "dev F1" score that should climb). Each finished experiment is copied to
`MyDrive/ws2_results/`. Do not close the tab.
"""),
    code("""
!python colab_runner.py --drive-dir /content/drive/MyDrive/ws2_results
"""),
    md("""
**7. Optional: repeat the main experiment with two more random starts** (about 15 minutes). Results can change a little with the random
start, so the improvement plan asks for three. Only run this if time allows.
"""),
    code("""
# !python colab_runner.py --only A_distilroberta_distill --seeds 1,2 --drive-dir /content/drive/MyDrive/ws2_results
"""),
    md("""
## When it is finished
1. Open **Google Drive**, find the folder **ws2_results**, and download it (right-click > Download; it is about 1-2 GB).
2. Put it in your project folder as `major/ws2_results/` and tell Claude it is there.
3. Claude will then measure the new models on the laptop (on test sentences they have never seen) and tell you honestly whether they are better.

If a cell shows a red error, copy the last 15 lines of the error and send them to Claude.
"""),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "WS2_Distillation_Colab.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")

# ── WS2b: licence-safe students without FIN, three seeds each, scored on Colab right after training ──
EVAL_SCRIPTS = ["common.py", "eval_ner.py", "run_all.py", "bootstrap.py", "label_maps.py", "download_data.py", "gliner_adapter.py"]
EVAL_DATA = ["MANIFEST.json"] + [f"ner/{d}/{f}" for d in ("wnut2017", "btc", "fin") for f in ("en_test.jsonl", "label.json")] + ["ner_foreign/uner_en_ewt_test.json"]
with zipfile.ZipFile(OUT / "ws2b_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(ROOT / "training" / "distill.py", "distill.py")
    z.write(ROOT / "training" / "colab_runner.py", "colab_runner.py")
    z.write(ROOT / "evaluation" / "data" / "train_mix.jsonl", "train_mix.jsonl")
    z.write(ROOT / "evaluation" / "data" / "dev_mix.jsonl", "dev_mix.jsonl")
    z.write(ROOT / "backend" / "ner_clean.py", "backend/ner_clean.py")
    for f in EVAL_SCRIPTS:
        z.write(ROOT / "evaluation" / f, f"evaluation/{f}")
    for f in EVAL_DATA:
        z.write(ROOT / "evaluation" / "data" / f, f"evaluation/data/{f}")

SEEDED = ",".join(f"{n}_s{s}" for n in ("A_noFIN_distill", "B_noFIN_plain") for s in (0, 1, 2))
MODELS = ",".join(f"{n}_s{s}=out/{n}_s{s}/best" for n in ("A_noFIN_distill", "B_noFIN_plain") for s in (0, 1, 2))
b_cells = [
    md("""
# WS2b: English students trained WITHOUT the FIN dataset (3 random starts each)

**Why.** The FIN dataset's original licence is not verified, so a model trained on it cannot be shipped. This notebook trains the two small English
models again **without FIN**:

| name | what it is |
|---|---|
| `A_noFIN_distill` | like student A: learns from the human labels AND from the big teacher model |
| `B_noFIN_plain` | like student B: learns from the human labels only. Student B was the best small model on the fair English test (English-EWT, 0.771) |

Each is trained three times with different random starts (`_s0`, `_s1`, `_s2`), so we can see how much the result moves by chance. The best checkpoint
is chosen on WNUT and BTC validation sentences only (no FIN). Right after training, every model is scored on the four English test sets.

**How long.** About 60-80 minutes on the free GPU. Keep the tab open.

**Before you start:**
1. **Runtime > Change runtime type > T4 GPU > Save.**
2. Have `ws2b_bundle.zip` ready (it is next to this notebook).
3. Your Google Drive needs about **2 GB free**: the six trained models are copied there as soon as each finishes.
"""),
    code("!nvidia-smi"),
    code("""
# 1. Install the library version that matches the laptop (about 1 minute).
!pip install -q "transformers==4.57.6" sentencepiece
"""),
    code("""
# 2. Upload ws2b_bundle.zip when the file picker appears, then unpack it.
from google.colab import files
uploaded = files.upload()
!unzip -o -q ws2b_bundle.zip && ls
"""),
    code("""
# 3. Connect Google Drive (finished models are copied to MyDrive/ws2b_results).
from google.colab import drive
drive.mount('/content/drive')
"""),
    code("""
# 4. Make a validation file WITHOUT FIN sentences, so FIN is not used even to pick the best checkpoint.
import json
rows = [l for l in open('dev_mix.jsonl', encoding='utf-8') if l.strip() and json.loads(l)['source'] != 'fin']
open('dev_mix_nofin.jsonl', 'w', encoding='utf-8').writelines(rows)
print(len(rows), 'validation sentences (WNUT + BTC)')
"""),
    md("**5. Train the six models** (the long step; progress lines show the loss and a validation F1 that should rise)."),
    code("""
!python colab_runner.py --only A_noFIN_distill,B_noFIN_plain --seeds 0,1,2 --dev dev_mix_nofin.jsonl --drive-dir /content/drive/MyDrive/ws2b_results
"""),
    md("**6. Score every model on the four English test sets** (about 5 minutes). One line per model and test set."),
    code(f"""
!python evaluation/run_all.py --models "{MODELS}" --datasets wnut2017,btc,fin,uner_en_ewt --full 2>&1 | grep -E "strict F1|skipped|Error|Traceback"
"""),
    code("""
# 7. Pack the predictions (a few MB) and download them.
!cd evaluation && zip -q -r ../ws2b_runs.zip runs
from google.colab import files
files.download('ws2b_runs.zip')
"""),
    md("""
## When it is finished
Extract `ws2b_runs.zip` to `<your folder>/ws2b_runs` and tell Claude. The trained models stay in Google Drive (`ws2b_results`);
do not download them yet: Claude will say which one, if any, is worth bringing to the laptop.
"""),
]
(OUT / "WS2b_noFIN_Seeds_Colab.ipynb").write_text(json.dumps({**nb, "cells": b_cells}, indent=1), encoding="utf-8")
print("wrote", OUT)
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}  {f.stat().st_size / 1e6:.1f} MB")
