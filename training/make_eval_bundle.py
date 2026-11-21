"""Create the Colab evaluation package: major/WS3_Colab/{WS3_Evaluate_Models_Colab.ipynb, colab_eval_bundle.zip}.

The bundle holds only what evaluation needs: backend/ner_clean.py (the app's entity clean-up, no web imports), the evaluation scripts and the
frozen Indian-language test sets, so run_all.py works in a bare Colab session.
"""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "WS3_Colab"
OUT.mkdir(exist_ok=True)
DATA = ROOT / "evaluation" / "data"
SCRIPTS = ["common.py", "eval_ner.py", "run_all.py", "bootstrap.py", "label_maps.py", "download_data.py", "error_analysis.py", "gliner_adapter.py"]
DATA_FILES = (["ner_hi/naamapadam/hi_test.json", "ner_hi/hiner/hi_test.json"]
              + [f"ner_indic/{l}_test.json" for l in ("as", "bn", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te")]
              + [f"ner_multi/mc_{l}_dev.json" for l in ("de", "es", "fr", "zh", "hi")]
              + ["ner_hi/hinglish/hi_test.json", "ner_hi/hinglish/deva_test.json"]
              + [f"ner_foreign/{n}_test.json" for n in ("uner_ru", "uner_de", "uner_zh", "uner_zhs", "anercorp", "stockmark_ja", "uner_ja")]
              + [f"ner_multi/mc_{l}_tune.json" for l in ("de", "es", "fr")]
              + ["ner/wikiann/fr_test.jsonl", "ner/wikiann/es_test.jsonl", "ner/wikiann/label.json"])

with zipfile.ZipFile(OUT / "colab_eval_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(ROOT / "backend" / "ner_clean.py", "backend/ner_clean.py")
    for f in SCRIPTS:
        z.write(ROOT / "evaluation" / f, f"evaluation/{f}")
    z.write(DATA / "MANIFEST.json", "evaluation/data/MANIFEST.json")
    for f in DATA_FILES:
        z.write(DATA / f, f"evaluation/data/{f}")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(True)}


DATASETS = "nm_as,nm_bn,nm_gu,nm_kn,nm_ml,nm_mr,nm_or,nm_pa,nm_ta,nm_te,naamapadam,hiner,mc_hi"
cells = [
    md("""
# WS3: measure the Indian-language entity models on a GPU

**What this does.** It scores three entity-recognition models on human-labelled test sentences in **Hindi and ten other Indian languages**
(all test sentences, not just the first 300 I could afford on the laptop):

| name | model | licence |
|---|---|---|
| `indicner` | ai4bharat/IndicNER (11 Indian languages) | MIT (gated: needs your Hugging Face token) |
| `indicbert_hi` | neuronsbyisshu/indicbert-hi-ner-naamapadam (trained on Hindi only) | CC0 |
| `davlan_multi` | Davlan/xlm-roberta-base-wikiann-ner (the model used before) | none declared |

**How long.** About 15-30 minutes on the free GPU. Keep the browser tab open.

**Before you start (once):**
1. **Runtime > Change runtime type > T4 GPU > Save.**
2. **Add your Hugging Face token as a Colab secret** (this keeps it out of the notebook):
   click the **key icon** on the left side of this page, then **+ Add new secret**, name it exactly `HF_TOKEN`, paste a **Read** token as the value
   (create a new one at huggingface.co > Settings > Access Tokens if you no longer have it), and switch **Notebook access** ON.
3. Have `colab_eval_bundle.zip` ready (it came with this notebook).

Then run the cells one at a time, top to bottom. **Never paste the token into a cell or send it to anyone.**
"""),
    code("""
# 1. Check that a GPU is attached (you should see a Tesla T4).
!nvidia-smi
"""),
    code("""
# 2. Install the library version that matches the laptop (about 1 minute). Warnings about gradio or diffusers are harmless.
!pip install -q "transformers==4.57.6" sentencepiece
"""),
    code("""
# 3. Log in to Hugging Face using the secret you added (the token is not shown).
from google.colab import userdata
from huggingface_hub import login, whoami
try:
    login(token=userdata.get('HF_TOKEN'))
    print("logged in as:", whoami()["name"])
except Exception as e:
    print("Could not log in. Did you add the secret named HF_TOKEN and switch 'Notebook access' on?", type(e).__name__)
"""),
    code("""
# 4. Upload colab_eval_bundle.zip when the file picker appears, then unpack it.
from google.colab import files
uploaded = files.upload()
!unzip -o -q colab_eval_bundle.zip && ls && ls evaluation | head -20
"""),
    md("""
**5. Quick check (about 2 minutes).** Scores IndicNER on the 51 Assamese sentences just to make sure the gated model downloads and the GPU is used.
If it ends with a line such as `nm_as/as/indicner: strict F1 ...` and no red error, continue.
"""),
    code("""
%env HF_HUB_DISABLE_XET=1
!python evaluation/run_all.py --models "indicner=ai4bharat/IndicNER" --datasets nm_as --full 2>&1 | grep -E "strict F1|skipped|Error|Traceback"
"""),
    md("**6. The real run.** One cell per model so you can see progress. Each prints one result line per test set."),
    code(f"""
!python evaluation/run_all.py --models "indicner=ai4bharat/IndicNER" --datasets {DATASETS} --full 2>&1 | grep -E "strict F1|skipped|Error|Traceback"
"""),
    code(f"""
!python evaluation/run_all.py --models "indicbert_hi=neuronsbyisshu/indicbert-hi-ner-naamapadam" --datasets {DATASETS} --full 2>&1 | grep -E "strict F1|skipped|Error|Traceback"
"""),
    code(f"""
!python evaluation/run_all.py --models "davlan_multi=Davlan/xlm-roberta-base-wikiann-ner" --datasets {DATASETS} --full 2>&1 | grep -E "strict F1|skipped|Error|Traceback"
"""),
    md("**7. Side-by-side table** (strict F1, higher is better). This is only for you to look at; the files below are what Claude will use."),
    code("""
import json, sys
sys.path.insert(0, "evaluation")
from run_all import summarize
models = ["indicner", "indicbert_hi", "davlan_multi"]
sets = {"nm_as":"Assamese","nm_bn":"Bengali","nm_gu":"Gujarati","nm_kn":"Kannada","nm_ml":"Malayalam","nm_mr":"Marathi","nm_or":"Odia",
        "nm_pa":"Punjabi","nm_ta":"Tamil","nm_te":"Telugu","naamapadam":"Hindi (Naamapadam)","hiner":"Hindi (HiNER)","mc_hi":"Hindi (MultiCoNER)"}
lang = {k: k[3:] if k.startswith(("nm_","mc_")) else "hi" for k in sets}
print(f"{'test set':22s}" + "".join(f"{m:>15s}" for m in models))
for ds, name in sets.items():
    row = f"{name:22s}"
    for m in models:
        try:
            rows = [json.loads(l) for l in open(f"evaluation/runs/{ds}__{lang[ds]}__{m}.jsonl", encoding="utf-8")]
            row += f"{summarize(rows)['strict'][2]:15.3f}"
        except FileNotFoundError:
            row += f"{'-':>15s}"
    print(row)
"""),
    code("""
# 8. Pack the saved predictions (a few MB) and download them.
!cd evaluation && zip -q -r ../ws3_runs.zip runs
from google.colab import files
files.download('ws3_runs.zip')
"""),
    md("""
## When it is finished
1. The download `ws3_runs.zip` goes to your Downloads folder. Extract it to a new folder named **ws3_results** in your user folder, for example `<your folder>/ws3_results` (not inside OneDrive).
2. Tell Claude it is there. Claude will compute the confidence intervals, compare the models language by language, and build the per-language status table.

If a cell shows a red error, copy the last 15 lines and send them to Claude.
"""),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "WS3_Evaluate_Models_Colab.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")

# ── second notebook: GLiNER zero-shot models (open licence, no login needed) ──
G_SETS = "nm_as,nm_bn,nm_gu,nm_kn,nm_ml,nm_mr,nm_or,nm_pa,nm_ta,nm_te,naamapadam,hiner,mc_hi,mc_de,mc_es,mc_fr,mc_zh"
g_cells = [
    md("""
# WS3b: measure the GLiNER models on the same test sentences

**What this does.** GLiNER models are *zero-shot*: instead of being trained on a fixed list of entity types, they are asked for types by name
("person", "organization", "location"). They are open (Apache-2.0). This notebook scores three of them on the **same human-labelled test sentences**
as before: Hindi and ten other Indian languages, plus German, Spanish, French and Chinese (the languages where the current model has no clear licence).

| name | model | licence |
|---|---|---|
| `gliner_x_base` | knowledgator/gliner-x-base | Apache-2.0 |
| `gliner_x_small` | knowledgator/gliner-x-small | Apache-2.0 |
| `gliner_multi` | urchade/gliner_multi-v2.1 | Apache-2.0 |

**How long.** About 20-40 minutes on the free GPU. Keep the browser tab open. **No Hugging Face login is needed** (the models are open).

**Before you start:** Runtime > Change runtime type > **T4 GPU** > Save. Have `colab_eval_bundle.zip` ready (the new one in the same folder).
The score threshold is GLiNER's default (0.5) and was **not tuned on test data**.
"""),
    code("""
# 1. GPU check (you should see a Tesla T4).
!nvidia-smi
"""),
    code("""
# 2. Install the GLiNER package (about 1-2 minutes). Red warnings about other packages are harmless; a real error would stop the cell.
!pip install -q gliner
"""),
    code("""
# 3. Upload colab_eval_bundle.zip when the file picker appears, then unpack it.
from google.colab import files
uploaded = files.upload()
!unzip -o -q colab_eval_bundle.zip && ls evaluation | head -20
"""),
    md("**4. Quick check (about 2 minutes).** Scores the smallest model on the 51 Assamese sentences to make sure everything works. If it ends with a result line and no red error, continue."),
    code("""
!python evaluation/run_all.py --models "gliner_x_small=gliner:knowledgator/gliner-x-small" --datasets nm_as --full 2>&1 | grep -E "strict F1|skipped|Error|Traceback|ImportError"
"""),
    md("**5. The real run.** One cell per model. Each prints one result line per test set."),
    code(f"""
!python evaluation/run_all.py --models "gliner_x_base=gliner:knowledgator/gliner-x-base" --datasets {G_SETS} --full 2>&1 | grep -E "strict F1|skipped|Error|Traceback"
"""),
    code(f"""
!python evaluation/run_all.py --models "gliner_x_small=gliner:knowledgator/gliner-x-small" --datasets {G_SETS} --full 2>&1 | grep -E "strict F1|skipped|Error|Traceback"
"""),
    code(f"""
!python evaluation/run_all.py --models "gliner_multi=gliner:urchade/gliner_multi-v2.1" --datasets {G_SETS} --full 2>&1 | grep -E "strict F1|skipped|Error|Traceback"
"""),
    code("""
# 6. Side-by-side table (strict F1; '-' = that run did not finish).
import json, sys
sys.path.insert(0, "evaluation")
from run_all import summarize
models = ["gliner_x_base", "gliner_x_small", "gliner_multi"]
sets = {"nm_as":"Assamese","nm_bn":"Bengali","nm_gu":"Gujarati","nm_kn":"Kannada","nm_ml":"Malayalam","nm_mr":"Marathi","nm_or":"Odia","nm_pa":"Punjabi",
        "nm_ta":"Tamil","nm_te":"Telugu","naamapadam":"Hindi (Naamapadam)","hiner":"Hindi (HiNER)","mc_hi":"Hindi (MultiCoNER)","mc_de":"German (MultiCoNER)",
        "mc_es":"Spanish (MultiCoNER)","mc_fr":"French (MultiCoNER)","mc_zh":"Chinese (MultiCoNER)"}
lang = {k: ("hi" if k in ("naamapadam","hiner") else k[3:]) for k in sets}
print(f"{'test set':24s}" + "".join(f"{m:>16s}" for m in models))
for ds, name in sets.items():
    row = f"{name:24s}"
    for m in models:
        try:
            rows = [json.loads(l) for l in open(f"evaluation/runs/{ds}__{lang[ds]}__{m}.jsonl", encoding="utf-8")]
            row += f"{summarize(rows)['strict'][2]:16.3f}"
        except FileNotFoundError:
            row += f"{'-':>16s}"
    print(row)
"""),
    code("""
# 7. Pack the saved predictions and download them.
!cd evaluation && zip -q -r ../ws3b_runs.zip runs
from google.colab import files
files.download('ws3b_runs.zip')
"""),
    md("""
## When it is finished
Extract `ws3b_runs.zip` to a new folder (for example `<your folder>/ws3b_runs`, **not inside OneDrive**) and tell Claude. Claude will compare GLiNER with the other models
language by language, with confidence intervals. If a cell shows a red error, copy the last 15 lines and send them to Claude.
"""),
]
# ── third notebook: retry the two knowledgator models (mT5-based; they need gliner[stanza] and sentencepiece) ──
X_SETS = G_SETS
r_cells = [
    md("""
# WS3c: retry the two GLiNER-X models

The first GLiNER notebook produced results for `gliner_multi` only: the two `knowledgator/gliner-x-*` models did not run. Their model cards say to
install **`gliner[stanza]`** (extra word-splitting tools for many languages), and they are built on mT5, which needs **sentencepiece**. This notebook installs both
and shows any error **in full** instead of hiding it.

**Before you start:** Runtime > Change runtime type > **T4 GPU** > Save. Have the newest `colab_eval_bundle.zip` ready. No Hugging Face login is needed.
"""),
    code("!nvidia-smi"),
    code("""
# 1. Install the packages the GLiNER-X models need (about 2-3 minutes). Warnings about other packages are harmless.
!pip install -q -U "gliner[stanza]" sentencepiece
"""),
    code("""
# 2. Upload colab_eval_bundle.zip, then unpack it.
from google.colab import files
uploaded = files.upload()
!unzip -o -q colab_eval_bundle.zip && ls evaluation | head
"""),
    md("**3. Quick check (about 3 minutes).** The smallest model on 51 Assamese sentences. This time the FULL output is shown: if it fails, copy the last 20 lines to Claude."),
    code("""
!python evaluation/run_all.py --models "gliner_x_small=gliner:knowledgator/gliner-x-small" --datasets nm_as --full 2>&1 | tail -25
"""),
    md("**4. The real run.** One cell per model; the last 12 lines of each are shown."),
    code(f"""
!python evaluation/run_all.py --models "gliner_x_base=gliner:knowledgator/gliner-x-base" --datasets {X_SETS} --full 2>&1 | grep -E "strict F1|skipped|Error|Traceback" | tail -25
"""),
    code(f"""
!python evaluation/run_all.py --models "gliner_x_small=gliner:knowledgator/gliner-x-small" --datasets {X_SETS} --full 2>&1 | grep -E "strict F1|skipped|Error|Traceback" | tail -25
"""),
    code("""
# 5. Pack and download the new predictions.
!cd evaluation && zip -q -r ../ws3c_runs.zip runs
from google.colab import files
files.download('ws3c_runs.zip')
"""),
    md("Extract `ws3c_runs.zip` to a new folder outside OneDrive (for example `<your folder>/ws3c_runs`) and tell Claude."),
]
nb3 = {**nb, "cells": r_cells}
(OUT / "WS3c_GLiNER_X_retry_Colab.ipynb").write_text(json.dumps(nb3, indent=1), encoding="utf-8")

# ── fourth notebook: only the runs that crashed in WS3c (GLiNER-X returned entities without character offsets) ──
d_cells = [
    md("""
# WS3d: the four GLiNER-X runs that crashed

Marathi and Tamil (both models) and German and French (base model) stopped with `TypeError: int() ... NoneType`: GLiNER-X sometimes returns an entity
whose start/end position is empty. The adapter now finds the entity's text in the sentence again (and prints how many it recovered or had to drop).

**Before you start:** Runtime > Change runtime type > **T4 GPU** > Save. Have the newest `colab_eval_bundle.zip` ready (it contains the fixed adapter). No Hugging Face login is needed.
"""),
    code('!pip install -q -U "gliner[stanza]" sentencepiece'),
    code("""
from google.colab import files
uploaded = files.upload()
!unzip -o -q colab_eval_bundle.zip && ls evaluation | head
"""),
    md("**Run both cells (about 5 minutes in total).**"),
    code("""
!python evaluation/run_all.py --models "gliner_x_base=gliner:knowledgator/gliner-x-base" --datasets nm_mr,nm_ta,mc_de,mc_fr --full 2>&1 | grep -E "strict F1|skipped|note:|Error|Traceback" | tail -25
"""),
    code("""
!python evaluation/run_all.py --models "gliner_x_small=gliner:knowledgator/gliner-x-small" --datasets nm_mr,nm_ta --full 2>&1 | grep -E "strict F1|skipped|note:|Error|Traceback" | tail -25
"""),
    code("""
!cd evaluation && zip -q -r ../ws3d_runs.zip runs
from google.colab import files
files.download('ws3d_runs.zip')
"""),
    md("Extract `ws3d_runs.zip` to a new folder outside OneDrive (for example `<your folder>/ws3d_runs`) and tell Claude."),
]
(OUT / "WS3d_GLiNER_X_fix_Colab.ipynb").write_text(json.dumps({**nb, "cells": d_cells}, indent=1), encoding="utf-8")

# ── fifth notebook: Hinglish (Hindi in English letters) ──
H_RUN = 'grep -E "strict F1|skipped|Error|Traceback"'
h_cells = [
    md("""
# WS3e: which model reads Hinglish best?

**Hinglish** is Hindi typed in English letters ("Rahul kal Mumbai ja raha hai"). The app cannot tell it is Hindi, so it sends it to the old multilingual model.
This notebook scores six ways of reading it on **125 test sentences (164 names, organisations and places)** that I wrote and that a Hindi speaker still has to check:

| name | what is scored |
|---|---|
| `davlan_multi` | what the app does today (the old multilingual model, no licence) |
| `english_bert` | the app's English model (dslim/bert-base-NER) on the Hinglish text |
| `indicbert_hi` | the Hindi model, given the English-letter text directly |
| `gliner_x_base`, `gliner_multi` | the two zero-shot GLiNER models on the English-letter text |
| `indicbert_hi` on `hinglish_deva` | the same sentences first converted to Devanagari (Azure transliteration, done on the laptop), then the Hindi model |

**How long.** About 10 minutes. **Before you start:** Runtime > Change runtime type > **T4 GPU** > Save. Have the newest `colab_eval_bundle.zip` ready. No Hugging Face login is needed.
"""),
    code("!nvidia-smi"),
    code("""
# 1. Install (1 minute).
!pip install -q "transformers==4.57.6" sentencepiece
"""),
    code("""
# 2. Upload colab_eval_bundle.zip, then unpack it.
from google.colab import files
uploaded = files.upload()
!unzip -o -q colab_eval_bundle.zip && ls evaluation | head -3
"""),
    md("**3. The three ordinary models on the English-letter text.**"),
    code(f"""
!python evaluation/run_all.py --models "davlan_multi=Davlan/xlm-roberta-base-wikiann-ner,english_bert=dslim/bert-base-NER,indicbert_hi=neuronsbyisshu/indicbert-hi-ner-naamapadam" --datasets hinglish --full 2>&1 | {H_RUN}
"""),
    md("**4. The Hindi model on the Devanagari version of the same sentences.**"),
    code(f"""
!python evaluation/run_all.py --models "indicbert_hi=neuronsbyisshu/indicbert-hi-ner-naamapadam" --datasets hinglish_deva --full 2>&1 | {{H_RUN}}
""".replace("{H_RUN}", H_RUN)),
    md("**5. The two GLiNER models** (installs one more package, about 2 minutes)."),
    code("""
!pip install -q -U "gliner[stanza]" sentencepiece
"""),
    code(f"""
!python evaluation/run_all.py --models "gliner_x_base=gliner:knowledgator/gliner-x-base,gliner_multi=gliner:urchade/gliner_multi-v2.1" --datasets hinglish --full 2>&1 | {H_RUN}
"""),
    code(f"""
!python evaluation/run_all.py --models "gliner_x_base=gliner:knowledgator/gliner-x-base" --datasets hinglish_deva --full 2>&1 | {H_RUN}
"""),
    code("""
# 6. Pack the predictions and download them.
!cd evaluation && zip -q -r ../ws3e_runs.zip runs
from google.colab import files
files.download('ws3e_runs.zip')
"""),
    md("Extract `ws3e_runs.zip` to a new folder outside OneDrive (for example `<your folder>/ws3e_runs`) and tell Claude."),
]
(OUT / "WS3e_Hinglish_Colab.ipynb").write_text(json.dumps({**nb, "cells": h_cells}, indent=1), encoding="utf-8")

# ── sixth notebook: Russian, Arabic, Japanese (and ordinary-text German / Chinese) ──
F_SETS = "uner_ru,uner_de,uner_zh,uner_zhs,anercorp,stockmark_ja"
F_GREP = 'grep -E "strict F1|skipped|note:|Error|Traceback"'
f_cells = [
    md("""
# WS3f: entity recognition for Russian, Arabic and Japanese

The app has never been measured on Russian, Arabic or Japanese. This notebook scores the current model and open-licence candidates on human-labelled test sets:

| test set | language | sentences | source (licence) |
|---|---|---|---|
| `uner_ru`, `uner_de`, `uner_zh` | Russian, German, Chinese (Traditional) | 1,000 each (the same news/Wikipedia sentences translated) | Universal NER, PUD (CC-BY-SA-4.0) |
| `uner_zhs` | Chinese (Simplified) | 500 | Universal NER, GSDSIMP (CC-BY-SA-4.0) |
| `anercorp` | Arabic | 924 (news) | ANERcorp, CAMeL Lab test split (CC-BY-SA-4.0) |
| `stockmark_ja` | Japanese | 5,343 (Wikipedia) | stockmark (CC-BY-SA-3.0) |

| name | model | licence |
|---|---|---|
| `davlan_multi` | Davlan/xlm-roberta-base-wikiann-ner (what the app uses now) | none declared |
| `hrl_base` | Davlan/xlm-roberta-base-ner-hrl (same size; trained on news in 10 languages incl. Arabic, German, Chinese) | AFL-3.0 (open) |
| `uner_all` | universalner/uner_all (XLM-R large, trained on Universal NER) | MIT |
| `camel_ar` | CAMeL-Lab/bert-base-arabic-camelbert-mix-ner (Arabic only) | Apache-2.0 |
| `gliner_x_base`, `gliner_multi` | zero-shot GLiNER models | Apache-2.0 |

**How long.** About 30-40 minutes (the Japanese set is large). Keep the tab open. **Before you start:** Runtime > Change runtime type > **T4 GPU** > Save. Have the newest `colab_eval_bundle.zip` ready. No Hugging Face login is needed.
"""),
    code("!nvidia-smi"),
    code("""
# 1. Install (1 minute).
!pip install -q "transformers==4.57.6" sentencepiece
"""),
    code("""
# 2. Upload colab_eval_bundle.zip, then unpack it.
from google.colab import files
uploaded = files.upload()
!unzip -o -q colab_eval_bundle.zip && ls evaluation | head -3
"""),
    md("**3. The ordinary models, one cell each** (about 3-8 minutes per cell)."),
    code(f"""
!python evaluation/run_all.py --models "davlan_multi=Davlan/xlm-roberta-base-wikiann-ner" --datasets {F_SETS} --full 2>&1 | {F_GREP}
"""),
    code(f"""
!python evaluation/run_all.py --models "hrl_base=Davlan/xlm-roberta-base-ner-hrl" --datasets {F_SETS},mc_de,mc_es,mc_fr,mc_zh --full 2>&1 | {F_GREP}
"""),
    code(f"""
!python evaluation/run_all.py --models "uner_all=universalner/uner_all" --datasets {F_SETS} --full 2>&1 | {F_GREP}
"""),
    code(f"""
!python evaluation/run_all.py --models "camel_ar=CAMeL-Lab/bert-base-arabic-camelbert-mix-ner" --datasets anercorp --full 2>&1 | {F_GREP}
"""),
    md("**4. The two GLiNER models** (installs one more package first)."),
    code("""
!pip install -q -U "gliner[stanza]" sentencepiece
"""),
    code(f"""
!python evaluation/run_all.py --models "gliner_x_base=gliner:knowledgator/gliner-x-base" --datasets {F_SETS} --full 2>&1 | {F_GREP}
"""),
    code(f"""
!python evaluation/run_all.py --models "gliner_multi=gliner:urchade/gliner_multi-v2.1" --datasets {F_SETS} --full 2>&1 | {F_GREP}
"""),
    code("""
# 5. Pack the predictions and download them.
!cd evaluation && zip -q -r ../ws3f_runs.zip runs
from google.colab import files
files.download('ws3f_runs.zip')
"""),
    md("Extract `ws3f_runs.zip` to a new folder outside OneDrive (for example `<your folder>/ws3f_runs`) and tell Claude. If a cell shows `skipped` or `Error`, send a screenshot of it as well."),
]
(OUT / "WS3f_Russian_Arabic_Japanese_Colab.ipynb").write_text(json.dumps({**nb, "cells": f_cells}, indent=1), encoding="utf-8")

# ── seventh notebook: GLiNER-X for de/fr/es with a tuned threshold, Otter and per-language specialists (items 1 and 3, research R4) ──
G = 'grep -E "strict F1|skipped|note:|not placed|Error|Traceback"'
TUNE = "mc_de_tune,mc_es_tune,mc_fr_tune"
DEFR = "mc_de,mc_es,mc_fr,uner_de"
R4 = "uner_ru,uner_ja,uner_zh,uner_zhs,mc_zh,anercorp,stockmark_ja"
WIKI = "--datasets wikiann --langs fr,es --n 3000"
LOW = "GLINER_THRESHOLD=0.1"
g7 = [
    md("""
# WS3g: which open model should replace the unlicensed one (German, French, Spanish, Russian, Arabic, Japanese, Chinese)?

Three things in one run:
1. **GLiNER-X for German, French and Spanish, done properly.** Its score threshold is chosen on a sample of the MultiCoNER *training* split
   (never on the test sets), and it is tested on cased text too: UNER German-PUD, and WikiANN French and Spanish (first 3,000 test sentences;
   WikiANN labels are machine-made "silver", and the current model was trained on WikiANN, so it has an advantage there).
2. **Otter** (`whoisjones/otter-cross-mmbert`, Apache-2.0), a newer zero-shot model, on every language.
3. **Per-language specialists** (all permissive): CamemBERT with dates (French, MIT), bert4ner (Chinese, Apache-2.0), CAMeLBERT-MSA (Arabic,
   Apache-2.0), llm-book BERT (Japanese, Apache-2.0; tested on UNER Japanese-PUD because it was trained on the stockmark set).
4. A re-run of four GLiNER-X sets with the improved adapter (it now finds more entities that come back without a position).

Zero-shot models are run once at a low threshold (0.1); the laptop then picks the threshold from the tuning sentences without running them again.

**How long.** About 60-90 minutes. Keep the tab open. **Before you start:** Runtime > Change runtime type > **T4 GPU** > Save; have the newest
`colab_eval_bundle.zip` ready. No Hugging Face login is needed. If a cell shows `skipped` or `Error`, carry on with the next cell and send Claude a screenshot.
"""),
    code("!nvidia-smi"),
    code("""
# 1. Install (about 2 minutes). fugashi + unidic-lite are needed by the Japanese model's word splitter.
!pip install -q "transformers==4.57.6" sentencepiece fugashi unidic-lite
"""),
    code("""
# 2. Upload colab_eval_bundle.zip, then unpack it.
from google.colab import files
uploaded = files.upload()
!unzip -o -q colab_eval_bundle.zip && ls evaluation | head -3
"""),
    md("**3. Models with a fixed label set** (no threshold to tune). One cell each."),
    code(f"""
# the current model and the open hrl model on cased French/Spanish (WikiANN) and on Japanese-PUD
!python evaluation/run_all.py --models "davlan_multi=Davlan/xlm-roberta-base-wikiann-ner,hrl_base=Davlan/xlm-roberta-base-ner-hrl" {WIKI} 2>&1 | {G}
!python evaluation/run_all.py --models "davlan_multi=Davlan/xlm-roberta-base-wikiann-ner,hrl_base=Davlan/xlm-roberta-base-ner-hrl,uner_all=universalner/uner_all" --datasets uner_ja --full 2>&1 | {G}
"""),
    code(f"""
# French specialist (CamemBERT, with dates)
!python evaluation/run_all.py --models "camembert_fr=Jean-Baptiste/camembert-ner-with-dates" --datasets mc_fr --full 2>&1 | {G}
!python evaluation/run_all.py --models "camembert_fr=Jean-Baptiste/camembert-ner-with-dates" --datasets wikiann --langs fr --n 3000 2>&1 | {G}
"""),
    code(f"""
# Chinese specialist (bert4ner, Simplified Chinese) and Arabic specialist (CAMeLBERT-MSA)
!python evaluation/run_all.py --models "bert4ner_zh=shibing624/bert4ner-base-chinese" --datasets uner_zhs,uner_zh,mc_zh --full 2>&1 | {G}
!python evaluation/run_all.py --models "camel_msa=CAMeL-Lab/bert-base-arabic-camelbert-msa-ner" --datasets anercorp --full 2>&1 | {G}
"""),
    code(f"""
# Japanese specialist (llm-book). NOT on stockmark: it was trained on that set.
!python evaluation/run_all.py --models "llmbook_ja=llm-book/bert-base-japanese-v3-ner-wikipedia-dataset" --datasets uner_ja --full 2>&1 | {G}
"""),
    md("**4. Otter** (runs code from its model page; the exact reviewed version is pinned in the adapter)."),
    code(f"""
!{LOW} python evaluation/run_all.py --models "otter=otter:whoisjones/otter-cross-mmbert" --datasets {TUNE},{DEFR},{R4} --full 2>&1 | {G}
!{LOW} python evaluation/run_all.py --models "otter=otter:whoisjones/otter-cross-mmbert" {WIKI} 2>&1 | {G}
"""),
    md("**5. The GLiNER models** (installs the gliner package first; it may update other libraries, which is why it comes last)."),
    code("""
!pip install -q -U "gliner[stanza]" sentencepiece
"""),
    code(f"""
!{LOW} python evaluation/run_all.py --models "gliner_x_base=gliner:knowledgator/gliner-x-base" --datasets {TUNE},{DEFR},{R4} --full 2>&1 | {G}
!{LOW} python evaluation/run_all.py --models "gliner_x_base=gliner:knowledgator/gliner-x-base" {WIKI} 2>&1 | {G}
"""),
    code(f"""
!{LOW} python evaluation/run_all.py --models "gliner_x_small=gliner:knowledgator/gliner-x-small" --datasets {TUNE},{DEFR},{R4} --full 2>&1 | {G}
!{LOW} python evaluation/run_all.py --models "gliner_x_small=gliner:knowledgator/gliner-x-small" {WIKI} 2>&1 | {G}
"""),
    code(f"""
# GLiNER multi: it produced no files in the last two runs, so this time every error line is shown
!{LOW} python evaluation/run_all.py --models "gliner_multi=gliner:urchade/gliner_multi-v2.1" --datasets {TUNE},{DEFR} --full 2>&1 | {G}
!{LOW} python evaluation/run_all.py --models "gliner_multi=gliner:urchade/gliner_multi-v2.1" {WIKI} 2>&1 | {G}
"""),
    md("**6. The improved adapter on the four sets that lost entities before** (default threshold 0.5, so it compares directly with the old run)."),
    code(f"""
!python evaluation/run_all.py --models "gliner_x_base_fix=gliner:knowledgator/gliner-x-base" --datasets nm_mr,nm_ta,mc_de,mc_fr --full 2>&1 | {G}
!python evaluation/run_all.py --models "gliner_x_small_fix=gliner:knowledgator/gliner-x-small" --datasets nm_mr,nm_ta --full 2>&1 | {G}
"""),
    code("""
# 7. Pack the predictions and download them.
!cd evaluation && zip -q -r ../ws3g_runs.zip runs
from google.colab import files
files.download('ws3g_runs.zip')
"""),
    md("Extract `ws3g_runs.zip` to `<your folder>/ws3g_runs` and tell Claude. Send screenshots of any cell that shows `skipped` or `Error`."),
]
(OUT / "WS3g_Replacement_Models_Colab.ipynb").write_text(json.dumps({**nb, "cells": g7}, indent=1), encoding="utf-8")

nb2 = {**nb, "cells": g_cells}
(OUT / "WS3b_GLiNER_Colab.ipynb").write_text(json.dumps(nb2, indent=1), encoding="utf-8")
print("wrote", OUT)
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}  {f.stat().st_size / 1e6:.1f} MB")
