"""Compare several translation models on the same WMT24++ segments, outside the app (the app code is not touched).

For each model it reports chrF++ and BLEU (as in eval_translation.py) plus two sanity signals that a single score hides:
  * length ratio   mean(len(output) / len(reference)); far from 1 means truncated or runaway output
  * suspect        share of outputs shorter than 0.4x or longer than 2.5x the reference (typical of hallucinated text)

    python evaluation/bakeoff_translation.py --lang ja --n 40 \
        --models Helsinki-NLP/opus-tatoeba-en-ja,staka/fugumt-en-ja,facebook/m2m100_418M
"""
import argparse
import time

from common import use_console_utf8
from eval_translation import BLEU_TOKENIZER, load_segments

M2M = "facebook/m2m100_"


def translate_all(model_id, lang, sources, prefix="", beams=2, batch=8):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).eval()
    extra = {}
    if model_id.startswith(M2M):
        tok.src_lang = "en"
        extra["forced_bos_token_id"] = tok.get_lang_id(lang)
    order = sorted(range(len(sources)), key=lambda i: len(sources[i]))
    out = [""] * len(sources)
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        enc = tok([prefix + sources[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=400)
        with torch.inference_mode():
            gen = model.generate(**enc, num_beams=beams, max_new_tokens=400, **extra)
        for i, text in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[i] = text.strip()
    return out


def main():
    use_console_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--models", required=True, help="comma-separated Hugging Face ids")
    ap.add_argument("--prefix", default="", help="sentence-initial target token for multi-target models, e.g. '>>cmn_Hans<<'")
    args = ap.parse_args()

    import sacrebleu
    segments = load_segments(args.lang, args.n)
    sources, refs = [s["source"] for s in segments], [s["target"] for s in segments]
    print("| model | chrF++ | BLEU | length ratio | suspect outputs | seconds |\n|---|---|---|---|---|---|")
    for model_id in (m.strip() for m in args.models.split(",")):
        t0 = time.time()
        hyps = translate_all(model_id, args.lang, sources, prefix=(args.prefix + " ") if args.prefix else "")
        dt = time.time() - t0
        chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=BLEU_TOKENIZER.get(args.lang, "13a")).score
        ratios = [len(h) / max(1, len(r)) for h, r in zip(hyps, refs)]
        suspect = sum(1 for x in ratios if x < 0.4 or x > 2.5)
        print(f"| `{model_id}` | {chrf:.1f} | {bleu:.1f} | {sum(ratios) / len(ratios):.2f} | {suspect}/{len(ratios)} | {dt:.0f} |", flush=True)


if __name__ == "__main__":
    main()
