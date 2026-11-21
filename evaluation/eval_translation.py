"""Translation quality of the app's providers, scored against human references.

Benchmark: WMT24++ (Deutsch et al., 2025; Apache-2.0) - English paragraphs from news, social media, speech and literary
text, each with a professionally post-edited translation in 55 languages. The English side is the source here, so this
measures English -> X. Segments flagged as bad sources and the benchmark's canary rows are skipped, and a fixed random
sample (seed 0) is used so runs are comparable.

Each provider is run through the app's real code path (`perform_translation`) with TRANSLATION_PROVIDERS pinned to that
single provider, so a fallback can never hide a failure: if a provider cannot translate a segment it is counted as
failed and scored as an empty translation.

Metrics (sacreBLEU):
    chrF++  character n-gram F-score with word bigrams - the primary metric (robust for Hindi, Arabic, Chinese)
    BLEU    shown for comparison with older papers (tokenizer: zh -> "zh", others "13a"; ja -> "char")

Usage (from the project root):
    python evaluation/eval_translation.py --langs hi --providers local,azure --n 60
    python evaluation/eval_translation.py --langs hi,fr,de,es,ru,zh,ar,ja --providers local --n 40 --out evaluation/translation_new.md   (then copy the table into RESULTS.md)

Cost / time notes: the local models translate roughly 30-60 characters per second on a laptop CPU, so 60 segments of
~250 characters take several minutes per language. Azure uses your own quota (a few thousand characters per run).
With 40-60 segments a difference of about 2 chrF++ points is within noise.
"""
import argparse
import json
import os
import random
import time

from common import DATA, download, load_backend, use_console_utf8

URL = "https://huggingface.co/datasets/google/wmt24pp/resolve/main/en-{code}.jsonl"
WMT_CODES = {"hi": "hi_IN", "fr": "fr_FR", "de": "de_DE", "es": "es_MX", "ru": "ru_RU", "zh": "zh_CN", "ar": "ar_SA", "ja": "ja_JP"}
BLEU_TOKENIZER = {"zh": "zh", "ja": "char"}
LABELS = {"azure": "Azure Translator", "local": "Local open-source models", "google": "Google (unofficial)", "mymemory": "MyMemory"}


def load_segments(lang, n):
    path = download(URL.format(code=WMT_CODES[lang]), DATA / "translation" / f"en-{WMT_CODES[lang]}.jsonl")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if not r["is_bad_source"] and r["domain"] != "canary" and 20 <= len(r["source"]) <= 700]
    random.Random(0).shuffle(rows)
    return rows[:n]


def main():
    use_console_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default="hi")
    ap.add_argument("--providers", default="local")
    ap.add_argument("--n", type=int, default=40, help="segments per language")
    ap.add_argument("--out", help="also write the table to this markdown file")
    args = ap.parse_args()

    import sacrebleu
    app = load_backend()
    rows = ["| language | provider | segments | failed | chrF++ | BLEU | characters/second |", "|---|---|---|---|---|---|---|"]
    samples = []
    for lang in (x.strip() for x in args.langs.split(",")):
        segments = load_segments(lang, args.n)
        refs = [s["target"] for s in segments]
        for provider in (p.strip() for p in args.providers.split(",")):
            os.environ["TRANSLATION_PROVIDERS"] = provider                   # exactly one provider: no hidden fallback
            for name in app._blocked_until:
                app._blocked_until[name] = 0.0
            hyps, failed, chars, t0 = [], 0, 0, time.time()
            for i, seg in enumerate(segments):
                try:
                    hyps.append(app.perform_translation("en", lang, seg["source"]))
                    chars += len(seg["source"])
                except Exception:
                    hyps.append("")
                    failed += 1
                if (i + 1) % 10 == 0:
                    print(f"  {lang}/{provider}: {i + 1}/{len(segments)}", flush=True)
            dt = max(time.time() - t0, 1e-9)
            chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
            bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=BLEU_TOKENIZER.get(lang, "13a")).score
            rows.append(f"| {lang} | {LABELS.get(provider, provider)} | {len(segments)} | {failed} | {chrf:.1f} | {bleu:.1f} | {chars / dt:.0f} |")
            print(f"{lang}/{provider}: chrF++ {chrf:.1f}  BLEU {bleu:.1f}  failed {failed}/{len(segments)}", flush=True)
            samples.append((lang, provider, segments[:3], hyps[:3]))

    report = ["## Translation quality, English -> X (WMT24++, human references)", "", *rows, "",
              "### Examples (first three sampled segments)", ""]
    for lang, provider, segs, hyps in samples:
        report.append(f"**{lang} / {LABELS.get(provider, provider)}**")
        report.append("")
        for seg, hyp in zip(segs, hyps):
            report += [f"* EN: {seg['source'][:160]}", f"  * system: {hyp[:160]}", f"  * reference: {seg['target'][:160]}"]
        report.append("")
    print("\n" + "\n".join(report))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
