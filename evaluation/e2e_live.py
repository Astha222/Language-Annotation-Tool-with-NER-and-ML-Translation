"""Live end-to-end check of a RUNNING backend with Hindi, the other offered Indian languages and Hinglish (WS3, item 6).

Start the backend first, preferably against throw-away databases, then run this script:

    # terminal 1 (backend folder)
    POSTGRES_URL=sqlite:///<scratch>/e2e.db MONGO_DB=annotationtool_e2e PRELOAD_MODELS=none python -m uvicorn main:app --port 8000
    # terminal 2
    python evaluation/e2e_live.py --base http://127.0.0.1:8000

What it does, through the real HTTP API: language list and statuses, register + login, language detection, entity recognition and
translation for one sentence per language, save / list / export an annotation, and a full document job (upload, analyze, poll).
The sentences were written for this check (one per language, with names the model should find); they are a smoke test, not an accuracy test:
"found" lists which expected names came back, so a miss is visible, but accuracy figures come from evaluation/RESULTS.md.
"""
import argparse
import json
import sys
import time
import uuid

import requests

SAMPLES = [
    ("hi", "प्रधानमंत्री नरेंद्र मोदी ने मंगलवार को मुंबई में टाटा समूह के अधिकारियों से मुलाकात की।", ["नरेंद्र मोदी", "मुंबई", "टाटा"]),
    ("mr", "मुख्यमंत्री एकनाथ शिंदे यांनी पुण्यात रिलायन्सच्या अधिकाऱ्यांची भेट घेतली.", ["एकनाथ शिंदे", "पुण्यात", "रिलायन्स"]),
    ("bn", "রবীন্দ্রনাথ ঠাকুর কলকাতায় জন্মগ্রহণ করেন এবং বিশ্বভারতী বিশ্ববিদ্যালয় প্রতিষ্ঠা করেন।", ["রবীন্দ্রনাথ ঠাকুর", "কলকাতা", "বিশ্বভারতী"]),
    ("ta", "சச்சின் டெண்டுல்கர் சென்னையில் இந்திய கிரிக்கெட் வாரியத்தின் கூட்டத்தில் கலந்து கொண்டார்.", ["சச்சின் டெண்டுல்கர்", "சென்னை", "இந்திய கிரிக்கெட் வாரிய"]),
    ("te", "మహేష్ బాబు హైదరాబాద్‌లో కొత్త సినిమా ప్రారంభించారు.", ["మహేష్ బాబు", "హైదరాబాద్"]),
    ("gu", "નરેન્દ્ર મોદી અમદાવાદમાં ગુજરાત યુનિવર્સિટીની મુલાકાતે ગયા.", ["નરેન્દ્ર મોદી", "અમદાવાદ", "ગુજરાત યુનિવર્સિટી"]),
    ("kn", "ಸಿದ್ದರಾಮಯ್ಯ ಬೆಂಗಳೂರಿನಲ್ಲಿ ಇನ್ಫೋಸಿಸ್ ಕಚೇರಿಗೆ ಭೇಟಿ ನೀಡಿದರು.", ["ಸಿದ್ದರಾಮಯ್ಯ", "ಬೆಂಗಳೂರ", "ಇನ್ಫೋಸಿಸ್"]),
    ("ml", "മമ്മൂട്ടി കൊച്ചിയിൽ പുതിയ സിനിമയുടെ ചിത്രീകരണം ആരംഭിച്ചു.", ["മമ്മൂട്ടി", "കൊച്ചി"]),
    ("pa", "ਭਗਵੰਤ ਮਾਨ ਨੇ ਅੰਮ੍ਰਿਤਸਰ ਵਿੱਚ ਪੰਜਾਬ ਯੂਨੀਵਰਸਿਟੀ ਦਾ ਦੌਰਾ ਕੀਤਾ।", ["ਭਗਵੰਤ ਮਾਨ", "ਅੰਮ੍ਰਿਤਸਰ", "ਪੰਜਾਬ ਯੂਨੀਵਰਸਿਟੀ"]),
    ("hi", "Rahul kal Mumbai ja raha hai Infosys ki meeting ke liye .", ["Rahul", "Mumbai", "Infosys"]),          # Hinglish
]
DOCUMENT = ("भारत की राजधानी नई दिल्ली है। प्रधानमंत्री नरेंद्र मोदी ने आज लखनऊ में एक सभा को संबोधित किया। "
            "रिलायंस इंडस्ट्रीस के अध्यक्ष मुकेश अंबानी ने मुंबई में नई योजना की घोषणा की। ") * 3


class Check:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail=""):
        self.rows.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""), flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--target", default="en", help="translation target")
    ap.add_argument("--skip-translation", action="store_true")
    ap.add_argument("--json-out", help="write every response to this file")
    args = ap.parse_args()
    B, c, log = args.base.rstrip("/"), Check(), {}

    r = requests.get(f"{B}/api/languages", timeout=30)
    langs = r.json()
    items = langs.get("languages", langs) if isinstance(langs, dict) else langs
    codes = {x.get("code"): x for x in items} if isinstance(items, list) else {}
    c.add("GET /api/languages", r.ok and all(k in codes for k in ("hi", "mr", "bn", "ta", "te", "gu", "kn", "ml", "pa")),
          f"{len(codes)} languages; hi status: {(codes.get('hi') or {}).get('ner', {}).get('status') if isinstance((codes.get('hi') or {}).get('ner'), dict) else (codes.get('hi') or {}).get('status')}")
    log["languages"] = langs

    user = "e2e" + uuid.uuid4().hex[:8]
    r = requests.post(f"{B}/api/auth/register", json={"username": user, "email": f"{user}@example.com", "password": "password123"}, timeout=30)
    c.add("register", r.status_code == 201, str(r.status_code))
    tok = requests.post(f"{B}/api/auth/login", json={"username": user, "password": "password123"}, timeout=30).json().get("access_token")
    H = {"Authorization": f"Bearer {tok}"}
    c.add("login", bool(tok))

    log["samples"] = []
    for lang, text, expected in SAMPLES:
        label = "Hinglish" if lang == "hi" and text.isascii() else lang
        t0 = time.perf_counter()
        det = requests.post(f"{B}/api/detect-language", json={"text": text}, headers=H, timeout=60).json()
        t1 = time.perf_counter()
        ner = requests.post(f"{B}/api/ner", json={"text": text, "lang": "auto"}, headers=H, timeout=300)
        t2 = time.perf_counter()
        ents = ner.json().get("entities", []) if ner.ok else []
        words = [e.get("word") or text[e["start"]:e["end"]] for e in ents]
        found = [x for x in expected if any(x in w or w in x for w in words)]
        c.add(f"{label}: detect-language", det.get("detected_language") == lang,
              f"got {det.get('detected_language')}" + (" (romanized)" if det.get("romanized") else "") + f", {1000 * (t1 - t0):.0f} ms")
        c.add(f"{label}: NER (auto language)", ner.ok and len(found) >= max(1, len(expected) - 1),
              f"{len(ents)} entities {[(e.get('entity_group'), w) for e, w in zip(ents, words)]}; expected names found {len(found)}/{len(expected)}; {1000 * (t2 - t1):.0f} ms")
        entry = {"lang": label, "detect": det, "ner_status": ner.status_code, "entities": ents}
        if not args.skip_translation:
            t3 = time.perf_counter()
            tr = requests.post(f"{B}/api/translate", json={"text": text, "source_lang": "auto", "target_lang": args.target}, headers=H, timeout=300)
            out = tr.json() if tr.ok else {"error": tr.text[:200]}
            changed = tr.ok and out.get("translated_text", "").strip() not in ("", text.strip())
            c.add(f"{label}: translate -> {args.target}", changed,
                  f"{out.get('provider', '')}: {str(out.get('translated_text', out.get('error')))[:110]}; {1000 * (time.perf_counter() - t3):.0f} ms")
            entry["translation"] = out
        log["samples"].append(entry)

    # save, list, export one Hindi annotation
    text = SAMPLES[0][1]
    ents = next(s["entities"] for s in log["samples"] if s["lang"] == "hi")
    r = requests.post(f"{B}/api/annotations", json={"original_text": text, "translated_text": None, "ner_tags": json.dumps(ents),
                                                     "source_lang": "hi", "target_lang": args.target}, headers=H, timeout=60)
    c.add("save annotation (Hindi)", r.status_code == 201, str(r.status_code) + (" " + r.text[:150] if r.status_code != 201 else ""))
    lst = requests.get(f"{B}/api/annotations", headers=H, timeout=60)
    c.add("list annotations", lst.ok and any(a.get("original_text") == text for a in lst.json()), f"{len(lst.json()) if lst.ok else lst.status_code} rows")
    ex = requests.get(f"{B}/api/export", headers=H, timeout=120)       # JSON; the app builds CSV/Excel/PDF from it in the browser
    c.add("export (JSON)", ex.ok and "नरेंद्र" in ex.content.decode("utf-8", "replace"), f"{ex.status_code}, {len(ex.content)} bytes")

    # document job
    up = requests.post(f"{B}/api/documents", files={"file": ("hindi_e2e.txt", DOCUMENT.encode("utf-8"), "text/plain")}, headers=H, timeout=120)
    c.add("upload document (Hindi .txt)", up.ok, (up.json().get("detected_language") if up.ok else up.text[:150]) or "")
    if up.ok:
        doc = up.json()["doc_id"]
        job = requests.post(f"{B}/api/documents/{doc}/analyze", json={"source_lang": "auto", "target_lang": args.target}, headers=H, timeout=60).json()
        state = {}
        for _ in range(180):
            state = requests.get(f"{B}/api/jobs/{job['job_id']}", headers=H, timeout=60).json()
            if state.get("status") in ("done", "error", "cancelled", "failed"):
                break
            time.sleep(2)
        res = state.get("result") or {}
        c.add("document job (detect + NER + translation)", state.get("status") == "done",
              f"status {state.get('status')}; source {res.get('source_lang')}; {res.get('entity_total')} entities ({res.get('entity_unique')} unique); "
              f"engines {res.get('translation_engines')}; translation {str(res.get('translated_preview', ''))[:80]!r}; warnings {state.get('warnings')}; error {state.get('error')}")
        log["document_job"] = state

    ok = sum(1 for _, k, _ in c.rows if k)
    print(f"\n{ok}/{len(c.rows)} checks passed")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"checks": c.rows, "log": log}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
