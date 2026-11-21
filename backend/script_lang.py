"""Language from the writing system (Unicode script) for Indian languages.

Most Indian languages have a script of their own, so the script alone identifies them: Odia, Punjabi (Gurmukhi), Gujarati, Tamil, Telugu, Kannada
and Malayalam. langdetect, used for everything else, has no Odia profile at all (it raised an error, so Odia text was treated as English), and it
can confuse short texts. Two scripts are shared:
  * Devanagari: Hindi, Marathi, Nepali ... -> "deva": the caller asks langdetect, which separates them (and falls back to Hindi).
  * Bengali script: Bengali or Assamese -> Assamese when the text contains the letters ৰ or ৱ, which Bengali does not use; else Bengali.
Text in which no single script has at least MIN_SHARE of the letters (for example English with a few Hindi words) is left to the caller.
"""
import unicodedata

MIN_SHARE = 0.5

SCRIPT_RANGES = [                       # (first code point, last code point, language or script tag)
    (0x0900, 0x097F, "deva"),           # Devanagari: needs langdetect
    (0x0980, 0x09FF, "beng"),           # Bengali / Assamese
    (0x0A00, 0x0A7F, "pa"),             # Gurmukhi
    (0x0A80, 0x0AFF, "gu"),             # Gujarati
    (0x0B00, 0x0B7F, "or"),             # Odia
    (0x0B80, 0x0BFF, "ta"),             # Tamil
    (0x0C00, 0x0C7F, "te"),             # Telugu
    (0x0C80, 0x0CFF, "kn"),             # Kannada
    (0x0D00, 0x0D7F, "ml"),             # Malayalam
]
ASSAMESE_LETTERS = ("ৰ", "ৱ")   # ৰ ৱ


def script_language(text: str):
    """A language code ("or", "ta" ...), "deva" (Devanagari: ask langdetect), or None (no dominant Indian script)."""
    counts, letters = {}, 0
    for ch in text or "":
        if not (ch.isalpha() or unicodedata.category(ch).startswith("M")):
            continue                    # letters and combining vowel signs count; digits, spaces, punctuation do not
        letters += 1
        cp = ord(ch)
        for lo, hi, tag in SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[tag] = counts.get(tag, 0) + 1
                break
    if not letters or not counts:
        return None
    tag, n = max(counts.items(), key=lambda kv: kv[1])
    if n / letters < MIN_SHARE:
        return None
    if tag == "beng":
        return "as" if any(c in text for c in ASSAMESE_LETTERS) else "bn"
    return tag
