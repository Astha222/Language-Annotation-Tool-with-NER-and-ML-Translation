"""Pre/post-processing for IndicTrans2 models, in pure Python.

This is a port of ``IndicProcessor`` from IndicTransToolkit (https://github.com/VarunGumma/IndicTransToolkit,
MIT License, Copyright (c) Varun Gumma). The original is a Cython extension that needs a C compiler to install;
this version has the same behaviour for the steps IndicTrans2 relies on but installs everywhere.

Differences from the original: the placeholder maps are returned explicitly by ``preprocess_batch`` and handed back
to ``postprocess_batch`` (the original keeps them in a shared queue, which is unsafe with concurrent callers).

Needs: regex, sacremoses, indic-nlp-library-itt.
"""
from typing import Dict, List, Optional, Tuple

import regex as re
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_detokenize, indic_tokenize
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer

# FLORES-200 language code -> ISO code used by indic-nlp-library
_FLORES_CODES = {
    "asm_Beng": "as", "awa_Deva": "hi", "ben_Beng": "bn", "bho_Deva": "hi", "brx_Deva": "hi", "doi_Deva": "hi",
    "eng_Latn": "en", "gom_Deva": "kK", "gon_Deva": "hi", "guj_Gujr": "gu", "hin_Deva": "hi", "hne_Deva": "hi",
    "kan_Knda": "kn", "kas_Arab": "ur", "kas_Deva": "hi", "kha_Latn": "en", "lus_Latn": "en", "mag_Deva": "hi",
    "mai_Deva": "hi", "mal_Mlym": "ml", "mar_Deva": "mr", "mni_Beng": "bn", "mni_Mtei": "hi", "npi_Deva": "ne",
    "ory_Orya": "or", "pan_Guru": "pa", "san_Deva": "hi", "sat_Olck": "or", "snd_Arab": "ur", "snd_Deva": "hi",
    "tam_Taml": "ta", "tel_Telu": "te", "urd_Arab": "ur", "unr_Deva": "hi",
}

# every Indic/Arabic-Indic digit -> ASCII digit
_DIGIT_TABLE = {}
for _ascii, _variants in {
    "0": "\u09e6\u0ae6\u0ce6\u0966\u0660\uabf0\u0b66\u0a66\u1c50\u06f0",
    "1": "\u09e7\u0ae7\u0967\u0ce7\u06f1\uabf1\u0b67\u0a67\u1c51\u0c67",
    "2": "\u09e8\u0ae8\u0968\u0ce8\u06f2\uabf2\u0b68\u0a68\u1c52\u0c68",
    "3": "\u09e9\u0ae9\u0969\u0ce9\u06f3\uabf3\u0b69\u0a69\u1c53\u0c69",
    "4": "\u09ea\u0aea\u096a\u0cea\u06f4\uabf4\u0b6a\u0a6a\u1c54\u0c6a",
    "5": "\u09eb\u0aeb\u096b\u0ceb\u06f5\uabf5\u0b6b\u0a6b\u1c55\u0c6b",
    "6": "\u09ec\u0aec\u096c\u0cec\u06f6\uabf6\u0b6c\u0a6c\u1c56\u0c6c",
    "7": "\u09ed\u0aed\u096d\u0ced\u06f7\uabf7\u0b6d\u0a6d\u1c57\u0c6d",
    "8": "\u09ee\u0aee\u096e\u0cee\u06f8\uabf8\u0b6e\u0a6e\u1c58\u0c6e",
    "9": "\u09ef\u0aef\u096f\u0cef\u06f9\uabf9\u0b6f\u0a6f\u1c59\u0c6f",
}.items():
    for _c in _variants:
        _DIGIT_TABLE[ord(_c)] = _ascii

_MULTISPACE = re.compile(r"[ ]{2,}")
_DIGIT_SPACE_PERCENT = re.compile(r"(\d) %")
_DOUBLE_QUOT_PUNC = re.compile(r"\"([,\.]+)")
_DIGIT_NBSP_DIGIT = re.compile(r"(\d) (\d)")
_END_BRACKET_SPACE_PUNC = re.compile(r"\) ([\.!:?;,])")

_URL = re.compile(r"\b(?<![\w/.])(?:(?:https?|ftp)://)?(?:(?:[\w-]+\.)+(?!\.))(?:[\w/\-?#&=%.]+)+(?!\.\w+)\b")
_NUMERAL = re.compile(
    r"(~?\d+\.?\d*\s?%?\s?-?\s?~?\d+\.?\d*\s?%|~?\d+%|\d+[-\/.,:']\d+[-\/.,:'+]\d+(?:\.\d+)?|\d+[-\/.:'+]\d+(?:\.\d+)?)"
)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}")
_OTHER = re.compile(r"[A-Za-z0-9]*[#|@]\w+")

_PUNC_REPLACEMENTS = [
    (re.compile(r"\r"), ""),
    (re.compile(r"\(\s*"), "("),
    (re.compile(r"\s*\)"), ")"),
    (re.compile(r"\s:\s?"), ":"),
    (re.compile(r"\s;\s?"), ";"),
    (re.compile(r"[`´‘‚’]"), "'"),
    (re.compile(r"[„“”«»]"), '"'),
    (re.compile(r"[–—]"), "-"),
    (re.compile(r"\.\.\."), "..."),
    (re.compile(r" %"), "%"),
    (re.compile(r"nº "), "nº "),
    (re.compile(r" ºC"), " ºC"),
    (re.compile(r" [?!;]"), lambda m: m.group(0).strip()),
    (re.compile(r", "), ", "),
]

# the model sometimes mangles the <IDn> placeholder of an entity; these are the known shapes of the damage
_INDIC_FAILURE_CASES = [
    "آی ڈی ", "ꯑꯥꯏꯗꯤ", "आईडी", "आई . डी . ", "आई . डी .", "आई. डी. ", "आई. डी.", "आय. डी. ", "आय. डी.",
    "आय . डी . ", "आय . डी .", "आइ . डी . ", "आइ . डी .", "आइ. डी. ", "आइ. डी.", "ऐटि", "آئی ڈی ",
    "ᱟᱭᱰᱤ ᱾", "आयडी", "ऐडि", "आइडि", "ᱟᱭᱰᱤ",
]


def _placeholder_variants(n: int) -> List[str]:
    out = []
    for tag in ("ID", "id"):
        out += [f"<{tag}{n}>", f"< {tag}{n} >", f"[{tag}{n}]", f"[ {tag}{n} ]", f"[{tag} {n}]",
                f"<{tag}{n}]", f"< {tag}{n}]", f"<{tag}{n} ]"]
    for c in _INDIC_FAILURE_CASES:
        out += [f"<{c}{n}>", f"< {c}{n} >", f"< {c} {n} >", f"<{c} {n}]", f"< {c} {n} ]",
                f"[{c}{n}]", f"[{c} {n}]", f"[ {c}{n} ]", f"[ {c} {n} ]", f"{c} {n}", f"{c}{n}"]
    return out


class IndicProcessor:
    def __init__(self):
        self._en_tok = MosesTokenizer(lang="en")
        self._en_norm = MosesPunctNormalizer()
        self._en_detok = MosesDetokenizer(lang="en")
        self._xlit = UnicodeIndicTransliterator()
        self._normalizers: Dict[str, object] = {}

    # ── preprocessing ────────────────────────────────────────────────────
    def _punc_norm(self, text: str) -> str:
        for pattern, repl in _PUNC_REPLACEMENTS:
            text = pattern.sub(repl, text)
        text = _MULTISPACE.sub(" ", text)
        text = _END_BRACKET_SPACE_PUNC.sub(r")\1", text)
        text = _DIGIT_SPACE_PERCENT.sub(r"\1%", text)
        text = _DOUBLE_QUOT_PUNC.sub(r'\1"', text)
        text = _DIGIT_NBSP_DIGIT.sub(r"\1.\2", text)
        return text.strip()

    def _wrap_placeholders(self, text: str) -> Tuple[str, Dict[str, str]]:
        """URLs, e-mails, numbers and #tags are swapped for <IDn> tokens so the model cannot corrupt them."""
        entity_map: Dict[str, str] = {}
        serial = 1
        for pattern in (_EMAIL, _URL, _NUMERAL, _OTHER):
            for match in set(pattern.findall(text)):
                if pattern is _URL and len(match.replace(".", "")) < 4:
                    continue
                if pattern is _NUMERAL and len(match.replace(" ", "").replace(".", "").replace(":", "")) < 4:
                    continue
                for variant in _placeholder_variants(serial):
                    entity_map[variant] = match
                text = text.replace(match, f"<ID{serial}>")
                serial += 1
        text = re.sub(r"\s+", " ", text).replace(">/", ">").replace("]/", "]")
        return text, entity_map

    def _normalizer(self, iso: str):
        if iso not in self._normalizers:
            self._normalizers[iso] = IndicNormalizerFactory().get_normalizer(iso)
        return self._normalizers[iso]

    def _preprocess(self, sent: str, src_lang: str, tgt_lang: Optional[str]) -> Tuple[str, Dict[str, str]]:
        iso = _FLORES_CODES.get(src_lang, "hi")
        script = src_lang.split("_")[1]
        sent = self._punc_norm(sent).translate(_DIGIT_TABLE)
        sent, entity_map = self._wrap_placeholders(sent)
        if iso == "en":
            tokens = self._en_tok.tokenize(self._en_norm.normalize(sent.strip()), escape=False)
            processed = " ".join(tokens)
        else:
            normed = self._normalizer(iso).normalize(sent.strip())
            processed = " ".join(indic_tokenize.trivial_tokenize(normed, iso))
            if script not in ("Arab", "Aran", "Olck", "Mtei", "Latn"):
                processed = self._xlit.transliterate(processed, iso, "hi").replace(" ् ", "्")
        return f"{src_lang} {tgt_lang} {processed.strip()}", entity_map

    def preprocess_batch(self, batch: List[str], src_lang: str, tgt_lang: str) -> Tuple[List[str], List[Dict[str, str]]]:
        """Returns the model inputs and, per sentence, the placeholder map that postprocess_batch needs."""
        results = [self._preprocess(s, src_lang, tgt_lang) for s in batch]
        return [r[0] for r in results], [r[1] for r in results]

    # ── postprocessing ───────────────────────────────────────────────────
    def _postprocess(self, sent: str, lang: str, entity_map: Dict[str, str]) -> str:
        lang_code, script = lang.split("_", 1)
        iso = _FLORES_CODES.get(lang, "hi")
        if script in ("Arab", "Aran"):
            sent = sent.replace(" ؟", "؟").replace(" ۔", "۔").replace(" ،", "،").replace("ٮ۪", "ؠ")
        if lang_code == "ory":
            sent = sent.replace("ଯ଼", "ୟ")
        for key, value in entity_map.items():
            sent = sent.replace(key, value)
        if lang == "eng_Latn":
            return self._en_detok.detokenize(sent.split(" "))
        return indic_detokenize.trivial_detokenize(self._xlit.transliterate(sent, "hi", iso), iso)

    def postprocess_batch(self, sents: List[str], lang: str, entity_maps: List[Dict[str, str]]) -> List[str]:
        return [self._postprocess(s, lang, m) for s, m in zip(sents, entity_maps)]
