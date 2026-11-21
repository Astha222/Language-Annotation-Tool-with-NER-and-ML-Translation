"""Offline, open-source translation: no API key, no quota, and the text never leaves the machine.

Engines (all downloaded once from the Hugging Face Hub, then cached):

* Hindi <-> English: IndicTrans2 "rotary" distilled 200M models by Varun Gumma, Pranjal Chitale and Kalika Bali
  (MIT License), https://huggingface.co/prajdabre/rotary-indictrans2-en-indic-dist-200M . Far more accurate for
  Hindi than generic models, and built for long documents. Needs ``trust_remote_code`` (the model ships its own
  Python code), so the revisions below are PINNED: a later change to the repository cannot alter what runs here.
  Set ``LOCAL_INDICTRANS=0`` to refuse that and use Marian for Hindi too.
* Every other pair: Helsinki-NLP OPUS-MT Marian models (Apache-2.0 / CC-BY-4.0), ~300 MB per direction.
* A pair that touches neither English nor a direct model (e.g. French -> Hindi) is pivoted through English.

Settings (environment variables, all optional)
    LOCAL_MT_BEAMS         beam-search width, default 2 (1 = fastest, 4 = best quality)
    LOCAL_MT_THREADS       CPU threads for PyTorch (default: PyTorch's choice)
    LOCAL_INDICTRANS       "0" disables the IndicTrans2 models (default "1")
    LOCAL_MT_MAX_LOADED    models kept in memory at once, default 2
"""
import logging
import os
import re
import threading
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── what is supported ────────────────────────────────────────────────────
INDICTRANS = {
    # direction -> (repository, pinned commit, source tag, target tag)
    ("en", "hi"): ("prajdabre/rotary-indictrans2-en-indic-dist-200M", "23249dd48107a6207a4ac1fba62789c884458f0f",
                   "eng_Latn", "hin_Deva"),
    ("hi", "en"): ("prajdabre/rotary-indictrans2-indic-en-dist-200M", "00213ee82929050694b162123bb26a8dc177cdae",
                   "hin_Deva", "eng_Latn"),
}
# Marian models are named Helsinki-NLP/opus-mt-<source>-<target>
_MARIAN_TARGETS = {"hi": "hi", "fr": "fr", "es": "es", "de": "de", "zh": "zh", "ar": "ar", "ru": "ru", "ja": "ja"}
# English -> Japanese: "opus-mt-en-jap" is NOT Japanese (ISO 639-3 "jap" is Jaruara) and its only published test set is
# Bible verses, so it produced unrelated text (chrF++ 4.4). The Tatoeba model is the Apache-2.0 en-ja Marian model.
# LOCAL_MT_JA_MODEL overrides it, e.g. to compare candidates with evaluation/eval_translation.py.
_MARIAN_MODEL_OVERRIDE = {("en", "ja"): "Helsinki-NLP/opus-tatoeba-en-ja"}
# Multi-target Marian models need a sentence-initial token that says which language to produce (see their model cards:
# "a sentence initial language token is required in the form of >>id<<"). Without it they may pick another script or dialect.
_TARGET_TOKEN = {("en", "zh"): ">>cmn_Hans<<",      # Simplified Chinese (the model can also write Traditional, Cantonese, ...)
                 ("en", "ar"): ">>ara<<"}           # Standard Arabic
_MARIAN_SOURCES = {"hi": "hi", "fr": "fr", "es": "es", "de": "de", "zh": "zh", "ar": "ar", "ru": "ru", "ja": "ja"}
SUPPORTED_LANGS = {"en"} | set(_MARIAN_TARGETS)

MAX_LINE_CHARS = 700            # longer lines are split at sentence ends so a batch never exceeds the model's limits
BATCH_SIZE = 8


class LocalTranslationError(Exception):
    pass


class Cancelled(Exception):
    pass


def supports(src: str, tgt: str) -> bool:
    return src != tgt and src in SUPPORTED_LANGS and tgt in SUPPORTED_LANGS


def _beams() -> int:
    try:
        return max(1, min(8, int(os.getenv("LOCAL_MT_BEAMS", "2"))))
    except ValueError:
        return 2


def _indictrans_enabled() -> bool:
    return os.getenv("LOCAL_INDICTRANS", "1").strip() != "0"


# ── text segmentation ────────────────────────────────────────────────────
_SENT_END = re.compile(r"(?<=[.!?।。！？])\s+")


def _split_line(line: str, limit: int) -> List[str]:
    """Split an over-long line into sentence groups of at most `limit` characters."""
    if len(line) <= limit:
        return [line]
    parts, cur = [], ""
    for sent in _SENT_END.split(line):
        while len(sent) > limit:                                   # a single enormous sentence: hard split on spaces
            cut = sent.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            piece, sent = sent[:cut], sent[cut:].lstrip()
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(piece)
        if cur and len(cur) + 1 + len(sent) > limit:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    return parts


# ── model cache ──────────────────────────────────────────────────────────
class _Engine:
    """One loaded translation direction."""
    def __init__(self, kind: str, src: str, tgt: str):
        self.kind, self.src, self.tgt = kind, src, tgt
        import torch
        threads = os.getenv("LOCAL_MT_THREADS", "").strip()
        if threads.isdigit():
            torch.set_num_threads(int(threads))
        self.torch = torch
        if kind == "indictrans":
            self._load_indictrans()
        else:
            self._load_marian()

    def _load_indictrans(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from indic_processor import IndicProcessor
        repo, rev, self.src_tag, self.tgt_tag = INDICTRANS[(self.src, self.tgt)]
        logger.info(f"Loading {repo}@{rev[:8]} (first use downloads ~0.9 GB)")
        self.tok = AutoTokenizer.from_pretrained(repo, revision=rev, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(repo, revision=rev, trust_remote_code=True).eval()
        # the model code predates transformers' Cache classes; make generate() use the legacy tuple cache it expects
        self.model._supports_default_dynamic_cache = lambda *a, **k: False
        self.proc = IndicProcessor()

    def _load_marian(self):
        from transformers import MarianMTModel, MarianTokenizer
        a = "en" if self.src == "en" else _MARIAN_SOURCES[self.src]
        b = _MARIAN_TARGETS[self.tgt] if self.src == "en" else "en"
        name = _MARIAN_MODEL_OVERRIDE.get((self.src, self.tgt), f"Helsinki-NLP/opus-mt-{a}-{b}")
        if (self.src, self.tgt) == ("en", "ja") and os.getenv("LOCAL_MT_JA_MODEL"):
            name = os.getenv("LOCAL_MT_JA_MODEL")
        logger.info(f"Loading {name} (first use downloads ~0.3 GB)")
        self.tok = MarianTokenizer.from_pretrained(name)
        self.model = MarianMTModel.from_pretrained(name).eval()
        self.prefix = _TARGET_TOKEN.get((self.src, self.tgt), "")
        if self.prefix and self.prefix not in self.tok.get_vocab():
            raise LocalTranslationError(f"{name} has no target token {self.prefix}")

    def translate_batch(self, texts: List[str], beams: int) -> List[str]:
        torch = self.torch
        if self.kind == "indictrans":
            pre, maps = self.proc.preprocess_batch(texts, self.src_tag, self.tgt_tag)
            enc = self.tok(pre, padding="longest", truncation=True, max_length=512, return_tensors="pt")
            with torch.inference_mode():
                out = self.model.generate(**enc, num_beams=beams, max_new_tokens=400,
                                          repetition_penalty=1.5 if beams > 1 else 1.2, early_stopping=beams > 1)
            dec = self.tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            return self.proc.postprocess_batch(dec, self.tgt_tag, maps)
        if self.prefix:
            texts = [f"{self.prefix} {t}" for t in texts]
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=400)
        with torch.inference_mode():
            out = self.model.generate(**enc, num_beams=beams, max_new_tokens=400)
        return self.tok.batch_decode(out, skip_special_tokens=True)


_engines: "OrderedDict[Tuple[str, str], _Engine]" = OrderedDict()
_lock = threading.Lock()        # one generation at a time: the models are not thread-safe and already use all cores


def _engine(src: str, tgt: str) -> _Engine:
    key = (src, tgt)
    if key in _engines:
        _engines.move_to_end(key)
        return _engines[key]
    kind = "indictrans" if key in INDICTRANS and _indictrans_enabled() else "marian"
    if kind == "marian" and "en" not in key:
        raise LocalTranslationError(f"no direct local model for {src}->{tgt}")
    eng = _Engine(kind, src, tgt)
    _engines[key] = eng
    try:
        limit = max(1, int(os.getenv("LOCAL_MT_MAX_LOADED", "2")))
    except ValueError:
        limit = 2
    while len(_engines) > limit:
        _engines.popitem(last=False)                           # drop the least recently used model
    return eng


# ── public API ───────────────────────────────────────────────────────────
def loaded_engines() -> List[str]:
    """The translation directions currently held in memory, e.g. ["en-hi", "en-fr"]."""
    return [f"{src}-{tgt}" for src, tgt in _engines]


def _translate_direct(src: str, tgt: str, text: str, cancelled: Optional[Callable[[], bool]]) -> str:
    """One model, one direction. Keeps the line structure of the input (headings, bullets, blank lines)."""
    lines = text.split("\n")
    jobs: List[Tuple[int, int, str]] = []                      # (line index, piece index, text)
    pieces: Dict[int, List[str]] = {}
    for i, line in enumerate(lines):
        if not any(ch.isalpha() for ch in line):
            continue                                           # blank, digit-only or symbol-only lines are kept as they are
        parts = _split_line(line.strip(), MAX_LINE_CHARS)
        pieces[i] = [""] * len(parts)
        jobs.extend((i, k, p) for k, p in enumerate(parts))

    order = sorted(range(len(jobs)), key=lambda n: len(jobs[n][2]))   # similar lengths together = less padding
    beams = _beams()
    with _lock:
        eng = _engine(src, tgt)
        for start in range(0, len(order), BATCH_SIZE):
            if cancelled and cancelled():
                raise Cancelled()
            idx = order[start:start + BATCH_SIZE]
            out = eng.translate_batch([jobs[n][2] for n in idx], beams)
            for n, translated in zip(idx, out):
                pieces[jobs[n][0]][jobs[n][1]] = translated.strip()

    return "\n".join(" ".join(pieces[i]) if i in pieces else line for i, line in enumerate(lines))


def translate(src: str, tgt: str, text: str, cancelled: Optional[Callable[[], bool]] = None) -> str:
    """Translate text between any two supported languages (pivoting through English when needed)."""
    if not supports(src, tgt):
        raise LocalTranslationError(f"unsupported language pair {src}->{tgt}")
    if src == "en" or tgt == "en":
        return _translate_direct(src, tgt, text, cancelled)
    return _translate_direct("en", tgt, _translate_direct(src, "en", text, cancelled), cancelled)


def is_available() -> bool:
    """True when the Python packages needed for local translation are installed."""
    try:
        import sentencepiece  # noqa: F401
        import sacremoses  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False
