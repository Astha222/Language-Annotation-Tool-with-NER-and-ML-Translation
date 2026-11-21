"""Make a zero-shot NER model (GLiNER or Otter) look like a Hugging Face NER pipeline, so run_all.py scores it with exactly the same code, tests and confidence intervals.

Zero-shot models are asked for entity types by name instead of having a fixed label set. Here they are asked for "person", "organization" and
"location", which are mapped to PER, ORG and LOC. Use them from run_all.py with a model name that starts with "gliner:" or "otter:":

    python evaluation/run_all.py --models "gliner_x_base=gliner:knowledgator/gliner-x-base" --datasets nm_bn --full
    python evaluation/run_all.py --models "otter=otter:whoisjones/otter-cross-mmbert" --datasets uner_de --full

The score threshold (default 0.5, GLINER_THRESHOLD overrides it for both kinds) is NOT tuned here. To tune it, run with a low threshold (e.g. 0.1):
every prediction keeps its score in the saved file, and evaluation/tune_threshold.py picks the threshold on a tuning split and applies it to the
test runs without running the model again.
The `gliner` package (and, for Otter, trust_remote_code) is only needed when this adapter is used; install it in a separate environment
(for example a Colab session), not in the main one.
"""
import os
import re
import unicodedata

LABELS = {"person": "PER", "organization": "ORG", "location": "LOC"}
# Otter runs code from its model repository: pin the exact revision that was reviewed (2026-11-18), so a later push cannot change what runs
OTTER_REVISIONS = {"whoisjones/otter-cross-mmbert": "8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d"}


def _norm(s):
    return unicodedata.normalize("NFC", s).casefold()


def locate(text, word, taken):
    """Character span of `word` in `text`, for an entity returned WITHOUT offsets. Tries, in order: the exact text; the same text after
    Unicode normalisation and case folding; the words of the entity with any whitespace between them. The first occurrence that is not already
    taken is used. Returns None when nothing matches."""
    if not word or not word.strip():
        return None
    candidates = []
    pos = text.find(word)
    while pos >= 0:
        candidates.append((pos, pos + len(word)))
        pos = text.find(word, pos + 1)
    if not candidates and len(_norm(text)) == len(text):            # case folding kept the length, so positions still line up
        nt, nw = _norm(text), _norm(word)
        pos = nt.find(nw)
        while pos >= 0:
            candidates.append((pos, pos + len(nw)))
            pos = nt.find(nw, pos + 1)
    if not candidates:
        pattern = r"\s*".join(re.escape(w) for w in word.split())
        candidates = [(m.start(), m.end()) for m in re.finditer(pattern, text, flags=re.IGNORECASE)]
    return next((c for c in candidates if c not in taken), None)


class GlinerPipeline:
    def __init__(self, model_id, threshold=None, device=None, kind="gliner"):
        import torch
        self.kind = kind
        if kind == "otter":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, revision=OTTER_REVISIONS.get(model_id))
            self.model.eval()
        else:
            from gliner import GLiNER
            self.model = GLiNER.from_pretrained(model_id)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = self.model.to(device)
        except Exception:                                   # some versions place themselves; the scores do not depend on the device
            pass
        self.threshold = float(threshold if threshold is not None else os.getenv("GLINER_THRESHOLD", "0.5"))
        self.recovered = 0          # entities whose character span was missing and was found again in the text
        self.unresolved = 0         # entities dropped because the span could not be found
        self.unresolved_examples = []

    def _convert(self, ents, text=""):
        """GLiNER-X sometimes returns an entity with start/end = None (it found the words but could not map them back to characters; seen in
        Marathi, Tamil, German and French). The span is then recovered with locate(). If that fails the entity is dropped and counted in
        self.unresolved, and its text is kept in self.unresolved_examples, so the number and the cases are visible."""
        out, taken = [], set()
        for e in ents:
            if e["label"] not in LABELS:
                continue
            start, end, word = e.get("start"), e.get("end"), e.get("text", "")
            if start is None or end is None:
                span = locate(text, word, taken)
                self.recovered += span is not None
                if span is None:
                    self.unresolved += 1
                    if len(self.unresolved_examples) < 10:
                        self.unresolved_examples.append((word, text[:120]))
                    continue
                start, end = span
            taken.add((int(start), int(end)))
            out.append({"entity_group": LABELS[e["label"]], "score": float(e["score"]), "start": int(start), "end": int(end),
                        "word": word or text[int(start):int(end)]})
        return out

    def __call__(self, texts, batch_size=16, **_ignored):
        """Same shape as a pipeline: a list of texts in, a list of entity lists out (the pipeline-only keyword arguments are ignored)."""
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        labels = list(LABELS)
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if self.kind == "otter":
                results = self.model.predict(batch, labels=labels, threshold=self.threshold, batch_size=batch_size)
            elif hasattr(self.model, "batch_predict_entities"):
                results = self.model.batch_predict_entities(batch, labels, threshold=self.threshold)
            else:
                results = [self.model.predict_entities(t, labels, threshold=self.threshold) for t in batch]
            out.extend(self._convert(r, t) for r, t in zip(results, batch))
        return out[0] if single else out
