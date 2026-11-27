"""Otter (whoisjones/otter-cross-mmbert, Apache-2.0): a zero-shot entity model, wrapped to look like a Hugging Face NER pipeline.

Otter is not a normal token-classification model: it is asked for entity types by name ("person", "organization", "location") and its model
class lives in the model's own repository, so loading it RUNS CODE FROM THAT REPOSITORY (`trust_remote_code`). That is why the registry
(ner_registry.py) only allows it for an entry that pins a full commit hash, and why NER_ALLOW_REMOTE_CODE=0 switches such models off.

The same wrapper is used for scoring in evaluation/gliner_adapter.py (kept as a separate file because the Colab bundles ship it on its own);
the results in evaluation/RESULTS.md (threshold 0.25, tuned on MultiCoNER train sentences) were produced with that copy.
"""
import re
import unicodedata

LABELS = {"person": "PER", "organization": "ORG", "location": "LOC"}


def _norm(s):
    return unicodedata.normalize("NFC", s).casefold()


def locate(text, word, taken):
    """Character span of `word` in `text` for an entity returned without offsets: exact text, then case-folded, then the words with any whitespace
    between them. The first occurrence not already taken is used; None when nothing matches."""
    if not word or not word.strip():
        return None
    candidates = []
    pos = text.find(word)
    while pos >= 0:
        candidates.append((pos, pos + len(word)))
        pos = text.find(word, pos + 1)
    if not candidates and len(_norm(text)) == len(text):
        nt, nw = _norm(text), _norm(word)
        pos = nt.find(nw)
        while pos >= 0:
            candidates.append((pos, pos + len(nw)))
            pos = nt.find(nw, pos + 1)
    if not candidates:
        pattern = r"\s*".join(re.escape(w) for w in word.split())
        candidates = [(m.start(), m.end()) for m in re.finditer(pattern, text, flags=re.IGNORECASE)]
    return next((c for c in candidates if c not in taken), None)


class OtterPipeline:
    def __init__(self, model, threshold=0.25):
        self.model = model
        self.threshold = float(threshold)

    def _convert(self, ents, text):
        out, taken = [], set()
        for e in ents:
            if e.get("label") not in LABELS:
                continue
            start, end, word = e.get("start"), e.get("end"), e.get("text", "")
            if start is None or end is None:
                span = locate(text, word, taken)
                if span is None:
                    continue
                start, end = span
            taken.add((int(start), int(end)))
            out.append({"entity_group": LABELS[e["label"]], "score": float(e["score"]), "start": int(start), "end": int(end),
                        "word": word or text[int(start):int(end)]})
        return out

    def __call__(self, texts, batch_size=16, **_ignored):
        """Same shape as a pipeline: texts in, one entity list per text out (pipeline-only keyword arguments such as aggregation_strategy are ignored)."""
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results = self.model.predict(batch, labels=list(LABELS), threshold=self.threshold, batch_size=batch_size)
            out.extend(self._convert(r, t) for r, t in zip(results, batch))
        return out[0] if single else out


def load(model_id: str, revision: str, threshold=0.25, device=None) -> OtterPipeline:
    """Download (once) and load Otter at the pinned commit. `revision` must be the reviewed commit hash."""
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, revision=revision)
    model.eval()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = model.to(device)
    except Exception:                       # some versions place themselves; the scores do not depend on the device
        pass
    return OtterPipeline(model, threshold)
