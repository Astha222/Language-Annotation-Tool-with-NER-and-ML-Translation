"""Turn raw token-classification output into tidy entities (shared by the app and the evaluation scripts, so both clean identically).

Kept free of any web, database or model imports so it also runs in a bare evaluation environment such as Colab.
"""
import unicodedata

_EDGE_PUNCT = " \t\r\n,.;:!?()[]{}\"'“”‘’।"

def _is_wordchar(c: str) -> bool:
    # Devanagari vowel signs etc. are category Mn/Mc and not isalnum(), but they belong to the word
    return c.isalnum() or unicodedata.category(c) in ("Mn", "Mc")

def _snap_to_token(chunk: str, s: int, e: int) -> tuple:
    """Grow an entity to cover a token the model cut in the middle: "Node" + ".js" -> "Node.js".
    Only a dot between word characters joins pieces, so "Google." and "Google's" are left alone."""
    n = len(chunk)
    while s > 0 and (_is_wordchar(chunk[s - 1]) or
                     (chunk[s - 1] == "." and s >= 2 and _is_wordchar(chunk[s - 2]) and _is_wordchar(chunk[s]))):
        s -= 1
    while e < n and (_is_wordchar(chunk[e]) or
                     (chunk[e] == "." and e + 1 < n and _is_wordchar(chunk[e + 1]) and _is_wordchar(chunk[e - 1]))):
        e += 1
    return s, e

def _clean_entities(chunk: str, entities: list, offset: int, snap: bool = True) -> list:
    """Model output -> tidy entities: exact input text, no edge punctuation, whole tokens, no overlapping duplicates."""
    spans = []
    for r in entities:
        s, e = int(r["start"]), int(r["end"])
        while s < e and chunk[s] in _EDGE_PUNCT: s += 1
        while e > s and chunk[e - 1] in _EDGE_PUNCT: e -= 1
        if s >= e:
            continue
        if snap:
            s, e = _snap_to_token(chunk, s, e)
        spans.append({"g": r["entity_group"], "score": float(r["score"]), "s": s, "e": e})
    spans.sort(key=lambda x: (x["s"], -x["e"]))
    merged = []
    for sp in spans:
        if merged and sp["s"] < merged[-1]["e"]:           # pieces of one token now overlap: keep one entity
            m = merged[-1]
            m["e"] = max(m["e"], sp["e"])
            if sp["score"] > m["score"]:
                m["score"], m["g"] = sp["score"], sp["g"]
        else:
            merged.append(sp)
    return [{"entity_group": m["g"], "score": round(m["score"], 4), "word": chunk[m["s"]:m["e"]],
             "start": m["s"] + offset, "end": m["e"] + offset} for m in merged]
