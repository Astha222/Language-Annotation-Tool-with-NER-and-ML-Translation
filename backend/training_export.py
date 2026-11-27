"""
Export helpers for machine learning training formats: CoNLL / BIO and JSONL.

Supported formats:
1. CoNLL / BIO:
   - One token per line: <token>\t<BIO_TAG>
   - Examples of tags: B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, O
   - Blank line between sentences
   - Splits normal languages using spaces and punctuation
   - For Chinese (zh) and Japanese (ja), uses one character per token
   - Determines BIO labels using stored entity start/end character offsets
   - A token beginning an entity gets B-TYPE
   - Subsequent tokens inside that entity get I-TYPE
2. JSONL:
   - One JSON object per line in the structure:
     {"text":"...","entities":[[start,end,"PER"],[start,end,"ORG"]],"lang":"..."}
   - Preserves exact stored offsets
"""

import json
from typing import List, Dict, Any, Tuple, Optional, Set
import regex

SENT_PUNCT = set(".!?।。！？")


def clean_entity_group(raw_group: str) -> str:
    """Normalize entity group / type (e.g. 'B-PER' -> 'PER')."""
    grp = (raw_group or "MISC").strip()
    if grp.startswith(("B-", "I-")):
        grp = grp[2:]
    return grp or "MISC"


def tokenize_for_conll(text: str, lang: str = "en") -> List[Tuple[str, int, int]]:
    """Return a list of (token_string, start_offset, end_offset) for the given text.

    For Chinese and Japanese, non-whitespace characters are tokenized one by one.
    For all other languages, text is split on spaces and punctuation using Unicode categories.
    """
    if not text:
        return []
    lang_lower = (lang or "en").lower().strip()
    if lang_lower.startswith(("zh", "ja", "chinese", "japanese")):
        return [(ch, i, i + 1) for i, ch in enumerate(text) if not ch.isspace()]
    return [(m.group(), m.start(), m.end()) for m in regex.finditer(r"\w+|[^\w\s]", text)]


def parse_entities_list(entities: Any) -> List[Dict[str, Any]]:
    """Normalize entities whether they are passed as a JSON string or a list of dicts."""
    if not entities:
        return []
    if isinstance(entities, str):
        try:
            entities = json.loads(entities)
        except Exception:
            return []
    if not isinstance(entities, list):
        return []

    cleaned: List[Dict[str, Any]] = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        if "start" in e and "end" in e:
            try:
                s = int(e["start"])
                end = int(e["end"])
            except (ValueError, TypeError):
                continue
            if s >= end:
                continue
            grp = e.get("entity_group") or e.get("type") or e.get("label") or "MISC"
            cleaned.append({
                "start": s,
                "end": end,
                "type": clean_entity_group(str(grp)),
            })
    cleaned.sort(key=lambda x: (x["start"], x["end"]))
    return cleaned


def annotation_to_conll(text: str, entities: Any, lang: str = "en") -> str:
    """Convert a single annotation to CoNLL/BIO format."""
    if not text or not text.strip():
        return ""
    tokens = tokenize_for_conll(text, lang)
    if not tokens:
        return ""

    cleaned_entities = parse_entities_list(entities)
    started_entities: Set[int] = set()
    result_lines: List[str] = []
    is_cjk = (lang or "en").lower().strip().startswith(("zh", "ja", "chinese", "japanese"))

    for i, (tok, t_start, t_end) in enumerate(tokens):
        matched_idx: Optional[int] = None
        matched_type: Optional[str] = None

        for idx, e in enumerate(cleaned_entities):
            # Overlap between token span and entity span
            if max(t_start, e["start"]) < min(t_end, e["end"]):
                matched_idx = idx
                matched_type = e["type"]
                break

        if matched_idx is not None:
            if matched_idx not in started_entities:
                tag = f"B-{matched_type}"
                started_entities.add(matched_idx)
            else:
                tag = f"I-{matched_type}"
        else:
            tag = "O"

        result_lines.append(f"{tok}\t{tag}")

        # Check for sentence boundary before the next token
        if i < len(tokens) - 1:
            next_start = tokens[i + 1][1]
            inter_text = text[t_end:next_start]
            inside_entity = any(e["start"] <= t_start and t_end < e["end"] for e in cleaned_entities)

            is_break = False
            if "\n" in inter_text:
                is_break = True
            elif not inside_entity and any(p in tok for p in SENT_PUNCT):
                if is_cjk or (inter_text.strip() == "" and len(inter_text) > 0):
                    is_break = True

            if is_break:
                result_lines.append("")

    return "\n".join(result_lines)


def export_conll(annotations: List[Dict[str, Any]]) -> str:
    """Export a list of annotation dictionaries to CoNLL/BIO format."""
    blocks: List[str] = []
    for a in annotations:
        txt = a.get("original_text") or ""
        ents = a.get("ner_tags") or []
        lng = a.get("source_lang") or "en"
        block = annotation_to_conll(txt, ents, lng)
        if block:
            blocks.append(block)
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def annotation_to_jsonl_dict(text: str, entities: Any, lang: str = "en") -> Dict[str, Any]:
    """Convert a single annotation to JSONL object structure:
    {"text":"...","entities":[[start,end,"PER"],[start,end,"ORG"]],"lang":"..."}
    """
    cleaned_entities = parse_entities_list(entities)
    ent_tuples = [[e["start"], e["end"], e["type"]] for e in cleaned_entities]
    return {
        "text": text or "",
        "entities": ent_tuples,
        "lang": lang or "en",
    }


def export_jsonl(annotations: List[Dict[str, Any]]) -> str:
    """Export a list of annotation dictionaries to JSONL format (one JSON object per line)."""
    lines: List[str] = []
    for a in annotations:
        txt = a.get("original_text") or ""
        ents = a.get("ner_tags") or []
        lng = a.get("source_lang") or "en"
        obj = annotation_to_jsonl_dict(txt, ents, lng)
        lines.append(json.dumps(obj, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")
