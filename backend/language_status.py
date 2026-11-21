"""Per-language quality status ("Stable", "Beta", "Experimental") for entity recognition.

The labels are computed, not typed: evaluation/make_language_status.py reads the evaluation results and writes language_status.json next to this
file; the API serves it (GET /api/languages) and the app shows it beside each language. The rules below are the ones agreed with the project owner:

    stable        strict F1 >= 0.75, the LOWER end of the 95% range >= 0.70, at least 500 human-labelled test sentences,
                  an open licence, and a test made by native speakers of the language
    beta          F1 between 0.50 and 0.75, or good scores that do not yet meet every condition for "stable" (the reasons say which)
    experimental  F1 below 0.50 (hidden by default in the app)
    blocked       the model has no open licence, so it cannot be offered by default whatever its score
"""
import json
from pathlib import Path
from typing import List, Optional, Tuple

STATUS_FILE = Path(__file__).resolve().parent / "language_status.json"

STABLE_F1 = 0.75
STABLE_CI_LOW = 0.70
STABLE_MIN_SENTENCES = 500
EXPERIMENTAL_BELOW = 0.50

THRESHOLDS = {"stable_f1": STABLE_F1, "stable_ci_low": STABLE_CI_LOW, "stable_min_sentences": STABLE_MIN_SENTENCES,
              "experimental_below": EXPERIMENTAL_BELOW}


def ner_status(f1: Optional[float], ci_low: Optional[float], sentences: int, open_licence: bool, native_test: bool) -> Tuple[str, List[str]]:
    """The status of one language's entity recognition and, in plain words, what is still missing for "stable"."""
    if f1 is None:
        return "not_measured", ["no test result yet"]
    if not open_licence:
        return "blocked", ["the model has no open licence"]
    if f1 < EXPERIMENTAL_BELOW:
        return "experimental", [f"F1 {f1:.2f} is below {EXPERIMENTAL_BELOW:.2f}"]
    missing = []
    if f1 < STABLE_F1:
        missing.append(f"F1 {f1:.2f} is below {STABLE_F1:.2f}")
    if ci_low is None or ci_low < STABLE_CI_LOW:
        missing.append(f"the lower end of the 95% range ({'unknown' if ci_low is None else f'{ci_low:.2f}'}) is below {STABLE_CI_LOW:.2f}")
    if sentences < STABLE_MIN_SENTENCES:
        missing.append(f"only {sentences} test sentences (needs {STABLE_MIN_SENTENCES})")
    if not native_test:
        missing.append("no test made by native speakers yet")
    return ("beta", missing) if missing else ("stable", [])


def load() -> dict:
    """The generated status table, or an empty one if it has not been generated."""
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"thresholds": THRESHOLDS, "languages": []}
