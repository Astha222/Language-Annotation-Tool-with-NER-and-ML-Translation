"""Shared helpers for the evaluation scripts: dataset download/caching and access to the backend code."""
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
DATA = Path(__file__).resolve().parent / "data"      # downloaded benchmark files (git-ignored)


def use_console_utf8():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def load_backend():
    """Import the app's backend module without touching your real databases."""
    os.environ["POSTGRES_URL"] = "sqlite:///:memory:"          # nothing is written; the evaluation only calls NER/translation
    os.environ["RESULT_CACHE_PATH"] = "off"                   # a cache hit would make timings (and any re-run) meaningless
    os.environ.setdefault("MONGO_DB", "annotationtool_evaluation")
    sys.path.insert(0, str(BACKEND))
    import logging
    logging.disable(logging.WARNING)
    import main
    return main


def download(url: str, dest: Path) -> Path:
    """Download once, then reuse the cached copy."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size == 0:
        print(f"  downloading {url.split('/resolve/')[-1]} ...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "annotation-tool-evaluation"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            f.write(r.read())
    return dest
