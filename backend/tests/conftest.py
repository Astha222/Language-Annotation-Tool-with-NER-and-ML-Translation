"""Shared test setup.

The suite never touches your real databases, never calls a translation service and never downloads a model:
* SQL runs on a throw-away SQLite file, MongoDB is replaced by an in-memory fake,
* translation providers and the NER pipeline are replaced by small deterministic stand-ins,
* tests that need real models are marked ``slow`` and only run with ``pytest --run-slow``.
"""
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

# Must be set before `main` is imported. Empty strings stop load_dotenv() from pulling in a developer's real .env.
_TMP = Path(tempfile.mkdtemp(prefix="annot_tests_"))
os.environ.update({
    "POSTGRES_URL": f"sqlite:///{(_TMP / 'test.db').as_posix()}",
    "MONGO_DB": "annotationtool_unit_tests",
    "JWT_SECRET": "unit-test-secret-" + "x" * 32,
    "TRANSLATION_PROVIDERS": "azure,local",
    "AZURE_TRANSLATOR_KEY": "",
    "AZURE_TRANSLATOR_REGION": "",
    "AZURE_TRANSLATOR_ENDPOINT": "",
    "MYMEMORY_EMAIL": "",
    "CORS_ORIGINS": "http://localhost:8081",
    "PRELOAD_MODELS": "none",              # tests must never load a real model at startup
    "RESULT_CACHE_PATH": "off",            # tests must not read or write a real cache file
})

import pytest  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="also run tests that download and run real translation/NER models")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs real models (downloads hundreds of MB); run with --run-slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="needs real models: run with --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


# ── fakes ────────────────────────────────────────────────────────────────
class FakeCollection:
    """The small slice of a pymongo collection that main.py uses."""
    def __init__(self):
        self.docs = []

    def insert_one(self, doc):
        self.docs.append(dict(doc))

    def find_one(self, query):
        return next((d for d in self.docs if all(d.get(k) == v for k, v in query.items())), None)

    def find(self, query):
        key, cond = next(iter(query.items()))
        wanted = set(cond["$in"]) if isinstance(cond, dict) else {cond}
        return [d for d in self.docs if d.get(key) in wanted]

    def update_one(self, query, update, upsert=False):
        doc = self.find_one(query)
        if doc is None and upsert:
            doc = dict(query)
            self.docs.append(doc)
        if doc is not None:
            doc.update(update.get("$set", {}))

    def delete_one(self, query):
        doc = self.find_one(query)
        if doc is not None:
            self.docs.remove(doc)


def fake_ner_pipeline(batch, batch_size=8):
    """Stand-in for a Hugging Face NER pipeline: every Capitalised word becomes a PER entity."""
    outputs = []
    for text in batch:
        outputs.append([
            {"entity_group": "PER", "score": 0.99, "word": m.group(0), "start": m.start(), "end": m.end()}
            for m in re.finditer(r"\b[A-Z][a-z]{2,}\b", text)
        ])
    return outputs


@pytest.fixture(scope="session")
def main_module():
    import main
    return main


@pytest.fixture(autouse=True)
def isolated(main_module, monkeypatch, tmp_path):
    """Fresh fake Mongo, document folder, NER stand-in and provider state for every test."""
    m = main_module
    monkeypatch.setattr(m, "ner_collection", FakeCollection())
    monkeypatch.setattr(m, "DOC_DIR", tmp_path)
    monkeypatch.setattr(m, "_ner_pipelines", {m.NER_MODEL_EN: fake_ner_pipeline, m.NER_MODEL_MULTI: fake_ner_pipeline, m.NER_MODEL_HI: fake_ner_pipeline})
    for name in m._blocked_until:
        m._blocked_until[name] = 0.0
    m.JOBS.clear()
    db = m.SessionLocal()                  # saved jobs (M6) must not leak from one test into the next
    db.query(m.JobDB).delete()
    db.commit()
    db.close()
    yield


@pytest.fixture
def providers(main_module, monkeypatch):
    """Replace every translation provider with a labelled stand-in and record the calls."""
    m = main_module
    calls = {"azure": [], "local": [], "google": [], "mymemory": []}

    def make(name, prefix):
        def fn(src, tgt, text, *args, **kwargs):
            calls[name].append((src, tgt, text))
            return prefix + text
        return fn

    monkeypatch.setattr(m.azure_translator, "translate", make("azure", "AZ:"))
    monkeypatch.setattr(m.local_translator, "translate", make("local", "LOC:"))
    monkeypatch.setattr(m.local_translator, "is_available", lambda: True)
    monkeypatch.setattr(m, "_google_translate", make("google", "G:"))
    monkeypatch.setattr(m, "_mymemory_translate", make("mymemory", "M:"))
    monkeypatch.setattr(m, "_google_wait_turn", lambda: None)
    monkeypatch.setattr(m, "_sleep_unless_cancelled", lambda seconds, cancelled=None: None)   # no real waiting
    return calls


@pytest.fixture
def azure_on(monkeypatch):
    monkeypatch.setenv("AZURE_TRANSLATOR_KEY", "test-key-not-real")


@pytest.fixture
def client(main_module):
    from fastapi.testclient import TestClient
    return TestClient(main_module.app)


@pytest.fixture
def auth(client):
    """Headers for a freshly registered user."""
    name = "user" + uuid.uuid4().hex[:8]
    r = client.post("/api/auth/register", json={"username": name, "email": f"{name}@example.com", "password": "password123"})
    assert r.status_code == 201, r.text
    token = client.post("/api/auth/login", json={"username": name, "password": "password123"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth2(client):
    """A second, unrelated user (for access-control tests)."""
    name = "other" + uuid.uuid4().hex[:8]
    client.post("/api/auth/register", json={"username": name, "email": f"{name}@example.com", "password": "password123"})
    token = client.post("/api/auth/login", json={"username": name, "password": "password123"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
