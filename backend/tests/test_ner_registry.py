"""The NER model registry: defaults, overrides, adding a model, and the fallback when a model cannot be loaded."""
import json

import pytest

import ner_registry


def _write(tmp_path, data, name="extra.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


# ── the registry itself ──────────────────────────────────────────────────
def test_bundled_registry_is_valid_and_routes_every_group():
    reg = ner_registry.load(env={})
    assert reg.group_source("en") == "muskan-424/distilroberta-ner-english-b-nofin"
    assert reg.group_source("indic") == "neuronsbyisshu/indicbert-hi-ner-naamapadam"
    assert reg.group_source("*") == "Davlan/xlm-roberta-base-ner-hrl"
    assert reg.language_routes() == {}


def test_every_bundled_route_and_fallback_names_a_model_in_the_file():
    data = json.loads(ner_registry.BUNDLED.read_text(encoding="utf-8"))
    for ref in data["routes"].values():
        assert ref in data["models"]
    for entry in data["models"].values():
        assert entry.get("licence"), "every model states its licence (or that it has none)"
        if entry.get("fallback"):
            assert entry["fallback"] in data["models"]


def test_legacy_settings_still_win_and_accept_name_id_or_folder():
    reg = ner_registry.load(env={"NER_MODEL_EN": "  C:/models/student/best ", "NER_MODEL_HI": "multi-davlan-hrl", "NER_MODEL_MULTI": "  "})
    assert reg.group_source("en") == "C:/models/student/best"            # a folder
    assert reg.group_source("indic") == "Davlan/xlm-roberta-base-ner-hrl"  # a registry name
    assert reg.group_source("*") == "Davlan/xlm-roberta-base-ner-hrl"      # blank = unchanged


def test_a_user_file_can_add_a_model_and_route_it(tmp_path):
    extra = _write(tmp_path, {"models": {"my-fr": {"source": "D:/models/fr-ner/best", "licence": "MIT"}},
                              "routes": {"fr": "my-fr", "en": "english-bert-base"}})
    reg = ner_registry.load(env={"NER_MODELS_FILE": extra})
    assert reg.language_routes() == {"fr": "D:/models/fr-ner/best"}
    assert reg.group_source("en") == "dslim/bert-base-NER"              # the file changed the English default
    assert reg.group_source("indic") == "neuronsbyisshu/indicbert-hi-ner-naamapadam"   # the rest is untouched


@pytest.mark.parametrize("bad", [
    {"models": {"x": {"licence": "MIT"}}},                       # no source
    {"models": {"x": {"source": "  "}}},                         # empty source
    {"routes": {"en": ""}},                                      # a group route emptied
])
def test_a_broken_registry_file_fails_loudly(tmp_path, bad):
    with pytest.raises(ValueError):
        ner_registry.load(env={"NER_MODELS_FILE": _write(tmp_path, bad)})


def test_unreadable_registry_file_fails_loudly(tmp_path):
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        ner_registry.load(env={"NER_MODELS_FILE": str(tmp_path / "bad.json")})
    with pytest.raises(ValueError):
        ner_registry.load(env={"NER_MODELS_FILE": str(tmp_path / "missing.json")})


def test_describe_lists_every_route_with_licence_and_fallback():
    rows = {r["route"]: r for r in ner_registry.load(env={}).describe({"hi", "mr"})}
    assert rows["en"]["fallback"] == "dslim/bert-base-NER" and "Apache" in rows["en"]["licence"]
    assert rows["indic"]["languages"] == ["hi", "mr"]
    assert rows["*"]["fallback"] is None


# ── the app uses it ──────────────────────────────────────────────────────
def test_a_per_language_route_beats_the_group(main_module, monkeypatch):
    monkeypatch.setattr(main_module, "_NER_LANG_ROUTES", {"fr": "D:/models/fr-ner/best"})
    assert main_module.ner_model_name("fr") == "D:/models/fr-ner/best"
    assert main_module.ner_model_name("de") == main_module.NER_MODEL_MULTI


def _loader(main_module, monkeypatch, failing=()):
    """Replace the real model loader; sources in `failing` raise like a missing repo does."""
    loads = []

    def fake(model, lang):
        loads.append(model)
        if model in failing:
            raise OSError(f"{model} is not a local folder and is not a valid model identifier")
        return lambda *a, **k: []
    monkeypatch.setattr(main_module, "_load_ner_pipeline", fake)
    monkeypatch.setattr(main_module, "_ner_pipelines", {})
    monkeypatch.setattr(main_module, "_ner_fallbacks", {})
    return loads


def test_english_falls_back_when_the_default_cannot_be_loaded(main_module, monkeypatch):
    m = main_module
    loads = _loader(m, monkeypatch, failing={m.NER_MODEL_EN})
    assert callable(m.get_ner_pipeline("en"))
    assert loads == [m.NER_MODEL_EN, "dslim/bert-base-NER"]
    assert m._ner_fallbacks == {m.NER_MODEL_EN: "dslim/bert-base-NER"}
    assert m.effective_ner_model("en") == "dslim/bert-base-NER"          # cached results are keyed by the model that really ran
    assert m.ner_model_name("en") == m.NER_MODEL_EN                       # the configured default is unchanged
    body = m.health_models()
    assert body["ner_fallbacks"] == {m.NER_MODEL_EN: "dslim/bert-base-NER"}
    assert {r["route"] for r in body["ner_routes"]} == {"en", "indic", "*"}


def test_no_fallback_when_the_default_loads(main_module, monkeypatch):
    m = main_module
    loads = _loader(m, monkeypatch)
    m.get_ner_pipeline("en")
    assert loads == [m.NER_MODEL_EN] and m._ner_fallbacks == {}
    assert m.effective_ner_model("en") == m.NER_MODEL_EN


def test_a_model_without_fallback_still_raises(main_module, monkeypatch):
    m = main_module
    _loader(m, monkeypatch, failing={m.NER_MODEL_HI})
    with pytest.raises(OSError):
        m.get_ner_pipeline("hi")
    assert m._ner_fallbacks == {}


def test_fallback_is_forgotten_once_the_default_loads_again(main_module, monkeypatch):
    m = main_module
    failing = {m.NER_MODEL_EN}
    loads = _loader(m, monkeypatch, failing=failing)
    m.get_ner_pipeline("en")
    assert m._ner_fallbacks
    failing.clear()                                  # the weights were published / the network is back
    m._ner_pipelines.clear()                         # e.g. the model was unloaded by the memory limit
    m.get_ner_pipeline("en")
    assert m._ner_fallbacks == {} and loads[-1] == m.NER_MODEL_EN


def test_result_cache_key_follows_the_model_that_really_answered(main_module, monkeypatch):
    m = main_module
    keys = []
    monkeypatch.setattr(m.result_cache, "get_json", lambda k: keys.append(k) or None)
    monkeypatch.setattr(m.result_cache, "put_json", lambda k, v: None)
    monkeypatch.setattr(m, "_ner_pipelines", {m.NER_MODEL_EN: lambda texts, **kw: [[] for _ in texts]})
    monkeypatch.setattr(m, "_ner_fallbacks", {})
    m.run_ner("Alice met Bob in Paris.", "en")
    monkeypatch.setattr(m, "_ner_fallbacks", {m.NER_MODEL_EN: "dslim/bert-base-NER"})
    m.run_ner("Alice met Bob in Paris.", "en")
    assert len(keys) == 2 and keys[0] != keys[1]


# ── models that run code from their own repository (Otter) ───────────────
def test_otter_is_registered_pinned_and_not_routed_anywhere():
    reg = ner_registry.load(env={})
    otter = reg.options_for("whoisjones/otter-cross-mmbert")
    assert otter["trust_remote_code"] is True and otter["loader"] == "otter"
    assert len(otter["revision"]) == 40 and otter["threshold"] == 0.25
    assert "whoisjones/otter-cross-mmbert" not in {r["source"] for r in reg.describe()}      # decision D2(a): no language uses it yet


@pytest.mark.parametrize("entry", [
    {"source": "x/y", "trust_remote_code": True},                                          # no pin
    {"source": "x/y", "trust_remote_code": True, "revision": "main"},                      # a branch is not a pin
    {"source": "x/y", "trust_remote_code": True, "revision": "8729188e"},                  # short hash
    {"source": "x/y", "loader": "otter", "revision": "8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d"},   # otter runs code: must say so
    {"source": "x/y", "loader": "magic"},                                                  # unknown loader
])
def test_remote_code_needs_a_full_commit_hash_and_an_explicit_flag(tmp_path, entry):
    with pytest.raises(ValueError):
        ner_registry.load(env={"NER_MODELS_FILE": _write(tmp_path, {"models": {"bad": entry}})})


def test_a_pinned_remote_code_model_is_accepted(tmp_path):
    ok = {"source": "x/y", "trust_remote_code": True, "revision": "a" * 40}
    reg = ner_registry.load(env={"NER_MODELS_FILE": _write(tmp_path, {"models": {"good": ok}, "routes": {"es": "good"}})})
    assert reg.language_routes() == {"es": "x/y"} and reg.options_for("x/y")["revision"] == "a" * 40


def test_a_model_given_only_by_id_or_folder_never_gets_remote_code(main_module, monkeypatch):
    monkeypatch.setattr(main_module, "_NER_REGISTRY", ner_registry.load(env={"NER_MODEL_MULTI": "someone/else-ner"}))
    assert main_module._remote_code_options("someone/else-ner") == {}


def test_remote_code_runs_only_with_the_pin_and_can_be_switched_off(main_module, monkeypatch):
    m = main_module
    monkeypatch.delenv("NER_ALLOW_REMOTE_CODE", raising=False)
    assert m._remote_code_options("whoisjones/otter-cross-mmbert") == {"trust_remote_code": True,
                                                                        "revision": "8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d"}
    monkeypatch.setenv("NER_ALLOW_REMOTE_CODE", "0")
    with pytest.raises(RuntimeError, match="switched off"):
        m._remote_code_options("whoisjones/otter-cross-mmbert")
    assert m._remote_code_options("Davlan/xlm-roberta-base-ner-hrl") == {}      # ordinary models are not affected


def test_otter_route_loads_through_the_otter_loader_with_its_pin_and_threshold(main_module, monkeypatch, tmp_path):
    m = main_module
    extra = _write(tmp_path, {"routes": {"es": "multi-otter"}})
    monkeypatch.setattr(m, "_NER_REGISTRY", ner_registry.load(env={"NER_MODELS_FILE": extra}))
    monkeypatch.setattr(m, "_NER_LANG_ROUTES", m._NER_REGISTRY.language_routes())
    assert m.ner_model_name("es") == "whoisjones/otter-cross-mmbert" and m.ner_model_name("fr") == m.NER_MODEL_MULTI
    seen = {}
    import otter_ner
    monkeypatch.setattr(otter_ner, "load",
                        lambda model_id, revision, threshold=0.25, device=None: seen.update(id=model_id, rev=revision, thr=threshold) or "otter-pipeline")
    monkeypatch.delenv("NER_ONNX_DIR", raising=False)
    monkeypatch.delenv("NER_ALLOW_REMOTE_CODE", raising=False)
    assert m._load_ner_pipeline("whoisjones/otter-cross-mmbert", "es") == "otter-pipeline"
    assert seen == {"id": "whoisjones/otter-cross-mmbert", "rev": "8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d", "thr": 0.25}
    monkeypatch.setenv("NER_ALLOW_REMOTE_CODE", "0")
    with pytest.raises(RuntimeError):
        m._load_ner_pipeline("whoisjones/otter-cross-mmbert", "es")


def test_cache_key_changes_with_the_pinned_commit_and_threshold():
    reg = ner_registry.load(env={})
    assert reg.variant_tag("Davlan/xlm-roberta-base-ner-hrl") == ""
    assert reg.variant_tag("whoisjones/otter-cross-mmbert") == "rev=8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d;thr=0.25"


def test_otter_wrapper_maps_labels_recovers_missing_offsets_and_ignores_pipeline_arguments():
    import otter_ner

    class FakeModel:
        def predict(self, texts, labels, threshold, batch_size):
            assert labels == ["person", "organization", "location"] and threshold == 0.25
            return [[{"text": "Rahul", "label": "person", "start": 0, "end": 5, "score": 0.9},
                     {"text": "paris", "label": "location", "start": None, "end": None, "score": 0.8},     # no offsets: found again, case-folded
                     {"text": "2025", "label": "date", "start": 20, "end": 24, "score": 0.9},              # not a wanted type
                     {"text": "ghost", "label": "person", "start": None, "end": None, "score": 0.9}] if t.startswith("Rahul") else [] for t in texts]

    nlp = otter_ner.OtterPipeline(FakeModel(), 0.25)
    out = nlp(["Rahul lives in Paris.", "nobody"], batch_size=1, aggregation_strategy="simple", stride=0)
    assert out[1] == []
    assert out[0] == [{"entity_group": "PER", "score": 0.9, "start": 0, "end": 5, "word": "Rahul"},
                      {"entity_group": "LOC", "score": 0.8, "start": 15, "end": 20, "word": "paris"}]
