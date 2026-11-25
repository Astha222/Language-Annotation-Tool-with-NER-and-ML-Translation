"""ONNX builds belong to ONE model: chosen per registry entry, refused for any other model, never a reason for a request to fail."""
import json
import logging
import sys
import types

import pytest

import ner_registry


def _registry(tmp_path, models, routes=None):
    path = tmp_path / "extra.json"
    path.write_text(json.dumps({"models": models, "routes": routes or {}}), encoding="utf-8")
    return ner_registry.load(env={"NER_MODELS_FILE": str(path)})


def _onnx_folder(tmp_path, name="onnx", source=None, files=("model.onnx",)):
    folder = tmp_path / name
    folder.mkdir()
    for f in files:
        (folder / f).write_bytes(b"x")
    if source:
        (folder / "export_info.json").write_text(json.dumps({"source": source}), encoding="utf-8")
    return folder


@pytest.fixture
def loader(main_module, monkeypatch):
    """Fake ONNX runtime and fake Hugging Face pipeline; records what was loaded."""
    calls = {"onnx": [], "torch": []}

    def load_pipeline(model_dir, file="model_int8.onnx", threads=None):
        calls["onnx"].append((str(model_dir), file))
        if calls.get("fail"):
            raise RuntimeError("onnxruntime is not installed")
        return lambda texts, **kw: [[] for _ in texts]
    monkeypatch.setitem(sys.modules, "onnx_ner", types.SimpleNamespace(load_pipeline=load_pipeline))
    monkeypatch.setattr(main_module, "_hf_ner_pipeline", lambda model: calls["torch"].append(model) or (lambda texts, **kw: [[] for _ in texts]))
    monkeypatch.setattr(main_module, "_ner_pipelines", {})
    monkeypatch.setattr(main_module, "_ner_onnx_used", {})
    monkeypatch.setattr(main_module, "_ner_fallbacks", {})
    monkeypatch.delenv("NER_ONNX_DIR", raising=False)
    monkeypatch.delenv("NER_ONNX_FILE", raising=False)
    return calls


def _use(main_module, monkeypatch, registry):
    monkeypatch.setattr(main_module, "_NER_REGISTRY", registry)
    monkeypatch.setattr(main_module, "_NER_LANG_ROUTES", registry.language_routes())
    monkeypatch.setattr(main_module, "NER_MODEL_EN", registry.group_source("en"))
    monkeypatch.setattr(main_module, "NER_MODEL_HI", registry.group_source("indic"))
    monkeypatch.setattr(main_module, "NER_MODEL_MULTI", registry.group_source("*"))


def test_a_registry_entry_gets_its_own_onnx_build_in_every_language_it_serves(main_module, monkeypatch, tmp_path, loader):
    folder = _onnx_folder(tmp_path, source="x/fr-model")
    reg = _registry(tmp_path, {"fr": {"source": "x/fr-model", "onnx_dir": str(folder), "onnx_file": "model.onnx"}}, {"fr": "fr"})
    _use(main_module, monkeypatch, reg)
    main_module.get_ner_pipeline("fr")
    assert loader["onnx"] == [(str(folder), "model.onnx")] and loader["torch"] == []
    assert main_module._ner_onnx_used == {"x/fr-model": (str(folder), "model.onnx")}


def test_onnx_file_defaults_to_the_fp32_build(main_module, monkeypatch, tmp_path, loader):
    folder = _onnx_folder(tmp_path, files=("model.onnx", "model_int8.onnx"))
    _use(main_module, monkeypatch, _registry(tmp_path, {"m": {"source": "x/m", "onnx_dir": str(folder)}}, {"fr": "m"}))
    main_module.get_ner_pipeline("fr")
    assert loader["onnx"][0][1] == "model.onnx"                      # identical output to PyTorch; INT8 is opt-in


def test_a_relative_onnx_dir_is_relative_to_the_backend_folder(main_module, monkeypatch, tmp_path, loader):
    _use(main_module, monkeypatch, _registry(tmp_path, {"m": {"source": "x/m", "onnx_dir": "models/m-onnx"}}, {"fr": "m"}))
    assert main_module._onnx_choice("x/m", "fr") is None             # nothing there: normal model
    target = main_module.Path(main_module.__file__).resolve().parent / "models" / "m-onnx"
    assert not target.exists()                                      # the test must not have created anything in the repository


def test_a_missing_onnx_file_means_the_normal_model_and_says_so(main_module, monkeypatch, tmp_path, loader, caplog):
    reg = _registry(tmp_path, {"m": {"source": "x/m", "onnx_dir": str(tmp_path / "nowhere")}}, {"fr": "m"})
    _use(main_module, monkeypatch, reg)
    with caplog.at_level(logging.WARNING):
        main_module.get_ner_pipeline("fr")
    assert loader["onnx"] == [] and loader["torch"] == ["x/m"]
    assert "does not exist" in caplog.text


def test_a_broken_onnx_build_falls_back_to_the_same_model_not_to_another_one(main_module, monkeypatch, tmp_path, loader, caplog):
    folder = _onnx_folder(tmp_path)
    reg = _registry(tmp_path, {"m": {"source": "x/m", "onnx_dir": str(folder), "fallback": "english-bert-base"}}, {"fr": "m"})
    _use(main_module, monkeypatch, reg)
    loader["fail"] = True
    with caplog.at_level(logging.WARNING):
        assert callable(main_module.get_ner_pipeline("fr"))
    assert loader["torch"] == ["x/m"]                                  # the same model, PyTorch, not the registry fallback
    assert main_module._ner_fallbacks == {} and main_module._ner_onnx_used == {}
    assert "could not be loaded" in caplog.text


def test_an_export_of_another_model_is_refused(main_module, monkeypatch, tmp_path, loader, caplog):
    folder = _onnx_folder(tmp_path, source="dslim/bert-base-NER")
    reg = _registry(tmp_path, {"m": {"source": "x/m", "onnx_dir": str(folder)}}, {"fr": "m"})
    _use(main_module, monkeypatch, reg)
    with caplog.at_level(logging.WARNING):
        main_module.get_ner_pipeline("fr")
    assert loader["onnx"] == [] and loader["torch"] == ["x/m"]
    assert "exported from dslim/bert-base-NER, not from x/m" in caplog.text


def test_the_old_setting_is_refused_for_a_changed_english_model(main_module, monkeypatch, tmp_path, loader, caplog):
    """The scenario this prevents: NER_ONNX_DIR still points to the bert-base export after the English default became another model."""
    folder = _onnx_folder(tmp_path, source="dslim/bert-base-NER", files=("model_int8.onnx",))
    monkeypatch.setenv("NER_ONNX_DIR", str(folder))
    with caplog.at_level(logging.WARNING):
        main_module.get_ner_pipeline("en")
    assert loader["onnx"] == [] and loader["torch"] == [main_module.NER_MODEL_EN]


def test_the_old_setting_still_works_for_the_model_it_was_made_from(main_module, monkeypatch, tmp_path, loader, caplog):
    folder = _onnx_folder(tmp_path, source=None, files=("model_int8.onnx",))       # an export made before export_info.json existed
    monkeypatch.setenv("NER_ONNX_DIR", str(folder))
    with caplog.at_level(logging.WARNING):
        main_module.get_ner_pipeline("en")
    assert loader["onnx"] == [(str(folder), "model_int8.onnx")]
    assert "NER_ONNX_DIR is the old way" in caplog.text and "cannot be checked" in caplog.text
    # an export made with the new script is checked against the English model
    monkeypatch.setattr(main_module, "_ner_pipelines", {})
    loader["onnx"].clear()
    verified = _onnx_folder(tmp_path, name="verified", source=main_module.NER_MODEL_EN, files=("model_int8.onnx",))
    monkeypatch.setenv("NER_ONNX_DIR", str(verified))
    main_module.get_ner_pipeline("en")
    assert loader["onnx"] == [(str(verified), "model_int8.onnx")]


def test_the_registry_beats_the_old_setting(main_module, monkeypatch, tmp_path, loader):
    new = _onnx_folder(tmp_path, name="new", source="x/en")
    old = _onnx_folder(tmp_path, name="old", files=("model_int8.onnx",))
    _use(main_module, monkeypatch, _registry(tmp_path, {"en": {"source": "x/en", "onnx_dir": str(new)}}, {"en": "en"}))
    monkeypatch.setenv("NER_ONNX_DIR", str(old))
    main_module.get_ner_pipeline("en")
    assert loader["onnx"] == [(str(new), "model.onnx")]


def test_the_result_cache_key_follows_the_onnx_build_in_use(main_module, monkeypatch, tmp_path, loader):
    m = main_module
    folder = _onnx_folder(tmp_path)
    keys = []
    monkeypatch.setattr(m.result_cache, "get_json", lambda k: keys.append(k) or None)
    monkeypatch.setattr(m.result_cache, "put_json", lambda k, v: None)
    m.run_ner("Alice met Bob in Paris.", "fr")                                   # PyTorch (no ONNX in the default registry)
    reg = _registry(tmp_path, {"m": {"source": m.NER_MODEL_MULTI, "onnx_dir": str(folder)}})
    _use(m, monkeypatch, reg)
    monkeypatch.setattr(m, "_ner_pipelines", {})
    m.run_ner("Alice met Bob in Paris.", "fr")                                   # now through the ONNX build of the same model
    assert len(keys) == 2 and keys[0] != keys[1]


def test_health_lists_the_models_that_run_from_onnx(main_module, monkeypatch, tmp_path, loader):
    folder = _onnx_folder(tmp_path)
    _use(main_module, monkeypatch, _registry(tmp_path, {"m": {"source": "x/m", "onnx_dir": str(folder)}}, {"fr": "m"}))
    assert main_module.health_models()["ner_onnx"] == {}
    main_module.get_ner_pipeline("fr")
    assert main_module.health_models()["ner_onnx"] == {"x/m": f"{folder}/model.onnx"}


@pytest.mark.parametrize("entry", [{"source": "x/y", "onnx_dir": "  "}, {"source": "x/y", "onnx_dir": "d", "onnx_file": "model.bin"}])
def test_bad_onnx_fields_stop_the_server_at_startup(tmp_path, entry):
    with pytest.raises(ValueError):
        _registry(tmp_path, {"bad": entry})


def test_the_export_script_records_which_model_the_folder_came_from(tmp_path):
    import export_onnx_ner
    info = export_onnx_ner.write_export_info(tmp_path, "muskan-424/some-model", ["model.onnx", "model_int8.onnx"], 17)
    saved = json.loads((tmp_path / "export_info.json").read_text(encoding="utf-8"))
    assert saved["source"] == "muskan-424/some-model" == info["source"] and saved["files"] == ["model.onnx", "model_int8.onnx"]


def test_an_entry_in_your_own_file_overrides_the_bundled_entry_with_the_same_source(tmp_path):
    folder = _onnx_folder(tmp_path)
    reg = _registry(tmp_path, {"my-hrl": {"source": "Davlan/xlm-roberta-base-ner-hrl", "onnx_dir": str(folder)}})
    assert reg.options_for("Davlan/xlm-roberta-base-ner-hrl")["onnx_dir"] == str(folder)
    assert "onnx_dir" not in ner_registry.load(env={}).options_for("Davlan/xlm-roberta-base-ner-hrl")
