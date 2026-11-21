"""Models are loaded and warmed once at startup (PRELOAD_MODELS), and /health/models reports what is loaded."""
import pytest
from fastapi.testclient import TestClient


def test_nothing_is_preloaded_when_switched_off(main_module, monkeypatch):
    called = []
    monkeypatch.setattr(main_module, "_warm_up", lambda name: called.append(name))
    monkeypatch.setenv("PRELOAD_MODELS", "none")
    main_module.MODEL_STATUS.clear()
    assert main_module.preload_models() == {} and called == []


def test_listed_models_are_warmed_in_order_and_timed(main_module, monkeypatch):
    called = []
    monkeypatch.setattr(main_module, "_warm_up", lambda name: called.append(name))
    monkeypatch.setenv("PRELOAD_MODELS", "ner_en, mt:en-hi")
    main_module.MODEL_STATUS.clear()
    status = main_module.preload_models()
    assert called == ["ner_en", "mt:en-hi"]
    assert status["ner_en"]["status"] == "ready" and "seconds" in status["mt:en-hi"]


def test_a_failing_model_does_not_stop_the_others_or_the_server(main_module, monkeypatch):
    def warm(name):
        if name == "ner_en":
            raise OSError("no space left on device")
    monkeypatch.setattr(main_module, "_warm_up", warm)
    monkeypatch.setenv("PRELOAD_MODELS", "ner_en,ner_multi")
    main_module.MODEL_STATUS.clear()
    status = main_module.preload_models()
    assert status["ner_en"]["status"] == "failed" and "no space left" in status["ner_en"]["error"]
    assert status["ner_multi"]["status"] == "ready"


def test_unknown_model_name_is_reported_as_failed(main_module, monkeypatch):
    monkeypatch.setenv("PRELOAD_MODELS", "banana")
    main_module.MODEL_STATUS.clear()
    assert main_module.preload_models()["banana"]["status"] == "failed"


def test_warm_up_runs_the_real_code_paths(main_module, monkeypatch):
    seen = []
    monkeypatch.setattr(main_module, "run_ner", lambda text, lang, *a, **k: seen.append(("ner", lang)))
    monkeypatch.setattr(main_module.local_translator, "translate", lambda s, t, text, *a, **k: seen.append(("mt", s, t)))
    for name in ("ner_en", "ner_multi", "mt:en-fr"):
        main_module._warm_up(name)
    assert seen == [("ner", "en"), ("ner", "fr"), ("mt", "en", "fr")]


def test_preload_happens_at_startup_before_requests_are_served(main_module, monkeypatch):
    called = []
    monkeypatch.setattr(main_module, "_warm_up", lambda name: called.append(name))
    monkeypatch.setenv("PRELOAD_MODELS", "ner_en")
    main_module.MODEL_STATUS.clear()
    with TestClient(main_module.app) as c:                    # entering the context runs the lifespan startup
        assert called == ["ner_en"]
        body = c.get("/health/models").json()
    assert body["preload"]["ner_en"]["status"] == "ready"


def test_health_models_lists_loaded_models(client, main_module, monkeypatch):
    monkeypatch.setitem(main_module._ner_pipelines, "dslim/bert-base-NER", object())
    monkeypatch.setattr(main_module.local_translator, "loaded_engines", lambda: ["en-hi"])
    body = client.get("/health/models").json()
    assert "dslim/bert-base-NER" in body["ner_loaded"] and body["translation_loaded"] == ["en-hi"]


# ── CPU threads ──────────────────────────────────────────────────────────
def _fake_torch(monkeypatch):
    import sys
    import types
    calls = []
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(set_num_threads=lambda n: calls.append(n)))
    return calls


def test_torch_threads_setting_wins(main_module, monkeypatch):
    calls = _fake_torch(monkeypatch)
    monkeypatch.setenv("TORCH_THREADS", "3")
    assert main_module.configure_threads() == 3 and calls == [3]


def test_default_is_the_number_of_physical_cores(main_module, monkeypatch):
    import sys
    import types
    calls = _fake_torch(monkeypatch)
    monkeypatch.delenv("TORCH_THREADS", raising=False)
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(cpu_count=lambda logical=True: 4 if not logical else 8))
    assert main_module.configure_threads() == 4 and calls == [4]


def test_bad_thread_setting_falls_back_to_the_default(main_module, monkeypatch):
    import sys
    import types
    calls = _fake_torch(monkeypatch)
    monkeypatch.setenv("TORCH_THREADS", "banana")
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(cpu_count=lambda logical=True: 2))
    assert main_module.configure_threads() == 2 and calls == [2]


def test_unknown_core_count_leaves_pytorch_alone(main_module, monkeypatch):
    import sys
    import types
    calls = _fake_torch(monkeypatch)
    monkeypatch.delenv("TORCH_THREADS", raising=False)
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(cpu_count=lambda logical=True: None))
    assert main_module.configure_threads() is None and calls == []


# ── optional ONNX build of the English NER model ─────────────────────────
def _fake_onnx_module(monkeypatch):
    import sys
    import types
    calls = []

    def load_pipeline(model_dir, file="model_int8.onnx", threads=None):
        calls.append((str(model_dir), file, threads))
        return lambda texts, **kw: []
    monkeypatch.setitem(sys.modules, "onnx_ner", types.SimpleNamespace(load_pipeline=load_pipeline))
    return calls


def _no_real_models(monkeypatch):
    """Replace `from transformers import pipeline` with a recorder, so no test can download a model."""
    import sys
    import types
    built = []
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(pipeline=lambda *a, **k: built.append(a) or object()))
    return built


def test_onnx_model_is_used_when_the_folder_is_configured(main_module, monkeypatch, tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"x")
    calls = _fake_onnx_module(monkeypatch)
    monkeypatch.setattr(main_module, "_ner_pipelines", {})
    monkeypatch.setenv("NER_ONNX_DIR", str(tmp_path))
    monkeypatch.setenv("NER_ONNX_FILE", "model.onnx")
    monkeypatch.setenv("TORCH_THREADS", "4")
    main_module.get_ner_pipeline("en")
    assert calls == [(str(tmp_path), "model.onnx", "4")]


def test_onnx_is_only_used_for_english(main_module, monkeypatch, tmp_path):
    (tmp_path / "model_int8.onnx").write_bytes(b"x")
    calls = _fake_onnx_module(monkeypatch)
    monkeypatch.setattr(main_module, "_ner_pipelines", {})
    monkeypatch.setenv("NER_ONNX_DIR", str(tmp_path))
    _no_real_models(monkeypatch)                              # the other languages must not download anything
    main_module.get_ner_pipeline("fr")
    assert calls == []


def test_missing_onnx_file_falls_back_to_pytorch(main_module, monkeypatch, tmp_path):
    calls = _fake_onnx_module(monkeypatch)                      # folder exists but holds no model file
    built = _no_real_models(monkeypatch)
    monkeypatch.setattr(main_module, "_ner_pipelines", {})
    monkeypatch.setenv("NER_ONNX_DIR", str(tmp_path))
    main_module.get_ner_pipeline("en")
    assert calls == [] and built


# ── how text is cut up before NER ────────────────────────────────────────
def _recording_pipeline(main_module, monkeypatch):
    seen = {"calls": []}

    def nlp(texts, **kw):
        seen["calls"].append((list(texts), kw))
        return [[] for _ in texts]
    monkeypatch.setattr(main_module, "get_ner_pipeline", lambda lang="en": nlp)
    return seen


def test_default_chunking_has_no_stride(main_module, monkeypatch):
    seen = _recording_pipeline(main_module, monkeypatch)
    monkeypatch.delenv("NER_STRIDE_TOKENS", raising=False)
    monkeypatch.delenv("NER_CHUNK_CHARS", raising=False)
    main_module.run_ner(("Rahul works in Pune. " * 100).strip(), "en")           # about 2,100 characters
    texts = [t for batch, _ in seen["calls"] for t in batch]
    assert all("stride" not in kw for _, kw in seen["calls"])
    assert all(len(t) <= main_module.NER_CHUNK_CHARS for t in texts) and len(texts) >= 3


def test_stride_setting_uses_large_pieces_and_overlapping_windows(main_module, monkeypatch):
    seen = _recording_pipeline(main_module, monkeypatch)
    monkeypatch.setenv("NER_STRIDE_TOKENS", "64")
    main_module.run_ner(("Rahul works in Pune. " * 100).strip(), "en")
    texts, kw = seen["calls"][0]
    assert kw["stride"] == 64 and len(texts) == 1                                 # one 2,100-character piece, not 3 chunks


def test_bad_or_huge_stride_is_sanitised(main_module, monkeypatch):
    seen = _recording_pipeline(main_module, monkeypatch)
    monkeypatch.setenv("NER_STRIDE_TOKENS", "banana")
    main_module.run_ner("Rahul works in Pune.", "en")
    assert "stride" not in seen["calls"][0][1]
    monkeypatch.setenv("NER_STRIDE_TOKENS", "9999")
    main_module.run_ner("Rahul works in Pune.", "en")
    assert seen["calls"][1][1]["stride"] == 256


def test_chunk_size_can_be_set(main_module, monkeypatch):
    seen = _recording_pipeline(main_module, monkeypatch)
    monkeypatch.delenv("NER_STRIDE_TOKENS", raising=False)
    monkeypatch.setenv("NER_CHUNK_CHARS", "300")
    main_module.run_ner(("Rahul works in Pune. " * 100).strip(), "en")
    texts = [t for batch, _ in seen["calls"] for t in batch]
    assert all(len(t) <= 300 for t in texts) and len(texts) >= 7


def test_ner_batch_size_defaults_to_one_and_can_be_set(main_module, monkeypatch):
    seen = _recording_pipeline(main_module, monkeypatch)
    monkeypatch.delenv("NER_BATCH_SIZE", raising=False)
    main_module.run_ner(("Rahul works in Pune. " * 100).strip(), "en")
    assert all(len(texts) == 1 for texts, kw in seen["calls"]) and all(kw["batch_size"] == 1 for _, kw in seen["calls"])
    seen["calls"].clear()
    monkeypatch.setenv("NER_BATCH_SIZE", "4")
    main_module.run_ner(("Rahul works in Pune. " * 100).strip(), "en")
    assert seen["calls"][0][1]["batch_size"] == 4 and len(seen["calls"][0][0]) == 3
    assert main_module.run_ner("Rahul works in Pune.", "en", batch_size=2) == []        # an explicit argument still wins


# ── disk space where models are stored ───────────────────────────────────
def test_model_cache_follows_hf_home(main_module, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models"))
    assert main_module.model_cache_dir() == tmp_path / "models"
    monkeypatch.delenv("HF_HOME")
    assert main_module.model_cache_dir().name == "huggingface"


def test_low_disk_is_reported_and_logged(main_module, monkeypatch, tmp_path, caplog):
    import shutil
    import types
    monkeypatch.setenv("HF_HOME", str(tmp_path / "does-not-exist-yet"))
    monkeypatch.setattr(shutil, "disk_usage", lambda p: types.SimpleNamespace(free=int(1.2e9)))
    monkeypatch.setenv("MIN_FREE_GB", "3")
    monkeypatch.setenv("PRELOAD_MODELS", "none")
    status = main_module.disk_status()
    assert status["low"] is True and status["free_gb"] == 1.2
    with caplog.at_level("WARNING"):
        main_module.preload_models()
    assert "1.2 GB free" in caplog.text


def test_enough_disk_is_not_flagged_and_health_reports_it(client, main_module, monkeypatch):
    import shutil
    import types
    monkeypatch.setattr(shutil, "disk_usage", lambda p: types.SimpleNamespace(free=int(50e9)))
    body = client.get("/health/models").json()
    assert body["disk"]["free_gb"] == 50.0 and body["disk"]["low"] is False


def test_english_ner_model_can_be_chosen_by_setting(main_module, monkeypatch):
    monkeypatch.delenv("NER_MODEL_EN", raising=False)
    assert main_module._default_ner_model() == "dslim/bert-base-NER"
    monkeypatch.setenv("NER_MODEL_EN", "  C:/models/student/best ")
    assert main_module._default_ner_model() == "C:/models/student/best"
    monkeypatch.setenv("NER_MODEL_EN", "   ")
    assert main_module._default_ner_model() == "dslim/bert-base-NER"


def test_multilingual_ner_model_can_be_chosen_by_setting_and_defaults_to_the_old_model(main_module, monkeypatch):
    assert main_module.NER_MODEL_MULTI == "Davlan/xlm-roberta-base-wikiann-ner"           # not switched: the owner decides
    monkeypatch.setenv("NER_MODEL_MULTI", " Davlan/xlm-roberta-base-ner-hrl ")
    assert main_module._env_model("NER_MODEL_MULTI", "x") == "Davlan/xlm-roberta-base-ner-hrl"
    monkeypatch.setenv("NER_MODEL_MULTI", "  ")
    assert main_module._env_model("NER_MODEL_MULTI", "x") == "x"


# ── Hindi NER model ──────────────────────────────────────────────────────
def test_each_language_group_has_its_own_ner_model(main_module):
    assert main_module.ner_model_name("en") == main_module.NER_MODEL_EN
    assert main_module.ner_model_name("hi") == main_module.NER_MODEL_HI == "neuronsbyisshu/indicbert-hi-ner-naamapadam"
    for lang in ("fr", "de", "zh", "ja", "ar", "ru", "es"):
        assert main_module.ner_model_name(lang) == main_module.NER_MODEL_MULTI


def test_hindi_uses_its_own_cache_entry_and_warm_up(main_module, monkeypatch):
    seen = []
    monkeypatch.setattr(main_module, "run_ner", lambda text, lang, *a, **k: seen.append(lang))
    main_module._warm_up("ner_hi")
    assert seen == ["hi"]


def test_tokenizer_from_a_newer_transformers_is_rebuilt_from_tokenizer_json(main_module, monkeypatch):
    import sys
    import types
    built = []

    def pipeline(task, model=None, tokenizer=None, aggregation_strategy=None):
        if tokenizer is None:
            raise ValueError("Tokenizer class TokenizersBackend does not exist or is not currently imported.")
        built.append(tokenizer)
        return "pipe"
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(pipeline=pipeline))
    monkeypatch.setattr(main_module, "_plain_fast_tokenizer", lambda model: "plain-tokenizer")
    assert main_module._hf_ner_pipeline("some/model") == "pipe" and built == ["plain-tokenizer"]


def test_other_loading_errors_are_not_swallowed(main_module, monkeypatch):
    import sys
    import types

    def pipeline(*a, **k):
        raise ValueError("something else is wrong")
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(pipeline=pipeline))
    with pytest.raises(ValueError, match="something else"):
        main_module._hf_ner_pipeline("some/model")


# ── Chinese and Japanese are written without spaces ──────────────────────
def test_entities_are_not_grown_to_the_nearest_space_when_snapping_is_off(main_module):
    text = "约瑟夫·斯大林本人同时将男人和女人"
    raw = [{"entity_group": "PER", "score": 0.9, "start": 0, "end": 7, "word": "约瑟夫·斯大林"}]
    kept = main_module._clean_entities(text, raw, 0, snap=False)
    assert [text[e["start"]:e["end"]] for e in kept] == ["约瑟夫·斯大林"]
    grown = main_module._clean_entities(text, raw, 0, snap=True)               # the default for spaced scripts
    assert len(text[grown[0]["start"]:grown[0]["end"]]) > 7


def test_latin_entities_are_still_snapped_to_whole_tokens(main_module):
    text = "We use Node.js daily"
    raw = [{"entity_group": "MISC", "score": 0.9, "start": 7, "end": 11, "word": "Node"}]
    assert [text[e["start"]:e["end"]] for e in main_module._clean_entities(text, raw, 0)] == ["Node.js"]


def test_chinese_and_japanese_use_word_piece_grouping(main_module, monkeypatch):
    seen = _recording_pipeline(main_module, monkeypatch)
    for lang in ("zh", "ja"):
        main_module.run_ner("约瑟夫·斯大林在莫斯科。", lang)
    main_module.run_ner("Rahul works in Pune.", "en")
    main_module.run_ner("राहुल पुणे में है।", "hi")
    kinds = [kw.get("aggregation_strategy") for _, kw in seen["calls"]]
    assert kinds == ["simple", "simple", None, None]


# ── the Indian-language NER model and translation messages ───────────────
def test_nine_indian_languages_share_the_open_model_and_two_are_left_out(main_module):
    for lang in ("hi", "mr", "bn", "ta", "te", "gu", "kn", "ml", "pa"):
        assert main_module.ner_model_name(lang) == main_module.NER_MODEL_HI
    for lang in ("or", "as", "fr", "ja"):                                    # Odia and Assamese are not offered
        assert main_module.ner_model_name(lang) == main_module.NER_MODEL_MULTI
    assert main_module.ner_model_name("en") == main_module.NER_MODEL_EN


def test_translation_error_says_what_to_do_when_azure_is_missing(main_module, monkeypatch):
    monkeypatch.setattr(main_module, "_azure_ready", lambda: False)
    msg = main_module._no_provider_message("en", "bn")
    assert "Offline translation is not available" in msg and "Azure" in msg
    assert "Offline translation is not available" not in main_module._no_provider_message("en", "fr")      # offline French exists
    monkeypatch.setattr(main_module, "_azure_ready", lambda: True)
    assert "Offline translation is not available" not in main_module._no_provider_message("en", "bn")


def test_translating_into_an_unsupported_language_gives_a_clear_502(main_module, monkeypatch):
    import pytest
    from fastapi import HTTPException
    monkeypatch.setattr(main_module, "_azure_ready", lambda: False)
    monkeypatch.setenv("TRANSLATION_PROVIDERS", "azure,local")
    with pytest.raises(HTTPException) as e:
        main_module.perform_translation("en", "bn", "Hello world.")
    assert e.value.status_code == 502 and "Azure" in e.value.detail


# ── NER memory limit (NER_MAX_MODELS) ───────────────────────────────────────
def _counting_loader(main_module, monkeypatch):
    loads = []
    monkeypatch.setattr(main_module, "_ner_pipelines", {})
    monkeypatch.setattr(main_module, "_hf_ner_pipeline", lambda model: loads.append(model) or object())
    monkeypatch.delenv("NER_ONNX_DIR", raising=False)
    return loads


def test_at_most_two_ner_models_stay_loaded_and_the_least_recently_used_goes(main_module, monkeypatch):
    m = main_module
    loads = _counting_loader(m, monkeypatch)
    monkeypatch.setenv("NER_MAX_MODELS", "2")
    m.get_ner_pipeline("en")
    m.get_ner_pipeline("hi")
    m.get_ner_pipeline("en")                       # English used again: now Hindi is the least recently used
    m.get_ner_pipeline("fr")                       # a third model: Hindi is unloaded
    assert list(m._ner_pipelines) == [m.NER_MODEL_EN, m.NER_MODEL_MULTI]
    assert loads == [m.NER_MODEL_EN, m.NER_MODEL_HI, m.NER_MODEL_MULTI]
    m.get_ner_pipeline("hi")                       # needed again: loaded again, English (least recent) goes
    assert list(m._ner_pipelines) == [m.NER_MODEL_MULTI, m.NER_MODEL_HI] and loads[-1] == m.NER_MODEL_HI


def test_a_loaded_model_is_reused_not_reloaded(main_module, monkeypatch):
    m = main_module
    loads = _counting_loader(m, monkeypatch)
    first = m.get_ner_pipeline("mr")
    assert m.get_ner_pipeline("hi") is first and loads == [m.NER_MODEL_HI]      # Marathi and Hindi share one model


def test_zero_means_no_limit_and_bad_values_fall_back_to_two(main_module, monkeypatch):
    m = main_module
    _counting_loader(m, monkeypatch)
    monkeypatch.setenv("NER_MAX_MODELS", "0")
    for lang in ("en", "hi", "fr"):
        m.get_ner_pipeline(lang)
    assert len(m._ner_pipelines) == 3
    monkeypatch.setenv("NER_MAX_MODELS", "lots")
    assert m._ner_max_models() == 2


def test_health_reports_the_limit_and_the_loaded_models_in_use_order(client, main_module, monkeypatch):
    monkeypatch.setenv("NER_MAX_MODELS", "2")
    body = client.get("/health/models").json()
    assert body["ner_max_models"] == 2 and isinstance(body["ner_loaded"], list)


def test_concurrent_requests_for_the_same_model_load_it_once(main_module, monkeypatch):
    import threading
    import time
    m = main_module
    loads = []
    monkeypatch.setattr(m, "_ner_pipelines", {})
    monkeypatch.setattr(m, "_hf_ner_pipeline", lambda model: (loads.append(model), time.sleep(0.2), object())[2])
    threads = [threading.Thread(target=m.get_ner_pipeline, args=("hi",)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert loads == [m.NER_MODEL_HI]
