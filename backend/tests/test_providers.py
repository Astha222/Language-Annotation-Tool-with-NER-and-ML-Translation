"""Provider selection, fallback, cool-downs and chunking for both interactive and document translation."""
import pytest
from fastapi import HTTPException

import azure_translator as az


def never():
    return False


# ── configuration ────────────────────────────────────────────────────────
@pytest.mark.parametrize("env,expected", [
    ("azure,local", ["azure", "local"]),
    ("local , azure", ["local", "azure"]),
    ("google", ["google"]),
    ("LOCAL,bogus,MyMemory", ["local", "mymemory"]),
    ("", ["local"]),
    ("nonsense", ["local"]),
])
def test_provider_order_parsing(main_module, monkeypatch, env, expected):
    monkeypatch.setenv("TRANSLATION_PROVIDERS", env)
    assert main_module._provider_order() == expected


def test_default_order_is_azure_then_local(main_module, monkeypatch):
    monkeypatch.delenv("TRANSLATION_PROVIDERS", raising=False)
    assert main_module._provider_order() == ["azure", "local"]


# ── which provider answers ───────────────────────────────────────────────
def test_azure_is_used_first_when_a_key_is_configured(main_module, providers, azure_on):
    used = []
    out = main_module.perform_translation("en", "hi", "Hello world.", used)
    assert out == "AZ:Hello world." and used == ["azure"] and not providers["local"]


def test_local_models_are_the_default_without_any_key(main_module, providers):
    used = []
    out = main_module.perform_translation("en", "hi", "Hello world.", used)
    assert out == "LOC:Hello world." and used == ["local"]
    assert not providers["azure"] and not providers["google"] and not providers["mymemory"]


def test_order_can_prefer_local_even_with_a_key(main_module, providers, azure_on, monkeypatch):
    monkeypatch.setenv("TRANSLATION_PROVIDERS", "local,azure")
    assert main_module.perform_translation("en", "hi", "Hello.") == "LOC:Hello."


def test_google_and_mymemory_are_opt_in(main_module, providers, monkeypatch):
    monkeypatch.setattr(main_module.local_translator, "is_available", lambda: False)
    with pytest.raises(HTTPException) as e:                          # default order never touches them
        main_module.perform_translation("en", "hi", "Hello.")
    assert e.value.status_code == 502 and not providers["google"] and not providers["mymemory"]
    monkeypatch.setenv("TRANSLATION_PROVIDERS", "mymemory,google")
    assert main_module.perform_translation("en", "hi", "Hello.") == "M:Hello."


def test_same_language_returns_the_text_untouched(main_module, providers):
    assert main_module.perform_translation("en", "en", "Hello.") == "Hello."
    assert not any(providers.values())


# ── fallback and cool-downs ──────────────────────────────────────────────
def test_azure_auth_failure_falls_back_to_local_and_azure_is_skipped_afterwards(main_module, providers, azure_on, monkeypatch):
    attempts = []

    def rejecting(*a, **k):
        attempts.append(1)
        raise az.AzureAuthError("HTTP 401")

    monkeypatch.setattr(main_module.azure_translator, "translate", rejecting)
    assert main_module.perform_translation("en", "hi", "First.") == "LOC:First."
    assert main_module.perform_translation("en", "hi", "Second.") == "LOC:Second."
    assert len(attempts) == 1                                        # blocked after the first rejection
    assert not main_module._provider_ok("azure")


def test_azure_throttling_falls_back_quickly_for_interactive_requests(main_module, providers, azure_on, monkeypatch):
    def throttled(*a, **k):
        raise az.AzureThrottled("HTTP 429", retry_after=7)

    monkeypatch.setattr(main_module.azure_translator, "translate", throttled)
    assert main_module.perform_translation("en", "hi", "Hello.") == "LOC:Hello."
    assert not main_module._provider_ok("azure")


def test_documents_wait_out_azure_throttling_instead_of_falling_back(main_module, providers, azure_on, monkeypatch):
    results = [az.AzureThrottled("HTTP 429", retry_after=1), az.AzureThrottled("HTTP 429"), "AZ:done"]

    def flaky(src, tgt, text, **k):
        r = results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(main_module.azure_translator, "translate", flaky)
    text, status, provider = main_module._translate_doc_chunk("en", "hi", "Hello.\n", never)
    assert (text, status, provider) == ("AZ:done\n", "ok", "azure") and not providers["local"]


def test_a_failing_local_model_is_put_on_cooldown(main_module, providers, monkeypatch):
    calls = []

    def broken(*a, **k):
        calls.append(1)
        raise RuntimeError("model could not be downloaded")

    monkeypatch.setattr(main_module.local_translator, "translate", broken)
    monkeypatch.setenv("TRANSLATION_PROVIDERS", "local,mymemory")
    assert main_module.perform_translation("en", "hi", "One.") == "M:One."
    assert main_module.perform_translation("en", "hi", "Two.") == "M:Two."
    assert len(calls) == 1 and not main_module._provider_ok("local")


def test_unsupported_language_for_local_moves_on_or_fails_clearly(main_module, providers):
    with pytest.raises(HTTPException) as e:
        main_module.perform_translation("ko", "hi", "안녕하세요.")
    assert e.value.status_code == 502 and "no translation provider" in e.value.detail
    assert not providers["local"]


def test_all_providers_failing_raises_502_for_interactive_requests(main_module, providers, azure_on, monkeypatch):
    monkeypatch.setenv("TRANSLATION_PROVIDERS", "azure,local,google,mymemory")
    for name in ("azure_translator.translate", "local_translator.translate"):
        owner, attr = name.split(".")
        monkeypatch.setattr(getattr(main_module, owner), attr, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(main_module, "_google_translate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(main_module, "_mymemory_translate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(HTTPException) as e:
        main_module.perform_translation("en", "hi", "Hello.")
    assert e.value.status_code == 502


def test_a_document_chunk_that_cannot_be_translated_keeps_its_text(main_module, providers, monkeypatch):
    monkeypatch.setattr(main_module.local_translator, "is_available", lambda: False)
    text, status, provider = main_module._translate_doc_chunk("en", "hi", "Keep me.\n", never)
    assert (text, status, provider) == ("Keep me.\n", "failed", None)


def test_non_text_chunks_are_skipped_without_a_request(main_module, providers):
    assert main_module._translate_doc_chunk("en", "hi", " 123 --- \n", never)[1] == "skipped"
    assert not any(providers.values())


def test_cancelling_stops_further_attempts(main_module, providers):
    text, status, provider = main_module._translate_doc_chunk("en", "hi", "Hello.", lambda: True)
    assert status == "failed" and not any(providers.values())


# ── chunking and formatting ──────────────────────────────────────────────
def test_chunk_size_follows_the_first_ready_provider(main_module, providers, monkeypatch):
    assert main_module._chunk_chars("en", "hi") == main_module.LOCAL_CHUNK_CHARS            # no key: local first
    monkeypatch.setenv("AZURE_TRANSLATOR_KEY", "k")
    assert main_module._chunk_chars("en", "hi") == main_module.AZURE_CHUNK_CHARS
    monkeypatch.setenv("TRANSLATION_PROVIDERS", "mymemory")
    assert main_module._chunk_chars("en", "hi") == main_module.TRANSLATE_CHUNK_CHARS


def test_long_text_is_split_and_the_pieces_stay_in_order(main_module, providers):
    text = "Sentence number one is here. " * 200                           # ~5,800 chars -> several local chunks
    out = main_module.perform_translation("en", "hi", text)
    assert len(providers["local"]) > 3
    assert all(len(t) <= main_module.LOCAL_CHUNK_CHARS for _, _, t in providers["local"])
    assert out.count("LOC:") == len(providers["local"])


def test_line_breaks_between_chunks_survive(main_module, providers):
    out = main_module.perform_translation("en", "hi", "Heading\nBody text here.\n\nNext paragraph.")
    assert "\n" in out


def test_job_progress_weights_translation_more_when_local_models_do_the_work(main_module, providers, monkeypatch):
    assert main_module._first_ready("en", "hi") == "local"
    monkeypatch.setenv("AZURE_TRANSLATOR_KEY", "k")
    assert main_module._first_ready("en", "hi") == "azure"
