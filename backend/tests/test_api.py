"""HTTP API: authentication, annotations, translation, documents and background jobs."""
import json
import time

import pytest


def wait_for(client, headers, job_id, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}", headers=headers).json()
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def upload(client, headers, name="note.txt", content=b"Pichai met Nadella in Seattle. " * 5):
    return client.post("/api/documents", files={"file": (name, content)}, headers=headers)


# ── authentication ───────────────────────────────────────────────────────
class TestAuth:
    def test_register_login_me(self, client):
        r = client.post("/api/auth/register", json={"username": "asha_k", "email": "Asha@Example.com", "password": "password123"})
        assert r.status_code == 201 and r.json()["email"] == "asha@example.com"          # stored lowercase
        token = client.post("/api/auth/login", json={"username": "asha_k", "password": "password123"}).json()["access_token"]
        me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200 and me.json()["username"] == "asha_k"

    @pytest.mark.parametrize("payload,fragment", [
        ({"username": "ab", "email": "a@b.co", "password": "password123"}, "Username"),
        ({"username": "bad name", "email": "a@b.co", "password": "password123"}, "Username"),
        ({"username": "gooduser", "email": "not-an-email", "password": "password123"}, "email"),
        ({"username": "gooduser", "email": "a@b.co", "password": "short"}, "8 characters"),
        ({"username": "gooduser", "email": "a@b.co", "password": "x" * 80}, "too long"),
    ])
    def test_registration_validation_returns_readable_strings(self, client, payload, fragment):
        r = client.post("/api/auth/register", json=payload)
        assert r.status_code == 400 and isinstance(r.json()["detail"], str) and fragment in r.json()["detail"]

    def test_duplicates_are_rejected_case_insensitively(self, client):
        client.post("/api/auth/register", json={"username": "Dupe_User", "email": "dupe@example.com", "password": "password123"})
        for username, email in (("dupe_user", "other@example.com"), ("someone_else", "DUPE@example.com")):
            r = client.post("/api/auth/register", json={"username": username, "email": email, "password": "password123"})
            assert r.status_code == 400

    def test_wrong_password_and_missing_token(self, client, auth):
        assert client.post("/api/auth/login", json={"username": "nobody", "password": "x"}).status_code == 401
        assert client.get("/api/auth/me").status_code == 401
        assert client.get("/api/auth/me", headers={"Authorization": "Bearer garbage"}).status_code == 401

    def test_cors_only_allows_the_configured_origin(self, client):
        ok = client.options("/api/auth/login", headers={"Origin": "http://localhost:8081", "Access-Control-Request-Method": "POST"})
        bad = client.options("/api/auth/login", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
        assert ok.headers.get("access-control-allow-origin") == "http://localhost:8081"
        assert "access-control-allow-origin" not in bad.headers


# ── annotations ──────────────────────────────────────────────────────────
class TestAnnotations:
    def create(self, client, auth, **extra):
        body = {"original_text": "Pichai met Nadella.", "translated_text": "X", "ner_tags": json.dumps([{"word": "Pichai"}]),
                "source_lang": "en", "target_lang": "hi", **extra}
        return client.post("/api/annotations", json=body, headers=auth)

    def test_crud(self, client, auth):
        a = self.create(client, auth, source_name="file.pdf")
        assert a.status_code == 201 and a.json()["source_name"] == "file.pdf"
        aid = a.json()["id"]
        assert client.get(f"/api/annotations/{aid}", headers=auth).json()["translated_text"] == "X"
        assert client.put(f"/api/annotations/{aid}", json={"translated_text": "Y"}, headers=auth).json()["translated_text"] == "Y"
        assert [x["id"] for x in client.get("/api/annotations", headers=auth).json()] == [aid]
        assert client.delete(f"/api/annotations/{aid}", headers=auth).status_code == 200
        assert client.get(f"/api/annotations/{aid}", headers=auth).status_code == 404

    def test_invalid_entity_json_is_a_400_and_nothing_is_stored(self, client, auth, main_module):
        r = client.post("/api/annotations", json={"original_text": "t", "ner_tags": "{not json"}, headers=auth)
        assert r.status_code == 400
        assert client.get("/api/annotations", headers=auth).json() == [] and main_module.ner_collection.docs == []

    def test_users_cannot_see_or_change_each_others_annotations(self, client, auth, auth2):
        aid = self.create(client, auth).json()["id"]
        assert client.get(f"/api/annotations/{aid}", headers=auth2).status_code == 404
        assert client.put(f"/api/annotations/{aid}", json={"translated_text": "hack"}, headers=auth2).status_code == 404
        assert client.delete(f"/api/annotations/{aid}", headers=auth2).status_code == 404
        assert client.get("/api/annotations", headers=auth2).json() == []

    def test_a_mongo_outage_rolls_the_sql_row_back(self, client, auth, main_module, monkeypatch):
        from pymongo.errors import PyMongoError

        def down(doc):
            raise PyMongoError("mongo is down")

        monkeypatch.setattr(main_module.ner_collection, "insert_one", down)
        r = self.create(client, auth)
        assert r.status_code == 503
        monkeypatch.undo()
        assert client.get("/api/annotations", headers=auth).json() == []

    def test_list_view_clips_documents_but_single_get_and_export_return_everything(self, client, auth, main_module):
        big = "word " * 10000                                                   # 50,000 characters
        tags = [{"entity_group": "PER", "word": f"n{i}", "score": 0.9} for i in range(500)]
        aid = self.create(client, auth, original_text=big, translated_text=big, ner_tags=json.dumps(tags)).json()["id"]
        listed = client.get("/api/annotations", headers=auth).json()[0]
        assert listed["truncated"] and len(listed["original_text"]) < 25000 and len(json.loads(listed["ner_tags"])) == 200
        full = client.get(f"/api/annotations/{aid}", headers=auth).json()
        assert not full["truncated"] and len(full["original_text"]) == len(big) and len(json.loads(full["ner_tags"])) == 500
        exported = client.get("/api/export", headers=auth).json()[0]
        assert len(exported["original_text"]) == len(big) and len(exported["ner_tags"]) == 500


# ── text analysis endpoints ──────────────────────────────────────────────
class TestTextEndpoints:
    def test_translate_reports_which_provider_did_it(self, client, auth, providers):
        r = client.post("/api/translate", json={"text": "Hello.", "source_lang": "en", "target_lang": "hi"}, headers=auth)
        assert r.status_code == 200 and r.json()["translated_text"] == "LOC:Hello."
        assert "Local" in r.json()["provider"]

    def test_translate_with_auto_source_detects_the_language(self, client, auth, providers):
        r = client.post("/api/translate", json={"text": "Bonjour tout le monde, comment allez-vous aujourd'hui ?",
                                                "source_lang": "auto", "target_lang": "en"}, headers=auth)
        assert r.status_code == 200 and providers["local"][0][0] == "fr"

    def test_translate_failure_is_a_502_with_a_message(self, client, auth, providers):
        r = client.post("/api/translate", json={"text": "안녕하세요.", "source_lang": "ko", "target_lang": "hi"}, headers=auth)
        assert r.status_code == 502 and isinstance(r.json()["detail"], str)

    def test_text_length_limit(self, client, auth, providers):
        r = client.post("/api/translate", json={"text": "a" * 20001, "source_lang": "en", "target_lang": "hi"}, headers=auth)
        assert r.status_code == 413

    def test_ner_endpoint_returns_offsets_into_the_text(self, client, auth):
        text = "Pichai met Nadella."
        ents = client.post("/api/ner", json={"text": text, "lang": "en"}, headers=auth).json()["entities"]
        assert [text[e["start"]:e["end"]] for e in ents] == ["Pichai", "Nadella"]

    def test_detect_language_returns_a_base_code(self, client, auth):
        r = client.post("/api/detect-language", json={"text": "Das ist ein ganz normaler deutscher Satz über das Wetter."}, headers=auth)
        assert r.json()["detected_language"] == "de"

    def test_endpoints_require_a_login(self, client):
        for path in ("/api/ner", "/api/translate", "/api/detect-language"):
            assert client.post(path, json={"text": "x"}).status_code == 401


# ── documents and jobs ───────────────────────────────────────────────────
class TestDocuments:
    def test_upload_returns_a_preview_and_stats(self, client, auth):
        r = upload(client, auth)
        body = r.json()
        assert r.status_code == 200 and body["chars"] > 100 and body["preview"].startswith("Pichai") and body["filename"] == "note.txt"

    @pytest.mark.parametrize("name,content,fragment", [
        ("old.doc", b"\xd0\xcf\x11\xe0", ".docx"),
        ("a.zip", b"PK\x03\x04", "Unsupported"),
        ("empty.txt", b"  \n ", "No text"),
        ("e.txt", b"", "empty"),
        ("bad.pdf", b"not a pdf", "PDF"),
    ])
    def test_unusable_files_are_rejected_with_a_reason(self, client, auth, name, content, fragment):
        r = upload(client, auth, name, content)
        assert r.status_code == 400 and fragment in r.json()["detail"]

    def test_a_locked_temporary_file_does_not_hide_the_real_error(self, client, auth, monkeypatch):
        """On Windows a scanner can lock the just-written upload; deleting it then fails. The user must still get the real reason (was a flaky failure)."""
        import pathlib
        real_unlink = pathlib.Path.unlink

        def locked(self, *args, **kwargs):
            if self.name.startswith("upload_"):
                raise PermissionError("[WinError 32] The process cannot access the file because it is being used by another process")
            return real_unlink(self, *args, **kwargs)
        monkeypatch.setattr(pathlib.Path, "unlink", locked)
        r = upload(client, auth, "e.txt", b"  \n ")
        assert r.status_code == 400 and "No text" in r.json()["detail"]
        ok = upload(client, auth, "fine.txt", "Rahul works in Delhi.".encode())
        assert ok.status_code == 200

    def test_oversized_uploads_are_refused(self, client, auth, main_module, monkeypatch):
        monkeypatch.setattr(main_module, "MAX_UPLOAD_BYTES", 1024)
        r = upload(client, auth, "big.txt", b"a" * 5000)
        assert r.status_code == 413

    def test_document_text_limit(self, client, auth, main_module, monkeypatch):
        monkeypatch.setattr(main_module, "MAX_DOC_CHARS", 100)
        assert upload(client, auth, "long.txt", b"a " * 200).status_code == 413

    def test_documents_are_private_to_their_owner(self, client, auth, auth2):
        doc = upload(client, auth).json()
        r = client.post(f"/api/documents/{doc['doc_id']}/analyze", json={"target_lang": "hi"}, headers=auth2)
        assert r.status_code == 404
        assert client.delete(f"/api/documents/{doc['doc_id']}", headers=auth2).status_code == 404

    def test_path_traversal_ids_are_rejected(self, client, auth):
        assert client.delete("/api/documents/..%2F..%2Fsecret", headers=auth).status_code in (404, 405)
        assert client.post("/api/documents/not-a-real-id/analyze", json={"target_lang": "hi"}, headers=auth).status_code == 404

    def test_delete_removes_the_document(self, client, auth):
        doc = upload(client, auth).json()
        assert client.delete(f"/api/documents/{doc['doc_id']}", headers=auth).status_code == 200
        assert client.post(f"/api/documents/{doc['doc_id']}/analyze", json={"target_lang": "hi"}, headers=auth).status_code == 404

    def test_analyze_translates_extracts_entities_and_saves_to_history(self, client, auth, providers):
        doc = upload(client, auth).json()
        job_id = client.post(f"/api/documents/{doc['doc_id']}/analyze", json={"source_lang": "en", "target_lang": "hi"},
                             headers=auth).json()["job_id"]
        job = wait_for(client, auth, job_id)
        assert job["status"] == "done" and job["progress"] == 1.0
        res = job["result"]
        assert res["translated_preview"].startswith("LOC:Pichai") and res["entity_total"] == 15 and res["entity_unique"] == 3
        assert res["translation_engines"] == ["Local open-source models (offline)"]
        saved = client.get(f"/api/annotations/{res['annotation_id']}", headers=auth).json()
        assert saved["source_name"] == "note.txt" and saved["translated_text"].startswith("LOC:") and saved["source_lang"] == "en"
        assert len(json.loads(saved["ner_tags"])) == 15

    def test_job_uses_azure_when_it_is_configured(self, client, auth, providers, azure_on):
        doc = upload(client, auth).json()
        job_id = client.post(f"/api/documents/{doc['doc_id']}/analyze", json={"source_lang": "en", "target_lang": "hi"}, headers=auth).json()["job_id"]
        res = wait_for(client, auth, job_id)["result"]
        assert res["translation_engines"] == ["Azure Translator"] and res["translated_preview"].startswith("AZ:")

    def test_same_language_skips_translation_with_a_warning(self, client, auth, providers):
        doc = upload(client, auth).json()
        job_id = client.post(f"/api/documents/{doc['doc_id']}/analyze", json={"source_lang": "en", "target_lang": "en"}, headers=auth).json()["job_id"]
        job = wait_for(client, auth, job_id)
        assert job["status"] == "done" and job["result"]["translated_chars"] == 0 and "same" in job["warnings"][0]
        assert not any(providers.values())

    def test_failed_translation_keeps_the_entities_and_warns(self, client, auth, providers, main_module, monkeypatch):
        monkeypatch.setattr(main_module.local_translator, "is_available", lambda: False)     # no provider can run
        doc = upload(client, auth).json()
        job_id = client.post(f"/api/documents/{doc['doc_id']}/analyze", json={"source_lang": "en", "target_lang": "hi"}, headers=auth).json()["job_id"]
        job = wait_for(client, auth, job_id)
        assert job["status"] == "done" and job["result"]["translated_chars"] == 0 and job["result"]["entity_total"] > 0
        assert any("No translation provider" in w for w in job["warnings"])

    def test_only_one_job_per_user_at_a_time(self, client, auth, providers, main_module, monkeypatch):
        gate = {"go": False}

        def slow(src, tgt, text, cancelled=None):
            while not gate["go"] and not (cancelled and cancelled()):
                time.sleep(0.01)
            return "LOC:" + text

        monkeypatch.setattr(main_module.local_translator, "translate", slow)
        doc = upload(client, auth).json()
        body = {"source_lang": "en", "target_lang": "hi"}
        job_id = client.post(f"/api/documents/{doc['doc_id']}/analyze", json=body, headers=auth).json()["job_id"]
        assert client.post(f"/api/documents/{doc['doc_id']}/analyze", json=body, headers=auth).status_code == 409
        gate["go"] = True
        assert wait_for(client, auth, job_id)["status"] == "done"
        assert client.post(f"/api/documents/{doc['doc_id']}/analyze", json=body, headers=auth).status_code == 202

    def test_jobs_can_be_cancelled(self, client, auth, providers, main_module, monkeypatch):
        def slow(src, tgt, text, cancelled=None):
            while not (cancelled and cancelled()):
                time.sleep(0.01)
            raise main_module.local_translator.Cancelled()

        monkeypatch.setattr(main_module.local_translator, "translate", slow)
        doc = upload(client, auth).json()
        job_id = client.post(f"/api/documents/{doc['doc_id']}/analyze", json={"source_lang": "en", "target_lang": "hi"}, headers=auth).json()["job_id"]
        client.post(f"/api/jobs/{job_id}/cancel", headers=auth)
        assert wait_for(client, auth, job_id)["status"] == "cancelled"

    def test_jobs_are_private_to_their_owner(self, client, auth, auth2, providers):
        doc = upload(client, auth).json()
        job_id = client.post(f"/api/documents/{doc['doc_id']}/analyze", json={"source_lang": "en", "target_lang": "hi"}, headers=auth).json()["job_id"]
        assert client.get(f"/api/jobs/{job_id}", headers=auth2).status_code == 404
        assert client.post(f"/api/jobs/{job_id}/cancel", headers=auth2).status_code == 404
        wait_for(client, auth, job_id)

    def test_unknown_job_is_a_404_with_a_hint(self, client, auth):
        r = client.get("/api/jobs/" + "0" * 32, headers=auth)
        assert r.status_code == 404 and "run it again" in r.json()["detail"]


class TestIndependentNerAndTranslation:
    def test_ner_only_flow(self, client, auth):
        # 1. Run NER directly
        ner_resp = client.post("/api/ner", json={"text": "Sundar Pichai leads Google in Mountain View .", "lang": "en"}, headers=auth)
        assert ner_resp.status_code == 200
        entities = ner_resp.json()["entities"]
        assert len(entities) > 0

        # 2. Save annotation without translation
        save_resp = client.post("/api/annotations", json={
            "original_text": "Sundar Pichai leads Google in Mountain View .",
            "translated_text": None,
            "ner_tags": json.dumps(entities),
            "source_lang": "en",
            "target_lang": "fr",
        }, headers=auth)
        assert save_resp.status_code == 201
        data = save_resp.json()
        assert data["translated_text"] is None
        assert data["ner_tags"] is not None

        # 3. Retrieve from history
        get_resp = client.get(f"/api/annotations/{data['id']}", headers=auth)
        assert get_resp.status_code == 200
        assert get_resp.json()["translated_text"] is None

    def test_translation_only_flow(self, client, auth, providers):
        # 1. Run Translation directly
        tr_resp = client.post("/api/translate", json={
            "text": "Hello world",
            "source_lang": "en",
            "target_lang": "fr",
        }, headers=auth)
        assert tr_resp.status_code == 200
        translated = tr_resp.json()["translated_text"]
        assert translated

        # 2. Save annotation without NER tags
        save_resp = client.post("/api/annotations", json={
            "original_text": "Hello world",
            "translated_text": translated,
            "ner_tags": json.dumps([]),
            "source_lang": "en",
            "target_lang": "fr",
        }, headers=auth)
        assert save_resp.status_code == 201
        data = save_resp.json()
        assert data["translated_text"] == translated
        assert data["ner_tags"] is None or data["ner_tags"] == "[]"

    def test_ner_then_translate_updates_annotation(self, client, auth, providers):
        # 1. Run NER first and save
        save_resp = client.post("/api/annotations", json={
            "original_text": "Satya Nadella works at Microsoft .",
            "translated_text": None,
            "ner_tags": json.dumps([{"word": "Satya Nadella", "entity_group": "PER", "score": 0.99, "start": 0, "end": 13}]),
            "source_lang": "en",
            "target_lang": "fr",
        }, headers=auth)
        ann_id = save_resp.json()["id"]

        # 2. Run Translation afterwards and update existing annotation
        tr_resp = client.post("/api/translate", json={"text": "Satya Nadella works at Microsoft .", "source_lang": "en", "target_lang": "fr"}, headers=auth)
        translated = tr_resp.json()["translated_text"]
        update_resp = client.put(f"/api/annotations/{ann_id}", json={
            "translated_text": translated,
            "target_lang": "fr",
        }, headers=auth)
        assert update_resp.status_code == 200
        updated_data = update_resp.json()
        assert updated_data["translated_text"] == translated
        assert updated_data["ner_tags"] is not None

    def test_document_job_mode_ner_only(self, client, auth, providers):
        doc = upload(client, auth).json()
        job_id = client.post(
            f"/api/documents/{doc['doc_id']}/analyze",
            json={"source_lang": "en", "target_lang": "fr", "mode": "ner"},
            headers=auth
        ).json()["job_id"]
        res = wait_for(client, auth, job_id)
        assert res["status"] == "done"
        assert res["result"]["mode"] == "ner"
        assert res["result"]["translated_chars"] == 0
        assert res["result"]["entity_total"] >= 0

    def test_document_job_mode_translate_only(self, client, auth, providers):
        doc = upload(client, auth).json()
        job_id = client.post(
            f"/api/documents/{doc['doc_id']}/analyze",
            json={"source_lang": "en", "target_lang": "hi", "mode": "translate"},
            headers=auth
        ).json()["job_id"]
        res = wait_for(client, auth, job_id)
        assert res["status"] == "done"
        assert res["result"]["mode"] == "translate"
        assert res["result"]["entity_total"] == 0
        assert res["result"]["translated_chars"] > 0

