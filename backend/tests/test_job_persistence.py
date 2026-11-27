"""M6: document jobs are saved in the database, survive a server restart, and unfinished ones are restarted once."""
import json
import time

from test_api import upload, wait_for


def _start(client, auth, target="hi"):
    doc = upload(client, auth).json()
    r = client.post(f"/api/documents/{doc['doc_id']}/analyze", json={"source_lang": "en", "target_lang": target}, headers=auth)
    return doc["doc_id"], r.json()["job_id"]


def _row(m, job_id):
    db = m.SessionLocal()
    try:
        return db.get(m.JobDB, job_id)
    finally:
        db.close()


def test_a_finished_job_can_still_be_read_after_a_restart(client, auth, providers, main_module):
    _, job_id = _start(client, auth)
    done = wait_for(client, auth, job_id)
    assert done["status"] == "done"
    main_module.JOBS.clear()                                     # what a restart does to the in-memory copy
    again = client.get(f"/api/jobs/{job_id}", headers=auth).json()
    assert again["status"] == "done" and again["result"]["annotation_id"] == done["result"]["annotation_id"]
    assert again["result"]["entity_total"] == done["result"]["entity_total"]


def test_the_saved_row_follows_the_job(client, auth, providers, main_module):
    _, job_id = _start(client, auth)
    wait_for(client, auth, job_id)
    row = _row(main_module, job_id)
    assert row.status == "done" and row.progress == 1.0 and json.loads(row.result)["annotation_id"] > 0 and row.attempts == 1


def test_an_interrupted_job_is_restarted_once_at_startup(client, auth, providers, main_module):
    m = main_module
    doc_id, job_id = _start(client, auth)
    wait_for(client, auth, job_id)
    # pretend the server stopped while the job was running
    db = m.SessionLocal()
    row = db.get(m.JobDB, job_id)
    row.status, row.stage, row.progress, row.result = "running", "Translating 40%", 0.4, None
    db.commit()
    db.close()
    m.JOBS.clear()
    assert m.recover_jobs() == {"restarted": 1, "failed": 0}
    job = wait_for(client, auth, job_id)
    assert job["status"] == "done" and _row(m, job_id).attempts == 2


def test_a_job_is_not_restarted_twice_or_without_its_document(client, auth, providers, main_module):
    m = main_module
    doc_id, job_id = _start(client, auth)
    wait_for(client, auth, job_id)
    db = m.SessionLocal()
    row = db.get(m.JobDB, job_id)
    row.status, row.attempts = "running", m.MAX_JOB_ATTEMPTS        # already restarted once
    other = m.JobDB(id="f" * 32, user_id=row.user_id, doc_id="e" * 32, filename="gone.txt", status="queued",
                    stage="Queued", progress=0.0, warnings="[]", attempts=1, created=time.time())   # its document no longer exists
    db.add(other)
    db.commit()
    db.close()
    m.JOBS.clear()
    assert m.recover_jobs() == {"restarted": 0, "failed": 2}
    first = client.get(f"/api/jobs/{job_id}", headers=auth).json()
    second = client.get("/api/jobs/" + "f" * 32, headers=auth).json()
    assert first["status"] == "error" and "already restarted once" in first["error"]
    assert second["status"] == "error" and "expired" in second["error"]


def test_list_shows_only_my_jobs_newest_first(client, auth, auth2, providers):
    _, first = _start(client, auth)
    wait_for(client, auth, first)
    _, second = _start(client, auth)
    wait_for(client, auth, second)
    _, theirs = _start(client, auth2)
    wait_for(client, auth2, theirs)
    mine = client.get("/api/jobs", headers=auth).json()["jobs"]
    assert [j["id"] for j in mine] == [second, first]
    assert all("user_id" not in j for j in mine)                 # the public view does not leak internals


def test_cancelling_a_finished_job_is_harmless(client, auth, providers, main_module):
    _, job_id = _start(client, auth)
    wait_for(client, auth, job_id)
    main_module.JOBS.clear()
    r = client.post(f"/api/jobs/{job_id}/cancel", headers=auth)
    assert r.status_code == 200 and "not running" in r.json()["message"]


def test_progress_writes_are_throttled_but_status_changes_are_not(main_module, monkeypatch):
    m = main_module
    monkeypatch.setattr(m, "JOB_SAVE_INTERVAL", 3600)
    db = m.SessionLocal()
    user = m.UserDB(username="throttle_u", email="throttle@example.com", hashed_password="x")
    db.add(user)
    db.commit()
    uid = user.id
    db.close()
    job = {"id": "a" * 32, "user_id": uid, "doc_id": "b" * 32, "filename": "x.txt", "source_lang": "en", "target_lang": "hi",
           "status": "running", "stage": "Reading", "progress": 0.1, "error": None, "warnings": [], "result": None, "attempts": 1,
           "created": time.time()}
    m._save_job(job, force=True)
    job["stage"], job["progress"] = "Translating 50%", 0.5
    m._save_job(job)                                             # within the interval: skipped
    assert _row(m, job["id"]).progress == 0.1
    job["status"] = "done"
    m._save_job(job, force=True)
    assert _row(m, job["id"]).status == "done" and _row(m, job["id"]).progress == 0.5
