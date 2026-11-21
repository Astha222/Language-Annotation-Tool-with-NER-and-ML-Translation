"""The Azure client, tested against a local stand-in server that follows Microsoft's documented REST API."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

import azure_translator as az

SECRET = "SECRET-KEY-1234567890"


@pytest.fixture
def azure_server(monkeypatch):
    state = {"requests": [], "queue": []}                       # queue: status codes to return before 200s

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            query = parse_qs(urlparse(self.path).query)
            state["requests"].append({"path": urlparse(self.path).path, "query": query,
                                      "headers": dict(self.headers), "body": body})
            code = state["queue"].pop(0) if state["queue"] else 200
            if code == 200:
                payload = [{"translations": [{"text": "AZ:" + item["Text"], "to": query["to"][0]}]} for item in body]
                extra = {}
            else:
                payload = {"error": {"code": code * 1000 + 1, "message": f"fake error {code}"}}
                extra = {"Retry-After": "1"} if code == 429 else {}
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            for k, v in extra.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("AZURE_TRANSLATOR_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("AZURE_TRANSLATOR_KEY", SECRET)
    monkeypatch.setenv("AZURE_TRANSLATOR_REGION", "eastasia")
    monkeypatch.setenv("AZURE_CHARS_PER_MINUTE", "6000000")
    monkeypatch.setattr(az, "_bucket", az._Bucket())
    yield state
    server.shutdown()


def test_request_follows_the_documented_format(azure_server):
    assert az.translate("en", "hi", "Hello world") == "AZ:Hello world"
    r = azure_server["requests"][0]
    assert r["path"] == "/translate"
    assert r["query"] == {"api-version": ["3.0"], "from": ["en"], "to": ["hi"]}
    assert r["headers"]["Ocp-Apim-Subscription-Key"] == SECRET
    assert r["headers"]["Ocp-Apim-Subscription-Region"] == "eastasia"
    assert "application/json" in r["headers"]["Content-Type"] and "utf-8" in r["headers"]["Content-Type"].lower()
    assert r["body"] == [{"Text": "Hello world"}]


def test_chinese_uses_script_qualified_codes(azure_server):
    az.translate("zh", "en", "你好")
    az.translate("en", "zh", "Hello")
    assert azure_server["requests"][0]["query"]["from"] == ["zh-Hans"]
    assert azure_server["requests"][1]["query"]["to"] == ["zh-Hans"]


def test_auto_source_is_left_to_azure(azure_server):
    az.translate("auto", "fr", "Hello")
    assert "from" not in azure_server["requests"][0]["query"]


def test_region_header_is_optional_for_global_resources(azure_server, monkeypatch):
    monkeypatch.setenv("AZURE_TRANSLATOR_REGION", "")
    az.translate("en", "fr", "Hello")
    assert "Ocp-Apim-Subscription-Region" not in azure_server["requests"][0]["headers"]


def test_non_latin_text_round_trips_as_utf8(azure_server):
    assert az.translate("hi", "en", "नमस्ते दुनिया") == "AZ:नमस्ते दुनिया"


@pytest.mark.parametrize("code,exc", [(429, az.AzureThrottled), (401, az.AzureAuthError), (403, az.AzureAuthError),
                                      (500, az.AzureError), (400, az.AzureError)])
def test_http_errors_map_to_exceptions(azure_server, code, exc):
    azure_server["queue"] = [code]
    with pytest.raises(exc) as e:
        az.translate("en", "hi", "x")
    assert type(e.value) is exc


def test_retry_after_is_parsed(azure_server):
    azure_server["queue"] = [429]
    with pytest.raises(az.AzureThrottled) as e:
        az.translate("en", "hi", "x")
    assert e.value.retry_after == 1.0


def test_oversized_text_is_rejected_locally(azure_server):
    with pytest.raises(az.AzureError, match="50000"):
        az.translate("en", "hi", "a" * 50001)
    assert azure_server["requests"] == []


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.setenv("AZURE_TRANSLATOR_KEY", "")
    assert not az.is_configured()
    with pytest.raises(az.AzureAuthError):
        az.translate("en", "hi", "x")


def test_rate_limiter_allows_a_burst_then_paces(monkeypatch):
    monkeypatch.setenv("AZURE_CHARS_PER_MINUTE", "60000")          # 1,000 chars/second, bucket of one minute
    bucket = az._Bucket()
    assert bucket.reserve(30000) == 0 and bucket.reserve(30000) == 0
    assert 4.5 <= bucket.reserve(5000) <= 5.5                       # over quota: waits ~5 s


def test_key_never_appears_in_error_messages(azure_server):
    azure_server["queue"] = [401]
    with pytest.raises(az.AzureAuthError) as e:
        az.translate("en", "hi", "x")
    assert SECRET not in str(e.value)
