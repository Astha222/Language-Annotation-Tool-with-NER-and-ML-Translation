"""Azure AI Translator (Text Translation REST API v3.0) client.

Configuration comes from environment variables (read at call time, so .env changes need only a restart):

    AZURE_TRANSLATOR_KEY        the resource key (required to enable Azure)
    AZURE_TRANSLATOR_REGION     the resource region, e.g. centralindia (required for regional / multi-service
                                resources, optional for a "Global" resource)
    AZURE_TRANSLATOR_ENDPOINT   optional; default https://api.cognitive.microsofttranslator.com
    AZURE_CHARS_PER_MINUTE      optional send-rate cap, default 30000 (the free F0 tier allows ~33,300/minute)

Limits (Microsoft docs): 50,000 characters per request; F0 is capped at 2,000,000 characters per hour.
"""
import os
import threading
import time
from typing import Callable, Optional

import requests

DEFAULT_ENDPOINT = "https://api.cognitive.microsofttranslator.com"
MAX_REQUEST_CHARS = 50000

# Azure uses script-qualified codes for Chinese
_LANG = {"zh": "zh-Hans"}


class AzureError(Exception):
    """Any failure talking to Azure (network, bad request, 5xx)."""


class AzureThrottled(AzureError):
    """HTTP 429: too many characters/requests too quickly."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class AzureAuthError(AzureError):
    """HTTP 401/403: wrong key/region, or the free quota is used up."""


def is_configured() -> bool:
    return bool(os.getenv("AZURE_TRANSLATOR_KEY", "").strip())


def _endpoint() -> str:
    return (os.getenv("AZURE_TRANSLATOR_ENDPOINT", "").strip() or DEFAULT_ENDPOINT).rstrip("/")


# ── send-rate limiter (token bucket) ─────────────────────────────────────
# Bursts up to one minute of quota, then sustained at the configured rate. Staying under the cap avoids 429s
# and, on the free tier, avoids burning the hourly allowance in one go.
class _Bucket:
    def __init__(self):
        self.lock = threading.Lock()
        self.tokens: Optional[float] = None
        self.stamp = time.monotonic()

    def reserve(self, n: int) -> float:
        """Reserve n characters; return how many seconds the caller must wait before sending."""
        try:
            per_min = max(1000.0, float(os.getenv("AZURE_CHARS_PER_MINUTE", "30000")))
        except ValueError:
            per_min = 30000.0
        rate = per_min / 60.0
        with self.lock:
            now = time.monotonic()
            if self.tokens is None:
                self.tokens = per_min
            self.tokens = min(per_min, self.tokens + (now - self.stamp) * rate)
            self.stamp = now
            self.tokens -= n                       # may go negative: that is reserved future capacity
            return max(0.0, -self.tokens / rate)


_bucket = _Bucket()
_session = requests.Session()


def translate(src: str, tgt: str, text: str, cancelled: Optional[Callable[[], bool]] = None, timeout: float = 30) -> str:
    """Translate one piece of text (up to 50,000 characters). Raises AzureError subclasses on failure."""
    key = os.getenv("AZURE_TRANSLATOR_KEY", "").strip()
    if not key:
        raise AzureAuthError("AZURE_TRANSLATOR_KEY is not set")
    if len(text) > MAX_REQUEST_CHARS:
        raise AzureError(f"Text is longer than {MAX_REQUEST_CHARS} characters")

    wait = _bucket.reserve(len(text))
    end = time.monotonic() + wait
    while time.monotonic() < end:
        if cancelled and cancelled():
            raise AzureError("cancelled")
        time.sleep(min(0.25, max(0.0, end - time.monotonic())))

    headers = {"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/json; charset=UTF-8"}
    region = os.getenv("AZURE_TRANSLATOR_REGION", "").strip()
    if region:
        headers["Ocp-Apim-Subscription-Region"] = region
    params = {"api-version": "3.0", "to": _LANG.get(tgt, tgt)}
    if src and src != "auto":
        params["from"] = _LANG.get(src, src)       # omitted -> Azure auto-detects

    try:
        resp = _session.post(f"{_endpoint()}/translate", params=params, headers=headers,
                             json=[{"Text": text}], timeout=timeout)
    except requests.RequestException as e:
        raise AzureError(f"network error: {type(e).__name__}")

    if resp.status_code == 200:
        try:
            out = resp.json()[0]["translations"][0]["text"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise AzureError("unexpected response format")
        if not out:
            raise AzureError("empty translation")
        return out

    try:
        detail = resp.json().get("error", {}).get("message", "")
    except ValueError:
        detail = resp.text[:200]
    detail = (detail or "").strip()[:200]

    if resp.status_code == 429:
        try:
            retry_after = float(resp.headers.get("Retry-After", ""))
        except ValueError:
            retry_after = None
        raise AzureThrottled(f"HTTP 429: {detail}", retry_after)
    if resp.status_code in (401, 403):
        raise AzureAuthError(f"HTTP {resp.status_code}: {detail}")
    raise AzureError(f"HTTP {resp.status_code}: {detail}")


def check() -> str:
    """Tiny live request used by check_translation.py to validate the key/region/endpoint."""
    return translate("en", "hi", "Hello, how are you?")
