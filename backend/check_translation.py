"""Check which translation services work from this machine.

Run from the backend folder:   python check_translation.py

It reads backend/.env, sends one tiny request ("Hello, how are you?" English -> Hindi) to each service and
prints what happened. Your key is never printed.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))
sys.path.insert(0, str(Path(__file__).parent))

import azure_translator  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")        # Hindi output on Windows consoles

SAMPLE = ("en", "hi", "Hello, how are you?")


def line(name: str, ok: bool, detail: str):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name:9} {detail}")


print("Azure Translator configuration")
key = os.getenv("AZURE_TRANSLATOR_KEY", "").strip()
print(f"  key:      {'set (' + str(len(key)) + ' characters)' if key else 'NOT SET'}")
print(f"  region:   {os.getenv('AZURE_TRANSLATOR_REGION', '').strip() or '(not set - fine for a Global resource)'}")
print(f"  endpoint: {os.getenv('AZURE_TRANSLATOR_ENDPOINT', '').strip() or azure_translator.DEFAULT_ENDPOINT}")
print(f"  rate cap: {os.getenv('AZURE_CHARS_PER_MINUTE', '30000')} characters/minute")
print("\nLive test (English -> Hindi)")

if key:
    try:
        line("Azure", True, azure_translator.translate(*SAMPLE))
    except azure_translator.AzureAuthError as e:
        hint = "check the key and the REGION (a regional resource needs AZURE_TRANSLATOR_REGION)"
        if "403" in str(e):
            hint += "; 403 can also mean the free quota is used up"
        line("Azure", False, f"{e}\n{'':17}-> {hint}")
    except Exception as e:
        line("Azure", False, str(e))
else:
    line("Azure", False, "skipped - set AZURE_TRANSLATOR_KEY in backend/.env")

try:
    import main  # noqa: E402  (imports the app: needs the databases from .env to be reachable)
except Exception as e:
    print(f"\n  (could not load main.py to test the free services: {type(e).__name__}: {str(e)[:100]})")
    sys.exit(0)

for name, fn in (("Google", main._google_translate), ("MyMemory", main._mymemory_translate)):
    try:
        line(name, True, fn(*SAMPLE))
    except Exception as e:
        line(name, False, str(e)[:150])
