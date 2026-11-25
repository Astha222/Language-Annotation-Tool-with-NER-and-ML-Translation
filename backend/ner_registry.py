"""Registry of the entity-recognition models: which models exist and which one each language uses.

The bundled list is backend/ner_models.json. Optional changes, in this order (later wins):
  1. NER_MODELS_FILE=<path to a JSON file>   same shape as ner_models.json; its models and routes are added to / replace the bundled ones
  2. NER_MODEL_EN / NER_MODEL_HI / NER_MODEL_MULTI   change the English / Indian-language / every-other-language route
     (a registry name, a Hugging Face model id or a local folder)

Optional fields of a model entry: "fallback" (another entry or source, used if this one cannot be loaded); "revision" (a Hugging Face commit to pin);
"trust_remote_code": true (the model runs Python code that lives in its repository; ONLY allowed together with a "revision" that is a full 40-character commit hash,
so a later push to the model's page cannot change what runs; a deployment can refuse all such models with NER_ALLOW_REMOTE_CODE=0);
"loader": "otter" (a zero-shot model that is not a normal token-classification model, see otter_ner.py) and its "threshold".
A model that is only given by NER_MODEL_* as an id or folder, and is not in the registry, never gets remote code.

Route keys: a language code ("en", "fr", ...) is that language only; "indic" is the group of Indian languages (INDIC_NER_LANGS in main.py);
"*" is every language without a more specific route.

    python backend/ner_registry.py        prints the routes and checks the setup
"""
import json
import os
import re
from pathlib import Path
from typing import Optional

BUNDLED = Path(__file__).with_name("ner_models.json")
GROUPS = ("en", "indic", "*")
LEGACY_ENV = {"en": "NER_MODEL_EN", "indic": "NER_MODEL_HI", "*": "NER_MODEL_MULTI"}
LOADERS = ("transformers", "otter")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"NER model registry {path}: cannot read it ({e})") from e
    if not isinstance(data, dict):
        raise ValueError(f"NER model registry {path}: the top level must be a JSON object")
    return data


class Registry:
    def __init__(self, models: dict, routes: dict):
        for name, entry in models.items():
            if not isinstance(entry, dict) or not str(entry.get("source", "")).strip():
                raise ValueError(f"NER model registry: model '{name}' needs a non-empty 'source' (a Hugging Face id or a folder)")
            loader = entry.get("loader", "transformers")
            if loader not in LOADERS:
                raise ValueError(f"NER model registry: model '{name}': 'loader' must be one of {', '.join(LOADERS)}")
            if loader == "otter" and entry.get("trust_remote_code") is not True:
                raise ValueError(f"NER model registry: model '{name}' uses the otter loader, which runs code from the model's repository: "
                                 f"set \"trust_remote_code\": true to confirm that this is wanted")
            if entry.get("trust_remote_code") and not _COMMIT.match(str(entry.get("revision", ""))):
                raise ValueError(f"NER model registry: model '{name}' runs remote code, so it needs a \"revision\" that is a full 40-character commit hash "
                                 f"(the commit that was reviewed), never a branch name")
        for group in GROUPS:
            if not str(routes.get(group, "")).strip():
                raise ValueError(f"NER model registry: routes need an entry for '{group}'")
        self.models = models
        self.routes = {k: str(v).strip() for k, v in routes.items()}

    def entry(self, ref: str) -> dict:
        """A registry name gives its entry; anything else is taken as a Hugging Face model id or a local folder."""
        ref = str(ref).strip()
        if ref in self.models:
            return {"name": ref, **self.models[ref]}
        return {"name": None, "source": ref}

    def source(self, ref: str) -> str:
        return str(self.entry(ref)["source"]).strip()

    def group_source(self, group: str) -> str:
        return self.source(self.routes[group])

    def language_routes(self) -> dict:
        """Routes for single languages other than English: {code: source}."""
        return {k: self.source(v) for k, v in self.routes.items() if k not in GROUPS}

    def options_for(self, source: str) -> dict:
        """The registry entry whose source is `source` (loader, revision, trust_remote_code, threshold ...); {} for a model that is not in the registry."""
        for entry in self.models.values():
            if str(entry["source"]).strip() == source:
                return entry
        return {}

    def variant_tag(self, source: str) -> str:
        """Settings that change a model's output without changing its source (pinned commit, score threshold); part of the result-cache key. Empty for ordinary models."""
        o = self.options_for(source)
        parts = [f"rev={o['revision']}" if o.get("revision") else "", f"thr={o['threshold']}" if o.get("threshold") is not None else ""]
        return ";".join(p for p in parts if p)

    def fallback_of(self, source: str) -> Optional[str]:
        """The source to use when `source` cannot be loaded, or None."""
        for name, entry in self.models.items():
            if str(entry["source"]).strip() == source and str(entry.get("fallback", "")).strip():
                fb = self.source(entry["fallback"])
                return fb if fb != source else None
        return None

    def describe(self, indic_langs=()) -> list:
        """One row per route, for GET /health/models and the command line."""
        rows = []
        for key, ref in self.routes.items():
            e = self.entry(ref)
            rows.append({"route": key, "model": e["name"], "source": self.source(ref), "licence": e.get("licence"),
                         "fallback": self.fallback_of(self.source(ref)),
                         "languages": sorted(indic_langs) if key == "indic" else None})
        return rows


def load(env=None, bundled: Path = BUNDLED) -> Registry:
    env = os.environ if env is None else env
    data = _read(bundled)
    models, routes = dict(data.get("models", {})), dict(data.get("routes", {}))
    extra = str(env.get("NER_MODELS_FILE", "")).strip()
    if extra:
        more = _read(Path(extra))
        models.update(more.get("models", {}))
        routes.update(more.get("routes", {}))
    for group, var in LEGACY_ENV.items():
        value = str(env.get(var, "")).strip()
        if value:
            routes[group] = value
    return Registry(models, routes)


if __name__ == "__main__":
    reg = load()
    problems = 0
    print(f"{'route':<8} {'model':<22} {'source':<52} licence")
    for row in reg.describe():
        print(f"{row['route']:<8} {str(row['model']):<22} {row['source']:<52} {row['licence']}")
        src = Path(row["source"])
        looks_local = src.is_absolute() or row["source"].startswith((".", "~")) or "\\" in row["source"]
        if looks_local and not src.exists():
            problems += 1
            print(f"         ^ folder not found: {src}")
        if row["fallback"]:
            print(f"         ^ falls back to {row['fallback']} if it cannot be loaded")
    print("OK" if not problems else f"{problems} problem(s)")
    raise SystemExit(1 if problems else 0)
