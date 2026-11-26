"""Settings needed on hosting platforms (Railway / Vercel): the database URL they hand out, and CORS for the frontend's address."""
import re

import pytest


@pytest.mark.parametrize("env, expected", [
    ({"POSTGRES_URL": "postgresql://u:p@h:5432/db"}, "postgresql://u:p@h:5432/db"),
    ({"DATABASE_URL": "postgresql://u:p@h:5432/railway"}, "postgresql://u:p@h:5432/railway"),          # Railway's name for it
    ({"DATABASE_URL": "postgres://u:p@h:5432/db"}, "postgresql://u:p@h:5432/db"),                     # old scheme, refused by SQLAlchemy 2
    ({"POSTGRES_URL": "sqlite:///x.db", "DATABASE_URL": "postgresql://other"}, "sqlite:///x.db"),     # an explicit POSTGRES_URL wins
    ({"POSTGRES_URL": "  ", "DATABASE_URL": ""}, "postgresql://postgres:postgres@localhost:5432/annotationtool"),
])
def test_database_url(main_module, env, expected):
    assert main_module.database_url(env) == expected


def _cors(main_module):
    return next(m for m in main_module.app.user_middleware if m.cls.__name__ == "CORSMiddleware")


def test_cors_stays_strict_by_default(main_module, client):
    assert _cors(main_module).kwargs["allow_origin_regex"] is None
    ask = {"Access-Control-Request-Method": "POST"}
    allowed = client.options("/api/auth/login", headers={"Origin": "http://localhost:8081", **ask})
    refused = client.options("/api/auth/login", headers={"Origin": "https://evil-annotation-tool.vercel.app", **ask})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:8081"
    assert "access-control-allow-origin" not in refused.headers


def test_the_documented_vercel_pattern_is_narrow():
    """The example in main.py / README must allow the project's own Vercel addresses and nothing else."""
    pattern = re.compile(r"^https://annotation-tool(-[a-z0-9-]+)?\.vercel\.app$")
    for ok in ("https://annotation-tool.vercel.app", "https://annotation-tool-git-main-team.vercel.app", "https://annotation-tool-3fk2x9-team.vercel.app"):
        assert pattern.fullmatch(ok)
    for bad in ("http://annotation-tool.vercel.app", "https://annotation-tool.vercel.app.evil.com", "https://evil.com/?annotation-tool.vercel.app",
                "https://my-annotation-tool.vercel.app", "https://annotation-tool.vercel.app/"):
        assert not pattern.fullmatch(bad)


def test_secrets_published_in_the_repository_are_refused(main_module):
    """docker-compose.yml (local development) ships a 45-character default secret; anyone can read it, so it must never sign real tokens."""
    import re
    from pathlib import Path
    compose = (Path(main_module.__file__).resolve().parents[1] / "docker-compose.yml")
    if compose.exists():
        defaults = re.findall(r"JWT_SECRET:\s*\$\{JWT_SECRET:-([^}]*)\}", compose.read_text(encoding="utf-8"))
        assert defaults, "docker-compose.yml changed how it writes JWT_SECRET: update this test so it still checks the default"
        for default in defaults:
            assert default in main_module._UNSAFE_SECRETS
    assert "annotationtool_jwt_secret_dev_key_32chars_min!" in main_module._UNSAFE_SECRETS
