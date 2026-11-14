"""
SETTINGS - read from environment variables (and an optional .env file).

Secrets (the API key) live ONLY in the environment or the .env file. The .env file is
git-ignored; .env.example shows the names without real values. The key is never logged,
never put in a URL, and never stored in chat history.
"""
import os
from dataclasses import dataclass
from pathlib import Path


def load_env_file(path=".env"):
    """Tiny .env reader: KEY=VALUE lines. Existing environment variables win."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split(" #")[0].strip().strip('"').strip("'")     # allow trailing comments
        os.environ.setdefault(key.strip(), value)


@dataclass
class Settings:
    provider: str = "none"            # none | gemini
    api_key: str = ""
    model: str = "gemini-3.5-flash"   # change in .env if Google renames it
    timeout_seconds: float = 15.0
    max_retries: int = 2
    mode: str = "ai_first"           # ai_first | rules_first (rules first, model only when rules do not understand)
    problem: str = ""                 # why AI is off even though it was requested (shown to the user)

    @property
    def ai_enabled(self):
        return self.provider == "gemini" and bool(self.api_key)


def _float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return int(default)


def load_settings(env_file=".env"):
    load_env_file(env_file)
    provider = os.environ.get("LLM_PROVIDER", "none").strip().lower() or "none"
    key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY") or "").strip()
    s = Settings(provider=provider, api_key=key,
                 model=os.environ.get("LLM_MODEL", "gemini-3.5-flash").strip() or "gemini-3.5-flash",
                 timeout_seconds=max(1.0, min(_float("LLM_TIMEOUT_SECONDS", 15), 60)),
                 max_retries=max(0, min(_int("LLM_MAX_RETRIES", 2), 5)),
                 mode="rules_first" if os.environ.get("LLM_MODE", "").strip().lower() == "rules_first" else "ai_first")
    if provider not in ("none", "gemini"):
        s.problem = f"LLM_PROVIDER '{provider}' is not supported. Use 'none' or 'gemini'. Running with rules only."
        s.provider = "none"
    elif provider == "gemini" and not key:
        s.problem = "LLM_PROVIDER=gemini but no GEMINI_API_KEY is set. Running with rules only."
    return s
