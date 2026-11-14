"""
FACTORY - builds an Assistant from settings.

    assistant, notice = build_assistant(catalog, store)

With no key (or LLM_PROVIDER=none) you get the rules planner only. With a Gemini key you get the
AI planner with automatic fallback to rules. `notice` is a one-line message that says which one is active.
"""
from app.chat.ai_planner import HybridPlanner
from app.chat.assistant import Assistant
from app.chat.llm_client import GeminiClient
from app.config import load_settings


def build_assistant(catalog, store, settings=None, client=None):
    settings = settings or load_settings()
    if client is None and settings.ai_enabled:
        client = GeminiClient(settings.api_key, settings.model, settings.timeout_seconds)
    if client is None:
        notice = settings.problem or "AI planner is off (LLM_PROVIDER=none). Using the built-in rules planner."
        return Assistant(catalog, store, planner=HybridPlanner(None)), notice
    planner = HybridPlanner(client, mode=getattr(settings, "mode", "ai_first"), max_retries=settings.max_retries)
    model = getattr(client, "model", "model")
    return Assistant(catalog, store, planner=planner), f"AI planner: {model} (checked by the validator, falls back to rules)."
