import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from cc_copilot.types.model_choice import ModelChoice

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
PHOENIX_URL: str = os.getenv("PHOENIX_URL", "http://127.0.0.1:6006")
POLICY_DIR: Path = Path(os.getenv("POLICY_DIR", PROJECT_ROOT / "data" / "policies"))


def _get_required_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise ValueError(f"{key} environment variable is required but not set")
    return value


# Settings that need secrets are read lazily, so tests and the offline PII/RAG
# features work without any API key configured.
def get_user_api_key() -> str:
    return _get_required_env("USER_API_KEY")


@lru_cache
def get_default_model_choice() -> ModelChoice:
    from pydantic_ai.models.fallback import FallbackModel
    from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
    from pydantic_ai.providers.google import GoogleProvider

    provider = GoogleProvider(api_key=_get_required_env("GEMINI_API_KEY"))
    model = GoogleModel(_get_required_env("DEFAULT_GOOGLE_MODEL"), provider=provider)
    if fallback_name := os.getenv("FALLBACK_GOOGLE_MODEL"):
        # If the main model is busy or down, pydantic-ai retries the same request on this one.
        model = FallbackModel(model, GoogleModel(fallback_name, provider=provider))
    return ModelChoice(
        model=model,
        model_settings=GoogleModelSettings(timeout=60),  # give up on a single Gemini call after 60 s
    )


def get_judge_model_choice() -> ModelChoice:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    model_name = os.getenv("EVAL_JUDGE_MODEL") or _get_required_env("DEFAULT_GOOGLE_MODEL")
    provider = GoogleProvider(api_key=_get_required_env("GEMINI_API_KEY"))
    return ModelChoice(model=GoogleModel(model_name, provider=provider))
