"""Configuration for the HomeGrid agent.

Only this module reads LLM environment variables. Application code receives
one ``LLMConfig`` and never needs to know a provider's API details.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)

SUPPORTED_PROVIDERS = frozenset({"openai", "gemini", "openrouter", "groq"})
# OpenRouter and Groq expose OpenAI-compatible APIs, but their endpoint must be
# explicit so a provider switch is fully represented by the .env file.
PROVIDERS_REQUIRING_BASE_URL = frozenset({"openrouter", "groq"})


@dataclass(frozen=True)
class LLMConfig:
    """Provider-neutral settings consumed by the LLM abstraction."""

    provider: str
    model: str
    api_key: str
    base_url: Optional[str] = None


def load_llm_config(env: Optional[Mapping[str, str]] = None) -> LLMConfig:
    """Load LLM settings from a mapping (normally ``os.environ``).

    ``OPENAI_API_KEY`` and ``OPENAI_MODEL`` remain read-only legacy fallbacks
    for existing installations. New configurations should use only ``LLM_*``.
    """
    values = os.environ if env is None else env
    provider = values.get("LLM_PROVIDER", "").strip().lower()
    model = values.get("LLM_MODEL", "").strip() or values.get("OPENAI_MODEL", "").strip()
    api_key = values.get("LLM_API_KEY", "").strip() or values.get("OPENAI_API_KEY", "").strip()
    base_url = values.get("LLM_BASE_URL", "").strip() or None

    # Preserve pre-abstraction Gemini .env files that stored a Gemini model in
    # OPENAI_MODEL. Explicit LLM_PROVIDER always takes precedence.
    if not provider and model.lower().startswith("gemini"):
        provider = "gemini"
    if not provider:
        provider = "openai"  # legacy OpenAI-only configuration

    return LLMConfig(provider=provider, model=model, api_key=api_key, base_url=base_url)


LLM = load_llm_config()
# Compatibility constants for the startup banner and older local scripts.
LLM_PROVIDER = LLM.provider
LLM_MODEL = LLM.model
LLM_API_KEY = LLM.api_key
LLM_BASE_URL = LLM.base_url
OPENAI_API_KEY = LLM_API_KEY
OPENAI_MODEL = LLM_MODEL

MAX_RETRIES = 2
MAX_STEPS_PER_RUN = 200
MAX_NAV_STEPS = 150

ENV_ID = "homegrid-task"
ENV_MAX_STEPS = 100
CURRICULUM_TASK_TYPES = ["find", "get", "open", "cleanup", "rearrange"]

SKILLS_FILE = PROJECT_ROOT / "skills" / "skills.json"
LOGS_DIR = PROJECT_ROOT / "logs"

# Visual rendering delay (seconds) between actions — set via .env or default.
HOMEGRID_RENDER_DELAY = float(os.getenv("HOMEGRID_RENDER_DELAY", "0.55"))
