"""Configuration for the HomeGrid agent.

Only this module reads LLM environment variables. Application code receives
one ``LLMConfig`` and never needs to know a provider's API details.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)

SUPPORTED_PROVIDERS = frozenset({"openai", "gemini", "openrouter", "groq", "ollama"})
# OpenRouter and Groq expose OpenAI-compatible APIs, but their endpoint must be
# explicit so a provider switch is fully represented by the .env file.
PROVIDERS_REQUIRING_BASE_URL = frozenset({"openrouter", "groq", "ollama"})

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


@dataclass(frozen=True)
class LLMConfig:
    """Provider-neutral settings consumed by the LLM abstraction."""

    provider: str
    model: str
    api_key: str
    base_url: Optional[str] = None


def load_llm_config(env: Optional[Mapping[str, str]] = None) -> LLMConfig:
    """Load LLM settings from a mapping (normally ``os.environ``).

    Supports unified ``LLM_*`` variables, provider-specific variables
    (``GEMINI_API_KEY``, ``GEMINI_MODEL``), and legacy ``OPENAI_*`` fallbacks.
    Explicit ``LLM_PROVIDER`` always takes precedence.
    """
    values = os.environ if env is None else env
    provider = values.get("LLM_PROVIDER", "").strip().lower()

    # Raw model and candidate values
    raw_model = values.get("LLM_MODEL", "").strip() or values.get("OPENAI_MODEL", "").strip()
    gemini_model = values.get("GEMINI_MODEL", "").strip()

    # Provider auto-detection if LLM_PROVIDER is not explicitly specified
    if not provider:
        if raw_model.lower().startswith("gemini") or gemini_model:
            provider = "gemini"
        elif values.get("GEMINI_API_KEY", "").strip() and not (
            values.get("LLM_API_KEY", "").strip() or values.get("OPENAI_API_KEY", "").strip()
        ):
            provider = "gemini"
        else:
            provider = "openai"  # legacy OpenAI-only configuration

    # Resolve model and api_key according to selected provider
    if provider == "gemini":
        model = gemini_model or raw_model or DEFAULT_GEMINI_MODEL
        api_key = (
            values.get("GEMINI_API_KEY", "").strip()
            or values.get("LLM_API_KEY", "").strip()
            or values.get("OPENAI_API_KEY", "").strip()
        )
    else:
        model = raw_model
        api_key = values.get("LLM_API_KEY", "").strip() or values.get("OPENAI_API_KEY", "").strip()

    base_url = values.get("LLM_BASE_URL", "").strip() or None

    return LLMConfig(provider=provider, model=model, api_key=api_key, base_url=base_url)


LLM = load_llm_config()
# Compatibility constants for the startup banner and older local scripts.
LLM_PROVIDER = LLM.provider
LLM_MODEL = LLM.model
LLM_API_KEY = LLM.api_key
LLM_BASE_URL = LLM.base_url
OPENAI_API_KEY = LLM_API_KEY
OPENAI_MODEL = LLM_MODEL
GEMINI_API_KEY = LLM.api_key if LLM.provider == "gemini" else os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = LLM.model if LLM.provider == "gemini" else os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

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

# Linear SMDP Q-Learning Configuration
SMDP_POLICY_FILE = PROJECT_ROOT / "skills" / "smdp_policy.json"
SMDP_GAMMA = float(os.getenv("SMDP_GAMMA", "0.95"))
SMDP_ALPHA = float(os.getenv("SMDP_ALPHA", "0.05"))
SMDP_EPSILON = float(os.getenv("SMDP_EPSILON", "0.15"))
