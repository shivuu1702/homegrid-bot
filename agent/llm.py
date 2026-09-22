"""The single provider-neutral LLM boundary for HomeGrid.

The rest of the project calls only :func:`ask_llm`. Provider SDKs, endpoint
formats, validation, and transport failures are isolated in this file.
"""

from abc import ABC, abstractmethod
import re

from config import LLM, LLMConfig, PROVIDERS_REQUIRING_BASE_URL, SUPPORTED_PROVIDERS

_SYSTEM_PROMPT = (
    "You are a task planner for a robot navigating a 3-room house. "
    "The house has a kitchen, living room, and dining room. "
    "Objects (bottle, fruit, papers, plates) can be picked up. "
    "Bins (recycling bin, trash bin, compost bin) can store objects after being opened. "
    "Each bin opens with a specific action: pedal, grasp, or lift. "
    "You respond with concise, structured plans only."
)


class LLMError(RuntimeError):
    """A safe, user-facing LLM failure that the agent can recover from."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class LLMConfigurationError(LLMError):
    """Raised before any network call when .env is incomplete or invalid."""


def validate_llm_config(config: LLMConfig = LLM) -> LLMConfig:
    """Validate a provider-neutral config and return it on success."""
    if config.provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise LLMConfigurationError(
            f"Unsupported LLM_PROVIDER '{config.provider}'. Supported providers: {supported}."
        )
    if not config.api_key:
        raise LLMConfigurationError(
            f"LLM_API_KEY is missing for provider '{config.provider}'. Set it in .env."
        )
    if not config.model:
        raise LLMConfigurationError("LLM_MODEL is missing. Set it in .env.")
    if config.provider in PROVIDERS_REQUIRING_BASE_URL and not config.base_url:
        raise LLMConfigurationError(
            f"LLM_BASE_URL is required for provider '{config.provider}'. Set it in .env."
        )
    return config


class LLMAdapter(ABC):
    """Small internal adapter interface; it is never used by agent logic."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Return the model's text response or raise ``LLMError``."""


class OpenAICompatibleAdapter(LLMAdapter):
    """Shared implementation for OpenAI, OpenRouter, and Groq."""

    def complete(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("The 'openai' package is required for this provider. Run: pip install openai") from exc

        kwargs = {"api_key": self.config.api_key}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        try:
            response = OpenAI(**kwargs).chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=400,
            )
            text = response.choices[0].message.content
            if not text or not text.strip():
                raise LLMError("Provider returned an empty response.")
            return text.strip()
        except LLMError:
            raise
        except Exception as exc:
            raise _normalise_provider_error(self.config.provider, exc) from exc


class GeminiAdapter(LLMAdapter):
    """Gemini's native SDK implementation, intentionally separate."""

    def complete(self, prompt: str) -> str:
        try:
            from google import genai
        except ImportError as exc:
            raise LLMError(
                "The 'google-genai' package is required for Gemini. "
                "Run: pip install google-genai"
            ) from exc

        try:
            client = genai.Client(api_key=self.config.api_key)

            response = client.models.generate_content(
                model=self.config.model,
                contents=f"{_SYSTEM_PROMPT}\n\n{prompt}",
            )

            text = getattr(response, "text", None)
            if not text or not text.strip():
                raise LLMError("Provider returned an empty response.")

            return text.strip()

        except LLMError:
            raise
        except Exception as exc:
            raise _normalise_provider_error("gemini", exc) from exc


def get_provider_adapter(config: LLMConfig = LLM) -> LLMAdapter:
    """Select the configured adapter without making a network request."""
    validate_llm_config(config)
    if config.provider == "gemini":
        return GeminiAdapter(config)
    return OpenAICompatibleAdapter(config)


def _normalise_provider_error(provider: str, exc: Exception) -> LLMError:
    """Convert SDK-specific errors to one safe, recoverable application error."""
    status = getattr(exc, "status_code", None)
    if status is None:
        match = re.search(r"\b(401|403|404|408|429|500|502|503|504)\b", str(exc))
        status = int(match.group(1)) if match else None
    messages = {
        401: "Authentication failed. Check LLM_API_KEY.",
        403: "Permission denied. The key may not access this model.",
        404: "Endpoint or model not found. Check LLM_MODEL and LLM_BASE_URL.",
        408: "Provider request timed out. Try again.",
        429: "Provider rate limit reached. Try again later.",
        500: "Provider server error. Try again later.",
        502: "Provider gateway error. Try again later.",
        503: "Provider temporarily unavailable. Try again later.",
        504: "Provider gateway timeout. Try again later.",
    }
    if status in messages:
        return LLMError(
            f"{provider}: {messages[status]}",
            retryable=status in {408, 429, 500, 502, 503, 504},
        )
    return LLMError(f"{provider}: request failed ({type(exc).__name__}).")


def ask_llm(prompt: str) -> str:
    """Ask the configured model. This is the only LLM API used by the agent."""
    return get_provider_adapter().complete(prompt)


def test_llm_connection() -> bool:
    """Optional single-request live check for the current .env provider."""
    try:
        ask_llm("Reply with exactly the word: OK")
        print("[LLM] Connection test passed.")
        return True
    except LLMError as exc:
        print(f"[LLM ERROR] Connection test failed: {exc}")
        return False
