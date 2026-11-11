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
    if not config.api_key and config.provider != "ollama":
        if config.provider == "gemini":
            raise LLMConfigurationError(
                "GEMINI_API_KEY (or LLM_API_KEY) is missing for provider 'gemini'. Set it in .env."
            )
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

    @abstractmethod
    def test_connection(self) -> bool:
        """Verify connectivity to the provider endpoint or raise ``LLMError``."""


class OpenAICompatibleAdapter(LLMAdapter):
    """Shared implementation for OpenAI, OpenRouter, and Groq."""

    def complete(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("The 'openai' package is required for this provider. Run: pip install openai") from exc

        kwargs = {"api_key": self.config.api_key or "ollama"}
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

    def test_connection(self) -> bool:
        """Verify OpenAI-compatible endpoint connectivity."""
        self.complete("Reply with exactly the word: OK")
        return True


class GeminiAdapter(LLMAdapter):
    """Google Gemini native SDK implementation using official google-genai."""

    def complete(self, prompt: str) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMError(
                "The 'google-genai' package is required for Gemini. "
                "Run: pip install google-genai"
            ) from exc

        if not self.config.api_key:
            raise LLMError(
                "Gemini API key is missing. Set GEMINI_API_KEY (or LLM_API_KEY) in .env."
            )

        model = (self.config.model or "").strip()
        if not model:
            raise LLMError("GEMINI_MODEL (or LLM_MODEL) is missing. Set it in .env.")

        try:
            client = genai.Client(api_key=self.config.api_key)
            config = types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=400,
            )
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        except LLMError:
            raise
        except Exception as exc:
            raise _normalise_provider_error("gemini", exc) from exc

        # Extract generated text safely
        try:
            text = getattr(response, "text", None)
        except Exception:
            text = None

        if not text or not text.strip():
            raise LLMError("Provider returned an empty response.")

        return text.strip()

    def test_connection(self) -> bool:
        """Dedicated Gemini connection check using google.genai.Client and generate_content."""
        try:
            from google import genai
        except ImportError as exc:
            raise LLMError(
                "The 'google-genai' package is required for Gemini. "
                "Run: pip install google-genai"
            ) from exc

        if not self.config.api_key:
            raise LLMError(
                "GEMINI_API_KEY (or LLM_API_KEY) is missing for provider 'gemini'. Set it in .env."
            )

        model = (self.config.model or "").strip()
        if not model:
            raise LLMError("GEMINI_MODEL (or LLM_MODEL) is missing. Set it in .env.")

        try:
            client = genai.Client(api_key=self.config.api_key)
            # Minimal generate_content ping without complex task-planning parameters
            response = client.models.generate_content(
                model=model,
                contents="ping",
            )
        except LLMError:
            raise
        except Exception as exc:
            raise _normalise_provider_error("gemini", exc) from exc

        # Safe verification of response structure
        text = None
        try:
            text = getattr(response, "text", None)
        except Exception:
            text = None

        has_candidates = bool(getattr(response, "candidates", None))
        if not text and not has_candidates:
            raise LLMError("Provider returned an empty response.")

        return True


def get_provider_adapter(config: LLMConfig = LLM) -> LLMAdapter:
    """Select the configured adapter without making a network request."""
    validate_llm_config(config)
    if config.provider == "gemini":
        return GeminiAdapter(config)
    return OpenAICompatibleAdapter(config)


def _normalise_provider_error(provider: str, exc: Exception) -> LLMError:
    """Convert SDK-specific errors to one safe, recoverable application error.

    Never echoes raw exception contents or URLs that might leak API keys.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        code_attr = getattr(exc, "code", None)
        if isinstance(code_attr, int):
            status = code_attr

    exc_str = str(exc).upper()
    if status is None:
        match = re.search(r"\b(400|401|403|404|408|429|500|502|503|504)\b", exc_str)
        status = int(match.group(1)) if match else None

    # Handle explicit API status names from Gemini / OpenAI if numeric status was not found
    if status is None:
        if "RESOURCE_EXHAUSTED" in exc_str or "RATE_LIMIT" in exc_str or "QUOTA" in exc_str:
            status = 429
        elif "API_KEY_INVALID" in exc_str or "INVALID_API_KEY" in exc_str:
            status = 401
        elif "PERMISSION_DENIED" in exc_str:
            status = 403
        elif "NOT_FOUND" in exc_str:
            status = 404

    key_hint = "GEMINI_API_KEY (or LLM_API_KEY)" if provider == "gemini" else "LLM_API_KEY"
    model_hint = (
        "GEMINI_MODEL (or LLM_MODEL)"
        if provider == "gemini"
        else "LLM_MODEL and LLM_BASE_URL"
    )
    not_found_msg = (
        f"Model not found or unsupported for generateContent. Check {model_hint}."
        if provider == "gemini"
        else f"Endpoint or model not found. Check {model_hint}."
    )

    messages = {
        400: f"Invalid request or API key. Check {key_hint}.",
        401: f"Authentication failed. Check {key_hint}.",
        403: "Permission denied. The key may not access this model.",
        404: not_found_msg,
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


def test_llm_connection(config: LLMConfig = LLM, log=print) -> bool:
    """Optional single-request live check for the configured provider."""
    try:
        adapter = get_provider_adapter(config)
        adapter.test_connection()
        log("[LLM] Connection test passed.")
        return True
    except LLMError as exc:
        log(f"[LLM ERROR] Connection test failed: {exc}")
        return False
